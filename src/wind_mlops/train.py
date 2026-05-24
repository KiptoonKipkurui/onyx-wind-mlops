from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from wind_mlops.config import ARTIFACT_DIR, DEFAULT_CONFIG, MLRUNS_DIR, RAW_DATA_DIR
from wind_mlops.data import build_multiclass_training_frame, load_penmanshiel_dataset


def candidate_classifiers(random_state: int) -> dict[str, ClassifierMixin]:
    return {
        "multinomial_logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            multi_class="multinomial",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }


def build_pipeline(model_name: str, classifier: ClassifierMixin) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "multinomial_logistic_regression":
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps=steps)


def model_metrics(
    pipeline: Pipeline,
    X_test,
    y_test: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "top_2_accuracy": float(top_k_accuracy_score(y_test, probabilities, k=2, labels=labels)),
    }
    return metrics, predictions, probabilities


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_profile(data_dir: Path) -> dict[str, object]:
    files = sorted(data_dir.rglob("*.csv"))
    return {
        "source_root": str(data_dir),
        "csv_count": len(files),
        "csv_sha256": {str(path): sha256_file(path) for path in files},
    }


def train(data_dir: Path | None = None) -> Path:
    cfg = DEFAULT_CONFIG
    source = data_dir or RAW_DATA_DIR
    combined, statuses = load_penmanshiel_dataset(source, max_rows_per_turbine=cfg.max_rows_per_turbine)
    X, y_labels, feature_names, _, _ = build_multiclass_training_frame(
        combined,
        future_steps=cfg.future_steps,
        min_class_rows=cfg.min_class_rows,
    )

    if len(X) < cfg.min_rows:
        raise ValueError(f"Need at least {cfg.min_rows} rows after cleaning; got {len(X)}.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    class_names = label_encoder.classes_.tolist()
    class_to_id = {name: int(label_encoder.transform([name])[0]) for name in class_names}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(cfg.experiment_name)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        labels = np.arange(len(class_names))
        comparison: list[dict[str, object]] = []
        best_name = ""
        best_pipeline: Pipeline | None = None
        best_predictions: np.ndarray | None = None
        best_metrics: dict[str, float] | None = None

        for model_name, classifier in candidate_classifiers(cfg.random_state).items():
            pipeline = build_pipeline(model_name, classifier)
            pipeline.fit(X_train, y_train)
            candidate_metrics, predictions, _ = model_metrics(pipeline, X_test, y_test, labels)
            comparison.append({"model_name": model_name, **candidate_metrics})
            mlflow.log_metrics(
                {f"{model_name}_{key}": value for key, value in candidate_metrics.items()}
            )

            if best_metrics is None or (
                candidate_metrics["balanced_accuracy"],
                candidate_metrics["top_2_accuracy"],
            ) > (
                best_metrics["balanced_accuracy"],
                best_metrics["top_2_accuracy"],
            ):
                best_name = model_name
                best_pipeline = pipeline
                best_predictions = predictions
                best_metrics = candidate_metrics

        if best_pipeline is None or best_predictions is None or best_metrics is None:
            raise RuntimeError("No candidate classifiers were trained.")

        comparison = sorted(
            comparison,
            key=lambda row: (row["balanced_accuracy"], row["top_2_accuracy"]),
            reverse=True,
        )
        metrics = {
            **best_metrics,
            "rows": int(len(X)),
            "feature_count": int(len(feature_names)),
            "class_count": int(len(class_names)),
        }
        mlflow.log_params(
            {
                "model_type": best_name,
                "candidate_models": ",".join(candidate_classifiers(cfg.random_state).keys()),
                "source_root": str(source),
                "future_steps": cfg.future_steps,
                "min_class_rows": cfg.min_class_rows,
                "features": ",".join(feature_names),
                "class_names": ",".join(class_names),
            }
        )
        mlflow.log_metrics(metrics)

        model_path = ARTIFACT_DIR / "model.joblib"
        label_encoder_path = ARTIFACT_DIR / "label_encoder.joblib"
        metadata_path = ARTIFACT_DIR / "metadata.json"
        report_path = ARTIFACT_DIR / "classification_report.json"
        comparison_path = ARTIFACT_DIR / "model_comparison.json"
        data_profile_path = ARTIFACT_DIR / "data_profile.json"

        joblib.dump(best_pipeline, model_path)
        joblib.dump(label_encoder, label_encoder_path)
        data_profile = {
            **source_profile(source),
            "scada_row_count": int(len(combined)),
            "status_row_count_after_dedup": int(len(statuses)),
            "training_row_count": int(len(X)),
            "timestamp_min": str(combined["timestamp"].min()),
            "timestamp_max": str(combined["timestamp"].max()),
            "feature_names": feature_names,
            "class_counts": y_labels.value_counts().to_dict(),
        }
        metadata = {
            "run_id": run.info.run_id,
            "model_name": cfg.registered_model_name,
            "model_type": best_name,
            "candidate_models": [row["model_name"] for row in comparison],
            "task": "multiclass_classification",
            "target": "next_event_type_60m",
            "target_definition": (
                "Highest-priority event/status type active within the next six 10-minute SCADA "
                "rows for the same turbine; rows already in major events excluded."
            ),
            "feature_names": feature_names,
            "input_name": "float_input",
            "label_output": "label",
            "probability_output": "probabilities",
            "class_names": class_names,
            "class_to_id": class_to_id,
            "positive_class": 1,
            "metrics": metrics,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        report_path.write_text(
            json.dumps(
                classification_report(
                    y_test,
                    best_predictions,
                    labels=labels,
                    target_names=class_names,
                    output_dict=True,
                    zero_division=0,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        data_profile_path.write_text(json.dumps(data_profile, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(label_encoder_path))
        mlflow.log_artifact(str(metadata_path))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(comparison_path))
        mlflow.log_artifact(str(data_profile_path))
        mlflow.sklearn.log_model(best_pipeline, artifact_path="sklearn_model")

    print(model_path)
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing Penmanshiel Turbine_Data_*.csv and Status_*.csv files.",
    )
    args = parser.parse_args()
    train(args.data_dir)


if __name__ == "__main__":
    main()
