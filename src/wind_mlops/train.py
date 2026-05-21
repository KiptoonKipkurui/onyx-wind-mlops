from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from wind_mlops.config import ARTIFACT_DIR, DEFAULT_CONFIG, MLRUNS_DIR, RAW_DATA_DIR
from wind_mlops.data import build_multiclass_training_frame, load_penmanshiel_dataset


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

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    multi_class="multinomial",
                    random_state=cfg.random_state,
                ),
            ),
        ]
    )

    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(cfg.experiment_name)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)

        labels = np.arange(len(class_names))
        metrics = {
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "top_2_accuracy": float(top_k_accuracy_score(y_test, probabilities, k=2, labels=labels)),
            "rows": int(len(X)),
            "feature_count": int(len(feature_names)),
            "class_count": int(len(class_names)),
        }
        mlflow.log_params(
            {
                "model_type": "multinomial_logistic_regression",
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
        data_profile_path = ARTIFACT_DIR / "data_profile.json"

        joblib.dump(pipeline, model_path)
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
                    predictions,
                    labels=labels,
                    target_names=class_names,
                    output_dict=True,
                    zero_division=0,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        data_profile_path.write_text(json.dumps(data_profile, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(label_encoder_path))
        mlflow.log_artifact(str(metadata_path))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(data_profile_path))
        mlflow.sklearn.log_model(pipeline, artifact_path="sklearn_model")

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
