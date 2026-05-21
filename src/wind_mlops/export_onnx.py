from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from wind_mlops.config import ARTIFACT_DIR, DEFAULT_CONFIG, MLRUNS_DIR, MODEL_REPOSITORY_DIR
from wind_mlops.model_repository import unique_version_dir


def export_onnx(model_path: Path | None = None, metadata_path: Path | None = None) -> Path:
    cfg = DEFAULT_CONFIG
    model_path = model_path or ARTIFACT_DIR / "model.joblib"
    metadata_path = metadata_path or ARTIFACT_DIR / "metadata.json"
    pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    feature_count = len(metadata["feature_names"])
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=[(metadata["input_name"], FloatTensorType([None, feature_count]))],
        target_opset=17,
        options={id(pipeline.steps[-1][1]): {"zipmap": False}},
    )

    model_root = MODEL_REPOSITORY_DIR / cfg.registered_model_name
    version, bundle_dir = unique_version_dir(model_root, metadata)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    metadata["version"] = version
    metadata["exported_at_utc"] = datetime.now(timezone.utc).isoformat()

    onnx_path = bundle_dir / "model.onnx"
    bundle_metadata_path = bundle_dir / "metadata.json"
    onnx_path.write_bytes(onnx_model.SerializeToString())
    bundle_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    with mlflow.start_run(run_id=metadata.get("run_id")):
        mlflow.log_artifact(str(onnx_path), artifact_path="onnx")
        mlflow.log_artifact(str(bundle_metadata_path), artifact_path="onnx")

    print(bundle_dir)
    return bundle_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    args = parser.parse_args()
    export_onnx(args.model, args.metadata)


if __name__ == "__main__":
    main()
