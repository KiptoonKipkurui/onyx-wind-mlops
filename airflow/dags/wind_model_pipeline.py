from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from airflow.sdk import dag, task


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dag(
    dag_id="wind_turbine_onnx_training",
    description="Train a wind-turbine event-type classifier, export ONNX, and publish to a local model repository.",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["wind", "onnx", "mlflow"],
)
def wind_model_pipeline():
    @task
    def train_model() -> str:
        from src.wind_mlops.train import train

        model_path = train()
        return str(model_path)

    @task
    def export_model(model_path: str) -> str:
        from src.wind_mlops.export_onnx import export_onnx

        bundle_dir = export_onnx(model_path=Path(model_path))
        return str(bundle_dir)

    @task
    def smoke_test(bundle_dir: str) -> None:
        from src.wind_mlops.smoke_test_onnx import smoke_test

        smoke_test(Path(bundle_dir))

    trained_model_path = train_model()
    exported_bundle_dir = export_model(trained_model_path)
    smoke_test(exported_bundle_dir)


wind_model_pipeline()
