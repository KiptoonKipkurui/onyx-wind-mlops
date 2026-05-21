from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_REPOSITORY_DIR = PROJECT_ROOT / "model_repository"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


@dataclass(frozen=True)
class TrainingConfig:
    experiment_name: str = "penmanshiel-event-type-classifier"
    registered_model_name: str = "penmanshiel-event-type-onnx"
    random_state: int = 42
    test_size: float = 0.2
    min_rows: int = 200
    min_class_rows: int = 30
    future_steps: int = 6
    max_rows_per_turbine: int | None = None


DEFAULT_CONFIG = TrainingConfig()
