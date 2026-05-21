from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def unique_version_dir(model_root: Path, metadata: dict[str, object]) -> tuple[str, Path]:
    run_id = str(metadata.get("run_id") or metadata.get("version") or "local")
    run_prefix = run_id[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = f"{run_prefix}-{timestamp}"
    candidate = model_root / version
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = model_root / f"{version}-{suffix:02d}"
    return candidate.name, candidate
