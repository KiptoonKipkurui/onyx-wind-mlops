from __future__ import annotations

from pathlib import Path

from wind_mlops.model_repository import unique_version_dir


def test_unique_version_dir_uses_run_prefix_and_avoids_overwrite(tmp_path: Path) -> None:
    model_root = tmp_path / "penmanshiel-event-type-onnx"
    metadata = {"run_id": "abcdef1234567890"}

    first_version, first_path = unique_version_dir(model_root, metadata)
    first_path.mkdir(parents=True)
    second_version, second_path = unique_version_dir(model_root, metadata)

    assert first_version.startswith("abcdef123456-")
    assert second_version.startswith(first_version)
    assert second_version != first_version
    assert second_path.name.endswith("-02")
