from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def smoke_test(bundle_dir: Path) -> None:
    metadata = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
    feature_count = len(metadata["feature_names"])
    session = ort.InferenceSession(str(bundle_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    sample = np.zeros((1, feature_count), dtype=np.float32)
    outputs = session.run(None, {metadata["input_name"]: sample})
    class_names = metadata.get("class_names") or [f"class_{idx}" for idx in range(len(outputs[1][0]))]
    probabilities = np.asarray(outputs[1])
    response = {
        "predicted_class_id": np.asarray(outputs[0]).tolist(),
        "class_probabilities": [
            dict(zip(class_names, row.astype(float).tolist(), strict=False)) for row in probabilities
        ],
    }
    print(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_dir", type=Path)
    args = parser.parse_args()
    smoke_test(args.bundle_dir)


if __name__ == "__main__":
    main()
