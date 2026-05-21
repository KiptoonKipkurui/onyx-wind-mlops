# Wind Turbine ONNX ML Take-Home

This is a production-shaped take-home for the ML Engineer exercise:

- train a Python model on Penmanshiel-style wind-farm SCADA data,
- track experiments and artifacts in MLflow,
- export the model to ONNX,
- publish a versioned bundle to a local model repository,
- serve the ONNX model from a .NET 10 controller-based Web API with ONNX Runtime.

The assignment is intentionally open-ended, so the default target is now a multiclass `next_event_type_60m` classifier. It predicts the highest-priority turbine event/status type expected in the next hour from current SCADA signals, while avoiding obvious leakage fields such as availability and lost-production columns.

## Project Layout

```text
src/wind_mlops/                 Python training, target building, export, ONNX smoke test
airflow/dags/                   Optional Airflow DAG for train -> export -> smoke test
api/WindInference.Api/          .NET 10 controller-based API using Microsoft.ML.OnnxRuntime
data/raw/                       Put the Penmanshiel CSV here
artifacts/                      Local sklearn model and training metadata
mlruns/                         Local MLflow tracking store
model_repository/               Versioned ONNX bundles for serving
```

## Data

Use the public Penmanshiel Wind Farm Data referenced in the interview brief. Put a SCADA CSV under:

```bash
wind-onnx-mlops-takehome/data/raw/
```

For the reusable Python pipeline, place the extracted folder containing `Turbine_Data_*.csv` and matching `Status_*.csv` files under `data/raw`, or pass the folder directly with `--data-dir`.

The training code resolves common wind-farm column names such as wind speed, active power, rotor speed, generator speed, pitch angle, nacelle/yaw, and temperature.

## Run The Python Pipeline

```bash
cd wind-onnx-mlops-takehome
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m wind_mlops.train --data-dir "$HOME/Downloads/Penmanshiel_SCADA_2016_WT01-10_3107"
python -m wind_mlops.export_onnx
```

The exported bundle is written to:

```text
model_repository/penmanshiel-event-type-onnx/<run-id-prefix>-<export-timestamp>/
  model.onnx
  metadata.json
```

MLflow also receives a `data_profile.json` artifact containing source CSV SHA-256 hashes, row counts, selected features, and target class balance. That gives you a lightweight data-versioning story without adding another service to the timeboxed exercise.
Each ONNX export creates a new version directory, so re-exporting from the same trained model does not overwrite earlier bundles.

Run an ONNX smoke test:

```bash
bundle="$(find model_repository/penmanshiel-event-type-onnx -mindepth 1 -maxdepth 1 -type d | sort -r | head -n 1)"
python -m wind_mlops.smoke_test_onnx "$bundle"
```

View MLflow locally:

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Optional Airflow Orchestration

The DAG in `airflow/dags/wind_model_pipeline.py` wraps the same three production entry points:

1. `python -m wind_mlops.train`
2. `python -m wind_mlops.export_onnx`
3. `python -m wind_mlops.smoke_test_onnx <latest bundle>`

For a short take-home, MLflow is the important MLOps layer and Airflow is kept deliberately thin. In a real deployment, this DAG would run after a validated data snapshot lands and would publish the ONNX bundle to a central registry or object store.

## Run The .NET 10 API

The API targets `net10.0` and uses ASP.NET Core controllers. The root routes are kept for quick testing, and conventional controller routes are available too:

```text
GET  /health
GET  /docs
GET  /openapi/v1.json
GET  /metadata
GET  /api/inference/metadata
POST /predict
POST /api/inference/predict
```

Redoc documentation is available at:

```text
http://localhost:5000/docs
```

The OpenAPI document is generated from controller routes, response attributes, and XML documentation comments in the C# code.

```bash
cd wind-onnx-mlops-takehome/api/WindInference.Api
dotnet restore
dotnet run
```

By default the API loads the local bundle at:

```text
../../model_repository/penmanshiel-event-type-onnx/7aa9f28371f1-20260521T171138Z
```

That path is relative to `api/WindInference.Api`. You can also set:

```bash
export MODEL_BUNDLE_PATH="$PWD/../../model_repository/penmanshiel-event-type-onnx/<version>"
```

The API uses an `IModelProvider` abstraction. The current implementation is `FileSystemModelProvider`, which resolves the bundle from disk. A future MLflow, S3, Azure Blob, or registry-backed provider can implement the same interface without changing the ONNX inference service.

## Run The API In Docker

Build the API image from the repository root:

```bash
docker build -t wind-inference-api ./api/WindInference.Api
```

Run it with the model bundle mounted read-only into `/models`:

```bash
docker run --rm -p 5000:8080 \
  -e MODEL_BUNDLE_PATH=/models \
  -v "$PWD/model_repository/penmanshiel-event-type-onnx/7aa9f28371f1-20260521T171138Z:/models:ro" \
  wind-inference-api
```

The container listens on port `8080`; the command maps it to `http://localhost:5000` on the host.

Example request with ordered features:

```bash
curl -s http://localhost:5000/api/inference/predict \
  -H "Content-Type: application/json" \
  -d '{
  "turbineId": "T01",
  "features": [7.5, 530.0, 12.1, 1420.0, 180.0, 2.0, 9.5]
}'
```

Example request with named features:

```bash
curl -s http://localhost:5000/api/inference/predict \
  -H "Content-Type: application/json" \
  -d '{
  "turbineId": "T01",
  "namedFeatures": {
    "Wind Speed": 7.5,
    "Active Power": 530.0,
    "Rotor Speed": 12.1
  }
}'
```

The model metadata endpoint shows the exact feature names expected by the trained artifact:

```bash
curl -s http://localhost:5000/api/inference/metadata
```

## Run Tests

Python:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/python
```

.NET:

```bash
dotnet test api/WindInference.Api/WindInference.Api.sln
```

## Interview Talking Points

- ONNX separates training framework choice from serving runtime, so the .NET service does not need sklearn.
- MLflow records parameters, metrics, reports, and the sklearn/ONNX artifacts for reproducibility.
- Data versioning is represented by the logged source-file hash and data profile artifact; in a larger system this would become a DVC/lakeFS/table-version integration.
- The model repository is versioned by MLflow run id prefix and contains both `model.onnx` and `metadata.json`, which is the serving contract.
- The API supports reusable inference patterns: metadata-driven feature ordering, model version decoupling through `MODEL_BUNDLE_PATH`, and structured prediction responses.
- The target is multiclass and forward-looking; the production value is in the end-to-end path and the ability to swap the model or event taxonomy later.
