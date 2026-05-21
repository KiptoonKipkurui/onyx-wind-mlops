$ErrorActionPreference = "Stop"

python -m pip install -e .
python -m wind_mlops.train
python -m wind_mlops.export_onnx

$bundle = Get-ChildItem -Directory "model_repository/penmanshiel-event-type-onnx" |
  Sort-Object Name -Descending |
  Select-Object -First 1

python -m wind_mlops.smoke_test_onnx $bundle.FullName
