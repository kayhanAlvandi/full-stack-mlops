# Image Classifier API

FastAPI service for tiled multi-channel image classification with majority voting.
Models are loaded directly from **MLflow** (registered model or run name), and every
prediction is optionally persisted to **PostgreSQL** for monitoring.

## Overview

Upload a large multi-channel image (e.g. 2000x2000, one file per channel) and the API will:
1. Load the model + config from MLflow (class names, crop size, channels auto-detected).
2. Canonicalize the channel order (ascending channel number) to match training.
3. Split the image into tiles (e.g. 224x224).
4. Run the trained model on each tile and aggregate to a whole-image prediction via majority vote.
5. If a database is configured, log the image/tile predictions and per-channel tile pixel stats
   (used by the drift/quality monitoring jobs).

The HTTP response returns the **whole-image** prediction (per-tile details are persisted to the DB,
not returned in the response).

## Setup

```bash
pip install -r requirements/api_req.txt
```

## Run

Pick **one** model source (priority: `model_name` > `run_name`):

```bash
# Option 1: Load from MLflow registered model (e.g. "TransferLearning/20" or ".../latest")
set API_MODEL_NAME=TransferLearning/20
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Option 2: Load from MLflow run name
set API_RUN_NAME=Vits_finetune_cosine_warmup_autoGradual_moredata
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The first model load can be slow; `/model` returns `503` until the model is ready (used as the
readiness gate in Kubernetes). See `../docker/IMAGES.md` and `../k8s/README.md` for the containerized
and clustered ways to run this service.

### Environment variables

Loaded via `api/config.py` (`Settings`, env prefix `API_`, also reads a `.env` file):

| Variable | Default | Description |
|---|---|---|
| `API_MODEL_NAME` | `""` | MLflow registered model, e.g. `TransferLearning/20` or `TransferLearning/latest` |
| `API_RUN_NAME` | `""` | MLflow run name (used if `API_MODEL_NAME` is unset) |
| `API_TRACKING_URI` | `http://mlflow:5000` | MLflow tracking URI |
| `API_EXPERIMENT_NAME` | `image_classifier` | MLflow experiment name (for run-name lookup) |
| `API_CROP_SIZE` | `224` | Tile size (auto-detected from MLflow artifacts when available) |
| `API_STRIDE` | `null` | Tile stride (defaults to `crop_size` = non-overlapping) |
| `API_DEVICE` | `cpu` | `cpu` or `cuda` |
| `API_ARTIFACT_ROOT` | `null` | Local path to the tracking server's artifact store (its `mlruns` dir). When set and the run's artifacts exist there, they're read straight off disk instead of HTTP-proxied through the tracking server. |
| `API_DB_URI` | `null` | PostgreSQL connection string. When set, predictions are logged to the DB; when unset, logging is skipped (predictions still work). |

## Input format

- **One file per channel.** Each uploaded file is treated as a single channel; files are stacked and
  reordered into ascending channel-number order to match training (upload order doesn't matter).
- **Supported file types:** `.tif` / `.tiff` and `.jxl` (other formats are attempted via OpenCV).
- **Filenames must follow the microscopy naming pattern** so the plate/well/field/channel can be
  parsed and stored:
  ```
  {plate}_{well}_T{time}F{field}L{layer}A{action}Z{z}C{channel}.{jxl|tif}
  ```
  Example: `MIG-Exp03-CP-40X-bin1X1_K07_T0001F001L01A01Z01C01.jxl`
  All files for one request must share the same plate/well/field (they differ only by channel).

## Endpoints

### `GET /health`
Liveness check. Returns:
```json
{"status": "ok", "model_loaded": true, "device": "cpu", "database_connected": true}
```

### `GET /model`
Detailed info about the loaded model (auto-detected from MLflow artifacts). Returns `503` until a
model is loaded:
```json
{
  "source": "registry:TransferLearning/20",
  "model_class": "TransferLearningModule",
  "backbone": "vit_small_patch16_224",
  "run_id": "a1b2c3...",
  "num_classes": 3,
  "class_names": ["class_A", "class_B", "class_C"],
  "in_channels": 5,
  "crop_size": 224,
  "stride": 224,
  "device": "cpu"
}
```

### `GET /db`
Database connection info. Returns `503` when no database is configured:
```json
{"connected": true, "uri": "postgresql://..."}
```

### `POST /predict`
Upload one image file per channel for tiled prediction.

**Form fields:**
- `files` (required): one or more image files, one per channel (ordered `C1..CN`, reordered server-side).
- `root_path` (required): root path recorded with the image metadata.

**Query params:**
- `crop_size` (optional): override tile size for this request.
- `stride` (optional): override tile stride for this request.

**Response** (whole-image prediction; per-tile results are logged to the DB, not returned):
```json
{
  "plate": "MIG-Exp03-CP-40X-bin1X1",
  "well": "K07",
  "field": 1,
  "run_id": "a1b2c3...",
  "predicted_class": "class_A",
  "total_tiles": 64,
  "vote_fraction": 0.78,
  "confidence": 0.95
}
```

## Example Usage

```python
import requests

# Check the loaded model
print(requests.get("http://localhost:8000/model").json())

# Multi-channel prediction: one file per channel + the required root_path form field
files = [
    ("files", open("MIG-Exp03-CP-40X-bin1X1_K07_T0001F001L01A01Z01C01.jxl", "rb")),
    ("files", open("MIG-Exp03-CP-40X-bin1X1_K07_T0001F001L01A01Z01C02.jxl", "rb")),
    ("files", open("MIG-Exp03-CP-40X-bin1X1_K07_T0001F001L01A01Z01C03.jxl", "rb")),
    ("files", open("MIG-Exp03-CP-40X-bin1X1_K07_T0001F001L01A01Z01C04.jxl", "rb")),
    ("files", open("MIG-Exp03-CP-40X-bin1X1_K07_T0001F001L01A01Z01C05.jxl", "rb")),
]
resp = requests.post(
    "http://localhost:8000/predict",
    files=files,
    data={"root_path": "/data/MIG-Exp03"},
)
print(resp.json())
```

To replay a dataset manifest against a running API as simulated production traffic, see
`scripts/simulate_live_predictions.py` (documented in `../monitoring/README.md`).
