# End-to-End MLOps: Train → Register → Serve → Monitor

> **A production-grade MLOps platform, not just a model.** Full lifecycle in one repo — PyTorch
> Lightning training, MLflow registry, a FastAPI serving layer with autoscaling and zero-downtime
> rollout, Postgres-backed prediction logging, and automated drift/quality monitoring, run on
> Kubernetes. Ships with a working multi-channel microscopy classifier as the reference implementation.

[![Tests-Training](https://github.com/kayhanAlvandi/full-stack-mlops/actions/workflows/ci_training.yml/badge.svg)](https://github.com/kayhanAlvandi/full-stack-mlops/actions/workflows/ci_training.yml)
[![Tests-Serving](https://github.com/kayhanAlvandi/full-stack-mlops/actions/workflows/ci_serving.yml/badge.svg)](https://github.com/kayhanAlvandi/full-stack-mlops/actions/workflows/ci_serving.yml)
[![Tests-Monitoring](https://github.com/kayhanAlvandi/full-stack-mlops/actions/workflows/ci_monitoring.yml/badge.svg)](https://github.com/kayhanAlvandi/full-stack-mlops/actions/workflows/ci_monitoring.yml)
[![Tests-DB](https://github.com/kayhanAlvandi/full-stack-mlops/actions/workflows/ci_db.yml/badge.svg)](https://github.com/kayhanAlvandi/full-stack-mlops/actions/workflows/ci_db.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-792ee5)
![MLflow](https://img.shields.io/badge/MLflow-registry-0194e2)
![FastAPI](https://img.shields.io/badge/FastAPI-serving-009688)
![Kubernetes](https://img.shields.io/badge/Kubernetes-kind-326ce5)

## Architecture

The whole lifecycle shares one backbone — **MLflow** (models + versions) and **Postgres**
(predictions + monitoring) — across three stages: **train**, **serve**, **monitor**. Serving and the
batch monitoring jobs run on Kubernetes; the stateful services stay external (host today, managed
cloud later via a one-line `ExternalName` swap).

![Architecture](docs/diagram/architecture.png)

> **Edges:** solid blue = data flow (the `train → register → serve → log → monitor` spine) ·
> dashed = control / autoscaling · dotted grey = image provenance (GHCR → workloads).

## Highlights

- **Full ML lifecycle in one system** — training, model registry, dataset versioning,
  serving, prediction storage, and post-deployment monitoring, wired together.
- **Config-driven training** — PyTorch Lightning loop with Hydra-composed experiments (`datamodule`/`model`/`optimizer`/`loss`/ `trainer` groups, no code edits per run), a model zoo from `simplecnn` to `vit_base`, and dual TensorBoard + MLflow logging.
- **Model & data versioning** — MLflow registry with resume-from-registered-model; every run logs a
  content-hashed dataset version + git commit for reproducibility.
- **Production serving** — FastAPI loading models straight from MLflow, on Kubernetes with
  **HPA autoscaling**, **readiness-gated zero-downtime rollout**, and self-healing pods.
- **Monitoring that distinguishes two questions** — unsupervised **drift** (does input/output still
  look like what the model was validated on?) vs. supervised **quality** (is it still accurate?)
  against a *frozen, model-independent benchmark* so scores are comparable across model versions.
- **Data engineering discipline** — normalized Postgres: shared content-hashed tile stacks, one
  live/reference/benchmark split via flags + SQL views, async label backfill, and drift metrics
  stored as queryable rows; all writes idempotent and resumable.
- **CI/CD & testing** — 5 scoped GitHub Actions workflows (lint → layered pytest → build/push images
  to GHCR); Training / API / DB / Monitoring test layers with GPU-aware markers.

<!---
## Visual tour

> _Screenshots (add under `docs/images/`):_
>
> | MLflow runs & registry | Evidently drift report | Kubernetes (HPA + pods) |
> |---|---|---|
> | _`docs/images/mlflow.png`_ | _`docs/images/drift.png`_ | _`docs/images/k8s.png`_ |
>
<!--- -->

## Features

**Training & experimentation**

- **PyTorch Lightning** training loop with a swappable `LightningModule` and `LightningDataModule`.
- **Hydra** config groups for `datamodule`, `model`, `optimizer`, `loss`, `callbacks`, and `trainer` — compose experiments from the CLI, no code edits.
- **MLflow** tracking (params, metrics, system metrics), artifact logging, and Model Registry with resume-from-registered-model support — plus **dataset versioning**: a content-hashed dataset version, manifest, and metadata logged per run for reproducibility.
- **Dual logging**: MLflow + TensorBoard.
- A model zoo of ready-to-use configs: `simplecnn`, `resnet18/50`, `efficientnet_b0/b3`, `vit_small/base` (+ fine-tune variants).

**Serving**

- **FastAPI** serving that loads a model directly from MLflow (registered name or run name) and performs tiled inference with majority voting.
- **PostgreSQL prediction logging** — every request persists image/tile predictions and per-channel pixel stats into a normalized schema (`database/`).
- **Kubernetes serving** (`k8s/`) — Deployment + Service + CPU-based HPA + Ingress, readiness-gated rolling updates on model change.

**Monitoring**

- **Ephemeral monitoring jobs** (`monitoring/`): benchmark registration, reference scoring, label backfill from later-added labels, and Evidently **drift** (unsupervised) + **quality** (supervised, vs. a frozen benchmark) reports — all reading/writing the same Postgres tables.
- One schema for live/reference/benchmark traffic, partitioned by flags + SQL views (`database/init/`).

**Platform & delivery**

- **Docker Compose** stacks for MLflow, GPU training, the API, and the jobs, orchestrated via `docker/makefile`.
- **GHCR-published images** (`api`, `monitoring`, `mlflow`, `training`) built by CI and pulled by the k8s manifests (`docker/IMAGES.md`).
- **Local `kind` cluster** (`k8s/`) — namespaces, `ExternalName` services to host Postgres/MLflow, ConfigMaps/Secrets, batch Job manifests, and traffic-simulation targets.
- **CI/CD** — path-scoped Actions workflows: ruff → layered pytest (unit + real-Postgres
  service containers for DB/API/monitoring) → GHCR push gated on green; training images are
  manual-dispatch only.
- **Layered test suite** — `tests/training` (unit + integration), `tests/api`, `tests/db`, `tests/monitoring` with pytest markers (`slow`, `gpu`) and a `Makefile` for local runs.

## Project Structure

```
full-stack-mlops/
├── configs/                    # Hydra config groups
│   ├── config.yaml             # Root config (composes the groups below)
│   ├── datamodule/             # Dataset + dataloader configs
│   ├── model/                  # Architecture configs (cnn, resnet, efficientnet, vit, ...)
│   ├── optimizer/              # Optimizer + scheduler configs
│   ├── loss/                   # Loss function configs
│   ├── callbacks/              # Lightning callback configs
│   └── trainer/                # Trainer configs (local / container)
├── src/
│   ├── config.py               # Config dataclasses
│   ├── dataset.py              # Multi-channel image dataset
│   ├── datamodule.py           # LightningDataModule
│   ├── transforms.py           # Image / batch transforms (incl. Mixup/CutMix)
│   ├── model.py                # LightningModule + architectures
│   ├── callbacks.py            # Custom callbacks (e.g. LogBestModelToMLflow)
│   └── dataset_versioning.py   # Dataset hashing, manifest, git tracking
├── api/                        # FastAPI serving (model from MLflow, logs to Postgres)
├── monitoring/                 # Batch jobs: benchmarks, references, drift/quality, labels
├── database/
│   ├── dblogger.py             # Prediction/metadata logging client (psycopg)
│   └── init/                   # Schema: prediction tables + live/ref/benchmark views
├── utils/                      # Filename parsing + MongoDB labels (shared across modules)
├── requirements/
│   ├── training_req.txt        # Training dependencies (torch, lightning, hydra, mlflow, …)
│   ├── api_req.txt             # Serving dependencies (fastapi, uvicorn, mlflow, psycopg, …)
│   ├── monitoring_req.txt      # Monitoring job dependencies (evidently, pydantic, …)
│   └── db_req.txt              # Minimal deps for DB-layer tests (psycopg)
├── docker/
│   ├── IMAGES.md               # Which image builds where, tags, GHCR auth
│   ├── makefile                # Compose orchestration (shared external network)
│   ├── services/               # Long-running: mlflow, api (+ api dev overlay)
│   └── jobs/                   # Ephemeral: train, monitoring, compute_references, build_dataset
├── k8s/                        # kind cluster: API deploy + HPA + ingress, job manifests
├── docs/
│   ├── architecture.py         # Renders docs/architecture.png (diagrams-as-code)
│   └── plans/                  # Design docs (kubernetes, airflow/terraform, monitoring, …)
├── tests/
│   ├── training/
│   │   ├── unit/               # Fast, isolated tests (transforms, dataset, model, …)
│   │   └── integration/        # End-to-end tests (training smoke, checkpoint, MLflow)
│   ├── api/                    # FastAPI endpoint tests via TestClient
│   ├── db/                     # Database logger tests (requires PostgreSQL)
│   ├── monitoring/             # Monitoring tests (unit / db / inference)
│   ├── diagnostics/            # Old exploratory scripts (excluded from collection)
│   ├── conftest.py             # Shared fixtures (synthetic images, tiled_datamodule)
│   └── Makefile                # Local test-runner shortcuts
├── .github/
│   ├── workflows/              # ci_training / ci_serving / ci_monitoring / ci_db / cd_mlflow
│   └── actions/                # Reusable composite actions (setup-env, build-push-image)
├── scripts/                    # build_dataset, simulate_live_predictions, MLflow/GHCR utils
├── data/
│   └── benchmark_v1/           # Example frozen dataset (`make create-benchmark NAME=benchmark_v1`)
│       ├── dataset_manifest.json    # Per-sample plate/well/field/label + channel files
│       └── dataset_metadata.json    # Classes, counts, version hash
├── benchmarks/                 # Dataloader & model benchmarks
├── train.py                    # Hydra entrypoint for training
├── pytest.ini                  # Pytest config (markers, norecursedirs)
└── README.md
```

## Usage

Everything runs in **containers** — there's no local Python setup. The two deployment paths
(**Docker Compose** for local iteration, **Kubernetes / kind** for a production-like setup) share the
same prerequisites and the same images. Deep details live in the sub-READMEs (`docker/IMAGES.md`,
`api/README.md`, `monitoring/README.md`, `k8s/README.md`); this section orchestrates and links.

> **Bring your own data.** The repo ships **no dataset**. The bundled microscopy example is a
> reference wired to an internal image store and label source (machine-specific paths in
> `configs/datamodule/tiled.yaml`, a private `tools` lib, and MongoDB for labels). Point the config
> at your data and swap in your own dataset/label logic — see [Using This Template](#using-this-template).

### Prerequisites

Shared by both the Docker and Kubernetes paths.

**Shared networks** — the compose stacks attach to two external Docker networks:

```bash
docker network create ml-platform
docker network create pg_network
```

**MLflow (tracking + registry)** — run the server from `docker/` (`make mlflow` also creates
`ml-platform`); compose/Dockerfile under `docker/services/mlflow/`:

```bash
cd docker
make mlflow            # http://localhost:5000
```

**Prediction database (Postgres)** — the API and monitoring jobs log to a PostgreSQL DB that runs
**outside** this repo. The schema lives in `database/init/` and Postgres applies it automatically on
first start when mounted into its init dir:

```yaml
# docker-compose.postgres.yaml  (run from the repo root)
services:
  postgres:                      # hostname used in the DB URIs below
    image: postgres:16
    container_name: postgres_server
    restart: unless-stopped
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: image_classifier     # must match the DB in your *_DB_URI
    ports: ["5432:5432"]
    volumes:
      - pg_data:/var/lib/postgresql/data
      - ./database/init:/docker-entrypoint-initdb.d:ro   # auto-applies the schema on first init
    networks: [pg_network]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
networks:
  pg_network:
    external: true
volumes:
  pg_data:
```

```bash
docker compose -f docker-compose.postgres.yaml up -d
```

Services read their connection string from `.env` files (templates:
`docker/services/api/.env.api.example`, `docker/jobs/monitoring/.env.monitoring.example`):

```
API_DB_URI=postgresql://postgres:postgres@postgres:5432/image_classifier
MONITORING_DB_URI=postgresql://postgres:postgres@postgres:5432/image_classifier
```

Keep the credentials/DB name in sync with the `POSTGRES_*` values. The schema only auto-applies on a
fresh `pg_data` volume — drop it to re-init, or apply `database/init/*.sql` manually.

### Train

Training is **Docker-only** — there is no Kubernetes training job (architecture/hyperparameters need
human judgment; only evaluation and promotion are automated). It's how you produce and register the
model that serving and monitoring then consume.

A run composes a Hydra config from one option per group — `datamodule`, `model`, `optimizer`, `loss`,
`callbacks`, `trainer` (see `configs/config.yaml`) — overridable on the CLI with no code edits. Model
and data code live in `src/`; the job is defined in `docker/jobs/train/`.

```bash
cd docker

# Default run
make train-run CMD="python train.py"

# Swap architecture / optimizer / loss / datamodule
make train-run CMD="python train.py model=resnet50"
make train-run CMD="python train.py model=vit_base_finetune optimizer=sgd loss=focal"

# Override individual values
make train-run CMD="python train.py seed=123 run_name=my_experiment tags=[baseline]"

# Resume / fine-tune from a registered MLflow model
make train-run CMD="python train.py resume_from_model=SimpleCNN/11"                                # full resume
make train-run CMD="python train.py resume_from_model=SimpleCNN/latest resume_weights_only=true"  # weights only
```

Each run logs params, metrics, system metrics, artifacts (Hydra config, dataset manifest/metadata),
and a content-hashed **dataset version** + git commit to MLflow (`http://localhost:5000`), and
registers the best model. TensorBoard curves/confusion-matrices are written under `logs/`.

### Serving & monitoring

Once a model is registered, serve it and run the monitoring jobs — with Docker Compose or on
Kubernetes. Both use the same images and the same external Postgres + MLflow.

#### Option A — Docker

**Serving** — point the API at a registered model/run in `docker/services/api/.env.api`
(`API_MODEL_NAME` or `API_RUN_NAME`), then (full endpoint docs in `api/README.md`):

```bash
cd docker
make serve        # MLflow + API at http://localhost:8000
make serve-dev    # same, but bind-mounts source + --reload for local iteration
```

**Monitoring** — each job is a `make` target (run after some traffic exists). See
`monitoring/README.md` for what each does and the order to run them:

```bash
make create-benchmark NAME=benchmark_v1                                      # build a frozen benchmark manifest
make register-benchmark-run MANIFEST=data/benchmark_v1/dataset_manifest.json  # register it (one-time)
make compute-predictions-references                                          # score val + benchmark references
make label-backfill-run                                                      # fill t_label from MongoDB
make drift-report-run    CMD="python -m monitoring.run_drift_report --window-days 14"
make quality-report-run  CMD="python -m monitoring.run_quality_report --window-days 14"
```

#### Option B — Kubernetes (kind)

Run the API and the batch jobs on a local `kind` cluster; Postgres/MLflow stay external (reached via
`host.docker.internal`). Full setup, secrets, and rationale: `k8s/README.md`.

```bash
cd k8s
make kind-up            # create the cluster
make bootstrap          # addons (ingress-nginx, metrics-server) + namespaces + config/secrets
#   make bootstrap-dev  # same, but build + load the local api image instead of pulling from GHCR
```

**Serving** — API Deployment + Service + HPA + Ingress:

```bash
kubectl apply -f api/
curl http://api.localtest.me/health     # via ingress (or port-forward svc/api-service)
```

**Monitoring** — the same jobs as Option A, as `batch/v1` Job manifests (immutable — delete before
re-running):

```bash
kubectl apply -f jobs/register-benchmark.yaml
kubectl apply -f jobs/compute-references.yaml
kubectl apply -f jobs/drift-report.yaml
kubectl delete -f jobs/drift-report.yaml   # before re-applying to re-run
```

Generate test traffic with `make simulate-internal` / `make simulate-external`.

## Testing & CI/CD

There's no separate "manual test suite" — whatever's run locally is exactly what CI runs, so this
covers both. Five scoped GitHub Actions workflows (`.github/workflows/`) run on push/PR to `main`,
each gated to the paths its module touches: ruff lint → pytest → (on success, push to `main`) build
and push a tagged image to GHCR. Both the lint and test steps use the reusable composite actions
`.github/actions/setup-env` (Python 3.11 + pip cache) and `.github/actions/build-push-image`
(tag/push — see `docker/IMAGES.md`).

| Workflow | Scope (triggers on) | Lint + test | Image build/push |
| --- | --- | --- | --- |
| `ci_training.yml` | `src/`, `train.py`, `tests/training/` | ruff → `tests/training/unit` (push/PR); `workflow_dispatch` picks `training` / `integration` / `all` | `training` — **dispatch-only**, `run_tests: all` |
| `ci_serving.yml` | `api/`, `tests/api/` | ruff → `tests/api` | `api` |
| `ci_monitoring.yml` | `monitoring/`, `tests/monitoring/`, `database/init/` | ruff → unit + DB (real Postgres service container) + inference | `monitoring` |
| `ci_db.yml` | `database/`, `tests/db/` | ruff → DB tests (real Postgres service container) | — (no image; the schema is consumed by `api`/`monitoring`) |
| `cd_mlflow.yml` | `docker/services/mlflow/**` | — (nothing to gate on) | `mlflow` |

**GPU tests** (`gpu` marker in `pytest.ini`) are excluded from CI — no CUDA on GitHub-hosted runners.
Run them locally on a CUDA machine before merging. `tests/diagnostics/` is old exploratory scripts,
excluded from collection.

### Running the same checks locally

```bash
# Training
ruff check src/ train.py tests/training/
pytest tests/training/unit -q           # matches the push/PR job
pytest tests/training/integration -q    # matches the 'integration' dispatch option
pytest -m "gpu" -q                      # GPU-only (run on a CUDA machine)

# Serving (api)
ruff check api/ tests/api/
pytest tests/api -q

# Monitoring — db/inference layers need a local Postgres (see Usage > Prerequisites)
ruff check monitoring/ tests/monitoring/
pytest tests/monitoring/unit -q
DB_TEST_URI=postgresql://admin:admin123456@localhost:5432/test pytest tests/monitoring/db -v
pytest tests/monitoring/inference -q

# Database layer — needs a local Postgres
ruff check database/ tests/db/
DB_TEST_URI=postgresql://admin:admin123456@localhost:5432/test pytest tests/db -v
```

## Using This Template

The **training side is fully generic** — swap these without touching anything else:

1. Click **"Use this template"** on GitHub (or clone).
2. Dataset + loading: `src/dataset.py` / `src/datamodule.py` and `configs/datamodule/`.
3. Architecture: `configs/model/` (or extend `src/model.py`).
4. Optimizer / loss / callbacks / trainer: `configs/`.
5. Train (`docker/jobs/train/`), track in MLflow, register the best model.

The **serving, database, and monitoring layers go deeper**: they share one sample-identity model —
`(plate, well, field, channel)` — baked into the DB schema (`database/init/`), the API's filename
parsing (`utils/filename_parser.py`, `api/main.py`), and label resolution (`utils/labels.py`, keyed
by `(plate, well)`). Reusing those layers for a different domain means adapting all three:

- `utils/filename_parser.py` — the naming-convention regex that recovers sample identity from
  uploaded filenames (see the pattern below).
- `utils/labels.py` — label lookup: MongoDB by `(plate, well)` (`resolve_labels_from_mongodb`), with
  a deterministic dummy fallback (`resolve_labels_dummy`) for tests/offline use.
- `database/init/01_prediction.sql` — the `image_metadata` columns (`plate`, `well`, `field`,
  `channel`, ...).

## Reference Implementation: Multi-Channel Microscopy

The bundled example classifies multi-channel microscopy images with configurable channel selection
and tiled cropping — it's what exercises the identity model above end to end: training's label
source, the API's filename parsing, the DB schema, and monitoring all key off the same
`(plate, well, field, channel)` tuple.

Image naming pattern (`utils/filename_parser.py`):
```
{plate}_{well}_T{time}F{field}L{layer}A{action}Z{z}C{channel}.{jxl|tif}
```
Example: `MIG-Exp03-CP-40X-bin1X1_K07_T0001F001L01A01Z01C01.jxl`

Labels are resolved per well from MongoDB (`resolve_labels_from_mongodb`) when
`datamodule.use_mongodb=true`, or deterministic dummy labels otherwise — see
`configs/datamodule/tiled.yaml`.
