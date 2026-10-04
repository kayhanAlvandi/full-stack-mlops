# Kubernetes (local `kind`)

Run this project's batch jobs and the prediction API on a local
[`kind`](https://kind.sigs.k8s.io/) cluster. Postgres and MLflow stay
**external** (on the host, reached via `host.docker.internal`); only this
project's workloads run in-cluster. See `docs/plans/kubernetes-plan.md` for the
full design rationale.

## What gets deployed

- **Namespaces:** `api` (the prediction API + the api-image `compute-references`
  job) and `monitoring` (drift / quality / label-backfill / register-benchmark
  jobs). Each namespace's default ServiceAccount carries the `ghcr-pull`
  imagePullSecret.
- **External services:** `postgres`, `mlflow`, `api` as `ExternalName` Services
  aliasing `host.docker.internal` — pods address them by stable in-cluster DNS,
  and only the alias changes when these move to managed cloud services later.
- **Config/secrets:** `api-config` / `monitoring-config` ConfigMaps and
  `api-secrets` / `monitoring-secrets` / `ghcr-pull` Secrets.
- **API serving:** `Deployment` + `Service` (ClusterIP) + `HorizontalPodAutoscaler`
  (CPU-based) + `Ingress` (`api.localtest.me`), behind ingress-nginx.
- **Jobs:** one-off `batch/v1` Job manifests in `jobs/`. These are plain Jobs,
  **not** CronJobs — the weekly scheduling mechanism (K8s CronJob vs. an Airflow
  DAG) is deferred to the Airflow phase (see
  `docs/plans/airflow-terraform-overview.md`).

## Prerequisites

- Docker Desktop (provides `host.docker.internal`), `kind`, `kubectl`, `make`.
- Python available on PATH (used by `kind-up` to template the cluster config,
  and by `simulate-external`).
- The external **postgres** and **mlflow** must already be running on the host
  (postgres on `5432` with the `prediction` DB; mlflow on `5000`).
- Host paths that get mounted into the kind node (see `kind-cluster.yaml`):
  - `<repo>/data`, `<repo>/mlflow-data/mlruns`, `<repo>/monitoring/reports`
    (derived automatically by the Makefile — the reports dir must already
    exist),
  - the O-drive image store and the external `tools` lib (set in `.env`, below).

## One-time setup (the only manual bits)

Everything else is in the manifests; these are the inputs they need:

1. **`k8s/.env`** — copy `.env.example` and set `O_DRIVE_PATH` and
   `TOOLS_LIB_PATH` for your machine. (`DATA_PATH` / `MLFLOW_DATA_PATH` /
   `REPORTS_PATH` are derived by the Makefile — don't set them.)
2. **Secrets** — copy each template in `config/secret-examples/` to
   `config/<name>.yaml` and fill in real values. These `config/*secret*.yaml`
   files are gitignored; the `*.example.yaml` templates are not.
   - `api-secrets.yaml` — `API_DB_URI` (points at the `prediction` DB).
   - `monitoring-secrets.yaml` — monitoring DB creds / Mongo settings.
   - `ghcr-pull-secret.yaml` — **only if the GHCR package is private.** It must
     exist in **both** the `api` and `monitoring` namespaces (a Secret is
     namespaced, and both default ServiceAccounts reference it).
3. Ensure `<repo>/monitoring/reports` exists (kind's extraMount requires the
   host path to be present before `kind-up`).

## Bring it up

```bash
cd k8s

# 1. Create the cluster (templates kind-cluster.yaml with the host paths).
make kind-up

# 2. Install addons + namespaces + config/secrets/external-services.
#    bootstrap     -> pulls the api image from GHCR (stable/prod-like)
#    bootstrap-dev -> builds the api image locally and `kind load`s it
make bootstrap        # or: make bootstrap-dev
```

`bootstrap` runs the ingress-nginx + metrics-server addons, applies
`namespaces/`, and applies the Kustomization (`config/` + `external-services/`).
It intentionally stops there: the cluster is now set up, and the API and jobs
are applied on demand as one-liners.

## Deploy the API and run jobs

Kustomize only handles cluster setup; the API Deployment and the Jobs are
applied directly:

```bash
# API Deployment + Service + HPA + Ingress
kubectl apply -f api/

# A monitoring/compute job (Jobs are immutable once created -- delete before
# re-running the same one)
kubectl apply -f jobs/drift-report.yaml
kubectl delete -f jobs/drift-report.yaml   # before re-applying to re-run
```

## What each job does

- **`register-benchmark`** (`monitoring`) — one-time, model-independent. Registers
  the fixed, curated benchmark set (samples never seen in training) into the DB
  from a frozen manifest: `image_metadata` + one `benchmark_dataset` row per
  `(plate, well, field)` with its known label + the member links. No model or
  inference involved — scoring the benchmark for a specific model is a separate
  step (`compute-references`).
- **`compute-references`** (`api`, uses the api image) — scores known-label
  samples with the **served model** and writes them as reference predictions:
  the run's own validation set (feeds drift detection) and/or the benchmark set
  (feeds the supervised-quality baseline). Uses the same predict path as the live
  API, and is resumable (skips samples already scored for this run).
- **`drift-report`** (`monitoring`) — compares a window of **live production**
  predictions against the model's validation reference set using Evidently,
  across three groups (image-level, tile-level, per-channel pixel stats). Needs
  no ground truth — it checks the predicted-label and feature distributions.
  Writes an HTML report + summary rows to Postgres.
- **`quality-report`** (`monitoring`) — the **supervised** signal. Compares the
  model's accuracy/F1 on the benchmark baseline against its accuracy/F1 on the
  *labeled* portion of a recent production window (Evidently
  `ClassificationPreset`). Skips cleanly if no production rows are labeled yet.
- **`label-backfill`** (`monitoring`) — fills missing production ground-truth
  labels (`t_label`) by looking up each `(plate, well)`'s treatment in MongoDB.
  Idempotent: only NULL labels are filled, reference/benchmark rows untouched.
- **`debug-simulate-live`** (`api`) — not a pipeline job; a `sleep infinity` pod
  with the data/O-drive mounts, used by `make simulate-internal` to generate
  traffic from inside the cluster (see below).

Typical order when validating from scratch: `register-benchmark` →
`compute-references` (benchmark + val) → generate some live traffic (simulate) →
`label-backfill` → `drift-report` / `quality-report`.

## Verify

```bash
kubectl get pods -A
kubectl get hpa -n api

# Reach the external services from inside the cluster
kubectl run -it --rm netcheck -n api --image=busybox --restart=Never -- \
  sh -c 'nc -zv host.docker.internal 5432; nc -zv host.docker.internal 5000'

# API health / loaded-model info (port-forward path)
kubectl port-forward -n api svc/api-service 8000:8000
curl localhost:8000/health
curl localhost:8000/model
```

The API's `startupProbe` allows a slow first model load (readiness is gated on
`/model`, which returns 503 until the model is loaded), so a pod can take a
while to become Ready on first start — that's expected.

## Access the API

- **Ingress:** `http://api.localtest.me` (`api.localtest.me` resolves to
  `127.0.0.1`; kind maps host ports 80/443 into the cluster).
- **Port-forward:** `kubectl port-forward -n api svc/api-service 8000:8000`.

## Simulate live prediction traffic

`scripts/simulate_live_predictions.py` replays a dataset manifest against the
API as randomized production traffic. Two Makefile targets wrap the flows:

```bash
# Inside the cluster: hits the API Service DNS directly (proves Service +
# readiness, no ingress involved). Spins up the debug-simulate-live pod, copies
# the script in, and runs it.
make simulate-internal
make simulate-internal INTERNAL_ARGS="--manifest /mnt/data/custom_v1/dataset_manifest.json --count 20"
make simulate-internal-clean        # remove the debug pod when done

# Outside the cluster: hits the ingress hostname from the host (proves
# ingress-nginx routes external traffic to the API). Needs `requests` installed
# on the host.
make simulate-external
make simulate-external EXTERNAL_ARGS="--manifest ../data/custom_v1/dataset_manifest.json --loop"
```

Both default to `data/benchmark_v1/dataset_manifest.json`. The debug pod mounts
`data/` at `/mnt/data` and the O-drive at `/mnt/O`, so internal manifest paths
are `/mnt/data/...`; external (host-side) paths are relative to the repo root
(`../data/...` from `k8s/`).

## Rollout on model change

The served model is pinned in `config/api-configs.yaml` (`API_MODEL_NAME`) and
resolved at pod startup. To promote a different model:

```bash
# 1. Edit API_MODEL_NAME in config/api-configs.yaml, then re-apply config:
kubectl apply -k .
# 2. Readiness-gated rolling restart (zero-downtime; halts if the new model
#    fails to load):
kubectl rollout restart deployment/api-deployment -n api
kubectl rollout status  deployment/api-deployment -n api
```

(Canary/shadow promotion is deferred to the Airflow phase.)

## Teardown

```bash
make kind-down
```

## Notes

- kind wipes the node's image store on every `kind-down`/`kind-up`. `bootstrap`
  relies on pulling the real image from GHCR; `bootstrap-dev` instead builds and
  `kind load`s the local api image so in-progress, not-yet-pushed changes are
  used. See `TODO.md` in this folder for the planned `imagePullPolicy` /
  Kustomize-overlay cleanup.
