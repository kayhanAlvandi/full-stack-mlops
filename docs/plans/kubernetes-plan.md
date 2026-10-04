# Plan: Kubernetes Implementation (CD → GHCR → kind cluster)

Detailed, ready-to-implement plan for the CD pipeline and the Kubernetes
migration: build/push images to GHCR, then run this project's jobs and
stateful services on a local `kind` cluster (cloud-portable design, cloud
migration deferred). Companion doc `airflow-terraform-overview.md` covers the
next tools at a conceptual level.

> Context: this project is a **template for practicing production-level MLOps**
> (see `AGENTS.md`). Its actual data volume does not *require* Kubernetes; we
> adopt it deliberately to practice real production patterns (see "Real-world
> Kubernetes use cases" below).

---

## Current-state facts (from codebase research)

Images built today (all `build.context: ../../..`, i.e. repo root):
- `image_classifier_api` — `docker/services/api/Dockerfile`, heavy (pytorch/cuda base). Bakes in `api/ utils/ database/`. Reused by the `compute_references` job.
- `image_classifier_monitoring` — `docker/jobs/monitoring/Dockerfile`, lightweight (no torch/mlflow). Bakes in `database/ monitoring/ utils/`. Shared by drift-report, quality-report, label-backfill, register-benchmark.
- mlflow — `docker/services/mlflow/Dockerfile`, trivial (`mlflow==3.10.0`).
- training — `docker/jobs/train/Dockerfile`; **done:** now bakes in `src/ configs/ utils/ train.py` (confirmed minimal set: `train.py` only imports `src.dataset_versioning`, `src/` only imports `utils.filename_parser`/`utils.labels` beyond stdlib, and `train.py`'s `@hydra.main(config_path="configs", ...)` needs `configs/` on disk; `database/`/`scripts/` are not on this import path). The dev compose still bind-mounts the whole repo root over `/workspace`, shadowing the baked-in code for local iteration.
- build_dataset — **new**, `docker/jobs/build_dataset/Dockerfile` (`python:3.11-slim`, lightweight, no torch/timm/mlflow). `scripts/build_dataset.py` and the `src` modules it uses (`sample_selection.py`, `dataset_versioning.py`) are explicitly torch-free, so running them no longer requires the ~12GB CUDA training image. Bakes in `scripts/ configs/ src/ utils/`. `docker/jobs/train/docker-compose.yaml`'s `build_dataset` service now builds from this Dockerfile instead of the training one; both `training` and `build_dataset` services also got explicit `image:` names (`image_classifier_training`, `image_classifier_build_dataset`) to avoid the compose-project-name drift issue we hit before (stale `docker-training`/`train-*` images from prior folder layouts).

Existing CI: `.github/workflows/ci_{db,monitoring,serving,training}.yml` + composite `.github/actions/setup-env`. All are test/lint only; **no build/push, no `cd.yml` exists yet.**

Networking today: two external docker networks `ml-platform` + `pg_network`. Services talk by container name (`mlflow`, `api`, `postgres`).

Config today: per-job `.env.*` files (`.env.api`, `.env.monitoring`, `.env.training`) with `*.example` templates.

Postgres: shared external infra, managed by its own compose (`D:\personal_project\pgSQL\docker-compose.yml`, stock `postgres:16`, container `postgres_server`, `pg_data` volume, `pg_network`, creds `admin`/`admin123`). Hosts multiple projects; this project only owns the `prediction` DB. Schema is `database/init/0{1..4}_*.sql`. Stays external (see Phase 2).

Data mounts: jobs use `${O_DRIVE_PATH}` (image files) and `${TOOLS_LIB_PATH}` (external `tools` lib, editable-installed at container start) — host bind mounts.

---

## Phase 1 — CD: build & push images to GHCR

**Goal:** every deployable image is built and published to `ghcr.io/<owner>/<image>:<tag>` by CI, so K8s (local kind and later cloud) can pull them instead of relying on local Docker.

### Steps
**Implementation approach taken (superseding the original single-`cd.yml` idea):** rather than one central `cd.yml` triggered via `workflow_run` (fiddly cross-workflow SHA/name matching), build/push jobs were added directly into each existing per-component `ci_*.yml` workflow, gated with `needs:` on that component's own test job(s) — same-run, same-commit gating, no timing issues, and each workflow's existing path filters are reused for free.

1. **Composite action `.github/actions/build-push-image/action.yml`** — shared build/push logic, called from each `ci_*.yml`:
   - Inputs: `image_name`, `dockerfile`, `extra_tag` (optional, for on-demand human-readable tags — see training below).
   - `docker/login-action@v3` to GHCR using `${{ secrets.GITHUB_TOKEN }}`.
   - `docker/setup-buildx-action@v3` (enables the `type=gha` cache backend used below — without it `build-push-action` falls back to the legacy builder).
   - `docker/metadata-action@v5` computes tags (`latest` on default branch, `sha-<shortsha>` always, semver on version tags, plus `extra_tag` when given) and OCI labels (including `org.opencontainers.image.revision` = full commit SHA, so any image — even one tagged with a human-readable name — is always traceable back to its exact commit).
   - `docker/build-push-action@v6`, `context: .` (repo root — build/push paths are always relative to `GITHUB_WORKSPACE`, *not* the compose-style "relative to the Dockerfile's own directory"), `cache-from/to: type=gha`.
2. **Per-component build-push jobs** (each `needs:` its own test job(s), gated `if: github.event_name == 'push' && github.ref == 'refs/heads/main'` so PRs never push images):
   - `ci_serving.yml` → `serving-build-push`: `image_name: api`, `dockerfile: docker/services/api/Dockerfile`.
   - `ci_monitoring.yml` → `monitoring-build-push`: `image_name: monitoring`, `dockerfile: docker/jobs/monitoring/Dockerfile`.
   - `mlflow`: no tests to gate on (it's just the server; connection code is covered by the training/serving/monitoring suites). Add a standalone `.github/workflows/cd_mlflow.yml` — `push` to `main` filtered to `docker/services/mlflow/**` (+ `workflow_dispatch`), no `needs:`, calling the composite action (`image_name: mlflow`, `dockerfile: docker/services/mlflow/Dockerfile`). Rebuilds only when the pinned mlflow version changes.
3. **`training` is deliberately excluded from the on-push CD path.** Rationale (see "Why `training` is excluded" below): it's baked into the image (`src/ configs/ utils/ train.py` — done, confirmed minimal set) purely so an image *can* be published, but it is **not built on every push**. Instead, `ci_training.yml`'s existing `workflow_dispatch` (`run_tests` choice input) gained:
   - A second input, `image_tag` (optional, human-readable, e.g. an experiment/config name).
   - A `training-build-push` job: `needs: test-dispatch` **and** `if: github.event_name == 'workflow_dispatch' && inputs.run_tests == 'all'` — both conditions matter, since `test-dispatch`'s per-suite steps are individually gated by `if:` and a job with *skipped* (not failed) steps still reports success, so `needs` alone wouldn't guarantee the full suite actually ran.
   - `extra_tag: ${{ inputs.image_tag }}` passed through to the composite action.
   - A companion lightweight image was also split out: `docker/jobs/build_dataset/Dockerfile` (`python:3.11-slim`, no torch/timm/mlflow) for `scripts/build_dataset.py`, which is explicitly torch-free and was previously (wastefully) run inside the ~12GB training image. Not part of CD/K8s yet — dev-only via `docker/jobs/train/docker-compose.yaml`'s `build_dataset` service.
4. **Docs:** add a short "Images & CD" section to `docker/` docs (not README, per project rule) describing image names/tags and how to pull.

### Why `training` is excluded from continuous build/push (design decision)

Unlike `api`/`monitoring` (stable artifacts meant to run unchanged until deliberately replaced), the training image is edited and rerun constantly during iteration (bind-mounted, hyperparameters/model code changed multiple times an hour). Baking it on every push would mean either constantly rebuilding a huge CUDA image for code still being iterated on, or falling back to the bind mount anyway — defeating the point either way. It also has no K8s payoff yet: the real justification for training-on-K8s is *shared GPU-pool scheduling* (node affinity/taints across many queued runs), which `kind` can't provide (no GPU) — so a training `Job` manifest today would just be a slower `make train-run` with no benefit. This also matches the project's own stance that training shouldn't be automated (hyperparameters need human judgment).

**Revisit when:** moving to a real cloud GPU node pool where shared-GPU scheduling is actually valuable (e.g. many training runs with different hyperparameters queued across a few GPU nodes). At that point, build the image **on-demand per experiment configuration** (the `workflow_dispatch` + `image_tag` pattern above already supports this), not per-commit — matching how real training pipelines version environments per "experiment," not per push. `docker/jobs/train.yaml` (a K8s `Job` manifest) is correspondingly **dropped from Milestone 2B** below until that point.

### Verification
- Push to `main` touching `api/**` or `monitoring/**`; confirm the corresponding `*-build-push` job runs after its tests pass and the package appears under the repo's GHCR packages.
- Manually dispatch `ci_training.yml` with `run_tests: all` and an `image_tag`; confirm `training-build-push` only runs when the full suite passed, and the image carries both the custom tag and a `sha-*` tag.
- `docker pull ghcr.io/<owner>/<image>:latest` locally succeeds for each pushed image.
- Confirm an image runs: `docker run --rm ghcr.io/.../monitoring:latest python -m monitoring.run_drift_report --help`.

### Open decisions
- Image naming scheme on GHCR (currently `image_name: api`/`monitoring` — confirm final `ghcr.io/<owner>/<image_name>` shape works for K8s manifests).
- Public vs private packages (private needs an imagePullSecret in K8s later).
- Whether to keep local compose `image:` names (`image_classifier_api`, `image_classifier_monitoring`, `image_classifier_training`, `image_classifier_build_dataset`) as-is (local dev) alongside the GHCR names (CD), or unify them.

---

## Phase 2 — Kubernetes (local kind first, cloud-portable)

**Goal:** run the ephemeral jobs as K8s `Job`/`CronJob`, and the API as a `Deployment` + `Service` + `HorizontalPodAutoscaler` (serving moves over once the jobs are proven; the API is what makes K8s' rolling-update/rollout story worth practicing — see Milestone 2D). **Postgres and mlflow stay external** (not in-cluster) — jobs and the API connect out to them. Plain manifests first (Kustomize/Helm deferred). Decisions locked with the user: **kind**, **postgres + mlflow external**, **plain manifests**, images pulled from **GHCR**.

Why external: Postgres is shared infra managed by its own compose (`D:\personal_project\pgSQL\docker-compose.yml`, stock `postgres:16`, `pg_data` volume) hosting multiple projects; this project only owns the `prediction` DB. mlflow is stateful via host files (`sqlite:////mlflow.db` + `/mlruns`). Keeping both external is also the realistic production pattern (managed data services outside the app cluster) and lets the whole DB lift to a cloud managed DB later.

Proposed layout: new `k8s/` dir with `namespace.yaml`, `jobs/`, `cronjobs/`, `serving/`, `config/` (ConfigMaps/Secrets), `README.md`. (No `postgres/`, no `mlflow/`.)

### Milestone 2A — cluster + external-service connectivity
1. `k8s/kind-cluster.yaml` — kind config: 1 control-plane + 1 worker. Document `kind create cluster --config`.
2. `k8s/namespace.yaml` — namespace `image-classifier`.
3. `k8s/config/` — translate `.env.*`:
   - `ConfigMap`s for non-secret settings (`API_TRACKING_URI`, `API_EXPERIMENT_NAME`, `MONITORING_BASE_URL`, `crop_size`).
   - `Secret`s for `*_DB_URI` and creds. Local kind: connection string targets the **`prediction`** DB with real creds `admin`/`admin123`. Generated from a gitignored `.env` via `kubectl create secret --from-env-file` (documented, not committed). Note the cloud Secret Manager / CSI path later.
4. **Reach external postgres + mlflow from pods:** both publish to the host (`5432`, `5000`), so pods use `host.docker.internal` (Docker Desktop). Wrap each in a `Service` of type `ExternalName` (e.g. `postgres.image-classifier.svc.cluster.local` → `host.docker.internal`) so job manifests reference a stable in-cluster name and only the ExternalName target changes for cloud (→ managed DB / mlflow endpoint). No schema seeding — the external postgres already has the `prediction` DB.

### Milestone 2B — translate the jobs
7. One-off `Job` manifests (mechanical translation of `docker compose run --rm`):
   - `jobs/register-benchmark.yaml`, `jobs/compute-references.yaml` (uses api image). **No `jobs/train.yaml`** — training-as-K8s-Job is deliberately deferred to the cloud phase (see "Why `training` is excluded" in Phase 1); on `kind` (no GPU) it would offer zero benefit over `make train-run`.
8. Weekly pipelines (`label-backfill`, `drift-report`, `quality-report`) ship
   as plain one-off `Job` manifests for now (triggered manually / on demand).
   **The scheduling mechanism is deferred** — whether the weekly cadence is
   driven by K8s `CronJob`s or by an Airflow DAG is an open decision moved to
   `airflow-terraform-overview.md` (DAG 1). Revisit once Airflow lands; if it
   stays on K8s, wrap these Jobs in `CronJob`s with `schedule`, `backoffLimit`,
   `activeDeadlineSeconds`, `concurrencyPolicy: Forbid`, `startingDeadlineSeconds`.
9. **Volumes/mounts:** the reports dir and data/image mounts (`${O_DRIVE_PATH}`, `${TOOLS_LIB_PATH}`, `monitoring/reports`, `data/`) become `hostPath` for kind-local testing (explicitly flagged as a stand-in) or `PVC`s; the external `tools` lib editable-install entrypoint carries over unchanged in the image. Prefer manifest mode for register-benchmark to avoid the image/tools/Mongo mounts entirely.
10. **imagePullPolicy / pull secrets:** reference GHCR images; if packages are private, add an `imagePullSecret`.

### Milestone 2C — validate end-to-end
11. Bring up cluster → apply namespace/config/ExternalName services → verify pods can reach external postgres + mlflow (`kubectl run --rm -it ... psql/curl host.docker.internal`).
12. Run each `Job` manually (`kubectl create -f ...` or `kubectl create job --from=cronjob/...`); confirm rows written to the external postgres and artifacts to external mlflow.
13. Let a `CronJob` fire on a short test schedule, confirm, then set real weekly schedules.
14. `k8s/README.md` documenting the full up/down/run workflow (kind create, kubectl apply order, teardown).

### Milestone 2D — API serving on K8s
Moves the API off docker compose once the jobs are proven (2A–2C). This is
also where K8s starts paying off beyond "compose but harder": replicas,
readiness-gated rolling updates, and autoscaling.
15. `k8s/serving/api-deployment.yaml` — `Deployment` (replicas: 2) running
    the `api` image from GHCR: `envFrom` the api ConfigMap
    (`API_TRACKING_URI` → the mlflow ExternalName service,
    `API_EXPERIMENT_NAME`, `API_MODEL_NAME` or `API_RUN_NAME`,
    `API_CROP_SIZE`, `API_DEVICE=cpu`) + Secret (`API_DB_URI`); resource
    requests/limits (torch inference is memory-heavy at model load).
    **Probes:** the model is loaded in the FastAPI `lifespan` handler
    before the server accepts connections (`api/main.py`), so a slow first
    artifact download must not trip liveness — `startupProbe` on `/health`
    with a generous `failureThreshold`, then `readinessProbe` +
    `livenessProbe` on `/health` (which reports `model_loaded`, `device`,
    `database_connected`).
16. `k8s/serving/api-service.yaml` + `k8s/serving/api-hpa.yaml` — `Service`
    (`ClusterIP`; `kubectl port-forward` or `NodePort` for local kind
    access) and `HorizontalPodAutoscaler` (`autoscaling/v2`, CPU-based,
    e.g. min 1 / max 3).
17. **Rollout on model change** (this phase's version of promotion): the
    model source is pinned in the ConfigMap and resolved at pod startup, so
    a champion change = update the ConfigMap value, then
    `kubectl rollout restart deployment/api` — a readiness-gated rolling
    restart: zero-downtime reload, and the rollout halts automatically if
    the new model fails to load. Document the two-command flow in
    `k8s/README.md`; the Airflow promotion step later upgrades this to
    canary/shadow (deferred item below).

### Deferred within Phase 2 (recorded, not built now)
- Canary/shadow serving for promotion (see rationale below) — the eventual
  reason serving is on K8s; arrives with the Airflow-driven promotion step.
  Milestone 2D already ships the simpler version: a readiness-gated rolling
  restart on model change.
- Kustomize base + `local`/`cloud` overlays — introduce once manifest duplication is felt.
- Real GPU scheduling (node affinity/taints/tolerations) — needs a GPU node, i.e. cloud phase.

---

## Real-world Kubernetes use cases (reference)

K8s shows up in ML production in several genuinely distinct ways — worth
recognizing which one you're building toward at each step:

1. **Online inference serving.** Model wrapped in an HTTP server (FastAPI,
   TorchServe, Triton, KServe, BentoML, vLLM), run as a `Deployment` with N
   replicas behind a `Service`/`Ingress`, scaled by an `HorizontalPodAutoscaler`
   on CPU/GPU/latency/queue-depth. Model updates ship as rolling updates with
   readiness-probe gating (no downtime, automatic halt on a bad rollout).
   This is the strongest, most direct justification for K8s and is the
   eventual target for this project's `/predict` API + champion promotion
   (canary/shadow traffic between champion and challenger before flipping the
   alias). Real tools: **KServe**, **Seldon Core**, **NVIDIA Triton**,
   **vLLM/TGI** for LLMs.
2. **Scale-to-zero for expensive GPU inference.** K8s + **Knative/KServe**
   serverless mode spins a GPU Pod up only on request and tears it down when
   idle — the standard cost-control pattern for big models (LLMs,
   diffusion/image-gen) used only occasionally.
3. **Batch / offline inference and data pipelines.** Nightly/weekly scoring or
   re-embedding jobs as K8s `Job`/`CronJob` — retries, parallelism
   (`completions`/`parallelism`), resource limits. **This is this project's
   Phase 2**: drift-report, quality-report, label-backfill,
   compute-references.
4. **Training / experimentation at scale.** Training runs as K8s `Job`s
   requesting `nvidia.com/gpu: N`, scheduled onto a shared GPU pool (queued
   when busy). Distributed multi-GPU/multi-node training uses the **Kubeflow
   Training Operator** (`PyTorchJob`/`TFJob`) to manage worker Pods and
   networking. This project's `train.py` as a K8s Job (Milestone 2B) is the
   shape-only version of this; real GPU scheduling via node
   affinity/taints/tolerations is a cloud-phase concern.
5. **Full ML platforms built on K8s.** At larger orgs, K8s is the substrate
   the whole MLOps stack runs on: **Kubeflow** (pipelines, notebooks, training
   operators, KServe), **Airflow/Flyte/Argo Workflows** (task Pods via
   `KubernetesPodOperator`), **Ray/KubeRay** (distributed training/inference),
   plus feature stores, vector DBs, MLflow, experiment trackers all deployed
   as K8s services.

## Why Kubernetes is justified for *this* project (design rationale)

This project's actual data volume does not *require* K8s; it's a template for
practicing production-level MLOps (see `AGENTS.md`). The justifications below
are the real production patterns K8s unlocks — lean into them so the practice
mirrors real systems rather than cargo-culting:

- **Ephemeral job execution at scale:** `docker compose run --rm` ≈ a K8s
  `Job`/`CronJob`, but K8s adds real retry policies, resource requests/limits,
  and **GPU scheduling** via node affinity/taints — the actual production
  pattern for GPU training, which compose can't express.
- **Serving with replicas + rolling updates:** the API is a single container
  today; a `Deployment` + `Service` + `HorizontalPodAutoscaler` enables
  zero-downtime rolling updates on champion promotion and autoscaling under
  load — neither native to compose.
- **Canary / shadow deployment for promotion** (the one worth leaning into,
  since it upgrades the promotion design): run the candidate as a second
  `Deployment` receiving mirrored/shadow traffic (or a small % via K8s-native
  traffic split / service mesh), log its predictions alongside the champion,
  and only flip the `@champion` alias + primary `Service` selector after it
  proves out online. A real online+offline evaluation pattern that requires
  K8s-style multi-deployment traffic management — compose has no good answer.
  Phased: Milestone 2D ships the simpler readiness-gated rolling restart on
  model change now; full shadow/canary arrives with the Airflow promotion
  step.
- **Config/secrets/resource isolation** across mlflow/postgres/api/jobs via
  namespaces, `ConfigMap`/`Secret`, resource quotas — practicing the K8s
  object model itself.

---

## Cross-cutting risks / considerations
- Heavy CUDA base images make CD builds slow/large — rely on buildx GHA cache + path filters.
- Baking training code into its image changes the current bind-mount dev workflow — keep both paths working.
- kind has no GPU — training is excluded from Phase 2 entirely (stays on host compose) until a real GPU node pool makes K8s-based training scheduling worthwhile (cloud phase).
- In-repo postgres manifest is authored fresh (no existing compose to mirror) — must match the credentials/schema the app expects.
- Private GHCR packages require an imagePullSecret in every namespace that pulls them.
- Secrets must never be committed — generate K8s Secrets from gitignored env files locally; document the cloud Secret Manager path.

## Suggested execution order
1. Phase 1 (CD → GHCR) end-to-end, including baking training code.
2. Phase 2A (cluster + postgres + mlflow), then 2B (jobs/cronjobs), then 2C (validate).
3. Phase 2D (API Deployment + Service + HPA + rollout-on-model-change) once jobs are proven.
4. Move on to `airflow-terraform-overview.md` once 1–3 are working — the Airflow promotion step is what upgrades rollout-on-change to canary/shadow.

---

## Progress log
- [~] Phase 1: CD → GHCR
  - [x] Bake training code into `docker/jobs/train/Dockerfile` (`src/ configs/ utils/ train.py`)
  - [x] Split out lightweight `docker/jobs/build_dataset/Dockerfile` (+ `requirements/build_dataset_req.txt`) for `scripts/build_dataset.py`
  - [x] `.github/actions/build-push-image` composite action (login, buildx, metadata, build-push)
  - [x] `serving-build-push` job in `ci_serving.yml` (`needs: test-push-serving`)
  - [x] `monitoring-build-push` job in `ci_monitoring.yml` (`needs: [monitoring-unit, monitoring-db, monitoring-inference]`)
  - [x] `training-build-push` job in `ci_training.yml`, on-demand via `workflow_dispatch` (`run_tests: all` + `image_tag`), not on every push
  - [x] `cd_mlflow.yml` standalone build-push (no tests to gate on)
  - [x] `token` passed as composite-action input (secrets unavailable in composite actions); PR-push guards; compose files pull from GHCR
  - [x] Images & CD docs (`docker/IMAGES.md`)
  - [ ] Prereq: create `GHCR_TOKEN` repo secret (PAT w/ `write:packages`)
  - [ ] End-to-end verification (trigger workflows, confirm GHCR packages, `docker pull`/`docker run`)
- [ ] Phase 2: Kubernetes
  - [ ] 2A: cluster + config + ExternalName services (postgres + mlflow stay external)
  - [x] 2B: Job manifests (plain `Job`s; CronJob scheduling deferred to the Airflow phase — see airflow-terraform-overview.md; no training Job — see Phase 1 rationale)
  - [ ] 2C: validate end-to-end
  - [ ] 2D: API Deployment + Service + HPA + rollout-on-model-change (canary/shadow deferred to the Airflow promotion step)
