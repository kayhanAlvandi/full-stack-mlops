# Project TODO

## Serving: model/API dependency version mismatch (unresolved)

**Problem:**
Models are logged via `mlflow.pytorch.log_model(..., code_paths=["src"])`
(<ref_snippet file="/d:/personal_project/image_classifier/src/callbacks.py" lines="114-122" />),
which bundles the model class source code with the artifact, but does **not**
pin the runtime dependency versions used during training (e.g. `timm`,
`torch`). The serving API (`api/predictor.py`) loads models in-process via
`mlflow.pytorch.load_model()` inside a single container built from
`requirements/api_req.txt`
(<ref_file file="/d:/personal_project/image_classifier/docker/services/api/Dockerfile" />).

If different developers train models with different dependency versions
(e.g. `timm==1.0.26` vs `timm==1.0.28`), one fixed API container cannot
correctly serve all of them. We hit this already and worked around it by
pinning `timm==1.0.26` in `api_req.txt`, but that's not a real fix — it just
makes the container match whichever model was trained most recently.

**Options discussed (see chat history for full breakdown, brainstormed on 2026-08-11):**
1. Log `pip_requirements` / `extra_pip_requirements` with the model at log-model
   time, and switch serving to `mlflow.pyfunc.load_model()` (supports env
   isolation) — requires refactoring how `predictor.py` calls the model.
2. Build a dedicated Docker image per model/run, using the model's logged
   `conda.yaml` to generate exact requirements — no `predictor.py` changes,
   but adds a build/deploy pipeline step.
3. Dynamically `pip install` the model's exact requirements at API container
   startup (before loading the model) — single container, adapts per model,
   but slow startup and can't serve two conflicting-version models at once.
4. Pin exact versions across `training_req.txt` and `api_req.txt` and enforce
   everyone uses the same environment — zero code changes, simplest, but
   doesn't scale with multiple developers/experiments.

**Recommendation (not yet implemented):** Do (1) regardless — always log
`pip_requirements` with the model, it's cheap and gives us metadata/options
later. Combine with (3) short-term for flexibility, revisit (2) if startup
latency becomes a problem or models need to run concurrently with
conflicting dependency versions.

**Status:** Not started. Revisit before onboarding more developers to
training or before relying on this for production serving.

## Drift detection: parallelisation approach (decided, shelved)

**Context:**
`TilePredictor.predict()` already batches tiles per image (~100 tiles
per image → one `model(batch_of_100)` call). The GPU is already
well-utilized per image, so cross-image batching gives negligible gain.

**Decision: shelve `batch_predict`**
A `batch_predict()` method was designed (see
`docs/plans/batch-predict.md`) but is shelved because `predict()`
already batches tiles at the GPU level. Adding cross-image batching
would add complexity for no meaningful speedup.

**Why threads don't work here:**
- Psycopg 3 connections are not thread-safe. A shared `DBLogger`
  cannot be used across threads — parallel `predict()` calls sharing
  `self.db_logger` would corrupt connection state and return wrong IDs.
- Multiple threads calling `self.model(...)` on one GPU serialize at
  the hardware level (CUDA queues kernels sequentially) — no speedup.
- Python GIL limits CPU-side parallelism for preprocessing/tiling/DB
  logging to ~1.3-1.5x, not true N×.

**Future approach: multi-process with `--shard` and `--device`**
True parallelism comes from **one process per hardware resource**,
each with its own `DBLogger` connection and `TilePredictor` instance,
processing disjoint shards of the samples. No locks, no shared state.

```
# Multi-GPU (true N× speedup):
python compute_predictions_references.py --shard 0/2 --device cuda:0
python compute_predictions_references.py --shard 1/2 --device cuda:1

# CPU-only multi-process (true N× on CPU):
python compute_predictions_references.py --shard 0/4 --device cpu
python compute_predictions_references.py --shard 1/4 --device cpu
python compute_predictions_references.py --shard 2/4 --device cpu
python compute_predictions_references.py --shard 3/4 --device cpu

# Mixed GPU + CPU (GPU gets most samples, CPU offloads a few):
python compute_predictions_references.py --shard 0/3 --device cuda:0   # GPU, ~70% of samples
python compute_predictions_references.py --shard 1/3 --device cpu      # CPU, ~15% of samples
python compute_predictions_references.py --shard 2/3 --device cpu      # CPU, ~15% of samples
```

**Why multiple processes/pods are safe but threads are not:**
- Each process/pod has its **own Psycopg connection**. PostgreSQL uses
  MVCC and handles hundreds of concurrent connections natively.
- A single Psycopg connection shared across threads corrupts because
  the connection has internal protocol state (current query, transaction,
  result buffer) that can't be interleaved.
- The rule: **one worker = one process = one connection.**

**Mixed CPU+GPU caveat:**
CPU inference is 10-50x slower than GPU for typical CNN models. The
CPU processes become the bottleneck unless:
- The model is small (CPU only 5-10x slower).
- There are many CPU cores (8+) and thousands of samples.
- The GPU process gets the majority of samples.
For hundreds of reference samples, single-GPU `predict()` in a loop
is sufficient. Mixed CPU+GPU is only worth it for very large jobs.

**What NOT to do:**
- Do not `ThreadPoolExecutor` parallel `predict()` with a shared
  `DBLogger` — Psycopg 3 will break.
- Do not run multiple processes on the **same GPU** — CUDA serializes
  kernels across processes, so you get the same throughput with extra
  memory overhead (two model copies on GPU) and context-switch cost.
  The only exception is NVIDIA MPS, which is complex and not worth it
  here.
- Do not `multiprocessing` with a shared `DBLogger` — connections are
  not picklable across processes.

**Status:** Decision made. `batch_predict` shelved. Multi-process
`--shard`/`--device` approach documented for future implementation
when `compute_predictions_references.py` needs to scale beyond a single GPU.
Kubernetes horizontal scaling (step 7) uses the same pattern at
larger scale with a job queue.

## Benchmark dataset tracking (future)

- [ ] Add persistent version and lineage tracking for benchmark datasets,
      potentially using MLflow experiment tracking. Benchmark registration
      currently reads the generated manifests and metadata from the mounted
      `data/` directory, but those artifacts are not tracked in a dedicated
      registry.

**Status:** Not required for the current smoke test; revisit when benchmark
versions need to be compared, audited, or reproduced across environments.

## Serving: queue-based inference (deferred)

**Context:**
`/predict` is currently synchronous request/response with a semaphore-bounded
threadpool + CPU-based HPA. That remains correct for interactive, seconds-level
inference. This item is parked, not prioritized.

**Idea:**
Move to job-submission semantics: `POST /predict` enqueues and returns
`202 + job_id`; workers (separate Deployment reusing `TilePredictor` +
`DBLogger`, no HTTP layer) pull jobs; `GET /predict/{job_id}` returns
status/result.

- **Queue**: Redis (self-hosted, matches current footprint) or SQS if this
  ever goes to cloud.
- **Autoscaling**: KEDA scaled-object on queue depth — a better signal than
  CPU utilization since queue depth *is* the backlog, CPU lags behind it.

**When to revisit:** inference grows to minutes-long per image (clients can't
hold a connection that long anyway), or traffic becomes batchy/bulk-submission
rather than interactive.

## README polish

Feedback from a recruiter's-eye review: the README convinces engineers who read
carefully, but lacks *evidence and accessibility* for someone who skims for
~30 seconds. Items A, B, D are **top priority**.

### A. Visual proof — screenshots + demo video (P1)

Uncomment and fill in the "Visual tour" section (README ~line 48) with real
screenshots under `docs/images/`: MLflow runs & registry, an Evidently drift
report, `kubectl get pods`/`hpa`.

Record a **30–60 second demo GIF/video** showing the full lifecycle:

```
1. train model
       ↓
2. MLflow shows run
       ↓
3. model registered
       ↓
4. Kubernetes deployment
       ↓
5. curl/API prediction
       ↓
6. PostgreSQL receives prediction
       ↓
7. simulate production traffic
       ↓
8. monitoring detects drift
```

### B. Demo path — `make demo` quickstart (P1)

Today there is no "5 commands to a working prediction": setup requires
creating networks, an external Postgres compose file, MongoDB labels, and
machine-specific config — readers have to take the README's word for it.

- [ ] Add a `make demo` (or `docker compose` demo stack) that spins up MLflow +
      Postgres + API with **dummy labels/data** (`resolve_labels_dummy`, a small
      synthetic dataset) so anyone can verify the pipeline end to end.
- [ ] Use the demo path as the basis for the demo video in (A).

### C. Density & representation (P2)

README is ~400 lines; the Usage section is heavy. Consider:

- A table of contents.
- Moving the Postgres `docker-compose` YAML blob (README ~lines 189–215) into
  the repo as an actual file the README links to.
- Tightening the Features/Highlights overlap — fold unique bullets up into
  Highlights and keep Features as the detailed inventory (or vice versa).

### D. Minor polish (P1)

- [ ] Rename `k8s/api/api-deployement.yaml` → `api-deployment.yaml` (typo).
- [ ] `tests/diagnostics/` advertised as "old exploratory scripts" (README
      lines ~137, ~342) undercuts the polish — either delete the directory or
      stop mentioning it in the README.
- [ ] Credentials in snippets (`admin:admin123456`, README ~line 360) — add a
      one-line "dev-only credentials" note.
- [ ] No license and no author/context line — add one sentence near the top
      ("built by X as a portfolio project demonstrating production MLOps
      patterns") so a recruiter knows the intent immediately.

### E. `docs/plans/` as an asset (P2)

Design docs (`airflow-terraform-overview.md`, `kubernetes-plan.md`,
`mlops-orchestration.md`, `batch-predict.md`) are good interview fodder but
invisible. Add one README line linking to them as "design decisions / roadmap".



