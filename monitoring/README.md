# Monitoring

Everything needed to keep a served model's predictions trustworthy after
deployment: input/prediction drift detection, and supervised quality tracking
once ground-truth labels become available. All jobs are ephemeral (run,
finish, exit) and share the normal `image_prediction`/`tile_prediction`
tables with production traffic -- distinguished only by two columns:

| | `is_reference` | `benchmark_id` |
|---|---|---|
| Live production traffic | `FALSE` | `NULL` |
| This run's own validation set | `TRUE` | `NULL` |
| The fixed, model-independent benchmark set | `FALSE` | not `NULL` |

Three canonical view pairs enforce this split so no query has to remember all
the flag combinations itself: `live_image_prediction`/`live_tile_prediction`,
`reference_image_prediction`/`reference_tile_prediction`,
`benchmark_image_prediction`/`benchmark_tile_prediction`
(`database/init/02_reference.sql`, `03_benchmark.sql`).

## Two baselines, two questions

- **Drift** (`is_reference`) asks *"does current input/output look like what
  this model was validated on?"* -- unsupervised, no ground truth needed,
  compares production traffic against **this run's own validation split**.
- **Quality** (`benchmark_id`) asks *"is this model still accurate?"* --
  supervised, needs labels, compares a labeled production window against
  **the fixed benchmark set**, so every model's score is on the same samples
  and directly comparable across runs.

These are deliberately two different baselines: a model's validation split
changes every time it's retrained, so it can't be used to compare quality
*across* models; the benchmark set never changes, so it can.

## The jobs, in the order you'd normally run them

### 1. `register_benchmark.py` -- one-time, model-independent
Registers the curated benchmark set (samples never in any model's training
set) into `benchmark_dataset`/`benchmark_dataset_member`/`image_metadata`.
No model, no inference -- just recording which samples exist and their known
`t_label`. Preferred input is a frozen manifest built by
`scripts/build_dataset.py` (labels + image shapes already resolved, so this
job touches neither the image files nor MongoDB); a fallback glob mode exists
for ad hoc use. Idempotent -- keyed on `(plate, well, field)`, safe to rerun
as the benchmark set grows.

```bash
python scripts/build_dataset.py dataset_spec=benchmark name=benchmark_v1
make register-benchmark-run MANIFEST=data/benchmark_v1/dataset_manifest.json
```

### 2. `compute_predictions_references.py` -- per served run
Scores **both** baselines for a specific model/run in one model load:

- `--target val`: this run's own validation samples (from its
  `dataset_manifest.json` MLflow artifact) -> reference rows
  (`is_reference=TRUE`).
- `--target benchmark`: the registered benchmark set -> benchmark rows
  (`benchmark_id=<id>`).
- `--target both` (default): both.

Resumable per target -- skips samples already scored for that `run_id`.

```bash
make compute-predictions-references
# or just one side:
make compute-predictions-references CMD="python -m monitoring.compute_predictions_references --target val"
```

### 3. Live traffic
Production predictions land in the same tables automatically through the
API's normal `/predict` path (`is_reference=FALSE`, `benchmark_id=NULL`,
`t_label=NULL` until backfilled). For testing/smoke-testing without a real
acquisition pipeline, `scripts/simulate_live_predictions.py` replays a
dataset-builder manifest against a running API at randomized intervals, one
sample at a time, from the host (not through Docker -- it stands in for an
external client):

```bash
python scripts/build_dataset.py dataset_spec=custom-dataset name=custom_v1
python scripts/simulate_live_predictions.py --manifest data/custom_v1/dataset_manifest.json --loop
```

### 4. `backfill_labels.py` -- whenever new ground truth is available
Finds production rows still missing `t_label`, resolves each `(plate,
well)`'s treatment from MongoDB, and writes it onto every matching row
(image- and tile-level). Global by well, not scoped to a `run_id` -- a label
is a property of the physical sample, so it updates every model's rows for
that well at once. Reference/benchmark rows already carry known labels and
are untouched. Idempotent -- only `NULL` labels are filled.

```bash
make label-backfill-run
```

### 5a. `run_drift_report.py` -- unsupervised, no labels needed
Compares a window of live traffic against the run's reference set with
Evidently, across three groups (`image_level`, `tile_level`,
`channel_stats` -- per-channel pixel statistics, since they have different
row cardinalities and can't share one DataFrame). Both sides are compared on
**predicted** label (`p_label`), not true label -- production traffic has no
ground truth. Writes HTML/JSON reports under `monitoring/reports/` and one
`drift_report` (+ `drift_report_column`) row per run.

```bash
make drift-report-run
# or: CMD="python -m monitoring.run_drift_report --window-days 14"
```

### 5b. `run_quality_report.py` -- supervised, needs backfilled labels
Compares the run's benchmark accuracy/F1 (the baseline) against the labeled
portion of a live production window, using Evidently's
`ClassificationPreset` (accuracy, F1, confusion matrix) on (`p_label`,
`t_label`). Skips silently if there are zero labeled current-window samples
-- run backfill first. Writes a report + one `quality_report` row.

```bash
make quality-report-run
```

Both report jobs resolve `run_id` the same way: an explicit `--run-id`, or
read off the running API's `/model` endpoint (`monitoring/config.py`) --
whichever model is actually being served is whichever model gets reported
on, no MLflow access needed by the report jobs themselves.

## Why the same tables, not parallel ones

A validation-set or benchmark prediction goes through the *exact same*
tiling/predict pipeline as a production one (`api/predictor.py`'s
`TilePredictor.predict()`), so it's structurally identical -- only the
`is_reference`/`benchmark_id`/`t_label` metadata passed alongside differs.
Mirroring the schema into separate reference/benchmark tables would mean
every query, index and migration exists twice for no benefit; a flag (plus
one nullable FK) does the same job with one set of tables, one set of
indexes, and reports that can `UNION`/compare across categories trivially.

## Environment

`monitoring/config.py`'s `MonitoringSettings` (env prefix `MONITORING_`,
loaded from `.env`):

| Variable | Default | Purpose |
|---|---|---|
| `MONITORING_DB_URI` | `None` | Postgres connection string |
| `MONITORING_BASE_URL` | `http://localhost:8000` | Running API, to resolve `run_id` off `/model` |

`compute_predictions_references.py` and `register_benchmark.py`'s fallback
glob mode additionally need model-serving settings (`API_MODEL_NAME` /
`API_RUN_NAME` / `API_DB_URI` / ...) via `api/config.py`'s `Settings`, since
they instantiate `TilePredictor` directly rather than going through the API.

## Running on Kubernetes

The `make` targets above run each job once via `docker compose` (good for local
iteration). The same jobs also ship as `batch/v1` Job manifests under `k8s/jobs/`
(`register-benchmark`, `compute-references`, `label-backfill`, `drift-report`,
`quality-report`), run against a local `kind` cluster with Postgres/MLflow/Mongo
kept external. See `k8s/README.md` for the cluster setup and the per-job
`kubectl apply -f jobs/<name>.yaml` workflow. Scheduling (K8s CronJob vs. an
Airflow DAG) is deferred — see `docs/plans/airflow-terraform-overview.md`.

## See also

- `docs/plans/drift-detection.md`, `docs/plans/quality-tracking.md`,
  `docs/plans/dataset-builder.md` -- original design docs (kept for context;
  flagged where later refactors superseded them).
- `TODO.md` -- open items (dependency-version mismatch, benchmark dataset
  version tracking, multi-process scaling for `compute_predictions_references.py`).
