# Makefile — independent services connected via external Docker network

# ── Network (shared by all services/jobs) ────────────────────────────────
network:
	-docker network create ml-platform

# ── Services (long-running) ──────────────────────────────────────────────
mlflow: network
	docker compose -f services/mlflow/docker-compose.yaml up -d

serve: network
	docker compose -f services/api/docker-compose.yaml up -d

# Dev: bind-mounts api/utils/database/monitoring source + --reload, no rebuild on edit
serve-dev: network
	docker compose -f services/api/docker-compose.yaml -f services/api/docker-compose.dev.yaml up -d --build

# ── Jobs (ephemeral, run-and-exit) ───────────────────────────────────────
# Usage: make train-run CMD="python train.py trainer=trainer_container model=vit_large"
train-run: network
	docker compose -f jobs/train/docker-compose.yaml run --rm training $(CMD)

# Usage: make compute-predictions-references (reads model/db config from
# jobs/compute_references/.env.api). Computes both this run's validation
# reference (drift baseline) and the global benchmark reference (quality
# baseline) in one model load. Override target with e.g.
# CMD="python -m monitoring.compute_predictions_references --target val"
compute-predictions-references: network
	docker compose -f jobs/compute_references/docker-compose.yaml run --rm compute-predictions-references \
		$(if $(CMD),$(CMD),python -m monitoring.compute_predictions_references)

# Usage: make drift-report-run  (or CMD="python -m monitoring.run_drift_report --window-days 14")
# Resolves the served model's run_id from the running api's /model endpoint.
drift-report-run: network
	docker compose -f jobs/monitoring/docker-compose.yaml run --rm drift-report \
		$(if $(CMD),$(CMD),python -m monitoring.run_drift_report)

# Usage: make quality-report-run  (or CMD="python -m monitoring.run_quality_report --window-days 14")
# Compares a served model's benchmark quality baseline against its labeled
# current-window quality. Resolves run_id the same way as drift-report-run.
quality-report-run: network
	docker compose -f jobs/monitoring/docker-compose.yaml run --rm quality-report \
		$(if $(CMD),$(CMD),python -m monitoring.run_quality_report)

# Usage: make label-backfill-run
#    or: make label-backfill-run CMD="python -m monitoring.backfill_labels --source csv --csv-path /mnt/O/plate_well_labels.csv --label-col Treatment"
# Backfills t_label on production predictions from MongoDB (default) or a CSV, global by well.
label-backfill-run: network
	docker compose -f jobs/monitoring/docker-compose.yaml run --rm label-backfill \
		$(if $(CMD),$(CMD),"python -m monitoring.backfill_labels --source csv --csv-path /mnt/O/plate_well_labels.csv --label-col Treatment")

# Usage: make create-benchmark NAME=benchmark_v1
#    (optional extra hydra overrides: ARGS="dataset_spec.max_samples_per_label=4")
create-benchmark:
	$(if $(NAME),,$(error NAME is required, e.g. make create-benchmark NAME=benchmark_v1))
	docker compose -f jobs/build_dataset/docker-compose.yaml run --rm build_dataset \
		python scripts/build_dataset.py dataset_spec=benchmark name=$(NAME) $(ARGS)

# Usage: make create-custom-dataset NAME=custom_v1
#    (optional extra hydra overrides: ARGS="dataset_spec.max_samples_per_label=4")
create-custom-dataset:
	$(if $(NAME),,$(error NAME is required, e.g. make create-custom-dataset NAME=custom_v1))
	docker compose -f jobs/build_dataset/docker-compose.yaml run --rm build_dataset \
		python scripts/build_dataset.py dataset_spec=custom-dataset name=$(NAME) $(ARGS)

# Usage: make register-benchmark-run MANIFEST=data/benchmark_v1/dataset_manifest.json
#    advanced/glob mode: make register-benchmark-run CMD="python -m monitoring.register_benchmark '/mnt/O/benchmark/*.jxl'"
# One-time/additive: registers curated benchmark samples (never in any
# training set) into the database. Model-independent, no inference.
# Build the manifest first: make create-benchmark NAME=<name>
register-benchmark-run: network
	$(if $(or $(MANIFEST),$(CMD)),,$(error provide MANIFEST=<path> (manifest mode) or CMD=... (advanced/glob mode)))
	docker compose -f jobs/monitoring/docker-compose.yaml run --rm register-benchmark \
		$(if $(CMD),$(CMD),python -m monitoring.register_benchmark --manifest $(MANIFEST) $(ARGS))

# Usage: make mlflow-gc                      (permanently purges soft-deleted runs + their registry entries)
#    or: make mlflow-gc ARGS="--dry-run"      (preview only, no changes)
#    or: make mlflow-gc ARGS="--older-than 30d"
# Deletes registered model versions pointing at runs already trashed in the
# MLflow UI, then runs `mlflow gc` to permanently remove those runs' metadata
# and artifacts and reclaim disk space. Piped over stdin -- no volume mount
# or image rebuild needed.
mlflow-gc:
	docker compose -f services/mlflow/docker-compose.yaml exec -T mlflow python - $(ARGS) < ../scripts/mlflow_purge_deleted.py

# ── Teardown (each is independent — won't affect others) ─────────────────
down-mlflow:
	docker compose -f services/mlflow/docker-compose.yaml down

down-serve:
	docker compose -f services/api/docker-compose.yaml down

down-all: down-serve down-mlflow