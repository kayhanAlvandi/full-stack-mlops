#!/usr/bin/env python
"""Permanently purge MLflow runs that are already soft-deleted (e.g. via the UI).

Soft-deleting a run in the MLflow UI only flips its lifecycle stage to
`deleted` -- the run's metadata and artifacts stay on disk. This script:

  1. Finds all runs currently in the `deleted` lifecycle stage.
  2. Deletes any registered model versions that point at those runs, so the
     registry doesn't end up with dangling references once artifacts are gone.
  3. Runs `mlflow gc` to permanently remove the runs' metadata and artifacts.

`mlflow gc` does not check the registry on its own, so step 2 has to happen
first -- otherwise a registered model version can keep pointing at a run
whose artifact files have just been deleted from disk.

Usage:
    python scripts/mlflow_purge_deleted.py
    python scripts/mlflow_purge_deleted.py --tracking-uri http://localhost:5000
    python scripts/mlflow_purge_deleted.py --dry-run
    python scripts/mlflow_purge_deleted.py --older-than 30d
"""
import argparse
import subprocess
import sys

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient


def find_deleted_runs(client: MlflowClient) -> list[str]:
    """Return run IDs currently in the `deleted` lifecycle stage, across all experiments."""
    run_ids = []
    for exp in client.search_experiments(view_type=ViewType.ALL):
        runs = client.search_runs(
            [exp.experiment_id],
            run_view_type=ViewType.DELETED_ONLY,
            max_results=50000,
        )
        run_ids.extend(r.info.run_id for r in runs)
    return run_ids


def find_model_versions_for_runs(client: MlflowClient, run_ids: set[str]) -> list[tuple[str, str, str]]:
    """Return (model_name, version, run_id) for registered model versions pointing at run_ids."""
    conflicts = []
    for model in client.search_registered_models():
        for mv in client.search_model_versions(f"name='{model.name}'"):
            if mv.run_id in run_ids:
                conflicts.append((model.name, mv.version, mv.run_id))
    return conflicts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracking-uri", default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument("--backend-store-uri", default="sqlite:////mlflow-data/mlflow.db", help="Backend store URI passed to `mlflow gc`")
    parser.add_argument("--artifacts-destination", default="/mlflow-data/mlruns", help="Artifact root passed to `mlflow gc`")
    parser.add_argument("--older-than", default=None, help="Only purge runs deleted more than this long ago, e.g. '30d'")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting anything")
    args = parser.parse_args()

    client = MlflowClient(tracking_uri=args.tracking_uri)

    deleted_run_ids = find_deleted_runs(client)
    print(f"Found {len(deleted_run_ids)} run(s) in the 'deleted' lifecycle stage.")
    if not deleted_run_ids:
        return

    conflicts = find_model_versions_for_runs(client, set(deleted_run_ids))
    if conflicts:
        print(f"\n{len(conflicts)} registered model version(s) point at deleted runs and will be removed:")
        for name, version, run_id in conflicts:
            print(f"  {name} v{version}  (run {run_id[:8]})")
    else:
        print("\nNo registered model versions reference the deleted runs.")

    if args.dry_run:
        print("\nDry run: no changes made. Re-run without --dry-run to apply.")
        return

    for name, version, _ in conflicts:
        client.delete_model_version(name, version)
    if conflicts:
        print(f"\nDeleted {len(conflicts)} model version(s) from the registry.")

    gc_cmd = [
        "mlflow", "gc",
        "--backend-store-uri", args.backend_store_uri,
        "--artifacts-destination", args.artifacts_destination,
        "--tracking-uri", args.tracking_uri,
    ]
    if args.older_than:
        gc_cmd += ["--older-than", args.older_than]

    print(f"\nRunning: {' '.join(gc_cmd)}")
    result = subprocess.run(gc_cmd, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
