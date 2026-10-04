#!/usr/bin/env python
"""Manually register a model for a run that finished training but never got
registered to the MLflow Model Registry (e.g. because LogBestModelToMLflow
failed silently -- see src/callbacks.py).

This re-creates what LogBestModelToMLflow.on_train_end would normally do,
using only artifacts already stored on the run (hydra_config.yaml +
checkpoints/*.ckpt) -- no retraining or dataset access required.

Usage:
    python scripts/register_missing_model.py <run_id>
    python scripts/register_missing_model.py <run_id> --tracking-uri http://localhost:5000
"""
import argparse
import sys
from pathlib import Path

import mlflow
import torch
from hydra.utils import get_class
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf


def register_missing_model(run_id: str, tracking_uri: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    existing = [
        mv for m in client.search_registered_models()
        for mv in client.search_model_versions(f"name='{m.name}'")
        if mv.run_id == run_id
    ]
    if existing:
        print(f"Run {run_id} is already registered: "
              f"{existing[0].name} v{existing[0].version}. Nothing to do.")
        return

    print(f"Loading hydra_config.yaml for run {run_id}...")
    config_path = Path(mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="hydra_config.yaml", tracking_uri=tracking_uri,
    ))
    cfg = OmegaConf.load(config_path)

    model_class = get_class(cfg.model._target_)
    registered_model_name = model_class.__name__
    print(f"Model class: {registered_model_name}")

    ckpt_artifacts = client.list_artifacts(run_id, path="checkpoints")
    ckpt_artifacts = [a for a in ckpt_artifacts if a.path.endswith(".ckpt") and "last" not in a.path]
    if not ckpt_artifacts:
        raise FileNotFoundError(
            f"No best checkpoint found under runs:/{run_id}/checkpoints/ -- "
            "nothing to register (the run likely never saved one)."
        )
    ckpt_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=ckpt_artifacts[0].path, tracking_uri=tracking_uri,
    )
    print(f"Downloaded checkpoint: {ckpt_path}")

    print("Loading model from checkpoint...")
    model = model_class.load_from_checkpoint(ckpt_path)
    model.eval()
    model.cpu()

    in_channels = len(cfg.datamodule.dataset.channels)
    crop_size = cfg.datamodule.dataset.crop_size
    example_input = torch.zeros(1, in_channels, crop_size, crop_size)
    with torch.no_grad():
        example_output = model(example_input)
    signature = infer_signature(example_input.numpy(), example_output.numpy())

    val_acc = run.data.metrics.get("val/acc")
    run_name = run.info.run_name or run_id[:8]

    with mlflow.start_run(run_id=run_id):
        model_info = mlflow.pytorch.log_model(
            model,
            name="model",
            signature=signature,
            input_example=example_input,
            registered_model_name=registered_model_name,
            code_paths=["src"],
            serialization_format="pickle",
        )

    version = model_info.registered_model_version
    description = (
        f"Run: {run_name}\n"
        f"Backbone: {cfg.model.get('backbone_name', '?')}\n"
        f"Val acc: {val_acc:.4f}\n" if val_acc is not None else f"Run: {run_name}\n"
    )
    client.update_model_version(registered_model_name, version, description=description)
    client.set_model_version_tag(registered_model_name, version, "run_name", run_name)
    client.set_model_version_tag(registered_model_name, version, "recovered_manually", "true")
    if val_acc is not None:
        client.set_model_version_tag(registered_model_name, version, "val_score", f"{val_acc:.4f}")

    print(f"\nRegistered {registered_model_name} v{version} (run_id={run_id[:8]}...)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", help="MLflow run ID whose model was never registered")
    parser.add_argument("--tracking-uri", default="http://localhost:5000", help="MLflow tracking URI")
    args = parser.parse_args()

    try:
        register_missing_model(args.run_id, args.tracking_uri)
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
