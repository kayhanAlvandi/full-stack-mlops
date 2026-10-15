"""Build a dataset definition (metadata + manifest) from a short spec.

A standalone counterpart to training's dataset artifacts: instead of describing
the dataset a model happened to train on, it describes any dataset defined purely
by selection filters -- primarily the fixed benchmark set used as the supervised
quality baseline (``docs/plans/quality-tracking.md``).

Run it independently (dev machine or the training image); it is not a monitoring
job. Output goes to ``<output_root>/<name>/``:

    dataset_metadata.json   version/hash, class_names, per-class counts, spec
    dataset_manifest.json   one entry per sample: plate/well/field/label/
                            root_path/channel_files/shape

Those two files are the frozen, reviewable definition of the dataset. Downstream
consumers only read them -- ``monitoring/register_benchmark.py --manifest ...``
inserts the samples into Postgres and needs no filesystem, MongoDB or MLflow
access of its own.

Usage:
    python scripts/build_dataset.py dataset_spec=benchmark name=benchmark_v1
    python scripts/build_dataset.py dataset_spec=benchmark \\
        dataset_spec.max_samples_per_label=4 name=benchmark_small
"""
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.dataset_versioning import (
    create_manifest_from_samples,
    create_metadata_from_samples,
)
from src.sample_selection import DatasetSpec, build_dataset


def spec_from_cfg(cfg: DictConfig, name: str) -> DatasetSpec:
    """Turn the resolved hydra ``dataset_spec`` node into a ``DatasetSpec``."""
    raw = OmegaConf.to_container(cfg.dataset_spec, resolve=True)

    def _wells(key):
        value = raw.get(key)
        return [tuple(w) for w in value] if value else None

    return DatasetSpec(
        root_dir=raw["root_dir"],
        channels=list(raw["channels"]),
        label_provider=raw.get("label_provider"),
        exclude_wells=_wells("exclude_wells"),
        include_wells=_wells("include_wells"),
        max_wells_per_label=raw.get("max_wells_per_label"),
        max_samples_per_label=raw.get("max_samples_per_label"),
        seed=raw.get("seed", 42),
        verbose=raw.get("verbose", False),
        name=name,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="build_dataset")
def main(cfg: DictConfig) -> None:
    name = cfg.name or cfg.dataset_spec.get("name") or "dataset"
    spec = spec_from_cfg(cfg, name)

    print("=" * 60)
    print(f"Building dataset '{name}'")
    print(OmegaConf.to_yaml(cfg.dataset_spec))
    print("=" * 60)

    samples = build_dataset(spec, shape_mode=cfg.shape_mode)
    if not samples:
        print("No samples selected -- nothing written. Check root_dir, channels, "
              "include_wells/exclude_wells and that the label source has these wells.")
        return

    metadata = create_metadata_from_samples(samples, spec, name=name)

    out_dir = Path(cfg.output_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "dataset_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    manifest_path = create_manifest_from_samples(samples, str(out_dir / "dataset_manifest.json"))

    print("=" * 60)
    print(f"Dataset version: {metadata['dataset_version']}")
    print(f"Classes ({metadata['num_classes']}): {metadata['class_names']}")
    print(f"Samples per class: {metadata['samples_per_class']}")
    print(f"Wells ({metadata['num_wells']}): {metadata['wells_used']}")
    print(f"Total samples: {metadata['total_samples']}")
    print(f"\nWrote {metadata_path}")
    print(f"Wrote {manifest_path}")
    print("\nReview these files, then commit them -- a benchmark every future "
          "model is compared against should be frozen, not rebuilt per use.")


if __name__ == "__main__":
    main()
