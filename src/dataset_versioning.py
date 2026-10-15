"""Dataset versioning utilities for tracking dataset composition and code.

Two entry points, one shape of output:

- ``create_dataset_metadata`` / ``create_dataset_manifest`` describe a *training*
  dataset (a Lightning datamodule with train/val splits), logged to MLflow.
- ``create_metadata_from_samples`` / ``create_manifest_from_samples`` describe any
  dataset built from a ``DatasetSpec`` (``src/sample_selection.py``) -- e.g. the
  fixed benchmark set -- saved locally under ``data/<name>/``.

Both emit the same per-sample entries, so a manifest can be consumed the same
way regardless of which produced it.
"""
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def get_git_commit_for_model_files() -> str | None:
    """Get the last git commit hash that modified model-related files."""
    model_files = [
        'src/model.py',
    ]
    
    try:
        git_hash = subprocess.check_output(
            ['git', 'log', '-1', '--format=%H', '--'] + model_files,
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        return git_hash if git_hash else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def check_model_uncommitted_changes() -> bool:
    """Check if there are uncommitted changes to model files."""
    model_files = ['src/model.py']
    try:
        result = subprocess.check_output(
            ['git', 'status', '--porcelain', '--'] + model_files,
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        return len(result) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_git_commit_for_dataset_files() -> str | None:
    """Get the last git commit hash that modified dataset-related files."""
    dataset_files = [
        'src/dataset.py',
        'src/datamodule.py',
        'src/transforms.py',
    ]
    
    try:
        # Get last commit that touched any of these files
        git_hash = subprocess.check_output(
            ['git', 'log', '-1', '--format=%H', '--'] + dataset_files,
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        return git_hash if git_hash else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def check_uncommitted_changes() -> bool:
    """Check if there are uncommitted changes to dataset files."""
    dataset_files = [
        'src/dataset.py',
        'src/datamodule.py',
        'src/transforms.py',
    ]
    
    try:
        # Check for uncommitted changes
        result = subprocess.check_output(
            ['git', 'status', '--porcelain', '--'] + dataset_files,
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        return len(result) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Compute hash of the datamodule configuration dict."""
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()


def compute_dataset_version(config_dict: dict[str, Any]) -> str:
    """
    Compute dataset version from git commit + config.
    
    Format: {git_commit[:8]}_{config_hash[:8]}
    Changes only when dataset code or data/dataloader config changes.
    """
    git_commit = get_git_commit_for_dataset_files()
    config_hash = compute_config_hash(config_dict)
    
    if git_commit:
        code_part = git_commit[:8]
    else:
        code_part = "no_git"
    
    # Warn about uncommitted changes
    if check_uncommitted_changes():
        print("[WARNING] Dataset code has uncommitted changes!")
        print("  Commit your changes for full reproducibility.")
        code_part = f"{code_part}_dirty"
    
    return f"{code_part}_{config_hash[:8]}"


def create_dataset_metadata(
    datamodule,
    config,
    train_samples: int,
    val_samples: int,
) -> dict[str, Any]:
    """
    Create comprehensive dataset metadata for MLflow logging.
    
    Args:
        datamodule: PyTorch Lightning DataModule instance
        config: Configuration object
        train_samples: Number of training samples
        val_samples: Number of validation samples
    
    Returns:
        Dictionary with dataset metadata
    """
    # Get label encoder
    label_encoder = datamodule.label_encoder
    
    # Collect wells used (if available)
    wells_used = []
    if hasattr(datamodule, 'train_dataset') and hasattr(datamodule.train_dataset, 'samples'):
        wells_set = set()
        for sample in datamodule.train_dataset.samples:
            if isinstance(sample, dict) and ('plate' in sample) and ('well' in sample):
                wells_set.add(f"{sample['plate']}/{sample['well']}")
        wells_used = sorted(wells_set)
        
        if not wells_used and len(datamodule.train_dataset.samples) > 0:
            # Debug: print first sample structure if wells_used is empty
            print(f"[WARNING] Could not extract wells. First sample structure: {list(datamodule.train_dataset.samples[0].keys())}")
    
    # Use the full datamodule config as-is for hashing and metadata
    from omegaconf import OmegaConf
    config_dict = OmegaConf.to_container(config.datamodule, resolve=True)
    
    metadata = {
        'dataset_version': compute_dataset_version(config_dict),
        'dataset_code_commit': get_git_commit_for_dataset_files(),
        'config_hash': compute_config_hash(config_dict),
        'has_uncommitted_changes': check_uncommitted_changes(),
        'num_classes': label_encoder.num_classes,
        'class_names': label_encoder.classes,
        'train_samples': train_samples,
        'val_samples': val_samples,
        'total_samples': train_samples + val_samples,
        'wells_used': wells_used,
        'datamodule_config': config_dict,
    }
    
    return metadata


def _manifest_entry(idx: int, sample: dict[str, Any]) -> dict[str, Any]:
    """Convert one selected sample into its portable manifest entry.

    Filenames are stored without their directory (combined with the entry's
    ``root_path`` at read time: ``Path(root_path) / filename``) so the manifest
    stays valid across machines and containers with different mount points.
    """
    root_paths = [path.parent for path in sample['channel_files'].values()]
    assert len(set(root_paths)) == 1, "All channel files should have the same root path"

    entry = {
        'index': idx,
        'plate': sample['plate'],
        'well': sample['well'],
        'field': sample['field'],
        "root_path": str(root_paths[0]),
        'channel_files': {
            channel: path.name for channel, path in sample['channel_files'].items()
        },
    }
    if 'label' in sample:
        entry['label'] = sample['label']
    if sample.get('image_size'):
        # (height, width) -- persisted so consumers (e.g. benchmark registration
        # writing image_metadata) don't have to reopen every image file.
        entry['shape'] = list(sample['image_size'])
    return entry


def create_dataset_manifest(
    datamodule,
    output_path: str = "dataset_manifest.json"
) -> str:
    """
    Create a detailed manifest of all samples in the dataset.
    
    Args:
        datamodule: PyTorch Lightning DataModule
        output_path: Path to save manifest
    
    Returns:
        Path to manifest file
    """
    manifest = {
        'train_samples': [],
        'val_samples': [],
    }

    for key, attr in (('train_samples', 'train_dataset'), ('val_samples', 'val_dataset')):
        dataset = getattr(datamodule, attr, None)
        if dataset is None:
            continue
        manifest[key] = [
            _manifest_entry(idx, sample) for idx, sample in enumerate(dataset.samples)
        ]

    # Save manifest
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return output_path


def create_metadata_from_samples(
    samples: list[dict[str, Any]],
    spec,
    name: str | None = None,
) -> dict[str, Any]:
    """Create dataset metadata for a spec-built (non-training) dataset.

    Mirrors ``create_dataset_metadata`` but takes a plain sample list plus the
    ``DatasetSpec`` that produced it, so it works for datasets that have no
    train/val split, no datamodule and no model -- e.g. the benchmark set.

    ``class_names`` uses the same ``sorted(set(...))`` rule as
    ``LabelEncoder.fit``, so it is directly comparable with a model's logged
    ``class_names`` (see the coverage warning in
    ``docs/plans/dataset-builder.md``).
    """
    from collections import Counter

    spec_dict = spec.to_dict() if hasattr(spec, 'to_dict') else dict(spec)
    wells_used = sorted({f"{s['plate']}/{s['well']}" for s in samples})

    # Unlabelled dataset (e.g. real production data with no ground truth
    # yet, built with label_provider=None): samples have no 'label' key at
    # all, so there is no class vocabulary to report.
    has_labels = bool(samples) and 'label' in samples[0]
    if has_labels:
        labels = [s['label'] for s in samples]
        class_names = sorted(set(labels))
        samples_per_class = dict(sorted(Counter(labels).items()))
    else:
        class_names = []
        samples_per_class = {}

    return {
        'name': name or spec_dict.get('name', 'dataset'),
        'dataset_version': compute_dataset_version(spec_dict),
        'dataset_code_commit': get_git_commit_for_dataset_files(),
        'config_hash': compute_config_hash(spec_dict),
        'has_uncommitted_changes': check_uncommitted_changes(),
        'num_classes': len(class_names),
        'class_names': class_names,
        'total_samples': len(samples),
        'samples_per_class': samples_per_class,
        'wells_used': wells_used,
        'num_wells': len(wells_used),
        'spec': spec_dict,
    }


def create_manifest_from_samples(
    samples: list[dict[str, Any]],
    output_path: str = "dataset_manifest.json",
) -> str:
    """Write a flat manifest for a spec-built dataset: ``{"samples": [...]}``.

    Entries have the same shape as the train/val entries in
    ``create_dataset_manifest``, so downstream loaders are interchangeable.
    """
    manifest = {
        'samples': [_manifest_entry(idx, sample) for idx, sample in enumerate(samples)],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return output_path

