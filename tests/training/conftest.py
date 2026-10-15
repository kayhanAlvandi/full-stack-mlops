"""Shared pytest fixtures for the training test suite.

Provides synthetic, on-disk multi-channel .tif images that match the real
dataset filename pattern, so unit tests can exercise the dataset / datamodule
code paths without needing the real microscopy dataset or MongoDB.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

IMG_SIZE = 64
CHANNELS = [1, 2, 3]


def _write_image(path: Path, size: int = IMG_SIZE, seed: int = 0) -> None:
    """Write a single synthetic uint16 grayscale .tif image."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 65535, size=(size, size), dtype=np.uint16)
    cv2.imwrite(str(path), img)


@pytest.fixture
def synthetic_image_dir(tmp_path):
    """Create synthetic multi-channel .tif images on disk.

    Layout: 1 plate, 4 wells (A01, A02, B01, B02), 1 field each, 3 channels.
    Filenames follow the pattern parsed by ``MultiChannelImageDataset``:
        {plate}_{well}_T001F001L01A01Z01C0{channel}.tif

    Returns a dict with the directory path and the plate/well/channel metadata.
    """
    plate = "PLATE1"
    wells = ["A01", "A02", "B01", "B02"]
    field = "001"
    seed = 0
    for well in wells:
        for ch in CHANNELS:
            fname = f"{plate}_{well}_T001F{field}L01A01Z01C0{ch}.tif"
            _write_image(tmp_path / fname, seed=seed)
            seed += 1
    return {
        "dir": tmp_path,
        "plate": plate,
        "wells": wells,
        "channels": list(CHANNELS),
        "img_size": IMG_SIZE,
    }


@pytest.fixture
def sample_chw():
    """A deterministic (C, H, W) float tensor for transform tests."""
    import torch

    g = torch.Generator().manual_seed(0)
    return torch.rand(3, IMG_SIZE, IMG_SIZE, generator=g)


@pytest.fixture
def balanced_labels(synthetic_image_dir):
    """Deterministic, balanced labels: 2 classes x 2 wells.

    Balanced counts let sklearn's stratified split succeed reliably.
    """
    plate = synthetic_image_dir["plate"]
    return {
        (plate, "A01"): "ClassA",
        (plate, "A02"): "ClassB",
        (plate, "B01"): "ClassA",
        (plate, "B02"): "ClassB",
    }


@pytest.fixture
def tiled_datamodule(synthetic_image_dir, balanced_labels, monkeypatch):
    """A ready-to-use MultiChannelDataModule over the synthetic images.

    Stubs out DummyLabelsProvider so labels are deterministic and balanced,
    and uses a small crop so each 64x64 image yields several 32x32 tiles.
    """
    from omegaconf import OmegaConf

    from src.datamodule import MultiChannelDataModule

    forced = dict(balanced_labels)

    class _StubProvider:
        def __init__(self, *args, **kwargs):
            pass

        def get_labels_for_wells(self, wells):
            return dict(forced)

        def get_labels(self, root_dir, exclude_wells=None):
            return dict(forced)

    # instantiate() resolves label_provider._target_ by importing the module and
    # reading the live attribute off it, so patching utils.labels.DummyLabelsProvider
    # (the actual _target_) is what's seen.
    monkeypatch.setattr("utils.labels.DummyLabelsProvider", _StubProvider)

    d = synthetic_image_dir
    dataset_cfg = OmegaConf.create({
        "_target_": "src.dataset.TiledMultiChannelDataset",
        "root_dir": str(d["dir"]),
        "channels": d["channels"],
        "crop_size": 32,
        "stride": 32,
        "cache_size": 8,
        "max_samples_per_label": None,
        "verbose": False,
    })
    transform_cfg = [{"name": "Normalize"}]
    label_provider_cfg = OmegaConf.create({"_target_": "utils.labels.DummyLabelsProvider"})

    return MultiChannelDataModule(
        dataset=dataset_cfg,
        train_transform=transform_cfg,
        val_transform=transform_cfg,
        batch_transform=None,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        train_val_split=0.5,
        label_provider=label_provider_cfg,
    )
