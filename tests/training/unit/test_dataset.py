"""Unit tests for LabelEncoder, DummyLabelsProvider and MultiChannelImageDataset."""
import pytest
import torch

from src.dataset import LabelEncoder, MultiChannelImageDataset
from utils.labels import DummyLabelsProvider

# ── LabelEncoder ────────────────────────────────────────────────────────────

def test_label_encoder_roundtrip():
    enc = LabelEncoder().fit(["cat", "dog", "cat", "bird"])
    assert enc.num_classes == 3
    # classes are sorted alphabetically
    assert enc.classes == ["bird", "cat", "dog"]
    assert enc.encode("bird") == 0
    assert enc.decode(0) == "bird"


def test_label_encoder_inverse_transform():
    enc = LabelEncoder().fit(["a", "b", "c"])
    idxs = enc.transform(["c", "a", "b"])
    assert enc.inverse_transform(idxs) == ["c", "a", "b"]


def test_label_encoder_transform_before_fit_raises():
    enc = LabelEncoder()
    with pytest.raises(ValueError):
        enc.transform(["cat"])


# ── DummyLabelsProvider ─────────────────────────────────────────────────────

def test_dummy_labels_provider_labels_every_well(synthetic_image_dir):
    provider = DummyLabelsProvider(class_names=["X", "Y"], seed=42)
    labels = provider.get_labels(str(synthetic_image_dir["dir"]))
    assert len(labels) == len(synthetic_image_dir["wells"])
    assert all(v in ("X", "Y") for v in labels.values())


def test_dummy_labels_provider_is_deterministic(synthetic_image_dir):
    d = str(synthetic_image_dir["dir"])
    a = DummyLabelsProvider(seed=7).get_labels(d)
    b = DummyLabelsProvider(seed=7).get_labels(d)
    assert a == b


def test_dummy_labels_provider_respects_exclude(synthetic_image_dir):
    d = synthetic_image_dir
    excluded = [(d["plate"], "A01")]
    labels = DummyLabelsProvider().get_labels(str(d["dir"]), exclude_wells=excluded)
    assert (d["plate"], "A01") not in labels
    assert len(labels) == len(d["wells"]) - 1


# ── MultiChannelImageDataset ────────────────────────────────────────────────

def test_dataset_len_matches_number_of_fields(synthetic_image_dir):
    d = synthetic_image_dir
    labels = {(d["plate"], w): "ClassA" for w in d["wells"]}
    enc = LabelEncoder().fit(list(labels.values()))
    ds = MultiChannelImageDataset(
        root_dir=str(d["dir"]),
        channels=d["channels"],
        labels_dict=labels,
        label_encoder=enc,
    )
    # one field per well
    assert len(ds) == len(d["wells"])


def test_dataset_getitem_shape_and_range(synthetic_image_dir):
    d = synthetic_image_dir
    labels = {(d["plate"], w): "ClassA" for w in d["wells"]}
    enc = LabelEncoder().fit(list(labels.values()))
    ds = MultiChannelImageDataset(
        root_dir=str(d["dir"]),
        channels=d["channels"],
        labels_dict=labels,
        label_encoder=enc,
    )
    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape[0] == len(d["channels"])
    assert image.dtype == torch.float32
    assert isinstance(label, int)
    assert 0 <= label < enc.num_classes
    # percentile-clip normalisation puts values in [0, 1]
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0
