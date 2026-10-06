"""Unit tests for api/predictor.py's pure logic: channel reordering, tiling,
and majority voting. These don't go through the HTTP layer or a model at
all (except make_predictor's StubModel, used only where a TilePredictor
instance -- not just its free functions -- is needed).
"""
import numpy as np
import pytest
import torch

from api.predictor import chans_reorder


def test_chans_reorder_canonicalizes_shuffled_channel_order():
    """Uploading channels out of order must still produce an (C, H, W) array
    and metadata canonicalized to ascending channel-number order, matching
    how training (src/dataset.py) stacks channels.
    """
    # Upload order is deliberately C03, C01, C02 -- not ascending.
    image_channels = [
        np.full((4, 4), 30, dtype=np.float32),
        np.full((4, 4), 10, dtype=np.float32),
        np.full((4, 4), 20, dtype=np.float32),
    ]
    image_metadata = {
        "channels": [3, 1, 2],
        "channel_files": [
            "PLATE1_A01_T0001F001L01A01Z01C03.tif",
            "PLATE1_A01_T0001F001L01A01Z01C01.tif",
            "PLATE1_A01_T0001F001L01A01Z01C02.tif",
        ],
    }

    image, reordered_metadata = chans_reorder(image_channels, image_metadata)

    assert reordered_metadata["channels"] == [1, 2, 3]
    assert reordered_metadata["channel_files"] == [
        "PLATE1_A01_T0001F001L01A01Z01C01.tif",
        "PLATE1_A01_T0001F001L01A01Z01C02.tif",
        "PLATE1_A01_T0001F001L01A01Z01C03.tif",
    ]
    # The image array's channel axis is reordered to match, not just the
    # metadata (channel 1's pixel value ends up at axis 0, etc).
    assert image[0].flat[0] == 10
    assert image[1].flat[0] == 20
    assert image[2].flat[0] == 30


def test_chans_reorder_rejects_duplicate_channel_numbers():
    """Two uploads parsed to the same channel number must be rejected, not
    silently collide/overwrite each other in the channel axis."""
    image_channels = [np.zeros((4, 4), dtype=np.float32), np.zeros((4, 4), dtype=np.float32)]
    image_metadata = {
        "channels": [1, 1],
        "channel_files": ["a_C01.tif", "b_C01.tif"],
    }
    with pytest.raises(ValueError, match="Duplicate channel"):
        chans_reorder(image_channels, image_metadata)


def test_chans_reorder_does_not_mutate_input_metadata():
    """The caller's metadata dict must stay untouched (e.g. manifest-derived
    metadata from compute_predictions_references shouldn't be mutated as a
    side effect of prediction)."""
    image_channels = [np.zeros((4, 4), dtype=np.float32), np.zeros((4, 4), dtype=np.float32)]
    image_metadata = {"channels": [2, 1], "channel_files": ["b.tif", "a.tif"]}
    original = dict(image_metadata)

    chans_reorder(image_channels, image_metadata)

    assert image_metadata == original


def test_tile_image_splits_into_expected_grid(predictor_factory):
    """A 64x64 image with crop_size=32, stride=32 should produce a 2x2 grid
    of non-overlapping tiles at the expected pixel offsets."""
    predictor = predictor_factory()
    image = torch.zeros(3, 64, 64)

    tiles = predictor.tile_image(image, crop_size=32, stride=32)

    assert len(tiles) == 4
    positions = {(t["row"], t["col"], t["x"], t["y"]) for t in tiles}
    assert positions == {(0, 0, 0, 0), (0, 1, 32, 0), (1, 0, 0, 32), (1, 1, 32, 32)}
    assert all(t["tile"].shape == (3, 32, 32) for t in tiles)


def test_majority_vote_picks_most_common_class_and_averages_its_confidence(predictor_factory):
    predictor = predictor_factory()
    tile_predictions = [
        {"predicted_class": "ClassA", "probabilities": {"ClassA": 0.9, "ClassB": 0.1}},
        {"predicted_class": "ClassA", "probabilities": {"ClassA": 0.7, "ClassB": 0.3}},
        {"predicted_class": "ClassB", "probabilities": {"ClassA": 0.2, "ClassB": 0.8}},
    ]

    result = predictor.majority_vote(tile_predictions)

    assert result["predicted_class"] == "ClassA"
    assert result["total_tiles"] == 3
    assert result["vote_counts"] == {"ClassA": 2, "ClassB": 1}
    assert result["vote_fraction"] == pytest.approx(2 / 3)
    # Averaged over ALL tiles, not just the ones that voted for the winner.
    assert result["confidence"] == pytest.approx((0.9 + 0.7 + 0.2) / 3)


def test_majority_vote_empty_tiles_returns_unknown(predictor_factory):
    predictor = predictor_factory()
    result = predictor.majority_vote([])
    assert result == {"predicted_class": "unknown", "confidence": 0.0, "vote_counts": {}}
