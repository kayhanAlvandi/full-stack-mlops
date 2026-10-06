"""Shared fixtures for api/predictor tests.

Provides a deterministic stub model so tests can build a *real*
``TilePredictor`` (exercising its actual preprocess/tile/predict/majority-vote/
db-logging code paths) without ever touching MLflow. This replaces a
hand-written ``FakePredictor`` that duplicated ``TilePredictor``'s behavior
and could silently fall out of sync with it (e.g. a new constructor arg or a
new db_logger call added to ``predict()`` would not be reflected in the fake).
"""
import pytest
import torch
from torch import nn

from api.predictor import TilePredictor

CLASS_NAMES = ["ClassA", "ClassB"]


class StubModel(nn.Module):
    """Deterministic stand-in for a trained model.

    Ignores the input and always scores class 0 ("ClassA") highest, so tests
    can assert on a known prediction without any trained weights.
    """

    def __init__(self, num_classes: int = len(CLASS_NAMES)):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(x.shape[0], self.num_classes)
        logits[:, 0] = 10.0
        return logits


def make_predictor(db_logger=None, crop_size: int = 32, in_channels: int = 3) -> TilePredictor:
    """Build a real TilePredictor around a StubModel, bypassing MLflow entirely."""
    model_info = {
        "source": "stub",
        "model_class": "StubModel",
        "backbone": None,
        "run_id": "deadbeef",
        "num_classes": len(CLASS_NAMES),
        "crop_size": crop_size,
        "in_channels": in_channels,
        "class_names": CLASS_NAMES,
    }
    return TilePredictor(StubModel(), model_info, device="cpu", db_logger=db_logger)


@pytest.fixture
def predictor_factory():
    """Factory fixture: predictor_factory(db_logger=...) -> TilePredictor."""
    return make_predictor
