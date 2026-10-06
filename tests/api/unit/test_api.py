"""API tests using FastAPI's TestClient (no database, no MLflow).

The real model loading (MLflow) is bypassed:
  - startup is forced into the "no model source" branch, so no network / MLflow
  - a real TilePredictor built around a StubModel (see ../conftest.py) is
    injected for the success-path tests, so these tests exercise the actual
    preprocess/tile/predict/majority-vote pipeline rather than a
    hand-maintained fake that can drift out of sync with it.

DB-logging behavior (what predict() actually writes, and in what order) is
covered against a real PostgreSQL schema in tests/api/db/, not here -- a
hand-written fake DB logger would only prove predict() calls *some* methods
with the right names, not that those calls match the real schema.
"""
import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api_main


class _StubDBLogger:
    """Presence-only stand-in for DBLogger: just enough to be non-None for
    the /health and /db connectivity checks, which never call a log_* method."""

    def close(self) -> None:
        """No-op close to satisfy the lifespan cleanup."""


@pytest.fixture(autouse=True)
def _no_real_model(monkeypatch):
    """Force the startup lifespan into the no-model branch so tests never hit MLflow."""
    monkeypatch.setattr(type(api_main.settings), "has_model_source", property(lambda self: False))


@pytest.fixture
def client():
    with TestClient(api_main.app) as c:
        yield c


def _npy_upload(shape=(3, 32, 32)):
    arr = np.zeros(shape, dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return {"files": ("img.npy", buf, "application/octet-stream")}


def _tif_file(filename: str, size=(32, 32)) -> tuple:
    """Create a single-channel .tif upload tuple with the given filename."""
    arr = np.zeros(size, dtype=np.uint16)
    ok, encoded = cv2.imencode(".tif", arr)
    assert ok, f"Failed to encode tif for {filename}"
    buf = io.BytesIO(encoded.tobytes())
    return ("files", (filename, buf, "image/tif"))


def _multi_channel_tif_files(n_channels=3, size=(32, 32)) -> list[tuple]:
    """Create multi-channel .tif uploads with microscopy filenames (C01..CN)."""
    files = []
    for ch in range(1, n_channels + 1):
        fname = f"PLATE1_A01_T0001F001L01A01Z01C0{ch}.tif"
        files.append(_tif_file(fname, size))
    return files


def _tif_file_with_value(filename: str, value: int, size=(32, 32)) -> tuple:
    """Create a single-channel .tif upload whose pixels are all `value`, so channels
    can be told apart by content (not just by upload position)."""
    arr = np.full(size, value, dtype=np.uint16)
    ok, encoded = cv2.imencode(".tif", arr)
    assert ok, f"Failed to encode tif for {filename}"
    buf = io.BytesIO(encoded.tobytes())
    return ("files", (filename, buf, "image/tif"))


def test_health_reports_no_model(client):
    api_main.predictor = None
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_model_endpoint_503_without_model(client):
    api_main.predictor = None
    r = client.get("/model")
    assert r.status_code == 503


def test_predict_503_without_model(client):
    api_main.predictor = None
    r = client.post("/predict", files=_npy_upload(), data={"root_path": "/tmp"})
    assert r.status_code == 503


def test_predict_success_with_mock(client, predictor_factory):
    api_main.predictor = predictor_factory()
    files = _multi_channel_tif_files(n_channels=3)
    r = client.post("/predict", files=files, data={"root_path": "/tmp"})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] == "ClassA"
    assert body["total_tiles"] == 1


def test_predict_rejects_bad_npy_shape(client, predictor_factory):
    api_main.predictor = predictor_factory()
    # 2D array is invalid: endpoint expects (C, H, W)
    r = client.post("/predict", files=_npy_upload(shape=(32, 32)), data={"root_path": "/tmp"})
    assert r.status_code == 400


def test_model_info_with_mock(client, predictor_factory):
    api_main.predictor = predictor_factory()
    r = client.get("/model")
    assert r.status_code == 200
    body = r.json()
    assert body["num_classes"] == 2
    assert body["class_names"] == ["ClassA", "ClassB"]


# ── Multi-channel upload tests ──────────────────────────────────────────────


def test_predict_multi_channel_success(client, predictor_factory):
    """Upload multiple .tif files (one per channel) and verify the API stacks them."""
    api_main.predictor = predictor_factory()
    files = _multi_channel_tif_files(n_channels=3)
    r = client.post("/predict", files=files, data={"root_path": "/tmp"})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] == "ClassA"
    assert body["total_tiles"] == 1


def test_predict_multi_channel_rejects_mismatched_shapes(client, predictor_factory):
    """Channels with different dimensions should return 400."""
    api_main.predictor = predictor_factory()
    files = [
        _tif_file("PLATE1_A01_T0001F001L01A01Z01C01.tif", size=(32, 32)),
        _tif_file("PLATE1_A01_T0001F001L01A01Z01C02.tif", size=(64, 64)),
    ]
    r = client.post("/predict", files=files, data={"root_path": "/tmp"})
    assert r.status_code == 400


# ── Channel-order canonicalization ──────────────────────────────────────────
# Training (src/dataset.py) always stacks channels sorted ascending by
# channel number. Inference must replicate that exact order regardless of
# upload order, or the model silently sees out-of-distribution input.
#
# The exact reordering behavior (pixel data + metadata) is unit-tested
# directly against chans_reorder in test_predictor.py; what's checked below
# is that main.py's upload handling wires an out-of-order upload through to
# a successful prediction. DB-logged ordering is checked for real against a
# live schema in tests/api/db/test_predictor_db.py.


def test_predict_canonicalizes_shuffled_channel_order(client, predictor_factory):
    """Uploading channels out of order still produces a successful prediction."""
    api_main.predictor = predictor_factory()
    # Upload order is deliberately C03, C01, C02 -- not ascending.
    files = [
        _tif_file_with_value("PLATE1_A01_T0001F001L01A01Z01C03.tif", value=30),
        _tif_file_with_value("PLATE1_A01_T0001F001L01A01Z01C01.tif", value=10),
        _tif_file_with_value("PLATE1_A01_T0001F001L01A01Z01C02.tif", value=20),
    ]
    r = client.post("/predict", files=files, data={"root_path": "/tmp"})
    assert r.status_code == 200
    assert r.json()["predicted_class"] == "ClassA"


def test_predict_rejects_duplicate_channel_numbers(client, predictor_factory):
    """Two files parsed to the same channel number should be rejected, not
    silently collide/overwrite each other in the channel axis."""
    api_main.predictor = predictor_factory()
    files = [
        _tif_file_with_value("PLATE1_A01_T0001F001L01A01Z01C01.tif", value=10),
        _tif_file_with_value("PLATE1_A01_T0002F001L01A01Z01C01.tif", value=20),
    ]
    r = client.post("/predict", files=files, data={"root_path": "/tmp"})
    assert r.status_code == 400


# ── Database connection tests ───────────────────────────────────────────────


def test_db_endpoint_503_without_db(client):
    """GET /db returns 503 when no database connection is configured."""
    api_main.db_logger = None
    r = client.get("/db")
    assert r.status_code == 503


def test_db_endpoint_with_db(client):
    """GET /db returns connection info when a db_logger is set."""
    api_main.db_logger = _StubDBLogger()
    r = client.get("/db")
    assert r.status_code == 200
    body = r.json()
    assert body["connected"] is True
    assert "uri" in body


def test_health_reports_database_connected(client):
    """GET /health includes database_connected=True when db_logger is set."""
    api_main.predictor = None
    api_main.db_logger = _StubDBLogger()
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["database_connected"] is True


def test_health_reports_database_disconnected(client):
    """GET /health includes database_connected=False when db_logger is None."""
    api_main.predictor = None
    api_main.db_logger = None
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["database_connected"] is False
