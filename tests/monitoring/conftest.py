"""Shared fixtures for monitoring tests.

Only manifest-building helpers live here -- monitoring unit tests are pure-
function tests that need no database/API/MLflow/MongoDB stand-ins at all.
Anything DB-facing is tested against real PostgreSQL under
tests/monitoring/db/, whose conftest applies the full schema set.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Manifest helpers
# ─────────────────────────────────────────────────────────────────────────────

PLATE = "MIG-Exp03-CP-40X-bin1X1"
WELL = "K07"
ROOT_PATH = "/data/benchmark"
SHAPE = [2048, 2048]


def _channel_files(field: int, well: str = WELL, plate: str = PLATE) -> dict:
    """Build a {channel_number_str: filename} dict for a 5-channel sample."""
    return {
        str(ch): f"{plate}_{well}_T0001F{field:03d}L01A01Z01C0{ch}.jxl"
        for ch in range(1, 6)
    }


@pytest.fixture
def flat_manifest(tmp_path: Path) -> Path:
    """A flat {"samples": [...]} manifest with 3 benchmark samples."""
    samples = []
    for i, well in enumerate(["K07", "K08", "K09"], start=1):
        samples.append({
            "plate": PLATE,
            "well": well,
            "field": 1,
            "label": f"Class{chr(64 + i)}",
            "root_path": ROOT_PATH,
            "shape": SHAPE,
            "channel_files": _channel_files(1, well=well),
        })
    manifest = {"samples": samples}
    path = tmp_path / "dataset_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.fixture
def train_val_manifest(tmp_path: Path) -> Path:
    """A training-style manifest with train_samples + val_samples."""
    def _sample(well: str, label: str):
        return {
            "plate": PLATE,
            "well": well,
            "field": 1,
            "label": label,
            "root_path": ROOT_PATH,
            "shape": SHAPE,
            "channel_files": _channel_files(1, well=well),
        }
    manifest = {
        "train_samples": [_sample("K07", "ClassA"), _sample("K08", "ClassB")],
        "val_samples": [_sample("K09", "ClassC")],
    }
    path = tmp_path / "train_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.fixture
def manifest_missing_shape(tmp_path: Path) -> Path:
    """A manifest entry without the 'shape' key (should raise on parse)."""
    manifest = {"samples": [{
        "plate": PLATE, "well": WELL, "field": 1, "label": "ClassA",
        "root_path": ROOT_PATH,
        "channel_files": _channel_files(1),
    }]}
    path = tmp_path / "no_shape.json"
    path.write_text(json.dumps(manifest))
    return path

