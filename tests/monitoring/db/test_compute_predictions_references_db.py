"""DB integration tests for monitoring/compute_predictions_references.py.

Two layers, both against real PostgreSQL (schemas 01-04):

- Method level: the DBLogger fetches the job's resume/grouping logic depends
  on -- get_reference_samples, fetch_benchmark_members,
  get_benchmark_predictions.
- Orchestration: compute_val_reference / compute_benchmark_reference
  end-to-end. The predictor is injected (only main() constructs a real
  TilePredictor, which needs MLflow+torch), so a stub supplies predictions
  while every DB interaction is real. _load_single_image is patched to a
  constant array because decoding real .jxl/.tif files is image-file IO, not
  the DB orchestration under test; the metadata it produces still flows
  through the real load_*_data loaders.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import monitoring.compute_predictions_references as cpr
from tests.monitoring.db.conftest import (
    PLATE,
    RUN_ID,
    insert_benchmark_sample,
)
from tests.monitoring.db.conftest import make_image_prediction_tuple as _pred


class _StubPredictor:
    """Duck-typed stand-in for TilePredictor at the injection boundary.

    predict() logs through the real DBLogger so resume checks
    (get_reference_samples / get_benchmark_predictions) see genuine rows --
    the stub replaces the model weights and tile pipeline, not the database.
    """

    def __init__(self, db_logger, run_id: str = RUN_ID, artifact_dir=None):
        self.db_logger = db_logger
        self.model_info = {"run_id": run_id}
        if artifact_dir is not None:
            self.model_info["artifact_dir"] = str(artifact_dir)
        self.predicted: list[dict] = []

    def predict(self, sample_images, image_metadata):
        self.predicted.append(image_metadata)
        self.db_logger.log_image_prediction((
            image_metadata["plate"], image_metadata["well"],
            int(image_metadata["field"]), self.model_info["run_id"],
            "positive", image_metadata.get("label"), 4, 0.75, 0.9,
            image_metadata.get("is_reference", False),
            image_metadata.get("benchmark_id"),
        ))
        return {"predicted_class": "positive"}


@pytest.fixture(autouse=True)
def _no_image_io(monkeypatch):
    """Patch the one seam that touches image files; everything else is real."""
    monkeypatch.setattr(cpr, "_load_single_image",
                        lambda _path: np.zeros((8, 8), dtype=np.float32))


def _val_manifest(tmp_path, samples: list[dict]):
    """Write a dataset_manifest.json into tmp_path; returns the artifact dir."""
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps({"val_samples": samples}))
    return tmp_path


def _val_sample(well: str, field=1, label="positive") -> dict:
    return {
        "plate": PLATE, "well": well, "field": field, "label": label,
        "root_path": "/data/val",
        "channel_files": {
            str(ch): f"{PLATE}_{well}_T0001F001L01A01Z01C0{ch}.jxl"
            for ch in range(1, 6)
        },
    }


def _benchmark_preds(db_logger, run_id: str = RUN_ID):
    with db_logger.pool.connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT well, benchmark_id, is_reference FROM image_prediction
            WHERE run_id = %s AND benchmark_id IS NOT NULL ORDER BY well
        """, (run_id,))
        return cur.fetchall()


# ─────────────────────────────────────────────────────────────────────────────
# DBLogger fetches the job depends on
# ─────────────────────────────────────────────────────────────────────────────

class TestGetReferenceSamples:
    def test_empty(self, db_logger):
        assert db_logger.get_reference_samples(RUN_ID) == []

    def test_returns_reference_keys(self, db_logger):
        db_logger.log_image_prediction(
            _pred(well="K07", field=1, is_reference=True))
        assert db_logger.get_reference_samples(RUN_ID) == [(PLATE, "K07", 1)]

    def test_excludes_production(self, db_logger):
        db_logger.log_image_prediction(_pred(well="K07", is_reference=False))
        assert db_logger.get_reference_samples(RUN_ID) == []

    def test_filters_by_run_id(self, db_logger):
        db_logger.log_image_prediction(
            _pred(well="K07", run_id="other_run", is_reference=True))
        assert db_logger.get_reference_samples(RUN_ID) == []


class TestFetchBenchmarkMembers:
    def test_joins_with_image_metadata(self, db_logger):
        insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        rows = db_logger.fetch_benchmark_members()
        assert len(rows) == 5  # 5 channels
        row = rows[0]
        assert row["benchmark_id"] > 0
        assert row["plate"] == PLATE
        assert row["well"] == "K07"
        assert row["field"] == 1
        assert row["t_label"] == "ClassA"
        assert "channel" in row
        assert "root_path" in row
        assert "file_name" in row

    def test_ordered_by_channel_index(self, db_logger):
        insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        rows = db_logger.fetch_benchmark_members()
        channel_indices = [row["channel_index"] for row in rows]
        assert channel_indices == sorted(channel_indices)

    def test_multiple_samples(self, db_logger):
        insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        insert_benchmark_sample(db_logger, well="K08", t_label="ClassB")
        rows = db_logger.fetch_benchmark_members()
        assert len(rows) == 10  # 2 samples x 5 channels
        benchmark_ids = {row["benchmark_id"] for row in rows}
        assert len(benchmark_ids) == 2


class TestGetBenchmarkPredictions:
    def test_empty(self, db_logger):
        assert db_logger.get_benchmark_predictions(RUN_ID) == []

    def test_returns_scored_benchmark_ids(self, db_logger):
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        db_logger.log_image_prediction(_pred(
            well="K07", p_label="ClassA", t_label="ClassA", benchmark_id=bid))
        assert bid in db_logger.get_benchmark_predictions(RUN_ID)

    def test_excludes_non_benchmark(self, db_logger):
        """A live prediction (benchmark_id=NULL) is not returned."""
        db_logger.log_image_prediction(_pred(benchmark_id=None))
        assert db_logger.get_benchmark_predictions(RUN_ID) == []

    def test_filters_by_run_id(self, db_logger):
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        db_logger.log_image_prediction(_pred(run_id="other_run", benchmark_id=bid))
        assert db_logger.get_benchmark_predictions(RUN_ID) == []
        assert bid in db_logger.get_benchmark_predictions("other_run")


# ─────────────────────────────────────────────────────────────────────────────
# compute_benchmark_reference -- orchestration
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkReference:
    def test_scores_all_registered_samples(self, db_logger):
        """Every registered benchmark sample is scored once, writing a real
        image_prediction row tagged with its benchmark_id."""
        bid1, _ = insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        bid2, _ = insert_benchmark_sample(db_logger, well="K08", t_label="ClassB")
        predictor = _StubPredictor(db_logger)

        cpr.compute_benchmark_reference(predictor, db_logger, RUN_ID)

        assert {m["well"] for m in predictor.predicted} == {"K07", "K08"}
        # metadata built by the real load_benchmark_data: benchmark_id set,
        # label taken from the benchmark's known t_label.
        assert {m["benchmark_id"] for m in predictor.predicted} == {bid1, bid2}
        rows = _benchmark_preds(db_logger)
        assert [(r[0], r[1], r[2]) for r in rows] == [
            ("K07", bid1, False), ("K08", bid2, False),
        ]

    def test_resume_skips_already_scored(self, db_logger):
        """A benchmark_id with an existing prediction for this run_id is
        skipped, so an interrupted run only scores what's missing."""
        bid1, _ = insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        bid2, _ = insert_benchmark_sample(db_logger, well="K08", t_label="ClassB")
        db_logger.log_image_prediction(_pred(well="K07", benchmark_id=bid1))
        predictor = _StubPredictor(db_logger)

        cpr.compute_benchmark_reference(predictor, db_logger, RUN_ID)

        assert [m["well"] for m in predictor.predicted] == ["K08"]
        assert set(db_logger.get_benchmark_predictions(RUN_ID)) == {bid1, bid2}

    def test_rerun_is_idempotent(self, db_logger):
        insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        predictor = _StubPredictor(db_logger)

        cpr.compute_benchmark_reference(predictor, db_logger, RUN_ID)
        cpr.compute_benchmark_reference(predictor, db_logger, RUN_ID)

        assert len(predictor.predicted) == 1
        assert len(_benchmark_preds(db_logger)) == 1

    def test_no_benchmark_samples(self, db_logger):
        predictor = _StubPredictor(db_logger)
        cpr.compute_benchmark_reference(predictor, db_logger, RUN_ID)
        assert predictor.predicted == []


# ─────────────────────────────────────────────────────────────────────────────
# compute_val_reference -- orchestration
# ─────────────────────────────────────────────────────────────────────────────

class TestValReference:
    def test_scores_manifest_val_samples(self, db_logger, tmp_path):
        """Each val_samples entry is scored and written as is_reference=TRUE."""
        artifact_dir = _val_manifest(tmp_path, [
            _val_sample("K07", label="ClassA"), _val_sample("K08", label="ClassB"),
        ])
        predictor = _StubPredictor(db_logger, artifact_dir=artifact_dir)

        cpr.compute_val_reference(predictor, db_logger, RUN_ID, "test-model")

        assert {m["well"] for m in predictor.predicted} == {"K07", "K08"}
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT well, t_label, is_reference FROM image_prediction
                WHERE run_id = %s AND is_reference = TRUE ORDER BY well
            """, (RUN_ID,))
            assert cur.fetchall() == [("K07", "ClassA", True), ("K08", "ClassB", True)]

    def test_resume_skips_logged_samples(self, db_logger, tmp_path):
        """A (plate, well, field) already logged as reference is skipped --
        including the string-field coercion ("001" -> 1) that older manifests
        relied on."""
        db_logger.log_image_prediction(_pred(well="K07", is_reference=True))
        artifact_dir = _val_manifest(tmp_path, [
            _val_sample("K07", field="001"), _val_sample("K08"),
        ])
        predictor = _StubPredictor(db_logger, artifact_dir=artifact_dir)

        cpr.compute_val_reference(predictor, db_logger, RUN_ID, "test-model")

        assert [m["well"] for m in predictor.predicted] == ["K08"]

    def test_missing_manifest_exits_cleanly(self, db_logger, tmp_path):
        """artifact_dir without dataset_manifest.json -> no scoring, no error."""
        predictor = _StubPredictor(db_logger, artifact_dir=tmp_path)
        cpr.compute_val_reference(predictor, db_logger, RUN_ID, "test-model")
        assert predictor.predicted == []

    def test_no_artifact_dir_exits_cleanly(self, db_logger):
        """model_info without artifact_dir (artifact download failed at load
        time) -> early return, no scoring."""
        predictor = _StubPredictor(db_logger)  # no artifact_dir key
        cpr.compute_val_reference(predictor, db_logger, RUN_ID, "test-model")
        assert predictor.predicted == []
