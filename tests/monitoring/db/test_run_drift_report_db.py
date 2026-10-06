"""DB integration tests for monitoring/run_drift_report.py.

Two layers, both against real PostgreSQL (schemas 01-04):

- Method level: the DBLogger fetches/writes the report is built on --
  fetch_reference/fetch_current at image and tile level, log_drift_report,
  log_drift_report_column.
- Orchestration: main() end-to-end -- the fetches, group building, a real
  Evidently DataDriftPreset run, and the drift_report + drift_report_column
  inserts. No fakes -- Evidently is part of requirements/monitoring_req.txt
  and already runs for real in the unit tests too.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta

import pytest

import monitoring.run_drift_report as drift
from tests.monitoring.db.conftest import (
    DB_TEST_URI,
    RUN_ID,
    insert_benchmark_sample,
    insert_image_and_tile_prediction,
)
from tests.monitoring.db.conftest import make_image_prediction_tuple as _pred


def _run_main(monkeypatch, *argv: str):
    monkeypatch.setenv("MONITORING_DB_URI", DB_TEST_URI)
    monkeypatch.setattr(sys, "argv", ["run_drift_report.py", *argv])
    drift.main()


def _window(days: int = 1):
    now = datetime.now()
    return now - timedelta(days=days), now + timedelta(days=1)


# ─────────────────────────────────────────────────────────────────────────────
# DBLogger fetches/writes
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchReferenceImageLevel:
    def test_empty(self, db_logger):
        assert db_logger.fetch_reference_image_level(RUN_ID) == []

    def test_returns_reference_rows(self, db_logger):
        db_logger.log_image_prediction(
            _pred(well="K07", is_reference=True, p_label="ClassA"))
        rows = db_logger.fetch_reference_image_level(RUN_ID)
        assert len(rows) == 1
        assert rows[0]["p_label"] == "ClassA"
        assert "vote_fraction" in rows[0]
        assert "avg_confidence" in rows[0]

    def test_excludes_production(self, db_logger):
        db_logger.log_image_prediction(_pred(is_reference=False))
        assert db_logger.fetch_reference_image_level(RUN_ID) == []

    def test_filters_by_run_id(self, db_logger):
        db_logger.log_image_prediction(_pred(run_id="other", is_reference=True))
        assert db_logger.fetch_reference_image_level(RUN_ID) == []


class TestFetchCurrentImageLevel:
    def test_empty(self, db_logger):
        assert db_logger.fetch_current_image_level(RUN_ID, *_window()) == []

    def test_returns_live_rows_in_window(self, db_logger):
        db_logger.log_image_prediction(_pred(well="K07"))
        rows = db_logger.fetch_current_image_level(RUN_ID, *_window())
        assert len(rows) == 1
        assert rows[0]["p_label"] == "positive"

    def test_excludes_reference(self, db_logger):
        """Reference rows don't appear in the current (live) window."""
        db_logger.log_image_prediction(_pred(well="K07", is_reference=True))
        assert db_logger.fetch_current_image_level(RUN_ID, *_window()) == []

    def test_excludes_benchmark(self, db_logger):
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="A")
        db_logger.log_image_prediction(_pred(well="K07", benchmark_id=bid))
        assert db_logger.fetch_current_image_level(RUN_ID, *_window()) == []

    def test_excludes_reference_wells(self, db_logger):
        """A live prediction for a well that also has a reference row is excluded."""
        db_logger.log_image_prediction(_pred(well="K07", is_reference=True))
        db_logger.log_image_prediction(_pred(well="K07", is_reference=False))
        assert db_logger.fetch_current_image_level(RUN_ID, *_window()) == []


class TestFetchReferenceTileLevel:
    def test_empty(self, db_logger):
        assert db_logger.fetch_reference_tile_level(RUN_ID) == []

    def test_returns_tile_rows_with_channel_stats_columns(self, db_logger):
        """One row per (tile_prediction, channel) -- the image_metadata ->
        tile_stack -> tile_stack_member chain has to be complete."""
        insert_image_and_tile_prediction(db_logger, well="K07", is_reference=True)
        rows = db_logger.fetch_reference_tile_level(RUN_ID)
        assert len(rows) == 5  # 1 tile x 5 channels
        row = rows[0]
        assert row["p_label"] == "positive"
        assert "confidence" in row
        assert "channel" in row
        # LEFT JOIN: stats columns present but NULL when none logged.
        assert "mean" in row and row["mean"] is None

    def test_excludes_production(self, db_logger):
        insert_image_and_tile_prediction(db_logger, well="K07", is_reference=False)
        assert db_logger.fetch_reference_tile_level(RUN_ID) == []

    def test_image_prediction_alone_not_enough(self, db_logger):
        """An image_prediction without the tile_stack chain produces no rows
        -- the fetch joins through tile_stack_member."""
        db_logger.log_image_prediction(_pred(well="K07", is_reference=True))
        assert db_logger.fetch_reference_tile_level(RUN_ID) == []


class TestFetchCurrentTileLevel:
    def test_returns_live_tile_rows_in_window(self, db_logger):
        insert_image_and_tile_prediction(db_logger, well="K07")
        rows = db_logger.fetch_current_tile_level(RUN_ID, *_window())
        assert len(rows) == 5  # 1 tile x 5 channels
        assert rows[0]["p_label"] == "positive"

    def test_excludes_reference(self, db_logger):
        insert_image_and_tile_prediction(db_logger, well="K07", is_reference=True)
        assert db_logger.fetch_current_tile_level(RUN_ID, *_window()) == []

    def test_excludes_reference_wells(self, db_logger):
        """Live tile rows for a well in this run's reference set are excluded."""
        insert_image_and_tile_prediction(db_logger, well="K07", is_reference=True)
        insert_image_and_tile_prediction(db_logger, well="K07", is_reference=False)
        assert db_logger.fetch_current_tile_level(RUN_ID, *_window()) == []


class TestLogDriftReport:
    def test_returns_positive_id(self, db_logger):
        now = datetime.now()
        report_id = db_logger.log_drift_report(
            (RUN_ID, now - timedelta(days=7), now, False, 2, 5, "/reports/drift_1"))
        assert isinstance(report_id, int)
        assert report_id > 0

    def test_stored_values(self, db_logger):
        now = datetime.now()
        report_id = db_logger.log_drift_report(
            (RUN_ID, now - timedelta(days=7), now, True, 3, 10, "/reports/drift_2"))
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT run_id, dataset_drift, n_columns_drifted, "
                "n_columns_total, report_path FROM drift_report WHERE id = %s",
                (report_id,))
            row = cur.fetchone()
        assert row == (RUN_ID, True, 3, 10, "/reports/drift_2")


class TestLogDriftReportColumn:
    def test_inserts_columns(self, db_logger):
        now = datetime.now()
        report_id = db_logger.log_drift_report(
            (RUN_ID, now - timedelta(days=7), now, False, 0, 3, "/reports/d"))
        columns = [
            (report_id, "vote_fraction", "image_level", 0.1, False, "ks"),
            (report_id, "avg_confidence", "image_level", 0.2, True, "ks"),
            (report_id, "channel_1_mean", "channel_stats", 0.3, True, "ks"),
        ]
        n = db_logger.log_drift_report_column(columns)
        assert n == 3
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM drift_report_column WHERE drift_report_id = %s",
                (report_id,))
            assert cur.fetchone()[0] == 3


# ─────────────────────────────────────────────────────────────────────────────
# run_drift_report.main() -- orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _seed_data(db_logger, n_ref: int = 6, n_cur: int = 6):
    """Reference rows for RUN_ID (validation wells) + live production rows in
    the default 7-day window (different wells, so the reference-well exclusion
    doesn't filter them out)."""
    for i in range(n_ref):
        insert_image_and_tile_prediction(
            db_logger, well=f"R{i:02d}", field=1, run_id=RUN_ID,
            p_label="positive" if i % 2 == 0 else "negative",
            confidence=0.7 + 0.05 * i, is_reference=True,
        )
    for i in range(n_cur):
        insert_image_and_tile_prediction(
            db_logger, well=f"P{i:02d}", field=1, run_id=RUN_ID,
            p_label="positive" if i % 3 else "negative",
            confidence=0.6 + 0.04 * i, is_reference=False,
        )


def _report_rows(db_logger):
    with db_logger.pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, run_id, n_columns_total, report_path FROM drift_report")
        reports = cur.fetchall()
        cur.execute("SELECT drift_report_id, column_name, column_group FROM drift_report_column")
        columns = cur.fetchall()
    return reports, columns


class TestMain:
    def test_full_report_writes_db_rows(self, monkeypatch, db_logger, tmp_path):
        """main() runs all available groups and persists one drift_report row
        plus per-column rows for it."""
        pytest.importorskip("evidently")  # real DataDriftPreset run
        _seed_data(db_logger)
        _run_main(monkeypatch, "--run-id", RUN_ID, "--reports-dir", str(tmp_path))

        reports, columns = _report_rows(db_logger)
        assert len(reports) == 1
        report_id, run_id, n_cols, report_path = reports[0]
        assert run_id == RUN_ID
        assert n_cols > 0
        assert len(columns) == n_cols
        assert all(cid == report_id for cid, *_ in columns)
        # image_level always present; tile_level present since we wrote tile rows.
        assert {"image_level", "tile_level"} <= {g for _, _, g in columns}

        # The report dir the DB row points at really holds the artifacts.
        from pathlib import Path
        out = Path(report_path)
        assert (out / "metrics.json").exists()
        assert (out / "image_level.html").exists()
        assert (out / "tile_level.html").exists()

    def test_no_reference_data_exits_cleanly(self, monkeypatch, db_logger, tmp_path):
        """No reference rows for the run -> clear error, nothing written."""
        for i in range(3):
            insert_image_and_tile_prediction(db_logger, well=f"P{i:02d}", run_id=RUN_ID)
        _run_main(monkeypatch, "--run-id", RUN_ID, "--reports-dir", str(tmp_path))
        reports, columns = _report_rows(db_logger)
        assert reports == []
        assert columns == []

    def test_no_current_data_exits_cleanly(self, monkeypatch, db_logger, tmp_path):
        """Reference exists but the window is empty -> nothing to compare."""
        for i in range(3):
            insert_image_and_tile_prediction(db_logger, well=f"R{i:02d}",
                                             run_id=RUN_ID, is_reference=True)
        _run_main(monkeypatch, "--run-id", RUN_ID, "--reports-dir", str(tmp_path))
        reports, _ = _report_rows(db_logger)
        assert reports == []

    def test_window_excludes_old_rows(self, monkeypatch, db_logger, tmp_path):
        """--window-start/--window-end bound which production rows count as
        current: with every live row backdated before the window, the job
        finds no current data and writes nothing."""
        _seed_data(db_logger, n_ref=3, n_cur=3)
        old = datetime.now() - timedelta(days=30)
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("UPDATE image_prediction SET created_at = %s "
                        "WHERE is_reference = FALSE", (old,))
            cur.execute("UPDATE tile_prediction SET created_at = %s "
                        "WHERE is_reference = FALSE", (old,))
        start = (datetime.now() - timedelta(days=1)).isoformat()
        end = datetime.now().isoformat()
        _run_main(monkeypatch, "--run-id", RUN_ID, "--reports-dir", str(tmp_path),
                  "--window-start", start, "--window-end", end)
        reports, _ = _report_rows(db_logger)
        assert reports == []
