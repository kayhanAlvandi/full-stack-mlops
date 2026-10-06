"""DB integration tests for monitoring/run_quality_report.py.

Two layers, both against real PostgreSQL (schemas 01-04):

- Method level: the DBLogger fetches/writes the report is built on --
  fetch_benchmark_quality, fetch_current_quality, log_quality_report.
- Orchestration: main() end-to-end -- benchmark predictions (real
  benchmark_id FK + t_label) as the Evidently reference, labeled live rows
  in the window as current, a real ClassificationPreset run, and the
  quality_report insert. Only the labels are seeded -- Evidently and the DB
  boundary are real.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta

import pytest

import monitoring.run_quality_report as quality
from tests.monitoring.db.conftest import (
    DB_TEST_URI,
    RUN_ID,
    insert_benchmark_sample,
)
from tests.monitoring.db.conftest import make_image_prediction_tuple as _pred


def _run_main(monkeypatch, *argv: str):
    monkeypatch.setenv("MONITORING_DB_URI", DB_TEST_URI)
    monkeypatch.setattr(sys, "argv", ["run_quality_report.py", *argv])
    quality.main()


def _window(days: int = 1):
    now = datetime.now()
    return now - timedelta(days=days), now + timedelta(days=1)


# ─────────────────────────────────────────────────────────────────────────────
# DBLogger fetches/writes
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchBenchmarkQuality:
    def test_empty(self, db_logger):
        assert db_logger.fetch_benchmark_quality(RUN_ID) == []

    def test_returns_benchmark_with_labels(self, db_logger):
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        db_logger.log_image_prediction(
            _pred(well="K07", p_label="ClassA", t_label="ClassA", benchmark_id=bid))
        rows = db_logger.fetch_benchmark_quality(RUN_ID)
        assert len(rows) == 1
        assert rows[0]["p_label"] == "ClassA"
        assert rows[0]["t_label"] == "ClassA"

    def test_excludes_unlabeled_benchmark(self, db_logger):
        """Benchmark rows without t_label are excluded."""
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        db_logger.log_image_prediction(_pred(well="K07", t_label=None, benchmark_id=bid))
        assert db_logger.fetch_benchmark_quality(RUN_ID) == []

    def test_excludes_production(self, db_logger):
        db_logger.log_image_prediction(_pred(well="K07", p_label="A", t_label="A"))
        assert db_logger.fetch_benchmark_quality(RUN_ID) == []

    def test_filters_by_run_id(self, db_logger):
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        db_logger.log_image_prediction(
            _pred(well="K07", run_id="other", t_label="ClassA", benchmark_id=bid))
        assert db_logger.fetch_benchmark_quality(RUN_ID) == []


class TestFetchCurrentQuality:
    def test_empty(self, db_logger):
        assert db_logger.fetch_current_quality(RUN_ID, *_window()) == []

    def test_returns_labeled_live_rows(self, db_logger):
        db_logger.log_image_prediction(_pred(well="K07", p_label="A", t_label="A"))
        rows = db_logger.fetch_current_quality(RUN_ID, *_window())
        assert len(rows) == 1
        assert rows[0]["p_label"] == "A"
        assert rows[0]["t_label"] == "A"

    def test_excludes_unlabeled(self, db_logger):
        db_logger.log_image_prediction(_pred(well="K07", t_label=None))
        assert db_logger.fetch_current_quality(RUN_ID, *_window()) == []

    def test_excludes_reference(self, db_logger):
        db_logger.log_image_prediction(
            _pred(well="K07", p_label="A", t_label="A", is_reference=True))
        assert db_logger.fetch_current_quality(RUN_ID, *_window()) == []

    def test_excludes_benchmark(self, db_logger):
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="A")
        db_logger.log_image_prediction(
            _pred(well="K07", p_label="A", t_label="A", benchmark_id=bid))
        assert db_logger.fetch_current_quality(RUN_ID, *_window()) == []

    def test_excludes_reference_wells(self, db_logger):
        """A labeled live row for a well in this run's reference set is excluded."""
        db_logger.log_image_prediction(_pred(well="K07", is_reference=True))
        db_logger.log_image_prediction(_pred(well="K07", p_label="A", t_label="A"))
        assert db_logger.fetch_current_quality(RUN_ID, *_window()) == []


class TestLogQualityReport:
    def test_returns_positive_id(self, db_logger):
        now = datetime.now()
        report_id = db_logger.log_quality_report(
            (RUN_ID, now - timedelta(days=7), now, 64, 20,
             0.85, 0.80, 0.70, 0.65, "/reports/quality_1"))
        assert isinstance(report_id, int)
        assert report_id > 0

    def test_stored_values(self, db_logger):
        now = datetime.now()
        report_id = db_logger.log_quality_report(
            (RUN_ID, now - timedelta(days=7), now, 64, 20,
             0.5781, 0.5051, 0.5821, 0.4611, "/reports/q"))
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT n_benchmark_samples, n_current_samples, "
                "benchmark_accuracy, benchmark_f1, current_accuracy, current_f1 "
                "FROM quality_report WHERE id = %s", (report_id,))
            row = cur.fetchone()
        assert row[0] == 64
        assert row[1] == 20
        assert abs(row[2] - 0.5781) < 1e-4
        assert abs(row[3] - 0.5051) < 1e-4
        assert abs(row[4] - 0.5821) < 1e-4
        assert abs(row[5] - 0.4611) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# run_quality_report.main() -- orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _seed_benchmark(db_logger, n: int = 10):
    """One registered benchmark sample + n scored predictions on it (mostly
    correct) so fetch_benchmark_quality returns labeled rows."""
    bid, _ = insert_benchmark_sample(db_logger, well="B01", t_label="positive")
    for i in range(n):
        # 8/10 correct -> benchmark accuracy 0.8, deterministic to assert on.
        db_logger.log_image_prediction(_pred(
            well="B01", field=i + 1, run_id=RUN_ID, t_label="positive",
            p_label="positive" if i < 8 else "negative",
            benchmark_id=bid,
        ))
    return bid


def _seed_current(db_logger, n: int = 6):
    """Labeled production rows in the default window (benchmark_id NULL,
    t_label NOT NULL) for fetch_current_quality."""
    for i in range(n):
        db_logger.log_image_prediction(_pred(
            well=f"P{i:02d}", run_id=RUN_ID, t_label="positive",
            p_label="positive" if i < 4 else "negative",
        ))


def _report_rows(db_logger):
    with db_logger.pool.connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT run_id, n_benchmark_samples, n_current_samples,
                   benchmark_accuracy, current_accuracy, report_path
            FROM quality_report
        """)
        return cur.fetchall()


class TestMain:
    def test_full_report_writes_db_row(self, monkeypatch, db_logger, tmp_path):
        """main() compares benchmark vs labeled production and persists one
        quality_report row with real accuracy numbers."""
        pytest.importorskip("evidently")  # real ClassificationPreset run
        _seed_benchmark(db_logger, n=10)
        _seed_current(db_logger, n=6)
        _run_main(monkeypatch, "--run-id", RUN_ID, "--reports-dir", str(tmp_path))

        reports = _report_rows(db_logger)
        assert len(reports) == 1
        run_id, n_bench, n_cur, bench_acc, cur_acc, report_path = reports[0]
        assert run_id == RUN_ID
        assert n_bench == 10
        assert n_cur == 6
        assert abs(bench_acc - 0.8) < 1e-6
        assert abs(cur_acc - 4 / 6) < 1e-6

        from pathlib import Path
        out = Path(report_path)
        assert (out / "report.html").exists()
        assert (out / "metrics.json").exists()

    def test_no_benchmark_predictions_exits_cleanly(self, monkeypatch, db_logger, tmp_path):
        """No scored benchmark rows for the run -> error, nothing written."""
        _seed_current(db_logger, n=3)
        _run_main(monkeypatch, "--run-id", RUN_ID, "--reports-dir", str(tmp_path))
        assert _report_rows(db_logger) == []

    def test_no_labeled_current_exits_cleanly(self, monkeypatch, db_logger, tmp_path):
        """Unlabeled production rows don't count as current quality data."""
        _seed_benchmark(db_logger, n=5)
        for i in range(3):
            db_logger.log_image_prediction(
                _pred(well=f"P{i:02d}", run_id=RUN_ID, t_label=None))
        _run_main(monkeypatch, "--run-id", RUN_ID, "--reports-dir", str(tmp_path))
        assert _report_rows(db_logger) == []
