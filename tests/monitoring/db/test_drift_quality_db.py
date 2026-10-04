"""DB integration tests for drift and quality report DB methods.

Covers DBLogger.fetch_reference_image_level, fetch_current_image_level,
fetch_benchmark_quality, fetch_current_quality, log_drift_report,
log_drift_report_column, and log_quality_report against a real PostgreSQL
with schemas 01-04.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from tests.monitoring.db.conftest import (
    RUN_ID,
    insert_benchmark_sample,
    make_image_prediction_tuple,
)

# ─────────────────────────────────────────────────────────────────────────────
# Drift report fetches
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchReferenceImageLevel:
    def test_empty(self, db_logger):
        assert db_logger.fetch_reference_image_level(RUN_ID) == []

    def test_returns_reference_rows(self, db_logger):
        db_logger.log_image_prediction(
            make_image_prediction_tuple(well="K07", is_reference=True, p_label="ClassA")
        )
        rows = db_logger.fetch_reference_image_level(RUN_ID)
        assert len(rows) == 1
        assert rows[0]["p_label"] == "ClassA"
        assert "vote_fraction" in rows[0]
        assert "avg_confidence" in rows[0]

    def test_excludes_production(self, db_logger):
        db_logger.log_image_prediction(make_image_prediction_tuple(is_reference=False))
        assert db_logger.fetch_reference_image_level(RUN_ID) == []

    def test_filters_by_run_id(self, db_logger):
        db_logger.log_image_prediction(
            make_image_prediction_tuple(run_id="other", is_reference=True)
        )
        assert db_logger.fetch_reference_image_level(RUN_ID) == []


class TestFetchCurrentImageLevel:
    def test_empty(self, db_logger):
        now = datetime.now()
        rows = db_logger.fetch_current_image_level(RUN_ID, now - timedelta(days=7), now)
        assert rows == []

    def test_returns_live_rows_in_window(self, db_logger):
        db_logger.log_image_prediction(make_image_prediction_tuple(well="K07"))
        now = datetime.now()
        rows = db_logger.fetch_current_image_level(RUN_ID, now - timedelta(days=1), now + timedelta(days=1))
        assert len(rows) == 1
        assert rows[0]["p_label"] == "positive"

    def test_excludes_reference(self, db_logger):
        """Reference rows don't appear in the current (live) window."""
        db_logger.log_image_prediction(
            make_image_prediction_tuple(well="K07", is_reference=True)
        )
        now = datetime.now()
        rows = db_logger.fetch_current_image_level(RUN_ID, now - timedelta(days=1), now + timedelta(days=1))
        assert rows == []

    def test_excludes_benchmark(self, db_logger):
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="A")
        db_logger.log_image_prediction(make_image_prediction_tuple(well="K07", benchmark_id=bid))
        now = datetime.now()
        rows = db_logger.fetch_current_image_level(RUN_ID, now - timedelta(days=1), now + timedelta(days=1))
        assert rows == []

    def test_excludes_reference_wells(self, db_logger):
        """A live prediction for a well that also has a reference row is excluded."""
        db_logger.log_image_prediction(
            make_image_prediction_tuple(well="K07", is_reference=True)
        )
        db_logger.log_image_prediction(
            make_image_prediction_tuple(well="K07", is_reference=False)
        )
        now = datetime.now()
        rows = db_logger.fetch_current_image_level(RUN_ID, now - timedelta(days=1), now + timedelta(days=1))
        assert rows == []


# ─────────────────────────────────────────────────────────────────────────────
# Quality report fetches
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchBenchmarkQuality:
    def test_empty(self, db_logger):
        assert db_logger.fetch_benchmark_quality(RUN_ID) == []

    def test_returns_benchmark_with_labels(self, db_logger):
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        db_logger.log_image_prediction(
            make_image_prediction_tuple(well="K07", p_label="ClassA", t_label="ClassA",
                                         benchmark_id=bid)
        )
        rows = db_logger.fetch_benchmark_quality(RUN_ID)
        assert len(rows) == 1
        assert rows[0]["p_label"] == "ClassA"
        assert rows[0]["t_label"] == "ClassA"

    def test_excludes_unlabeled_benchmark(self, db_logger):
        """Benchmark rows without t_label are excluded."""
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        db_logger.log_image_prediction(
            make_image_prediction_tuple(well="K07", t_label=None, benchmark_id=bid)
        )
        assert db_logger.fetch_benchmark_quality(RUN_ID) == []

    def test_excludes_production(self, db_logger):
        db_logger.log_image_prediction(
            make_image_prediction_tuple(well="K07", p_label="A", t_label="A")
        )
        assert db_logger.fetch_benchmark_quality(RUN_ID) == []


class TestFetchCurrentQuality:
    def test_empty(self, db_logger):
        now = datetime.now()
        assert db_logger.fetch_current_quality(RUN_ID, now - timedelta(days=7), now) == []

    def test_returns_labeled_live_rows(self, db_logger):
        db_logger.log_image_prediction(
            make_image_prediction_tuple(well="K07", p_label="A", t_label="A")
        )
        now = datetime.now()
        rows = db_logger.fetch_current_quality(RUN_ID, now - timedelta(days=1), now + timedelta(days=1))
        assert len(rows) == 1
        assert rows[0]["p_label"] == "A"
        assert rows[0]["t_label"] == "A"

    def test_excludes_unlabeled(self, db_logger):
        db_logger.log_image_prediction(make_image_prediction_tuple(well="K07", t_label=None))
        now = datetime.now()
        assert db_logger.fetch_current_quality(RUN_ID, now - timedelta(days=1), now + timedelta(days=1)) == []

    def test_excludes_reference(self, db_logger):
        db_logger.log_image_prediction(
            make_image_prediction_tuple(well="K07", p_label="A", t_label="A", is_reference=True)
        )
        now = datetime.now()
        assert db_logger.fetch_current_quality(RUN_ID, now - timedelta(days=1), now + timedelta(days=1)) == []

    def test_excludes_benchmark(self, db_logger):
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="A")
        db_logger.log_image_prediction(
            make_image_prediction_tuple(well="K07", p_label="A", t_label="A", benchmark_id=bid)
        )
        now = datetime.now()
        assert db_logger.fetch_current_quality(RUN_ID, now - timedelta(days=1), now + timedelta(days=1)) == []


# ─────────────────────────────────────────────────────────────────────────────
# Report logging
# ─────────────────────────────────────────────────────────────────────────────

class TestLogDriftReport:
    def test_returns_positive_id(self, db_logger):
        now = datetime.now()
        report_id = db_logger.log_drift_report(
            (RUN_ID, now - timedelta(days=7), now, False, 2, 5, "/reports/drift_1")
        )
        assert isinstance(report_id, int)
        assert report_id > 0

    def test_stored_values(self, db_logger):
        now = datetime.now()
        start = now - timedelta(days=7)
        report_id = db_logger.log_drift_report(
            (RUN_ID, start, now, True, 3, 10, "/reports/drift_2")
        )
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT run_id, window_start, window_end, dataset_drift, "
                "n_columns_drifted, n_columns_total, report_path "
                "FROM drift_report WHERE id = %s", (report_id,)
            )
            row = cur.fetchone()
        assert row[0] == RUN_ID
        assert row[3] is True
        assert row[4] == 3
        assert row[5] == 10
        assert row[6] == "/reports/drift_2"


class TestLogDriftReportColumn:
    def test_inserts_columns(self, db_logger):
        now = datetime.now()
        report_id = db_logger.log_drift_report(
            (RUN_ID, now - timedelta(days=7), now, False, 0, 3, "/reports/d")
        )
        columns = [
            (report_id, "vote_fraction", "image_level", 0.1, False, "ks"),
            (report_id, "avg_confidence", "image_level", 0.2, True, "ks"),
            (report_id, "channel_1_mean", "channel_stats", 0.3, True, "ks"),
        ]
        n = db_logger.log_drift_report_column(columns)
        assert n == 3
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM drift_report_column WHERE drift_report_id = %s", (report_id,))
            assert cur.fetchone()[0] == 3


class TestLogQualityReport:
    def test_returns_positive_id(self, db_logger):
        now = datetime.now()
        report_id = db_logger.log_quality_report(
            (RUN_ID, now - timedelta(days=7), now, 64, 20,
             0.85, 0.80, 0.70, 0.65, "/reports/quality_1")
        )
        assert isinstance(report_id, int)
        assert report_id > 0

    def test_stored_values(self, db_logger):
        now = datetime.now()
        start = now - timedelta(days=7)
        report_id = db_logger.log_quality_report(
            (RUN_ID, start, now, 64, 20, 0.5781, 0.5051, 0.5821, 0.4611, "/reports/q")
        )
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT n_benchmark_samples, n_current_samples, "
                "benchmark_accuracy, benchmark_f1, current_accuracy, current_f1 "
                "FROM quality_report WHERE id = %s", (report_id,)
            )
            row = cur.fetchone()
        assert row[0] == 64
        assert row[1] == 20
        assert abs(row[2] - 0.5781) < 1e-4
        assert abs(row[3] - 0.5051) < 1e-4
        assert abs(row[4] - 0.5821) < 1e-4
        assert abs(row[5] - 0.4611) < 1e-4
