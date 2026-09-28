"""DB integration tests for benchmark registration and scoring methods.

Covers DBLogger.log_benchmark_sample, log_benchmark_members,
get_benchmark_samples, fetch_benchmark_members, and
get_benchmark_predictions against a real PostgreSQL with schemas 01-04.
"""
from __future__ import annotations

from tests.monitoring.db.conftest import (
    PLATE,
    RUN_ID,
    insert_benchmark_sample,
    insert_images,
    make_image_prediction_tuple,
)


class TestLogBenchmarkSample:
    def test_returns_positive_id(self, db_logger):
        bid = db_logger.log_benchmark_sample((PLATE, "K07", 1, "ClassA"))
        assert isinstance(bid, int)
        assert bid > 0

    def test_idempotent_same_key(self, db_logger):
        """Re-registering the same (plate, well, field) returns the same id."""
        id1 = db_logger.log_benchmark_sample((PLATE, "K07", 1, "ClassA"))
        id2 = db_logger.log_benchmark_sample((PLATE, "K07", 1, "ClassA"))
        assert id1 == id2

    def test_updates_label_on_reregister(self, db_logger):
        """Re-registering with a different t_label updates the row."""
        db_logger.log_benchmark_sample((PLATE, "K07", 1, "ClassA"))
        bid = db_logger.log_benchmark_sample((PLATE, "K07", 1, "ClassB"))
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT t_label FROM benchmark_dataset WHERE id = %s", (bid,))
            assert cur.fetchone()[0] == "ClassB"

    def test_different_samples_different_ids(self, db_logger):
        id1 = db_logger.log_benchmark_sample((PLATE, "K07", 1, "ClassA"))
        id2 = db_logger.log_benchmark_sample((PLATE, "K08", 1, "ClassB"))
        assert id1 != id2


class TestLogBenchmarkMembers:
    def test_members_inserted(self, db_logger):
        img_ids = insert_images(db_logger)
        bid = db_logger.log_benchmark_sample((PLATE, "K07", 1, "ClassA"))
        members = [(bid, img_id, i) for i, img_id in enumerate(img_ids)]
        db_logger.log_benchmark_members(members)
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM benchmark_dataset_member WHERE benchmark_id = %s", (bid,))
            assert cur.fetchone()[0] == len(img_ids)

    def test_channel_index_stored(self, db_logger):
        img_ids = insert_images(db_logger)
        bid = db_logger.log_benchmark_sample((PLATE, "K07", 1, "ClassA"))
        members = [(bid, img_ids[i], i) for i in range(len(img_ids))]
        db_logger.log_benchmark_members(members)
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT channel_index FROM benchmark_dataset_member "
                "WHERE benchmark_id = %s ORDER BY channel_index", (bid,))
            indices = [row[0] for row in cur.fetchall()]
        assert indices == list(range(len(img_ids)))


class TestGetBenchmarkSamples:
    def test_empty(self, db_logger):
        assert db_logger.get_benchmark_samples() == []

    def test_returns_registered_keys(self, db_logger):
        insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        insert_benchmark_sample(db_logger, well="K08", t_label="ClassB")
        samples = db_logger.get_benchmark_samples()
        assert len(samples) == 2
        keys = {(s[0], s[1], s[2]) for s in samples}
        assert (PLATE, "K07", 1) in keys
        assert (PLATE, "K08", 1) in keys


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
        pred = make_image_prediction_tuple(
            well="K07", p_label="ClassA", t_label="ClassA",
            benchmark_id=bid,
        )
        db_logger.log_image_prediction(pred)
        result = db_logger.get_benchmark_predictions(RUN_ID)
        assert bid in result

    def test_excludes_non_benchmark(self, db_logger):
        """A live prediction (benchmark_id=NULL) is not returned."""
        db_logger.log_image_prediction(make_image_prediction_tuple(benchmark_id=None))
        assert db_logger.get_benchmark_predictions(RUN_ID) == []

    def test_filters_by_run_id(self, db_logger):
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        db_logger.log_image_prediction(
            make_image_prediction_tuple(run_id="other_run", benchmark_id=bid)
        )
        assert db_logger.get_benchmark_predictions(RUN_ID) == []
        assert bid in db_logger.get_benchmark_predictions("other_run")
