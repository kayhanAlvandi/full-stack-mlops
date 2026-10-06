"""DB integration tests for monitoring/register_benchmark.py.

Two layers, both against real PostgreSQL (schemas 01-04):

- Method level: the DBLogger registration primitives
  (log_benchmark_sample / log_benchmark_members / get_benchmark_samples).
- Orchestration: register_benchmark.main() end-to-end in --manifest mode,
  which needs neither image files nor MongoDB (labels and shapes come from
  the manifest), so the whole DB path -- group samples, skip already
  registered, insert image_metadata + benchmark_dataset + members -- runs
  for real. The glob fallback's label resolution needs MongoDB, which stays
  an external-dependency boundary; its parsing/grouping half is covered by
  unit tests.
"""
from __future__ import annotations

import sys

import pytest

from monitoring import register_benchmark
from tests.monitoring.db.conftest import (
    DB_TEST_URI,
    PLATE,
    insert_benchmark_sample,
    insert_images,
)


def _run_main(monkeypatch, *argv: str):
    monkeypatch.setenv("MONITORING_DB_URI", DB_TEST_URI)
    monkeypatch.setattr(sys, "argv", ["register_benchmark.py", *argv])
    register_benchmark.main()


def _count(db_logger, table: str) -> int:
    with db_logger.pool.connection() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")  # table names are fixed literals
        return cur.fetchone()[0]


# ─────────────────────────────────────────────────────────────────────────────
# DBLogger registration primitives
# ─────────────────────────────────────────────────────────────────────────────

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
            cur.execute(
                "SELECT COUNT(*) FROM benchmark_dataset_member WHERE benchmark_id = %s",
                (bid,))
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


# ─────────────────────────────────────────────────────────────────────────────
# register_benchmark.main() -- orchestration (manifest mode)
# ─────────────────────────────────────────────────────────────────────────────

class TestMain:
    def test_no_db_uri(self, monkeypatch, flat_manifest):
        """No MONITORING_DB_URI -> main() exits before touching the database."""
        monkeypatch.setenv("MONITORING_DB_URI", "")
        monkeypatch.setattr(sys, "argv",
                            ["register_benchmark.py", "--manifest", str(flat_manifest)])
        register_benchmark.main()

    def test_no_input_errors(self, monkeypatch):
        """Neither --manifest nor globs -> argparse error (SystemExit)."""
        monkeypatch.setenv("MONITORING_DB_URI", DB_TEST_URI)
        monkeypatch.setattr(sys, "argv", ["register_benchmark.py"])
        with pytest.raises(SystemExit):
            register_benchmark.main()

    def test_registers_manifest_samples(self, monkeypatch, db_logger, flat_manifest):
        """main() writes image_metadata, benchmark_dataset, and member rows for
        every manifest sample, with labels from the manifest."""
        _run_main(monkeypatch, "--manifest", str(flat_manifest))

        assert _count(db_logger, "image_metadata") == 15          # 3 samples x 5 channels
        assert _count(db_logger, "benchmark_dataset") == 3
        assert _count(db_logger, "benchmark_dataset_member") == 15

        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT plate, well, field, t_label FROM benchmark_dataset ORDER BY well")
            assert cur.fetchall() == [
                (PLATE, "K07", 1, "ClassA"),
                (PLATE, "K08", 1, "ClassB"),
                (PLATE, "K09", 1, "ClassC"),
            ]

    def test_resume_skips_already_registered(self, monkeypatch, db_logger, flat_manifest):
        """Samples already in benchmark_dataset are skipped, so a rerun only
        registers what's missing -- safe to re-run as the benchmark grows."""
        insert_benchmark_sample(db_logger, well="K07")  # same key as manifest's K07 sample
        _run_main(monkeypatch, "--manifest", str(flat_manifest))

        assert _count(db_logger, "benchmark_dataset") == 3
        # K07 was skipped entirely: its 5 member images are the ones
        # insert_benchmark_sample already wrote (image_metadata upserts on
        # file_name, so re-inserting wouldn't error anyway -- but the skip
        # means the files are never even read).
        assert _count(db_logger, "benchmark_dataset_member") == 15

    def test_rerun_is_fully_idempotent(self, monkeypatch, db_logger, flat_manifest):
        """Second run registers nothing new."""
        _run_main(monkeypatch, "--manifest", str(flat_manifest))
        _run_main(monkeypatch, "--manifest", str(flat_manifest))

        assert _count(db_logger, "benchmark_dataset") == 3
        assert _count(db_logger, "benchmark_dataset_member") == 15
