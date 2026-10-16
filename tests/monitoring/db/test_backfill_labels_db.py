"""DB integration tests for monitoring/backfill_labels.py.

Two layers, both against real PostgreSQL (schemas 01-04):

- Method level: DBLogger.fetch_unlabeled_wells / update_t_label -- the two
  DB primitives the job is built on. Edge cases (reference/benchmark
  exclusion, no-overwrite, multi-row updates) are pinned down here where they
  can be exercised directly.
- Orchestration: backfill_labels.main() end-to-end -- fetch unlabeled wells,
  resolve labels, write them back. Only the MongoDB lookup is mocked;
  everything DB-side is real.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import monitoring.backfill_labels as backfill
from tests.monitoring.db.conftest import (
    DB_TEST_URI,
    PLATE,
    insert_benchmark_sample,
    insert_image_and_tile_prediction,
)
from tests.monitoring.db.conftest import make_image_prediction_tuple as _pred


def _set_db_uri(monkeypatch):
    monkeypatch.setenv("MONITORING_DB_URI", DB_TEST_URI)


def _run_main(monkeypatch, labels=None):
    """Run backfill.main() with --source mongodb, stubbing the resolved labels.

    --source mongodb needs no extra required CLI args (unlike csv's
    --csv-path), and MongoDBLabelsProvider is patched before construction so
    no real MongoDB/tools import happens.
    """
    monkeypatch.setattr("sys.argv", ["backfill_labels", "--source", "mongodb"])
    mock_provider = MagicMock()
    mock_provider.get_labels_for_wells.return_value = labels or {}
    with patch("monitoring.backfill_labels.MongoDBLabelsProvider",
               return_value=mock_provider):
        backfill.main()
    return mock_provider.get_labels_for_wells


# ─────────────────────────────────────────────────────────────────────────────
# DBLogger.fetch_unlabeled_wells
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchUnlabeledWells:
    def test_empty(self, db_logger):
        assert db_logger.fetch_unlabeled_wells() == []

    def test_returns_unlabeled_production(self, db_logger):
        db_logger.log_image_prediction(_pred(well="K07", t_label=None))
        wells = db_logger.fetch_unlabeled_wells()
        assert (PLATE, "K07") in wells

    def test_excludes_labeled_production(self, db_logger):
        db_logger.log_image_prediction(_pred(well="K07", t_label="ClassA"))
        assert db_logger.fetch_unlabeled_wells() == []

    def test_excludes_reference(self, db_logger):
        """Reference rows (is_reference=TRUE) are not returned even if unlabeled."""
        db_logger.log_image_prediction(_pred(well="K07", t_label=None, is_reference=True))
        assert db_logger.fetch_unlabeled_wells() == []

    def test_excludes_benchmark(self, db_logger):
        """Benchmark rows (benchmark_id NOT NULL) are not returned even if unlabeled."""
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        db_logger.log_image_prediction(_pred(well="K07", t_label=None, benchmark_id=bid))
        assert db_logger.fetch_unlabeled_wells() == []

    def test_deduplicates_wells(self, db_logger):
        """Multiple predictions for the same well return one (plate, well) pair."""
        db_logger.log_image_prediction(_pred(well="K07", field=1))
        db_logger.log_image_prediction(_pred(well="K07", field=2))
        wells = db_logger.fetch_unlabeled_wells()
        assert len(wells) == 1
        assert wells[0] == (PLATE, "K07")


# ─────────────────────────────────────────────────────────────────────────────
# DBLogger.update_t_label
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateTLabel:
    def test_updates_image_prediction(self, db_logger):
        db_logger.log_image_prediction(_pred(well="K07", t_label=None))
        n_img, _ = db_logger.update_t_label(PLATE, "K07", "ClassA")
        assert n_img == 1
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT t_label FROM image_prediction WHERE plate=%s AND well=%s",
                        (PLATE, "K07"))
            assert cur.fetchone()[0] == "ClassA"

    def test_does_not_overwrite_existing(self, db_logger):
        """update_t_label only fills NULL t_labels, never overwrites."""
        db_logger.log_image_prediction(_pred(well="K07", t_label="OldLabel"))
        db_logger.update_t_label(PLATE, "K07", "NewLabel")
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT t_label FROM image_prediction WHERE plate=%s AND well=%s",
                        (PLATE, "K07"))
            assert cur.fetchone()[0] == "OldLabel"

    def test_excludes_reference(self, db_logger):
        """Reference rows are not updated by backfill."""
        db_logger.log_image_prediction(_pred(well="K07", t_label=None, is_reference=True))
        n_img, _ = db_logger.update_t_label(PLATE, "K07", "ClassA")
        assert n_img == 0
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT t_label FROM image_prediction WHERE is_reference=TRUE")
            assert cur.fetchone()[0] is None

    def test_excludes_benchmark(self, db_logger):
        """Benchmark rows are not updated by backfill."""
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        db_logger.log_image_prediction(_pred(well="K07", t_label=None, benchmark_id=bid))
        n_img, _ = db_logger.update_t_label(PLATE, "K07", "ClassB")
        assert n_img == 0

    def test_updates_multiple_rows(self, db_logger):
        """All production rows for a well are updated in one call."""
        db_logger.log_image_prediction(_pred(well="K07", field=1, t_label=None))
        db_logger.log_image_prediction(_pred(well="K07", field=2, t_label=None))
        db_logger.log_image_prediction(_pred(well="K07", field=3, t_label=None))
        n_img, _ = db_logger.update_t_label(PLATE, "K07", "ClassA")
        assert n_img == 3

    def test_updates_tile_predictions(self, db_logger):
        """Tile predictions for the updated image rows also get t_label."""
        img_pred_id = insert_image_and_tile_prediction(db_logger, well="K07", t_label=None)
        n_img, n_tile = db_logger.update_t_label(PLATE, "K07", "ClassA")
        assert n_img == 1
        assert n_tile == 1
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT t_label FROM tile_prediction WHERE image_pred_id = %s",
                        (img_pred_id,))
            assert cur.fetchone()[0] == "ClassA"

    def test_global_by_well(self, db_logger):
        """update_t_label updates all run_ids for a well, not just one."""
        db_logger.log_image_prediction(_pred(well="K07", run_id="run1", t_label=None))
        db_logger.log_image_prediction(_pred(well="K07", run_id="run2", t_label=None))
        n_img, _ = db_logger.update_t_label(PLATE, "K07", "ClassA")
        assert n_img == 2


# ─────────────────────────────────────────────────────────────────────────────
# backfill_labels.main() -- orchestration
# ─────────────────────────────────────────────────────────────────────────────

class TestMain:
    def test_no_db_uri(self, monkeypatch):
        """Without MONITORING_DB_URI, main() prints an error and returns.

        Set to empty rather than deleted: a local .env could still supply a
        URI after delenv, while an explicit empty env var takes precedence and
        makes has_db_uri deterministically False.
        """
        monkeypatch.setenv("MONITORING_DB_URI", "")
        monkeypatch.setattr("sys.argv", ["backfill_labels", "--source", "mongodb"])
        assert backfill.main() is None

    def test_db_connection_failure(self, monkeypatch):
        """An unreachable DB URI makes main() return cleanly instead of raising."""
        monkeypatch.setenv("MONITORING_DB_URI", "postgresql://baduser:badpass@localhost:1/nope")
        monkeypatch.setattr("sys.argv", ["backfill_labels", "--source", "mongodb"])
        assert backfill.main() is None

    def test_no_unlabeled_wells(self, monkeypatch, db_logger):
        """With no unlabeled production rows, the provider is never queried."""
        _set_db_uri(monkeypatch)
        mock_get_labels = _run_main(monkeypatch)
        mock_get_labels.assert_not_called()

    def test_no_labels_resolved_writes_nothing(self, monkeypatch, db_logger):
        """When the provider resolves no labels, no t_label is written."""
        _set_db_uri(monkeypatch)
        db_logger.log_image_prediction(_pred(well="K07", t_label=None))
        _run_main(monkeypatch, labels={})
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT t_label FROM image_prediction WHERE well = %s", ("K07",))
            assert cur.fetchone()[0] is None

    def test_labels_updated_writes_real_rows(self, monkeypatch, db_logger):
        """Resolved labels are written back to image_prediction (and its tile
        rows) for every matching well, via a real DBLogger."""
        _set_db_uri(monkeypatch)
        db_logger.log_image_prediction(_pred(well="K07", t_label=None))
        db_logger.log_image_prediction(_pred(well="K08", t_label=None))
        labels = {(PLATE, "K07"): "ClassA", (PLATE, "K08"): "ClassB"}
        mock_get_labels = _run_main(monkeypatch, labels=labels)

        called_wells = set(mock_get_labels.call_args[0][0])
        assert {(PLATE, "K07"), (PLATE, "K08")} <= called_wells

        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT well, t_label FROM image_prediction ORDER BY well")
            assert cur.fetchall() == [("K07", "ClassA"), ("K08", "ClassB")]

    def test_labels_updated_never_overwrites_existing_label(self, monkeypatch, db_logger):
        """A production row that already has a t_label is left untouched, even
        if the provider resolves a (possibly different) label for its well."""
        _set_db_uri(monkeypatch)
        db_logger.log_image_prediction(_pred(well="K07", t_label="AlreadySet"))
        # fetch_unlabeled_wells only returns wells with a NULL t_label row, so
        # this well wouldn't normally reach the provider at all; simulate the
        # defensive case where update_t_label is still asked to write to it.
        _run_main(monkeypatch, labels={(PLATE, "K07"): "ClassA"})
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT t_label FROM image_prediction WHERE well = %s", ("K07",))
            assert cur.fetchone()[0] == "AlreadySet"

    def test_tile_predictions_updated_too(self, monkeypatch, db_logger):
        """The label lands on both the image_prediction row and its child
        tile_prediction rows -- update_t_label keeps them in one transaction."""
        _set_db_uri(monkeypatch)
        img_pred_id = insert_image_and_tile_prediction(db_logger, well="K07", t_label=None)
        _run_main(monkeypatch, labels={(PLATE, "K07"): "ClassA"})
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT t_label FROM image_prediction WHERE id = %s", (img_pred_id,))
            assert cur.fetchone()[0] == "ClassA"
            cur.execute("SELECT DISTINCT t_label FROM tile_prediction WHERE image_pred_id = %s",
                        (img_pred_id,))
            assert cur.fetchall() == [("ClassA",)]

    def test_reference_and_benchmark_rows_untouched(self, monkeypatch, db_logger):
        """Unlabeled reference/benchmark rows for the same well are neither
        passed to the provider (fetch_unlabeled_wells only sees production
        rows) nor updated (update_t_label filters them again defensively)."""
        _set_db_uri(monkeypatch)
        db_logger.log_image_prediction(_pred(well="K07", t_label=None))                     # production
        db_logger.log_image_prediction(_pred(well="K07", t_label=None, is_reference=True))  # reference
        bid, _ = insert_benchmark_sample(db_logger, well="K08", t_label="ClassB")
        db_logger.log_image_prediction(_pred(well="K08", t_label=None, benchmark_id=bid))   # benchmark
        mock_get_labels = _run_main(monkeypatch, labels={(PLATE, "K07"): "ClassA"})

        # Only the production well is looked up.
        assert set(mock_get_labels.call_args[0][0]) == {(PLATE, "K07")}

        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT is_reference, benchmark_id IS NOT NULL, t_label
                FROM image_prediction ORDER BY id
            """)
            assert cur.fetchall() == [
                (False, False, "ClassA"),  # production: labeled
                (True, False, None),       # reference: untouched
                (False, True, None),       # benchmark: untouched
            ]

    def test_rerun_is_idempotent(self, monkeypatch, db_logger):
        """A second main() run finds no unlabeled wells left, so the provider
        is never queried and nothing is rewritten."""
        _set_db_uri(monkeypatch)
        db_logger.log_image_prediction(_pred(well="K07", t_label=None))
        _run_main(monkeypatch, labels={(PLATE, "K07"): "ClassA"})
        mock_get_labels = _run_main(monkeypatch)
        mock_get_labels.assert_not_called()
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT DISTINCT t_label FROM image_prediction")
            assert cur.fetchall() == [("ClassA",)]
