"""DB integration tests for label backfill methods.

Covers DBLogger.fetch_unlabeled_wells and update_t_label against a real
PostgreSQL with schemas 01-04, verifying that reference/benchmark rows are
excluded and existing labels are never overwritten.
"""
from __future__ import annotations

from tests.monitoring.db.conftest import (
    PLATE,
    RUN_ID,
    insert_benchmark_sample,
    insert_images,
    make_image_prediction_tuple,
)


class TestFetchUnlabeledWells:
    def test_empty(self, db_logger):
        assert db_logger.fetch_unlabeled_wells() == []

    def test_returns_unlabeled_production(self, db_logger):
        db_logger.log_image_prediction(make_image_prediction_tuple(well="K07", t_label=None))
        wells = db_logger.fetch_unlabeled_wells()
        assert (PLATE, "K07") in wells

    def test_excludes_labeled_production(self, db_logger):
        db_logger.log_image_prediction(make_image_prediction_tuple(well="K07", t_label="ClassA"))
        assert db_logger.fetch_unlabeled_wells() == []

    def test_excludes_reference(self, db_logger):
        """Reference rows (is_reference=TRUE) are not returned even if unlabeled."""
        db_logger.log_image_prediction(
            make_image_prediction_tuple(well="K07", t_label=None, is_reference=True)
        )
        assert db_logger.fetch_unlabeled_wells() == []

    def test_excludes_benchmark(self, db_logger):
        """Benchmark rows (benchmark_id NOT NULL) are not returned even if unlabeled."""
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        db_logger.log_image_prediction(
            make_image_prediction_tuple(well="K07", t_label=None, benchmark_id=bid)
        )
        assert db_logger.fetch_unlabeled_wells() == []

    def test_deduplicates_wells(self, db_logger):
        """Multiple predictions for the same well return one (plate, well) pair."""
        db_logger.log_image_prediction(make_image_prediction_tuple(well="K07", field=1))
        db_logger.log_image_prediction(make_image_prediction_tuple(well="K07", field=2))
        wells = db_logger.fetch_unlabeled_wells()
        assert len(wells) == 1
        assert wells[0] == (PLATE, "K07")


class TestUpdateTLabel:
    def test_updates_image_prediction(self, db_logger):
        db_logger.log_image_prediction(make_image_prediction_tuple(well="K07", t_label=None))
        n_img, _ = db_logger.update_t_label(PLATE, "K07", "ClassA")
        assert n_img == 1
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT t_label FROM image_prediction WHERE plate=%s AND well=%s",
                        (PLATE, "K07"))
            assert cur.fetchone()[0] == "ClassA"

    def test_does_not_overwrite_existing(self, db_logger):
        """update_t_label only fills NULL t_labels, never overwrites."""
        db_logger.log_image_prediction(make_image_prediction_tuple(well="K07", t_label="OldLabel"))
        db_logger.update_t_label(PLATE, "K07", "NewLabel")
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT t_label FROM image_prediction WHERE plate=%s AND well=%s",
                        (PLATE, "K07"))
            assert cur.fetchone()[0] == "OldLabel"

    def test_excludes_reference(self, db_logger):
        """Reference rows are not updated by backfill."""
        db_logger.log_image_prediction(
            make_image_prediction_tuple(well="K07", t_label=None, is_reference=True)
        )
        n_img, _ = db_logger.update_t_label(PLATE, "K07", "ClassA")
        assert n_img == 0
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT t_label FROM image_prediction WHERE is_reference=TRUE")
            assert cur.fetchone()[0] is None

    def test_excludes_benchmark(self, db_logger):
        """Benchmark rows are not updated by backfill."""
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        db_logger.log_image_prediction(
            make_image_prediction_tuple(well="K07", t_label=None, benchmark_id=bid)
        )
        n_img, _ = db_logger.update_t_label(PLATE, "K07", "ClassB")
        assert n_img == 0

    def test_updates_multiple_rows(self, db_logger):
        """All production rows for a well are updated in one call."""
        db_logger.log_image_prediction(make_image_prediction_tuple(well="K07", field=1, t_label=None))
        db_logger.log_image_prediction(make_image_prediction_tuple(well="K07", field=2, t_label=None))
        db_logger.log_image_prediction(make_image_prediction_tuple(well="K07", field=3, t_label=None))
        n_img, _ = db_logger.update_t_label(PLATE, "K07", "ClassA")
        assert n_img == 3

    def test_updates_tile_predictions(self, db_logger):
        """Tile predictions for the updated image rows also get t_label."""
        from utils.filename_parser import clean_tiles_metadata
        img_ids = insert_images(db_logger)
        tiles = [{"row": 0, "col": 0, "x": 0, "y": 0, "crop_size": 512}]
        tile_stack_metadata = clean_tiles_metadata(tiles, img_ids)
        tile_stack_ids = db_logger.log_tile_stack(tile_stack_metadata)
        members = [(tile_stack_ids[0], img_id, i) for i, img_id in enumerate(img_ids)]
        db_logger.log_tile_stack_member(members)
        img_pred_id = db_logger.log_image_prediction(
            make_image_prediction_tuple(well="K07", t_label=None)
        )
        tile_preds = [(img_pred_id, tile_stack_ids[0], RUN_ID, "positive", None, 0.9, False, None)]
        db_logger.log_tile_prediction(tile_preds)

        n_img, n_tile = db_logger.update_t_label(PLATE, "K07", "ClassA")
        assert n_img == 1
        assert n_tile == 1
        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT t_label FROM tile_prediction WHERE image_pred_id = %s", (img_pred_id,))
            assert cur.fetchone()[0] == "ClassA"

    def test_global_by_well(self, db_logger):
        """update_t_label updates all run_ids for a well, not just one."""
        db_logger.log_image_prediction(
            make_image_prediction_tuple(well="K07", run_id="run1", t_label=None)
        )
        db_logger.log_image_prediction(
            make_image_prediction_tuple(well="K07", run_id="run2", t_label=None)
        )
        n_img, _ = db_logger.update_t_label(PLATE, "K07", "ClassA")
        assert n_img == 2
