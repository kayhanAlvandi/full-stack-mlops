"""DB integration tests for live/reference/benchmark view partitioning.

Verifies that the SQL views in 02_reference.sql and 03_benchmark.sql correctly
partition image_prediction and tile_prediction rows into three disjoint sets:
live (production), reference (validation), and benchmark.
"""
from __future__ import annotations

from tests.monitoring.db.conftest import (
    RUN_ID,
    insert_benchmark_sample,
    insert_images,
    make_image_prediction_tuple,
)


def _count(db_logger, view, where=""):
    sql = f"SELECT COUNT(*) FROM {view}"
    if where:
        sql += f" WHERE {where}"
    with db_logger.pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()[0]


class TestLiveView:
    def test_live_excludes_reference(self, db_logger):
        db_logger.log_image_prediction(make_image_prediction_tuple(is_reference=True))
        assert _count(db_logger, "live_image_prediction") == 0

    def test_live_excludes_benchmark(self, db_logger):
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        db_logger.log_image_prediction(make_image_prediction_tuple(benchmark_id=bid))
        assert _count(db_logger, "live_image_prediction") == 0

    def test_live_includes_production(self, db_logger):
        db_logger.log_image_prediction(make_image_prediction_tuple())
        assert _count(db_logger, "live_image_prediction") == 1

    def test_live_excludes_both_reference_and_benchmark(self, db_logger):
        """A row that is both reference and benchmark should not appear in live."""
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        db_logger.log_image_prediction(
            make_image_prediction_tuple(is_reference=True, benchmark_id=bid)
        )
        assert _count(db_logger, "live_image_prediction") == 0


class TestReferenceView:
    def test_reference_includes_reference(self, db_logger):
        db_logger.log_image_prediction(make_image_prediction_tuple(is_reference=True))
        assert _count(db_logger, "reference_image_prediction") == 1

    def test_reference_excludes_production(self, db_logger):
        db_logger.log_image_prediction(make_image_prediction_tuple(is_reference=False))
        assert _count(db_logger, "reference_image_prediction") == 0

    def test_reference_includes_benchmark_that_is_also_reference(self, db_logger):
        """reference_image_prediction filters only on is_reference=TRUE, so a
        benchmark row with is_reference=TRUE appears here too. This is an edge
        case that shouldn't happen in practice (benchmark rows have
        is_reference=FALSE), but the view definition is correct per its
        documented contract (filter on is_reference only)."""
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        db_logger.log_image_prediction(
            make_image_prediction_tuple(is_reference=True, benchmark_id=bid)
        )
        assert _count(db_logger, "reference_image_prediction") == 1


class TestBenchmarkView:
    def test_benchmark_includes_benchmark(self, db_logger):
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="ClassA")
        db_logger.log_image_prediction(make_image_prediction_tuple(benchmark_id=bid))
        assert _count(db_logger, "benchmark_image_prediction") == 1

    def test_benchmark_excludes_production(self, db_logger):
        db_logger.log_image_prediction(make_image_prediction_tuple(benchmark_id=None))
        assert _count(db_logger, "benchmark_image_prediction") == 0

    def test_benchmark_excludes_reference(self, db_logger):
        db_logger.log_image_prediction(make_image_prediction_tuple(is_reference=True))
        assert _count(db_logger, "benchmark_image_prediction") == 0


class TestPartitionDisjoint:
    def test_live_plus_reference_plus_benchmark_equals_total(self, db_logger):
        """The three views partition all image_prediction rows."""
        # 3 live
        db_logger.log_image_prediction(make_image_prediction_tuple(well="K01"))
        db_logger.log_image_prediction(make_image_prediction_tuple(well="K02"))
        db_logger.log_image_prediction(make_image_prediction_tuple(well="K03"))
        # 2 reference
        db_logger.log_image_prediction(make_image_prediction_tuple(well="K04", is_reference=True))
        db_logger.log_image_prediction(make_image_prediction_tuple(well="K05", is_reference=True))
        # 2 benchmark
        bid1, _ = insert_benchmark_sample(db_logger, well="K07", t_label="A")
        bid2, _ = insert_benchmark_sample(db_logger, well="K08", t_label="B")
        db_logger.log_image_prediction(make_image_prediction_tuple(well="K07", benchmark_id=bid1))
        db_logger.log_image_prediction(make_image_prediction_tuple(well="K08", benchmark_id=bid2))

        total = _count(db_logger, "image_prediction")
        live = _count(db_logger, "live_image_prediction")
        ref = _count(db_logger, "reference_image_prediction")
        bench = _count(db_logger, "benchmark_image_prediction")
        assert live + ref + bench == total
        assert live == 3
        assert ref == 2
        assert bench == 2


class TestTilePredictionViews:
    """Same partitioning applies to tile_prediction views."""

    def _setup_tile_pred(self, db_logger, is_reference=False, benchmark_id=None):
        """Insert prerequisites and one tile_prediction row."""
        img_ids = insert_images(db_logger)
        # Need a tile_stack for the tile_prediction FK
        from utils.filename_parser import clean_tiles_metadata
        tiles = [{"row": 0, "col": 0, "x": 0, "y": 0, "crop_size": 512}]
        tile_stack_metadata = clean_tiles_metadata(tiles, img_ids)
        tile_stack_ids = db_logger.log_tile_stack(tile_stack_metadata)
        members = [(tile_stack_ids[0], img_id, i) for i, img_id in enumerate(img_ids)]
        db_logger.log_tile_stack_member(members)
        img_pred_id = db_logger.log_image_prediction(
            make_image_prediction_tuple(is_reference=is_reference, benchmark_id=benchmark_id)
        )
        tile_preds = [(img_pred_id, tile_stack_ids[0], RUN_ID, "positive", None, 0.9,
                       is_reference, benchmark_id)]
        db_logger.log_tile_prediction(tile_preds)
        return img_pred_id

    def test_live_tile_excludes_reference(self, db_logger):
        self._setup_tile_pred(db_logger, is_reference=True)
        assert _count(db_logger, "live_tile_prediction") == 0
        assert _count(db_logger, "reference_tile_prediction") == 1

    def test_live_tile_excludes_benchmark(self, db_logger):
        bid, _ = insert_benchmark_sample(db_logger, well="K07", t_label="A")
        self._setup_tile_pred(db_logger, benchmark_id=bid)
        assert _count(db_logger, "live_tile_prediction") == 0
        assert _count(db_logger, "benchmark_tile_prediction") == 1

    def test_live_tile_includes_production(self, db_logger):
        self._setup_tile_pred(db_logger)
        assert _count(db_logger, "live_tile_prediction") == 1
