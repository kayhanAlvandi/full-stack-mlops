"""DB integration tests for TilePredictor.predict() against a real PostgreSQL
schema (schemas 01-03 via conftest.py).

These call TilePredictor.predict() directly (built around the StubModel from
tests/api/conftest.py, same as the unit tests) with a real DBLogger, and
assert on rows actually inserted. This is the boundary that a hand-written
fake DB logger can't verify: that predict()'s log_* calls match the real
DBLogger method signatures and the real schema, not just a method name.
"""
import numpy as np


def _channels(values: list[int], size: int = 32) -> list[np.ndarray]:
    return [np.full((size, size), v, dtype=np.float32) for v in values]


def _image_metadata(channels: list[int], channel_files: list[str], **overrides) -> dict:
    metadata = {
        "plate": "PLATE1",
        "well": "A01",
        "field": 1,
        "channels": channels,
        "channel_files": channel_files,
        "shape": (32, 32),
        "root_path": "/data/images",
    }
    metadata.update(overrides)
    return metadata


def _fetch_all(db_logger, query: str) -> list[tuple]:
    with db_logger.pool.connection() as conn, conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()


def test_predict_writes_one_row_per_log_call(predictor_factory, db_logger):
    """A single-tile, 3-channel prediction should write exactly one row (or
    one row per channel, for channel-scoped tables) to every table predict()
    logs to."""
    predictor = predictor_factory(db_logger=db_logger)
    channels = _channels([10, 20, 30])
    metadata = _image_metadata(
        channels=[1, 2, 3],
        channel_files=[
            "PLATE1_A01_T0001F001L01A01Z01C01.tif",
            "PLATE1_A01_T0001F001L01A01Z01C02.tif",
            "PLATE1_A01_T0001F001L01A01Z01C03.tif",
        ],
    )

    result = predictor.predict(channels, metadata)

    assert result["predicted_class"] == "ClassA"
    assert result["total_tiles"] == 1

    assert len(_fetch_all(db_logger, "SELECT * FROM image_metadata")) == 3
    assert len(_fetch_all(db_logger, "SELECT * FROM tile_stack")) == 1
    assert len(_fetch_all(db_logger, "SELECT * FROM tile_stack_member")) == 3
    assert len(_fetch_all(db_logger, "SELECT * FROM tile_channel_stats")) == 3
    image_predictions = _fetch_all(db_logger, "SELECT p_label FROM image_prediction")
    assert image_predictions == [("ClassA",)]
    tile_predictions = _fetch_all(db_logger, "SELECT p_label FROM tile_prediction")
    assert tile_predictions == [("ClassA",)]


def test_predict_logs_channels_in_canonical_ascending_order(predictor_factory, db_logger):
    """Channels uploaded out of order must be logged to image_metadata in
    ascending channel-number order, matching how training (src/dataset.py)
    stacks channels -- checked here against the real schema/insert path
    rather than via an attribute on a fake."""
    predictor = predictor_factory(db_logger=db_logger)
    # Upload order is deliberately C03, C01, C02 -- not ascending.
    channels = _channels([30, 10, 20])
    metadata = _image_metadata(
        channels=[3, 1, 2],
        channel_files=[
            "PLATE1_A01_T0001F001L01A01Z01C03.tif",
            "PLATE1_A01_T0001F001L01A01Z01C01.tif",
            "PLATE1_A01_T0001F001L01A01Z01C02.tif",
        ],
    )

    predictor.predict(channels, metadata)

    rows = _fetch_all(db_logger, "SELECT channel, file_name FROM image_metadata ORDER BY id")
    assert rows == [
        (1, "PLATE1_A01_T0001F001L01A01Z01C01.tif"),
        (2, "PLATE1_A01_T0001F001L01A01Z01C02.tif"),
        (3, "PLATE1_A01_T0001F001L01A01Z01C03.tif"),
    ]


def test_predict_without_db_logger_does_not_touch_the_database(predictor_factory, db_logger):
    """Sanity check: a predictor with no db_logger attached must not write
    anything, confirming the rows asserted above come from predict()'s own
    logging calls and not test pollution."""
    predictor = predictor_factory(db_logger=None)
    channels = _channels([10, 20, 30])
    metadata = _image_metadata(
        channels=[1, 2, 3],
        channel_files=[
            "PLATE1_A01_T0001F001L01A01Z01C01.tif",
            "PLATE1_A01_T0001F001L01A01Z01C02.tif",
            "PLATE1_A01_T0001F001L01A01Z01C03.tif",
        ],
    )

    result = predictor.predict(channels, metadata)

    assert result["predicted_class"] == "ClassA"
    assert _fetch_all(db_logger, "SELECT * FROM image_metadata") == []
