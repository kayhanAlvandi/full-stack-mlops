"""Database fixtures for monitoring DB integration tests.

Applies all four schema files (01-04) in order, since monitoring views and
columns depend on the full schema set: live_* and benchmark_* views need
benchmark_id (03_benchmark.sql), and quality_report needs 04_quality.sql.

Requires a running PostgreSQL instance with DB_TEST_URI set (same convention
as tests/db/test_dblogger.py).
"""
from __future__ import annotations

import os
from pathlib import Path

import psycopg
import pytest

from database.dblogger import DBLogger

DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "admin123456")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "test")

DB_TEST_URI = os.getenv(
    "DB_TEST_URI",
    f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)

SCHEMA_DIR = Path(__file__).resolve().parents[3] / "database" / "init"
SCHEMA_FILES = [
    "01_prediction.sql",
    "02_reference.sql",
    "03_benchmark.sql",
    "04_quality.sql",
]


def _apply_schemas(conn):
    """Apply all schema files in order onto a fresh public schema."""
    conn.execute("DROP SCHEMA public CASCADE")
    conn.execute("CREATE SCHEMA public")
    for fname in SCHEMA_FILES:
        path = SCHEMA_DIR / fname
        with open(path, encoding="utf-8") as f:
            conn.execute(f.read())


@pytest.fixture(scope="session", autouse=True)
def _apply_db_schemas():
    """Apply all monitoring schemas (01-04) once at the start of the session."""
    conn = psycopg.connect(DB_TEST_URI, autocommit=True)
    _apply_schemas(conn)
    conn.close()
    yield
    conn = psycopg.connect(DB_TEST_URI, autocommit=True)
    conn.execute("DROP SCHEMA public CASCADE")
    conn.execute("CREATE SCHEMA public")
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def db_logger():
    """A connected DBLogger with the full schema applied; truncated between tests."""
    logger = DBLogger(db_uri=DB_TEST_URI)
    logger.connect()
    yield logger
    # Truncate all tables that tests insert into. image_metadata CASCADE
    # reaches most FK dependents, but benchmark_dataset has no FK to
    # image_metadata (only benchmark_dataset_member does), so it must be
    # truncated explicitly to prevent rows accumulating across tests.
    with logger.pool.connection() as conn:
        conn.execute(
            "TRUNCATE image_metadata, benchmark_dataset, drift_report, quality_report CASCADE"
        )
    logger.pool.close()


# ─────────────────────────────────────────────────────────────────────────────
# Shared constants and helpers
# ─────────────────────────────────────────────────────────────────────────────

PLATE = "MIG-Exp03-CP-40X-bin1X1"
WELL = "K07"
ROOT_PATH = "/data/images"
SHAPE = (2048, 2048)
RUN_ID = "abc123def456abc123def456abc123de"


def make_channel_filenames(well: str = WELL, field: int = 1, plate: str = PLATE) -> list[str]:
    """Build 5 single-channel filenames following the naming convention."""
    return [
        f"{plate}_{well}_T0001F{field:03d}L01A01Z01C0{ch}.jxl"
        for ch in range(1, 6)
    ]


def make_image_metadata_tuples(well: str = WELL, field: int = 1,
                                 root_path: str = ROOT_PATH,
                                 shape: tuple = SHAPE) -> list[tuple]:
    """Build DB-ready image_metadata tuples for a 5-channel sample."""
    filenames = make_channel_filenames(well=well, field=field)
    return [
        (PLATE, well, field, ch, root_path, fname, shape[0], shape[1])
        for ch, fname in enumerate(filenames, start=1)
    ]


def insert_images(db_logger, well: str = WELL, field: int = 1,
                  root_path: str = ROOT_PATH, shape: tuple = SHAPE) -> list[int]:
    """Insert image_metadata rows and return their ids."""
    rows = make_image_metadata_tuples(well=well, field=field, root_path=root_path, shape=shape)
    return db_logger.log_image_metadata(rows)


def make_image_prediction_tuple(plate: str = PLATE, well: str = WELL, field: int = 1,
                                run_id: str = RUN_ID, p_label: str = "positive",
                                t_label: str | None = None, total_tiles: int = 4,
                                vote_fraction: float = 0.75, avg_confidence: float = 0.9,
                                is_reference: bool = False,
                                benchmark_id: int | None = None) -> tuple:
    """Build an 11-field image_prediction tuple (matches DBLogger.log_image_prediction)."""
    return (plate, well, field, run_id, p_label, t_label,
            total_tiles, vote_fraction, avg_confidence, is_reference, benchmark_id)


def insert_benchmark_sample(db_logger, well: str = WELL, field: int = 1,
                            t_label: str = "ClassA") -> tuple[int, list[int]]:
    """Register a benchmark sample (image_metadata + benchmark_dataset + members).

    Returns (benchmark_id, image_ids).
    """
    img_ids = insert_images(db_logger, well=well, field=field)
    benchmark_id = db_logger.log_benchmark_sample((PLATE, well, field, t_label))
    members = [
        (benchmark_id, img_id, channel_index)
        for channel_index, img_id in enumerate(img_ids)
    ]
    db_logger.log_benchmark_members(members)
    return benchmark_id, img_ids
