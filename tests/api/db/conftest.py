"""Database fixtures for api/predictor DB integration tests.

Applies schemas 01-03: log_image_prediction/log_tile_prediction insert
is_reference and benchmark_id columns added by 03_benchmark.sql, and
log_tile_channel_stats needs the tile_channel_stats table from
02_reference.sql (same schema set as tests/db/test_dblogger.py).

Requires a running PostgreSQL instance with DB_TEST_URI set (same convention
as tests/db/test_dblogger.py and tests/monitoring/db/conftest.py).
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
SCHEMA_FILES = ["01_prediction.sql", "02_reference.sql", "03_benchmark.sql"]


def _apply_schemas(conn):
    conn.execute("DROP SCHEMA public CASCADE")
    conn.execute("CREATE SCHEMA public")
    for fname in SCHEMA_FILES:
        path = SCHEMA_DIR / fname
        with open(path, encoding="utf-8") as f:
            conn.execute(f.read())


@pytest.fixture(scope="session", autouse=True)
def _apply_db_schemas():
    """Apply the prediction schema set once at the start of the session."""
    conn = psycopg.connect(DB_TEST_URI, autocommit=True)
    _apply_schemas(conn)
    conn.close()
    yield
    conn = psycopg.connect(DB_TEST_URI, autocommit=True)
    conn.execute("DROP SCHEMA public CASCADE")
    conn.execute("CREATE SCHEMA public")
    conn.close()


@pytest.fixture
def db_logger():
    """A connected DBLogger with the schema applied; truncated between tests."""
    logger = DBLogger(db_uri=DB_TEST_URI)
    logger.connect()
    yield logger
    with logger.pool.connection() as conn:
        conn.execute("TRUNCATE image_metadata CASCADE")
    logger.pool.close()
