"""Tests for DBLogger.log_image_metadata.

Requires a running PostgreSQL instance with the prediction schema applied.
Schemas 01-03 are applied (prediction + reference + benchmark) because
log_image_prediction/log_tile_prediction now insert is_reference and
benchmark_id columns added by those later migrations.

Set the DB_TEST_URI environment variable to point to your test database, e.g.:
    DB_TEST_URI=postgresql://postgres:postgres@localhost:5432/image_classifier_test
"""
import os

import psycopg
import pytest

from database.dblogger import DBLogger
from utils.filename_parser import (
    clean_image_metadata,
    clean_tiles_metadata,
    extract_info_from_filename,
)

DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "admin123")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "prediction_test")

DB_TEST_URI = os.getenv(
    "DB_TEST_URI",
    f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)
# URI to the default 'postgres' DB (used to create/drop the test DB)
DB_ADMIN_URI = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/postgres"

SCHEMA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "database", "init"
)
SCHEMA_FILES = ["01_prediction.sql", "02_reference.sql", "03_benchmark.sql"]


def setup_module():
    """Apply schemas in a fresh public schema."""
    conn = psycopg.connect(DB_TEST_URI, autocommit=True)
    conn.execute("DROP SCHEMA public CASCADE")
    conn.execute("CREATE SCHEMA public")
    for fname in SCHEMA_FILES:
        path = os.path.join(SCHEMA_DIR, fname)
        with open(path, encoding="utf-8") as f:
            conn.execute(f.read())
    conn.close()


def teardown_module():
    """Drop the test database after all tests finish."""
    conn = psycopg.connect(DB_TEST_URI, autocommit=True)
    conn.execute("DROP SCHEMA public CASCADE")
    conn.execute("CREATE SCHEMA public")
    conn.close()


@pytest.fixture
def db_logger():
    logger = DBLogger(db_uri=DB_TEST_URI)
    logger.connect()
    yield logger
    # Truncate between tests for isolation, then close
    with logger.pool.connection() as conn:
        conn.execute("TRUNCATE image_metadata CASCADE")
    logger.pool.close()


# Filenames follow the pattern: {plate}_{well}_T{time}F{field}L{layer}A{action}Z{z}C{channel}.jxl
PLATE = "MIG-Exp03-CP-40X-bin1X1"
WELL = "K07"
ROOT_PATH = "/data/images"
SHAPE = (2048, 2048)

# Five single-channel files matching the naming convention
CHANNEL_FILENAMES = [
    f"{PLATE}_{WELL}_T0001F001L01A01Z01C0{ch}.jxl" for ch in range(1, 6)
]


def _make_upload_metadata(filenames: list[str], root_path: str = ROOT_PATH, shape: tuple = SHAPE) -> list[tuple]:
    """Build DB-ready tuples via the utils parser, same as the API does.

    Parses each filename to extract plate/well/field/channel, then assembles
    the single dict shape that clean_image_metadata expects (one dict with
    channels/channel_files lists, not a list of per-file dicts).
    """
    infos = [extract_info_from_filename(fname) for fname in filenames]
    # Sort by channel ascending (matches training/inference convention)
    sorted_pairs = sorted(zip(filenames, infos), key=lambda p: p[1]["channel"])
    sorted_filenames = [p[0] for p in sorted_pairs]
    sorted_infos = [p[1] for p in sorted_pairs]
    raw = {
        "plate": sorted_infos[0]["plate"],
        "well": sorted_infos[0]["well"],
        "field": sorted_infos[0]["field"],
        "channels": [info["channel"] for info in sorted_infos],
        "channel_files": sorted_filenames,
        "root_path": root_path,
        "shape": shape,
    }
    return clean_image_metadata(raw)


class TestLogImageMetadata:
    """Tests for DBLogger.log_image_metadata."""

    def test_single_insert_returns_list_with_one_id(self, db_logger):
        """Single file upload returns a list with one positive integer id."""
        metadata = _make_upload_metadata([CHANNEL_FILENAMES[0]])
        ids = db_logger.log_image_metadata(metadata)
        assert ids is not None
        assert isinstance(ids, list)
        assert len(ids) == 1
        assert isinstance(ids[0], int)
        assert ids[0] > 0

    def test_multi_channel_returns_ids_for_each_channel(self, db_logger):
        """Five channel files return five distinct ids."""
        metadata = _make_upload_metadata(CHANNEL_FILENAMES)
        ids = db_logger.log_image_metadata(metadata)
        assert len(ids) == 5
        assert len(set(ids)) == 5, "Each channel file should get a distinct id"

    def test_duplicate_insert_returns_same_id(self, db_logger):
        """Inserting same filenames twice returns the same ids (ON CONFLICT)."""
        metadata = _make_upload_metadata(CHANNEL_FILENAMES)
        ids_first = db_logger.log_image_metadata(metadata)
        ids_second = db_logger.log_image_metadata(metadata)
        assert ids_first == ids_second

    def test_duplicate_updates_root_path(self, db_logger):
        """ON CONFLICT should update root_path to the new value."""
        metadata = _make_upload_metadata([CHANNEL_FILENAMES[0]])
        db_logger.log_image_metadata(metadata)

        new_root = "/new/path"
        metadata_updated = _make_upload_metadata([CHANNEL_FILENAMES[0]], root_path=new_root)
        db_logger.log_image_metadata(metadata_updated)

        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT root_path FROM image_metadata WHERE file_name = %s",
                (CHANNEL_FILENAMES[0],),
            )
            row = cur.fetchone()
        assert row[0] == new_root

    def test_channel_parsed_correctly_from_filename(self, db_logger):
        """Channel number is correctly extracted from filename pattern."""
        metadata = _make_upload_metadata(CHANNEL_FILENAMES)
        ids = db_logger.log_image_metadata(metadata)

        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT file_name, channel FROM image_metadata WHERE id = ANY(%s) ORDER BY channel",
                (ids,),
            )
            rows = cur.fetchall()

        assert len(rows) == 5
        for i, row in enumerate(rows):
            assert row[1] == i + 1, f"Expected channel {i+1}, got {row[1]} for file {row[0]}"

    def test_partial_overlap_reuses_ids(self, db_logger):
        """Two 5-channel batches from different wells share 3 channels in common.

        Batch A (well K07): channels 1-5
        Batch B (well K08): channels 3-7  (channels 3,4,5 overlap with batch A)

        After inserting batch A then batch B:
        - ids for channels 3,4,5 must be the same across both batches (same file → same row)
        - ids for channels 1,2 (only in A) and 6,7 (only in B) must be distinct
        """
        well_a = "K07"
        well_b = "K08"

        # channels 3,4,5 of well_a == channels 3,4,5 of well_b only if same well;
        # wells differ so all 10 are distinct filenames — use same well to get real overlap:
        shared = [
            f"{PLATE}_{well_a}_T0001F001L01A01Z01C0{ch}.jxl" for ch in range(3, 6)
        ]
        unique_a = [
            f"{PLATE}_{well_a}_T0001F001L01A01Z01C0{ch}.jxl" for ch in range(1, 3)
        ]
        unique_b = [
            f"{PLATE}_{well_b}_T0001F001L01A01Z01C0{ch}.jxl" for ch in range(1, 3)
        ]

        batch_a = _make_upload_metadata(unique_a + shared)        # 5 files
        batch_b = _make_upload_metadata(unique_b + shared)        # 5 files (3 shared with batch_a)

        ids_a = db_logger.log_image_metadata(batch_a)
        ids_b = db_logger.log_image_metadata(batch_b)

        assert len(ids_a) == 5
        assert len(ids_b) == 5

        # shared files are at positions 2,3,4 in both batches
        shared_ids_from_a = set(ids_a[2:])
        shared_ids_from_b = set(ids_b[2:])
        assert shared_ids_from_a == shared_ids_from_b, "Shared filenames must map to same db ids"

        # unique files must have ids not present in the other batch
        unique_ids_a = set(ids_a[:2])
        unique_ids_b = set(ids_b[:2])
        assert unique_ids_a.isdisjoint(unique_ids_b), "Unique filenames must get different ids"
        assert unique_ids_a.isdisjoint(shared_ids_from_b), "Unique A ids must not collide with shared ids"
        assert unique_ids_b.isdisjoint(shared_ids_from_a), "Unique B ids must not collide with shared ids"

    def test_plate_well_field_parsed_correctly(self, db_logger):
        """Plate, well, and field are correctly extracted from filename."""
        metadata = _make_upload_metadata([CHANNEL_FILENAMES[0]])
        ids = db_logger.log_image_metadata(metadata)

        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT plate, well, field, channel, root_path, shape_x, shape_y FROM image_metadata WHERE id = %s",
                (ids[0],),
            )
            row = cur.fetchone()

        assert row[0] == PLATE
        assert row[1] == WELL
        assert row[2] == 1
        assert row[3] == 1
        assert row[4] == ROOT_PATH
        assert row[5] == SHAPE[0]
        assert row[6] == SHAPE[1]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared across tile/prediction tests
# ─────────────────────────────────────────────────────────────────────────────

RUN_ID = "abc123def456abc123def456abc123de"
CROP_SIZE = 512


def _insert_images(db_logger, filenames=None, root_path=ROOT_PATH, shape=SHAPE):
    """Insert image_metadata rows and return their ids."""
    filenames = filenames or CHANNEL_FILENAMES
    metadata = _make_upload_metadata(filenames, root_path=root_path, shape=shape)
    return db_logger.log_image_metadata(metadata)


def _make_tiles(n_tiles=4, crop_size=CROP_SIZE):
    """Build synthetic tile dicts as tile_image() would produce."""
    tiles = []
    for row in range(2):
        for col in range(n_tiles // 2):
            tiles.append({
                "row": row,
                "col": col,
                "x": col * crop_size,
                "y": row * crop_size,
                "crop_size": crop_size,
            })
    return tiles


def _make_image_prediction_tuple(plate=PLATE, well=WELL, field=1, run_id=RUN_ID,
                                  p_label="positive", t_label=None,
                                  total_tiles=4, vote_fraction=0.75, avg_confidence=0.9,
                                  is_reference=False, benchmark_id=None):
    return (plate, well, field, run_id, p_label, t_label,
            total_tiles, vote_fraction, avg_confidence, is_reference, benchmark_id)


# ─────────────────────────────────────────────────────────────────────────────
class TestLogTileStack:
    """Tests for DBLogger.log_tile_stack."""

    def test_insert_returns_ids_for_each_tile(self, db_logger):
        """Each tile gets its own stack_id."""
        img_ids = _insert_images(db_logger)
        tiles = _make_tiles(4)
        tile_stack_metadata = clean_tiles_metadata(tiles, img_ids)
        tile_stack_ids = db_logger.log_tile_stack(tile_stack_metadata)

        assert tile_stack_ids is not None
        assert len(tile_stack_ids) == 4
        assert len(set(tile_stack_ids)) == 4, "Each tile position must get a unique stack id"

    def test_duplicate_hash_returns_same_id(self, db_logger):
        """Inserting same tiles twice returns the same tile_stack ids (ON CONFLICT)."""
        img_ids = _insert_images(db_logger)
        tiles = _make_tiles(4)
        tile_stack_metadata = clean_tiles_metadata(tiles, img_ids)

        ids_first = db_logger.log_tile_stack(tile_stack_metadata)
        ids_second = db_logger.log_tile_stack(tile_stack_metadata)
        assert ids_first == ids_second

    def test_stored_values_match_input(self, db_logger):
        """row_ind, col_ind, x_left, y_top, crop_size are stored correctly."""
        img_ids = _insert_images(db_logger)
        tiles = _make_tiles(2)
        tile_stack_metadata = clean_tiles_metadata(tiles, img_ids)
        tile_stack_ids = db_logger.log_tile_stack(tile_stack_metadata)

        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT row_ind, col_ind, x_left, y_top, crop_size FROM tile_stack WHERE id = %s",
                (tile_stack_ids[0],),
            )
            row = cur.fetchone()

        assert row[0] == tiles[0]["row"]
        assert row[1] == tiles[0]["col"]
        assert row[2] == tiles[0]["x"]
        assert row[3] == tiles[0]["y"]
        assert row[4] == CROP_SIZE


# ─────────────────────────────────────────────────────────────────────────────
class TestLogTileStackMember:
    """Tests for DBLogger.log_tile_stack_member."""

    def test_members_inserted_for_all_channels(self, db_logger):
        """Each (tile_stack_id, image_id, channel_index) triple is inserted as a member."""
        img_ids = _insert_images(db_logger)
        tiles = _make_tiles(2)
        tile_stack_metadata = clean_tiles_metadata(tiles, img_ids)
        tile_stack_ids = db_logger.log_tile_stack(tile_stack_metadata)

        members = [
            (tile_stack_id, img_id, channel_index)
            for tile_stack_id in tile_stack_ids
            for channel_index, img_id in enumerate(img_ids)
        ]
        member_ids = db_logger.log_tile_stack_member(members)

        assert member_ids is not None
        assert len(member_ids) == len(members)

        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM tile_stack_member")
            count = cur.fetchone()[0]

        assert count == len(tile_stack_ids) * len(img_ids)

    def test_channel_index_stored_correctly(self, db_logger):
        """channel_index is stored per member, matching the model's input channel-axis position."""
        img_ids = _insert_images(db_logger)
        tiles = _make_tiles(2)
        tile_stack_metadata = clean_tiles_metadata(tiles, img_ids)
        tile_stack_ids = db_logger.log_tile_stack(tile_stack_metadata)

        members = [
            (tile_stack_ids[0], img_id, channel_index)
            for channel_index, img_id in enumerate(img_ids)
        ]
        db_logger.log_tile_stack_member(members)

        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT image_id, channel_index FROM tile_stack_member "
                "WHERE tile_stack_id = %s ORDER BY channel_index",
                (tile_stack_ids[0],),
            )
            rows = cur.fetchall()

        assert [row[0] for row in rows] == img_ids
        assert [row[1] for row in rows] == list(range(len(img_ids)))

    def test_duplicate_members_ignored(self, db_logger):
        """Inserting the same members twice does not raise and count stays the same."""
        img_ids = _insert_images(db_logger)
        tiles = _make_tiles(2)
        tile_stack_metadata = clean_tiles_metadata(tiles, img_ids)
        tile_stack_ids = db_logger.log_tile_stack(tile_stack_metadata)

        members = [(tile_stack_ids[0], img_ids[0], 0)]
        db_logger.log_tile_stack_member(members)
        db_logger.log_tile_stack_member(members)  # duplicate — must not raise

        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM tile_stack_member WHERE tile_stack_id = %s AND image_id = %s",
                (tile_stack_ids[0], img_ids[0]),
            )
            count = cur.fetchone()[0]
        assert count == 1


# ─────────────────────────────────────────────────────────────────────────────
class TestLogImagePrediction:
    """Tests for DBLogger.log_image_prediction."""

    def test_insert_returns_positive_id(self, db_logger):
        pred = _make_image_prediction_tuple()
        pred_id = db_logger.log_image_prediction(pred)
        assert isinstance(pred_id, int)
        assert pred_id > 0

    def test_each_insert_creates_new_row(self, db_logger):
        """Duplicate predictions must each get a new row (no ON CONFLICT)."""
        pred = _make_image_prediction_tuple()
        id1 = db_logger.log_image_prediction(pred)
        id2 = db_logger.log_image_prediction(pred)
        assert id1 != id2

    def test_stored_values_match_input(self, db_logger):
        """All columns are stored correctly, created_at is auto-set."""
        pred = _make_image_prediction_tuple()
        pred_id = db_logger.log_image_prediction(pred)

        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT plate, well, field, run_id, p_label, t_label, "
                "total_tiles, vote_fraction, avg_confidence, created_at "
                "FROM image_prediction WHERE id = %s",
                (pred_id,),
            )
            row = cur.fetchone()

        assert row[0] == PLATE
        assert row[1] == WELL
        assert row[2] == 1
        assert row[3] == RUN_ID
        assert row[4] == "positive"
        assert row[5] is None
        assert row[6] == 4
        assert abs(row[7] - 0.75) < 1e-6
        assert abs(row[8] - 0.9) < 1e-6
        assert row[9] is not None, "created_at should be auto-set by the DB"

    def test_t_label_can_be_set(self, db_logger):
        """t_label (true label) can be stored when provided."""
        pred = _make_image_prediction_tuple(t_label="negative")
        pred_id = db_logger.log_image_prediction(pred)

        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT t_label FROM image_prediction WHERE id = %s", (pred_id,))
            row = cur.fetchone()
        assert row[0] == "negative"


# ─────────────────────────────────────────────────────────────────────────────
class TestLogTilePrediction:
    """Tests for DBLogger.log_tile_prediction."""

    def _setup_prerequisites(self, db_logger):
        """Insert image_metadata, tile_stack, tile_stack_members, image_prediction."""
        img_ids = _insert_images(db_logger)
        tiles = _make_tiles(4)
        tile_stack_metadata = clean_tiles_metadata(tiles, img_ids)
        tile_stack_ids = db_logger.log_tile_stack(tile_stack_metadata)
        members = [
            (tile_stack_id, img_id, channel_index)
            for tile_stack_id in tile_stack_ids
            for channel_index, img_id in enumerate(img_ids)
        ]
        db_logger.log_tile_stack_member(members)
        img_pred_id = db_logger.log_image_prediction(_make_image_prediction_tuple())
        return tile_stack_ids, img_pred_id

    def test_insert_one_per_tile(self, db_logger):
        """One tile_prediction row is created per tile."""
        tile_stack_ids, img_pred_id = self._setup_prerequisites(db_logger)

        tile_preds = [
            (img_pred_id, ts_id, RUN_ID, "positive", None, 0.85, False, None)
            for ts_id in tile_stack_ids
        ]
        db_logger.log_tile_prediction(tile_preds)

        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM tile_prediction WHERE image_pred_id = %s",
                (img_pred_id,),
            )
            count = cur.fetchone()[0]
        assert count == len(tile_stack_ids)

    def test_stored_values_correct(self, db_logger):
        """p_label, confidence, run_id are stored correctly."""
        tile_stack_ids, img_pred_id = self._setup_prerequisites(db_logger)

        tile_preds = [(img_pred_id, tile_stack_ids[0], RUN_ID, "negative", None, 0.6, False, None)]
        db_logger.log_tile_prediction(tile_preds)

        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT image_pred_id, tile_stack_id, run_id, p_label, t_label, confidence "
                "FROM tile_prediction WHERE image_pred_id = %s",
                (img_pred_id,),
            )
            row = cur.fetchone()

        assert row[0] == img_pred_id
        assert row[1] == tile_stack_ids[0]
        assert row[2] == RUN_ID
        assert row[3] == "negative"
        assert row[4] is None
        assert abs(row[5] - 0.6) < 1e-6

    def test_created_at_auto_set(self, db_logger):
        """created_at is automatically set to a non-null timestamp."""
        tile_stack_ids, img_pred_id = self._setup_prerequisites(db_logger)
        tile_preds = [(img_pred_id, tile_stack_ids[0], RUN_ID, "positive", None, 0.9, False, None)]
        db_logger.log_tile_prediction(tile_preds)

        with db_logger.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT created_at FROM tile_prediction WHERE image_pred_id = %s",
                (img_pred_id,),
            )
            row = cur.fetchone()
        assert row[0] is not None, "created_at should be auto-set by the DB"
