from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


class DBLogger:
    """Logs predictions to a database.

    Uses a connection pool rather than a single shared connection: each
    method checks out its own connection for the duration of its
    transaction. This makes DBLogger safe to call from multiple threads
    concurrently (e.g. the API's inference threadpool) -- a single shared
    connection cannot be used that way, since one connection has one
    transaction/protocol state that can't be interleaved between callers.
    """
    def __init__(self, db_uri: str):
        self.db_uri = db_uri

    def connect(self):
        self.pool = ConnectionPool(self.db_uri, open=True)
        # open=True only schedules connection attempts on background workers;
        # wait() blocks until one succeeds and re-raises the real error if
        # every attempt fails -- without it, a bad URI surfaces 30s later as
        # PoolTimeout on the first query, so try/except around connect()
        # would never catch a failed connection.
        self.pool.wait()

    def close(self):
        self.pool.close()

    def log_image_metadata(self, image_metadata: list[tuple]):
        """Insert image metadata rows into the database.

        Args:
            image_metadata: list of tuples, each:
                (plate, well, field, channel, root_path, file_name, shape_x, shape_y)
        """
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.executemany("""
            INSERT INTO image_metadata (plate, well , field, channel , root_path
            ,file_name, shape_x, shape_y)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (file_name) DO UPDATE SET root_path = EXCLUDED.root_path
            RETURNING id
        """, image_metadata, returning=True)
            img_ids = []
            for _ in cursor.results():
                img_ids.append(cursor.fetchone()[0])
            return img_ids

    def log_tile_stack(self, tile_stack_metadata: list[tuple]):
        """Insert tile stack metadata rows into the database.

        Args:
            tile_stack_metadata: list of tuples, each:
                (stack_hash, row_ind, col_ind, x_left, y_top, crop_size)
        """
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.executemany("""
            INSERT INTO tile_stack (stack_hash, row_ind, col_ind, x_left, y_top, crop_size)
            VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (stack_hash) DO UPDATE SET row_ind = EXCLUDED.row_ind, col_ind = EXCLUDED.col_ind, x_left = EXCLUDED.x_left, y_top = EXCLUDED.y_top, crop_size = EXCLUDED.crop_size
            RETURNING id
        """, tile_stack_metadata, returning=True)
            tile_stack_ids = []
            for _ in cursor.results():
                tile_stack_ids.append(cursor.fetchone()[0])
            return tile_stack_ids

    def log_tile_stack_member(self, tile_stack_members: list[tuple]):
        """Insert tile stack member rows into the database.

        Args:
            tile_stack_members: list of tuples, each:
                (tile_stack_id, img_id, channel_index)
        """
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.executemany("""
            INSERT INTO tile_stack_member (tile_stack_id, image_id, channel_index)
            VALUES (%s, %s, %s) ON CONFLICT (tile_stack_id, image_id)
                DO UPDATE SET channel_index = EXCLUDED.channel_index
            RETURNING id
        """, tile_stack_members, returning=True)
            tile_stack_member_ids = []
            for _ in cursor.results():
                tile_stack_member_ids.append(cursor.fetchone()[0])
            return tile_stack_member_ids

    def log_image_prediction(self, image_prediction: tuple):
        """Insert image prediction row into the database.

        Args:
            image_prediction: tuple, each:
                (plate, well, field, run_id, p_label, t_label, total_tiles, vote_fraction, avg_confidence, is_reference, benchmark_id)
        """
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
            INSERT INTO image_prediction (plate, well, field, run_id, p_label, t_label, total_tiles, vote_fraction, avg_confidence, is_reference, benchmark_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, image_prediction)
            return cursor.fetchone()[0]

    def log_tile_prediction(self, tile_predictions: list[tuple]):
        """Insert tile prediction rows into the database.

        Args:
            tile_predictions: list of tuples, each:
                (image_pred_id, tile_stack_id, run_id, p_label, t_label, confidence, is_reference, benchmark_id)
        """
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.executemany("""
            INSERT INTO tile_prediction (image_pred_id, tile_stack_id, run_id, p_label, t_label, confidence, is_reference, benchmark_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, tile_predictions)
            return len(tile_predictions)  # return number of rows inserted

    def log_tile_channel_stats(self, tile_channel_stats: list[tuple]):
        """Insert tile channel stats rows into the database.

        Args:
            tile_channel_stats: list of tuples, each:
                (tile_stack_member_id, mean, std, p1, p5, p95, p99)
        """
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.executemany("""
            INSERT INTO tile_channel_stats (tile_stack_member_id, mean, std, p1, p5, p95, p99)
            VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (tile_stack_member_id) DO UPDATE SET
                mean = EXCLUDED.mean,
                std = EXCLUDED.std,
                p1 = EXCLUDED.p1,
                p5 = EXCLUDED.p5,
                p95 = EXCLUDED.p95,
                p99 = EXCLUDED.p99
                RETURNING id
            """, tile_channel_stats, returning=True)
            tile_channel_stats_ids = []
            for _ in cursor.results():
                tile_channel_stats_ids.append(cursor.fetchone()[0])
            return tile_channel_stats_ids

    def get_reference_samples(self, run_id: str) -> list[tuple]:
        """Return the (plate, well, field) of every validation sample already logged as
        reference for this run_id, so reference computation can resume and skip only the
        samples it already has instead of redoing (or fully skipping) the whole run.

        Args:
            run_id: MLflow run id to check

        Returns:
            List of (plate, well, field) tuples for reference rows that already exist for
            this run
        """
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
            SELECT plate, well, field
            FROM image_prediction
            WHERE run_id = %s AND is_reference = TRUE
            """, (run_id,))
            return cursor.fetchall()

    # ── Drift reporting: reads for reference vs. current comparison ─────────

    def _fetch_dicts(self, query: str, params: tuple) -> list[dict]:
        """Run a read query and return rows as a list of dicts (column -> value)."""
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()

    def fetch_reference_image_level(self, run_id: str) -> list[dict]:
        """Image-level reference rows for a run: (p_label, vote_fraction, avg_confidence)."""
        return self._fetch_dicts("""
            SELECT p_label, vote_fraction, avg_confidence
            FROM reference_image_prediction
            WHERE run_id = %s
        """, (run_id,))

    def fetch_current_image_level(self, run_id: str, window_start, window_end) -> list[dict]:
        """Image-level production rows in [window_start, window_end) for a run.

        Excludes wells that belong to this run's reference set, so the drift
        window is never compared partly against the reference itself (a
        validation well can legitimately be re-imaged and predicted in prod).
        """
        return self._fetch_dicts("""
            SELECT l.p_label, l.vote_fraction, l.avg_confidence
            FROM live_image_prediction l
            WHERE l.run_id = %s
              AND l.created_at >= %s
              AND l.created_at <  %s
              AND NOT EXISTS (
                  SELECT 1 FROM reference_image_prediction r
                  WHERE r.run_id = l.run_id
                    AND r.plate  = l.plate
                    AND r.well   = l.well
                    AND r.field  = l.field
              )
        """, (run_id, window_start, window_end))

    def fetch_reference_tile_level(self, run_id: str) -> list[dict]:
        """Tile-level reference rows + per-channel input stats for a run.

        One row per (tile_prediction, channel): tile_pred_id and confidence
        repeat across the tile's channels; caller dedups for confidence and
        pivots the channel stats long -> wide.
        """
        return self._fetch_dicts("""
            SELECT t.id AS tile_pred_id,
                   t.p_label,
                   t.confidence,
                   im.channel AS channel,
                   s.mean, s.std, s.p1, s.p5, s.p95, s.p99
            FROM reference_tile_prediction t
            JOIN tile_stack_member tsm     ON tsm.tile_stack_id = t.tile_stack_id
            JOIN image_metadata im         ON im.id = tsm.image_id
            LEFT JOIN tile_channel_stats s ON s.tile_stack_member_id = tsm.id
            WHERE t.run_id = %s
        """, (run_id,))

    def fetch_current_tile_level(self, run_id: str, window_start, window_end) -> list[dict]:
        """Tile-level production rows + per-channel input stats in the window.

        Same reference-well exclusion as fetch_current_image_level, applied via
        the parent image_prediction row.
        """
        return self._fetch_dicts("""
            SELECT t.id AS tile_pred_id,
                   t.p_label,
                   t.confidence,
                   im.channel AS channel,
                   s.mean, s.std, s.p1, s.p5, s.p95, s.p99
            FROM live_tile_prediction t
            JOIN live_image_prediction l   ON l.id = t.image_pred_id
            JOIN tile_stack_member tsm     ON tsm.tile_stack_id = t.tile_stack_id
            JOIN image_metadata im         ON im.id = tsm.image_id
            LEFT JOIN tile_channel_stats s ON s.tile_stack_member_id = tsm.id
            WHERE t.run_id = %s
              AND t.created_at >= %s
              AND t.created_at <  %s
              AND NOT EXISTS (
                  SELECT 1 FROM reference_image_prediction r
                  WHERE r.run_id = l.run_id
                    AND r.plate  = l.plate
                    AND r.well   = l.well
                    AND r.field  = l.field
              )
        """, (run_id, window_start, window_end))

    # ── Drift reporting: writes ──────────────────────────────────────────────

    def log_drift_report(self, drift_report: tuple):
        """Insert one drift-report row and return its id.

        Args:
            drift_report: tuple
                (run_id, window_start, window_end, dataset_drift,
                 n_columns_drifted, n_columns_total, report_path)
        """
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
            INSERT INTO drift_report
                (run_id, window_start, window_end, dataset_drift,
                 n_columns_drifted, n_columns_total, report_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, drift_report)
            return cursor.fetchone()[0]

    def log_drift_report_column(self, drift_report_columns: list[tuple]):
        """Insert per-column drift results for a drift report.

        Args:
            drift_report_columns: list of tuples, each:
                (drift_report_id, column_name, column_group, drift_score,
                 drifted, stat_test)
        """
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.executemany("""
            INSERT INTO drift_report_column
                (drift_report_id, column_name, column_group,
                 drift_score, drifted, stat_test)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, drift_report_columns)
            return len(drift_report_columns)

    # ── Label backfill: production ground-truth from MongoDB ────────────────

    def fetch_unlabeled_wells(self) -> list[tuple]:
        """(plate, well) pairs of production predictions still missing t_label.

        Excludes reference and benchmark rows -- both already carry known labels.
        """
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
            SELECT DISTINCT plate, well
            FROM image_prediction
            WHERE is_reference = FALSE
              AND benchmark_id IS NULL
              AND t_label IS NULL
            """)
            return cursor.fetchall()

    def update_t_label(self, plate: str, well: str, t_label: str) -> tuple[int, int]:
        """Backfill t_label for all production rows of a (plate, well).

        Updates image_prediction and its child tile_prediction rows in one
        transaction so they can't drift out of sync. Only fills rows where
        t_label IS NULL (never overwrites an existing label); reference and
        benchmark rows are left untouched. This is global by well -- it updates
        every matching production row regardless of run_id, since a MongoDB
        label is a property of the physical sample, not of which model ran.

        Returns (n_image_rows_updated, n_tile_rows_updated).
        """
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
            UPDATE image_prediction
            SET t_label = %s
            WHERE plate = %s AND well = %s
              AND is_reference = FALSE
              AND benchmark_id IS NULL
              AND t_label IS NULL
            RETURNING id
            """, (t_label, plate, well))
            image_ids = [row[0] for row in cursor.fetchall()]
            n_tiles = 0
            if image_ids:
                cursor.execute("""
                UPDATE tile_prediction
                SET t_label = %s
                WHERE image_pred_id = ANY(%s::int[])
                  AND t_label IS NULL
                """, (t_label, image_ids))
                n_tiles = cursor.rowcount
            return (len(image_ids), n_tiles)

    # ── Benchmark dataset: registration + per-run scoring ───────────────────

    def log_benchmark_sample(self, sample: tuple) -> int:
        """Insert (or update) one benchmark_dataset row, return its id.

        Args:
            sample: (plate, well, field, t_label)

        Idempotent on (plate, well, field): re-registering the same sample
        updates its label rather than creating a duplicate.
        """
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
            INSERT INTO benchmark_dataset (plate, well, field, t_label)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (plate, well, field) DO UPDATE SET t_label = EXCLUDED.t_label
            RETURNING id
            """, sample)
            return cursor.fetchone()[0]

    def log_benchmark_members(self, members: list[tuple]):
        """Insert benchmark_dataset_member rows, return their ids.

        Args:
            members: list of tuples, each (benchmark_id, image_id, channel_index)
        """
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.executemany("""
            INSERT INTO benchmark_dataset_member (benchmark_id, image_id, channel_index)
            VALUES (%s, %s, %s) ON CONFLICT (benchmark_id, image_id)
                DO UPDATE SET channel_index = EXCLUDED.channel_index
            RETURNING id
        """, members, returning=True)
            member_ids = []
            for _ in cursor.results():
                member_ids.append(cursor.fetchone()[0])
            return member_ids

    def get_benchmark_samples(self) -> list[tuple]:
        """(plate, well, field) of every registered benchmark sample.

        Lets register_benchmark skip samples it already registered instead of
        re-reading their image files.
        """
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute("SELECT plate, well, field FROM benchmark_dataset")
            return cursor.fetchall()

    def fetch_benchmark_members(self) -> list[dict]:
        """Every benchmark sample joined with its channel image files.

        One row per (benchmark sample, channel image); the caller groups by
        benchmark_id. Ordered so channels come out in channel_index order,
        matching the model's input channel-axis order.
        """
        return self._fetch_dicts("""
            SELECT b.id AS benchmark_id, b.plate, b.well, b.field, b.t_label,
                   m.channel_index, im.channel, im.root_path, im.file_name
            FROM benchmark_dataset b
            JOIN benchmark_dataset_member m ON m.benchmark_id = b.id
            JOIN image_metadata im          ON im.id = m.image_id
            ORDER BY b.id, m.channel_index
        """, ())

    def get_benchmark_predictions(self, run_id: str) -> list[int]:
        """benchmark_dataset ids already scored (have an image_prediction) for run_id.

        Lets compute_predictions_references resume without re-scoring samples it already has.
        """
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
            SELECT DISTINCT benchmark_id
            FROM image_prediction
            WHERE run_id = %s AND benchmark_id IS NOT NULL
            """, (run_id,))
            return [row[0] for row in cursor.fetchall()]

    # ── Supervised quality reporting ────────────────────────────────────────

    def fetch_benchmark_quality(self, run_id: str) -> list[dict]:
        """(p_label, t_label) for a model's benchmark predictions -- the quality baseline."""
        return self._fetch_dicts("""
            SELECT p_label, t_label
            FROM image_prediction
            WHERE run_id = %s
              AND benchmark_id IS NOT NULL
              AND t_label IS NOT NULL
        """, (run_id,))

    def fetch_current_quality(self, run_id: str, window_start, window_end) -> list[dict]:
        """(p_label, t_label) for labeled production rows in [window_start, window_end).

        Production only (is_reference = FALSE via live_image_prediction, and
        benchmark_id IS NULL), labeled (t_label IS NOT NULL), with the same
        reference-well exclusion the drift job uses so a re-imaged validation
        well isn't scored against itself.
        """
        return self._fetch_dicts("""
            SELECT l.p_label, l.t_label
            FROM live_image_prediction l
            WHERE l.run_id = %s
              AND l.benchmark_id IS NULL
              AND l.t_label IS NOT NULL
              AND l.created_at >= %s
              AND l.created_at <  %s
              AND NOT EXISTS (
                  SELECT 1 FROM reference_image_prediction r
                  WHERE r.run_id = l.run_id
                    AND r.plate  = l.plate
                    AND r.well   = l.well
                    AND r.field  = l.field
              )
        """, (run_id, window_start, window_end))

    def log_quality_report(self, quality_report: tuple):
        """Insert one quality-report row and return its id.

        Args:
            quality_report: tuple
                (run_id, window_start, window_end, n_benchmark_samples,
                 n_current_samples, benchmark_accuracy, benchmark_f1,
                 current_accuracy, current_f1, report_path)
        """
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
            INSERT INTO quality_report
                (run_id, window_start, window_end, n_benchmark_samples,
                 n_current_samples, benchmark_accuracy, benchmark_f1,
                 current_accuracy, current_f1, report_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, quality_report)
            return cursor.fetchone()[0]
