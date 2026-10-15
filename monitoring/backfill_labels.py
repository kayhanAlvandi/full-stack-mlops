"""Backfill production ground-truth labels (t_label) from MongoDB or a CSV.

Production predictions are logged before their true label is known. This job
finds production ``image_prediction`` rows still missing ``t_label``, looks up
each ``(plate, well)``'s label (via the external ``tools.loading.getCategories``
MongoDB lookup, exactly as training does in ``src/dataset.py``, or a
``Plate,Well,label`` CSV with ``--source csv``), and writes the label back onto
every matching production row -- image-level and its child tile-level rows.

Label resolution itself lives in ``utils/labels.py`` (``MongoDBLabelsProvider``/
``CSVLabelsProvider``), the same provider classes training's
``src.datamodule.MultiChannelDataModule`` and the dataset builder instantiate
from a Hydra ``label_provider`` config, so there is one implementation per
source rather than one per caller.

Label backfill is global by well: a label is a property of the physical sample
``(plate, well)``, not of which model predicted it, so a resolved label updates
all matching production rows regardless of ``run_id``. Reference and benchmark
rows are left untouched (they already carry known labels), and an existing
non-null ``t_label`` is never overwritten.

Idempotent and safe to rerun: only NULL labels are filled, so a second run with
no newly-available labels is a no-op. The MongoDB source needs the external
``tools`` package (installed at container start from the mounted ``/app/tools``)
and network access to the shared MongoDB server (connection string hardcoded
inside ``tools.loading``); the CSV source needs only the file itself.
"""
from __future__ import annotations

import argparse

from database.dblogger import DBLogger
from monitoring.config import MonitoringSettings
from utils.labels import CSVLabelsProvider, MongoDBLabelsProvider


def main():
    parser = argparse.ArgumentParser(description="Backfill t_label on production predictions.")
    parser.add_argument("--source", choices=["mongodb", "csv"], default="csv",
                        help="Label source (default: csv).")
    parser.add_argument("--csv-path",
                        help="Path to the label CSV, required with --source csv.")
    parser.add_argument("--plate-col", default="Plate", help="CSV source only.")
    parser.add_argument("--well-col", default="Well", help="CSV source only.")
    parser.add_argument("--label-col", default="label", help="CSV source only.")
    parser.add_argument("--collection", default="tags", help="MongoDB source only.")
    args = parser.parse_args()

    if args.source == "csv":
        if not args.csv_path:
            parser.error("--source csv requires --csv-path.")
        provider = CSVLabelsProvider(
            args.csv_path, plate_col=args.plate_col, well_col=args.well_col, label_col=args.label_col
        )
    else:
        provider = MongoDBLabelsProvider(collection=args.collection)

    settings = MonitoringSettings()

    if not settings.has_db_uri:
        print("ERROR: No database URI configured. Set MONITORING_DB_URI.")
        return

    db_logger = DBLogger(db_uri=settings.db_uri)
    try:
        db_logger.connect()
    except Exception as e:  # noqa: BLE001
        print(f"Failed to connect to database: {e}")
        return

    try:
        wells = db_logger.fetch_unlabeled_wells() or []
        print(f"{len(wells)} (plate, well) pairs have unlabeled production predictions.")
        if not wells:
            print("Nothing to backfill.")
            return

        labels = provider.get_labels_for_wells(wells)
        print(f"{args.source} resolved {len(labels)} / {len(wells)} wells to a label.")
        if not labels:
            print("No labels available yet -- nothing to write.")
            return

        n_wells, n_image_rows, n_tile_rows = 0, 0, 0
        for (plate, well), treatment in labels.items():
            n_img, n_tile = db_logger.update_t_label(plate, well, treatment)
            if n_img:
                n_wells += 1
                n_image_rows += n_img
                n_tile_rows += n_tile
                print(f"  {plate}/{well} -> {treatment} ({n_img} image rows, {n_tile} tile rows)")

        print(f"Done. {len(wells)} wells checked, {len(labels)} resolved, "
              f"{n_wells} newly labeled: {n_image_rows} image rows / {n_tile_rows} tile rows updated.")
    finally:
        db_logger.close()


if __name__ == "__main__":
    main()
