"""Compute reference predictions: validation-set (per run) and/or benchmark (global).

Both are "reference" predictions -- known-label samples scored by a specific
model and written into the normal image_prediction/tile_prediction tables,
distinguished only by is_reference/benchmark_id, never live traffic:

- Validation reference: this run's own held-out validation samples, read from
  its dataset_manifest.json (MLflow artifact). Feeds drift detection
  (run_drift_report.py compares this against a production window).
- Benchmark reference: the fixed, model-independent benchmark set registered
  once by register_benchmark.py. Feeds supervised quality baselines
  (run_quality_report.py compares this against labeled production).

Both use the same loading path as the API's TilePredictor and the same
predict() pipeline used in production, so results are directly comparable.

--target {val,benchmark,both} selects which reference(s) to compute (default
both). The model is loaded and the DB connection opened once regardless of
target, so `--target both` isn't double the model-load cost of running two
separate scripts.

Resumable per target: skips validation samples that already have a reference
row for this run_id (DBLogger.get_reference_samples), and benchmark samples
already scored for this run_id (DBLogger.get_benchmark_predictions), so an
interrupted run only processes what's missing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from api.config import Settings
from database.dblogger import DBLogger

if TYPE_CHECKING:
    # TilePredictor (api.predictor) transitively imports torch/mlflow/timm;
    # only main() actually instantiates one, everywhere else it's a type
    # hint, so the heavy import is deferred there and this module stays
    # importable with just requirements/monitoring_req.txt.
    from api.predictor import TilePredictor


def _load_single_image(file_path: Path) -> np.ndarray:
    """Load a single raw (unnormalized) image file from disk."""
    import cv2
    import pillow_jxl  # noqa: F401  register JXL support with PIL
    from PIL import Image

    suffix = file_path.suffix.lower()
    if suffix == ".tif":
        image_source = cv2.imread(str(file_path), -1)
    elif suffix == ".jxl":
        image_source = Image.open(file_path)
    else:
        raise ValueError("image path should end with .tif or .jxl")
    if image_source is None:
        raise ValueError(f"Failed to load image: {file_path}")
    return np.array(image_source, dtype=np.float32)


def load_validation_data(val_samples: list[dict]) -> tuple[list[list[np.ndarray]], list[dict]]:
    """Load raw pixel data + build predict()-shaped metadata for each val sample.

    Args:
        val_samples: list of sample dicts from dataset_manifest.json, each with
            keys plate, well, field, label, root_path, channel_files
            ({channel_number(str): filename}).

    Returns:
        (sample_images, images_metadata):
            sample_images: list (one per sample) of list[np.ndarray] (H, W) raw
                pixel arrays, one per channel.
            images_metadata: list of dicts matching TilePredictor.predict()'s
                expected image_metadata shape (plate, well, field, root_path,
                shape, channels, channel_files, label, is_reference).
    """
    images_metadata = []
    sample_images = []
    for sample in val_samples:
        root_path = Path(sample["root_path"])
        sample_image = []
        channel_files = []
        channels = []
        for channel, channel_file in sample["channel_files"].items():
            img = _load_single_image(root_path / channel_file)
            sample_image.append(img)
            channel_files.append(channel_file)
            channels.append(int(channel))

        image_metadata = sample.copy()
        image_metadata["channel_files"] = channel_files
        image_metadata["channels"] = channels
        image_metadata["field"] = int(sample["field"])
        image_metadata["shape"] = sample_image[0].shape
        image_metadata["is_reference"] = True
        images_metadata.append(image_metadata)
        sample_images.append(sample_image)

    return sample_images, images_metadata


def load_benchmark_data(samples: list[dict]) -> tuple[list[list[np.ndarray]], list[dict]]:
    """Load raw pixel data + build predict()-shaped metadata for each sample.

    Args:
        samples: grouped benchmark samples, each a dict with keys benchmark_id,
            plate, well, field, t_label, and channels (list of dicts with
            channel, root_path, file_name in channel_index order).

    Returns:
        (sample_images, images_metadata) matching TilePredictor.predict()'s
        expected shapes, with label = the benchmark t_label, benchmark_id set,
        and is_reference = False.
    """
    images_metadata = []
    sample_images = []
    for sample in samples:
        sample_image = []
        channel_files = []
        channels = []
        for ch in sample["channels"]:
            img = _load_single_image(Path(ch["root_path"]) / ch["file_name"])
            sample_image.append(img)
            channel_files.append(ch["file_name"])
            channels.append(int(ch["channel"]))

        image_metadata = {
            "plate": sample["plate"],
            "well": sample["well"],
            "field": sample["field"],
            "root_path": sample["channels"][0]["root_path"],
            "channels": channels,
            "channel_files": channel_files,
            "shape": sample_image[0].shape,
            "label": sample["t_label"],
            "is_reference": False,
            "benchmark_id": sample["benchmark_id"],
        }
        images_metadata.append(image_metadata)
        sample_images.append(sample_image)

    return sample_images, images_metadata


def _group_members(member_rows: list[dict]) -> list[dict]:
    """Group flat (sample, channel) rows into one dict per benchmark sample."""
    grouped: dict[int, dict] = {}
    for row in member_rows:
        bid = row["benchmark_id"]
        if bid not in grouped:
            grouped[bid] = {
                "benchmark_id": bid,
                "plate": row["plate"],
                "well": row["well"],
                "field": row["field"],
                "t_label": row["t_label"],
                "channels": [],
            }
        grouped[bid]["channels"].append({
            "channel": row["channel"],
            "root_path": row["root_path"],
            "file_name": row["file_name"],
        })
    return list(grouped.values())


def _score_samples(predictor: TilePredictor, sample_images: list[list[np.ndarray]],
                    images_metadata: list[dict]) -> tuple[int, int]:
    """Run predict() over already-loaded samples, printing per-sample results."""
    n_ok, n_failed = 0, 0
    total = len(sample_images)
    for i, (sample_image, image_metadata) in enumerate(zip(sample_images, images_metadata), start=1):
        key = (image_metadata["plate"], image_metadata["well"], image_metadata["field"])
        try:
            result = predictor.predict(sample_image, image_metadata)
            n_ok += 1
            print(f"[{i}/{total}] {key} -> {result['predicted_class']} "
                  f"(label={image_metadata.get('label')})")
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            print(f"[{i}/{total}] {key} FAILED: {e}")
    return n_ok, n_failed


def compute_val_reference(predictor: TilePredictor, db_logger: DBLogger, run_id: str, source: str) -> None:
    """Score this run's own validation samples (dataset_manifest.json) as reference rows."""
    try:
        artifact_dir = predictor.model_info["artifact_dir"]
        manifest_path = Path(artifact_dir) / "dataset_manifest.json"
        if not manifest_path.exists():
            print(f"dataset_manifest.json not found in run {run_id} artifacts "
                  f"(looked in {artifact_dir})")
            return
        with open(manifest_path) as f:
            manifest = json.load(f)
    except KeyError:
        print(f"No artifact_dir available for run {run_id} -- could not "
              "download run artifacts when the model was loaded.")
        return
    except Exception as e:  # noqa: BLE001
        print(f"Failed to read dataset manifest: {e}")
        return

    val_samples = manifest.get("val_samples", [])
    print(f"Found {len(val_samples)} validation samples for model: {source}")
    if not val_samples:
        print("No val_samples found in dataset_manifest.json -- nothing to do.")
        return

    # Skip samples that already have a reference row for this run_id, so an
    # interrupted run resumes instead of re-inserting duplicate rows.
    # field is coerced to int on both sides: image_metadata.field is INTEGER,
    # while older manifests stored the raw zero-padded string ("001"), which
    # would never compare equal and silently defeat the resume.
    already_done = {
        (plate, well, int(field)) for plate, well, field in
        (db_logger.get_reference_samples(run_id) or [])
    }
    remaining = [
        s for s in val_samples
        if (s["plate"], s["well"], int(s["field"])) not in already_done
    ]
    print(f"{len(already_done)} / {len(val_samples)} validation samples already have "
          f"reference rows for run_id={run_id}; processing {len(remaining)} remaining.")
    if not remaining:
        print("Nothing to do.")
        return

    sample_images, images_metadata = load_validation_data(remaining)
    print(f"Loaded {len(sample_images)} validation sample images")

    n_ok, n_failed = _score_samples(predictor, sample_images, images_metadata)
    print(f"Validation reference done. {n_ok} succeeded, {n_failed} failed out of {len(remaining)}.")


def compute_benchmark_reference(predictor: TilePredictor, db_logger: DBLogger, run_id: str) -> None:
    """Score the registered global benchmark set for this run_id."""
    member_rows = db_logger.fetch_benchmark_members() or []
    samples = _group_members(member_rows)
    print(f"Found {len(samples)} registered benchmark samples.")
    if not samples:
        print("No benchmark samples registered -- run register_benchmark first. Nothing to do.")
        return

    # Skip samples already scored for this run_id, so an interrupted run resumes.
    already_done = set(db_logger.get_benchmark_predictions(run_id) or [])
    remaining = [s for s in samples if s["benchmark_id"] not in already_done]
    print(f"{len(already_done)} / {len(samples)} benchmark samples already scored for "
          f"run_id={run_id}; processing {len(remaining)} remaining.")
    if not remaining:
        print("Nothing to do.")
        return

    sample_images, images_metadata = load_benchmark_data(remaining)
    print(f"Loaded {len(sample_images)} benchmark sample images")

    n_ok, n_failed = _score_samples(predictor, sample_images, images_metadata)
    print(f"Benchmark reference done. {n_ok} succeeded, {n_failed} failed out of {len(remaining)}.")


def main():
    # Deferred to here: api.predictor transitively imports torch/mlflow/timm,
    # and main() is the only path that needs them -- keeping it function-local
    # leaves the module importable with just requirements/monitoring_req.txt.
    from api.predictor import TilePredictor

    parser = argparse.ArgumentParser(
        description="Compute reference predictions (validation and/or benchmark) for the served model."
    )
    parser.add_argument("--target", choices=["val", "benchmark", "both"], default="both",
                        help="Which reference to compute: this run's own validation samples "
                             "('val'), the global benchmark set ('benchmark'), or both (default).")
    args = parser.parse_args()

    settings = Settings()
    db_logger: DBLogger | None = None

    if not settings.has_model_source:
        print("WARNING: No model source configured.")
        print("Set one of: API_MODEL_NAME or API_RUN_NAME")
        return

    if not settings.has_db_uri:
        print("ERROR: No database URI configured.")
        print("Set API_DB_URI to enable reference logging.")
        return

    source = settings.model_name or settings.run_name
    print(f"Loading model: {source}")

    db_logger = DBLogger(db_uri=settings.db_uri)
    print(f"Connecting to database: {settings.db_uri}")
    try:
        db_logger.connect()
    except Exception as e:  # noqa: BLE001
        print(f"Failed to connect to database: {e}")
        return

    try:
        predictor = TilePredictor.from_mlflow(
            tracking_uri=settings.tracking_uri,
            experiment_name=settings.experiment_name,
            model_name=settings.model_name,
            run_name=settings.run_name,
            crop_size=settings.crop_size,
            stride=settings.effective_stride,
            device=settings.device,
            db_logger=db_logger,
        )
        run_id = predictor.model_info["run_id"]

        if args.target in ("val", "both"):
            print("=" * 60)
            print(f"Validation reference (run_id={run_id})")
            print("=" * 60)
            compute_val_reference(predictor, db_logger, run_id, source)

        if args.target in ("benchmark", "both"):
            print("=" * 60)
            print(f"Benchmark reference (run_id={run_id})")
            print("=" * 60)
            compute_benchmark_reference(predictor, db_logger, run_id)

        print(f"Results saved in database at {settings.db_uri}")
    finally:
        if db_logger is not None:
            db_logger.close()


if __name__ == "__main__":
    main()
