"""Model loading and prediction logic for the API.

Supports loading models from:
  1. MLflow registered model: e.g. "TransferLearning/20" or "TransferLearning/latest"
  2. MLflow run name: e.g. "Vits_finetune_cosine_warmup_..."

Models are loaded via ``mlflow.pytorch.load_model``; the model class code is
bundled inside the MLflow artifact (logged with ``code_paths``), so the API does
not import any training code from ``src/``.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from omegaconf import OmegaConf

from utils.filename_parser import clean_image_metadata, clean_tiles_metadata

if TYPE_CHECKING:
    from database.dblogger import DBLogger



class Normalize:
    """Per-channel zero-mean, unit-variance normalization.

    Copied from src/transforms.py to keep the API decoupled from training code.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        return (x - mean) / (std + self.eps)


def compute_stats(pixels: torch.Tensor) -> tuple[float, float, float, float, float, float]:
    """Compute pixel-intensity summary statistics for one channel of one tile.

    Used for input-drift monitoring: mean/std/percentiles of a single-channel
    tile's raw (preprocessed) pixel values, stored in tile_channel_stats.

    Args:
        pixels: Tensor of shape (crop_size, crop_size) -- one channel of one tile.

    Returns:
        (mean, std, p1, p5, p95, p99) as plain Python floats.
    """
    flat = pixels.reshape(-1).float()
    mean = flat.mean().item()
    std = flat.std().item()
    p1, p5, p95, p99 = torch.quantile(
        flat, torch.tensor([0.01, 0.05, 0.95, 0.99], dtype=flat.dtype)
    ).tolist()
    return mean, std, p1, p5, p95, p99

def chans_reorder(image_channels: list[np.ndarray], image_metadata: dict) -> tuple[np.ndarray, dict]:
    """Canonicalize channel order to match training.

    src/dataset.py always stacks channels sorted ascending by channel number
    (`self.channels = sorted(channels)`), so inference must use that exact
    same order regardless of upload order, or the model sees
    out-of-distribution input whenever a caller doesn't happen to upload
    files in C1..CN order.

    Returns a new (C, H, W) ndarray and a *shallow copy* of image_metadata
    with `channels` and `channel_files` reordered. The caller's dict is left
    untouched so manifest-derived metadata (e.g. from compute_predictions_references) is
    not mutated as a side effect of prediction.
    """
    parsed_channels = image_metadata['channels']
    if len(set(parsed_channels)) != len(parsed_channels):
        raise ValueError(f"Duplicate channel numbers in upload: {parsed_channels}")
    order = sorted(range(len(image_channels)), key=lambda i: parsed_channels[i])
    ordered_image_channels = [image_channels[i] for i in order]
    # shallow copy so the caller's dict is not mutated in place
    reordered_metadata = dict(image_metadata)
    reordered_metadata['channels'] = [image_metadata['channels'][i] for i in order]
    reordered_metadata['channel_files'] = [image_metadata['channel_files'][i] for i in order]

    return np.stack(ordered_image_channels, axis=0), reordered_metadata

class TilePredictor:
    """Loads a trained model and performs tiled prediction on images.
    
    Workflow:
        1. Load model from MLflow (registered model, run name, or checkpoint)
        2. Auto-detect config (crop_size, channels, class names) from MLflow artifacts
        3. Accept a multi-channel image (C, H, W) numpy array
        4. Tile it into crop_size x crop_size patches
        5. Run inference on each tile
        6. Return per-tile predictions + majority vote
        7. Save results to prediction database
    """
    
    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlflow.db",
        experiment_name: str = "image_classifier",
        model_name: str = "",
        run_name: str = "",
        crop_size: int = 224,
        stride: int | None = None,
        device: str = "cpu",
        db_logger: DBLogger | None = None
    ):
        self.device = torch.device(device)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Load model and extract config from MLflow artifacts
        self.model, self.model_info = self._load_model(
            model_name=model_name,
            run_name=run_name,
        )
        # torchmetrics.Metric submodules (e.g. an accuracy metric logged as
        # part of the LightningModule) keep a plain `.device` attribute
        # recorded at save time -- unlike real tensors, mlflow's
        # map_location doesn't touch it. If the model was trained/saved on
        # GPU and this predictor targets CPU, that stale attribute makes
        # Metric._apply() probe a `cuda` tensor during .to(), crashing with
        # "no NVIDIA driver" on CPU-only hosts even though the target device
        # is cpu. Force it to match before calling .to() so no submodule
        # tries to allocate on a device we're not actually using.
        for module in self.model.modules():
            if isinstance(getattr(module, "device", None), torch.device):
                module.device = self.device
        self.model.to(self.device)
        self.model.eval()
        
        # Use config from MLflow artifacts if available, otherwise use provided values
        self.crop_size = self.model_info.get("crop_size", crop_size)
        self.stride = stride if stride is not None else self.crop_size
        self.class_names = self.model_info["class_names"]
        self.in_channels = self.model_info.get("in_channels", 5)
        
        # Preprocessing: normalize per-channel (zero mean, unit variance)
        self.normalize = Normalize()
        self.db_logger = db_logger
    
    def _load_model(self, model_name: str, run_name: str):
        """Load model with priority: model_name > run_name.
        
        For MLflow sources, also downloads hydra_config.yaml and dataset_manifest.json
        to auto-configure class names, crop size, channels, etc.
        """
        run_id = None
        model = None
        info = {}
        
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # ── 1. Load from MLflow registered model (e.g. "TransferLearning/20") ──
        if model_name:
            run_id, model = self._load_from_registry(model_name)
            info["source"] = f"registry:{model_name}"
        
        # ── 2. Load from MLflow run name ──
        elif run_name:
            run_id, model = self._load_from_run_name(run_name)
            info["source"] = f"run_name:{run_name}"
        
        else:
            raise ValueError(
                "No model source specified. Set one of: "
                "model_name (e.g. 'TransferLearning/20') or "
                "run_name (e.g. 'my_training_run')"
            )
        
        # Extract config from MLflow run artifacts if we have a run_id.
        # class_names order must match the model's output logits exactly, so
        # this must come from dataset_metadata.json -- no silent fallback to
        # a guessed/differently-ordered class list, or every prediction gets
        # mislabeled without any indication something went wrong.
        if not run_id:
            raise ValueError("No run_id resolved; cannot load artifacts")
        info.update(self._load_run_config(run_id))
        info["run_id"] = run_id
        info["model_class"] = model.__class__.__name__
        info["num_classes"] = model.num_classes
        
        return model, info
    
    def _load_from_registry(self, model_ref: str):
        """Load from MLflow model registry. e.g. 'TransferLearning/20' or 'TransferLearning/latest'."""
        client = mlflow.MlflowClient()
        
        ref = model_ref.removeprefix("models:/")
        parts = ref.split("/")
        name = parts[0]
        version = parts[1] if len(parts) > 1 else "latest"
        
        if version == "latest":
            versions = client.get_latest_versions(name)
            if not versions:
                raise ValueError(f"No versions found for registered model '{name}'")
            version = versions[0].version
            print(f"Resolved 'latest' -> version {version} for model '{name}'")
        
        mv = client.get_model_version(name, version)
        run_id = mv.run_id
        print(f"Loading {name}/v{version} (run_id={run_id[:8]}...)")
        
        model = self._load_model_from_run(run_id, client)
        return run_id, model
    
    def _load_from_run_name(self, run_name: str):
        """Load from MLflow run by its display name."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{self.experiment_name}' not found")
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"run_name = '{run_name}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            raise ValueError(f"No run found with name '{run_name}' in experiment '{self.experiment_name}'")
        
        run_id = runs.iloc[0].run_id
        print(f"Found run '{run_name}' (run_id={run_id[:8]}...)")
        
        client = mlflow.MlflowClient()
        model = self._load_model_from_run(run_id, client)
        return run_id, model
    
    def _load_model_from_run(self, run_id: str, client):
        """Load model from a run: run artifact -> registry.
        
        Models are logged via ``mlflow.pytorch.log_model`` with the model class
        code bundled (``code_paths``), so no training code import is required.
        """
        # 1. Try runs:/{run_id}/model (logged via mlflow.pytorch.log_model)
        try:
            model_uri = f"runs:/{run_id}/model"
            print(f"  Trying: {model_uri}")
            model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
            print("  ✓ Loaded from run artifact")
            return model
        except Exception:  # noqa: BLE001, S110
            pass
        
        # 2. Try model registry (find version linked to this run)
        try:
            for mv in client.search_model_versions():
                if mv.run_id == run_id:
                    model_uri = f"models:/{mv.name}/{mv.version}"
                    print(f"  Trying registry: {model_uri}")
                    model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
                    print("  ✓ Loaded from model registry")
                    return model
        except Exception as e:  # noqa: BLE001
            print(f"  Registry lookup failed: {e}")
        
        raise FileNotFoundError(
            f"No logged model found for run {run_id} (checked run artifact and registry)"
        )
    
    def _download_artifact(self, run_id: str):
        """Download and parse hydra_config.yaml from MLflow run artifacts."""
        artifact_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id, tracking_uri=self.tracking_uri,
        )
        return artifact_dir
    
    def _load_run_config(self, run_id: str) -> dict:
        """Extract crop_size, channels, class_names from MLflow run artifacts."""
        info = {}
        try:
            info['artifact_dir'] = self._download_artifact(run_id)
        except Exception as e: 
            raise RuntimeError(
                f"Could not download artifacts for run {run_id}"
            ) from e
        
        try:
            config_path = Path(info['artifact_dir']) / "hydra_config.yaml"
            cfg = OmegaConf.load(config_path)
            ds_cfg = cfg.datamodule.dataset
            info["crop_size"] = ds_cfg.get("crop_size", 224)
            info["in_channels"] = len(list(ds_cfg.get("channels", [1, 2, 3, 4, 5])))
            
            backbone = cfg.model.get("backbone_name", cfg.model.get("_target_", "unknown"))
            info["backbone"] = backbone
        except Exception as e:
            raise RuntimeError(
                f"Could not load hydra config for run {run_id}: {e}"
            ) from e
        

        metadata_path = Path(info['artifact_dir']) / "dataset_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"dataset_metadata.json not found in run {run_id} artifacts "
                f"({metadata_path}); cannot determine class_names order."
            )
        with open(metadata_path) as f:
            dataset_metadata = json.load(f)
        class_names = dataset_metadata.get("class_names")
        if not class_names:
            raise ValueError(
                f"dataset_metadata.json for run {run_id} has no class_names."
            )
        info["class_names"] = class_names

        return info
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess raw multi-channel image.
        
        Args:
            image: numpy array of shape (C, H, W), float32, raw pixel values
            
        Returns:
            Tensor of shape (C, H, W) with per-channel percentile normalization
        """
        # Percentile-clip per channel (same as dataset.py)
        processed_channels = []
        for c in range(image.shape[0]):
            ch = image[c]
            p_lo, p_hi = np.percentile(ch, [1, 99.5])
            if p_hi - p_lo > 0:
                ch = np.clip(ch, p_lo, p_hi)
                ch = ((ch - p_lo) / (p_hi - p_lo)).astype(np.float32)
            else:
                ch = np.zeros_like(ch, dtype=np.float32)
            processed_channels.append(ch)
        
        tensor = torch.from_numpy(np.stack(processed_channels, axis=0))
        return tensor
    
    def tile_image(self, image: torch.Tensor) -> list[dict]:
        """Split image into non-overlapping (or strided) tiles.
        
        Args:
            image: Tensor of shape (C, H, W)
            
        Returns:
            List of dicts with keys: 'tile' (C, crop_size, crop_size), 'row', 'col', 'y', 'x'
        """
        _, h, w = image.shape
        tiles = []
        
        for row_idx, y in enumerate(range(0, h - self.crop_size + 1, self.stride)):
            for col_idx, x in enumerate(range(0, w - self.crop_size + 1, self.stride)):
                tile = image[:, y:y + self.crop_size, x:x + self.crop_size]
                tiles.append({
                    "tile": tile,
                    "row": row_idx,
                    "col": col_idx,
                    "y": y,
                    "x": x,
                    "crop_size": self.crop_size,
                })
        
        return tiles
    
    @torch.no_grad()
    def predict_tiles(self, tiles: list[dict]) -> list[dict]:
        """Run model inference on all tiles.
        
        Args:
            tiles: List from tile_image()
            
        Returns:
            List of dicts with prediction info per tile
        """
        if not tiles:
            return []
        
        # Batch all tiles together
        batch = torch.stack([t["tile"] for t in tiles]).to(self.device)
        
        # Apply normalization per tile
        normalized = []
        for i in range(batch.shape[0]):
            normalized.append(self.normalize(batch[i]))
        batch = torch.stack(normalized)
        
        # Forward pass
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        results = []
        for i, tile_info in enumerate(tiles):
            results.append({
                "row": tile_info["row"],
                "col": tile_info["col"],
                "y": tile_info["y"],
                "x": tile_info["x"],
                "predicted_class": self.class_names[preds[i].item()],
                "predicted_idx": preds[i].item(),
                "confidence": probs[i, preds[i]].item(),
                "probabilities": {
                    name: probs[i, j].item()
                    for j, name in enumerate(self.class_names)
                },
            })
        
        return results
    
    def majority_vote(self, tile_predictions: list[dict]) -> dict:
        """Compute majority vote across all tile predictions.
        
        Args:
            tile_predictions: List from predict_tiles()
            
        Returns:
            Dict with overall prediction and vote counts
        """
        if not tile_predictions:
            return {"predicted_class": "unknown", "confidence": 0.0, "vote_counts": {}}
        
        votes = [t["predicted_class"] for t in tile_predictions]
        counter = Counter(votes)
        winner, winner_count = counter.most_common(1)[0]
        
        # Average the winner's probability across ALL tiles (not just tiles that voted winner)
        winner_confidences = [t["probabilities"][winner] for t in tile_predictions]
        avg_confidence = sum(winner_confidences) / len(winner_confidences)
        
        return {
            "predicted_class": winner,
            "confidence": avg_confidence,
            "total_tiles": len(tile_predictions),
            "vote_counts": dict(counter),
            "vote_fraction": winner_count / len(tile_predictions),
        }
    
    def predict(self, channels_image: list[np.ndarray], image_metadata: dict) -> dict:
        """Full prediction pipeline: preprocess -> tile -> predict -> majority vote.
        
        Args:
            channels_image: list of numpy arrays of shape (H, W), raw pixel values
            image_metadata: dictionary containg plate,well,field, root_path ,shape, channels [list of channel names in order of image channels], channel_files [list of filenames in order of image channels], and optionally label and is_reference
            
        Returns:
            Dict with 'tile_predictions' and 'image_prediction' (majority vote)
        """

        
        img_ids = None
        tile_stack_ids = None
        ## one source to reorder the input metdata and image channels to ascending order
        image, image_metadata = chans_reorder(channels_image, image_metadata)


        if self.db_logger:
            cleaned_image_metadata = clean_image_metadata(image_metadata)
            img_ids = self.db_logger.log_image_metadata(cleaned_image_metadata)

        # Preprocess
        tensor = self.preprocess_image(image)

        # Tile
        tiles = self.tile_image(tensor)


        if self.db_logger:
            # Log tile stack metadata (one stack_hash per tile position, shared by all channels)
            tiles_metadata = clean_tiles_metadata(tiles, img_ids)
            tile_stack_ids = self.db_logger.log_tile_stack(tiles_metadata)
            # Each tile stack has all channel images as members. channel_index
            # is img_ids' position, which is the model's input channel-axis
            # position. img_ids is built from image_metadata, which is
            # canonicalized to ascending channel-number order by
            # chans_reorder() at the top of predict() -- stored explicitly so
            # it doesn't need to be reconstructed later.
            tile_stack_members = [
                (tile_stack_id, img_id, channel_index)
                for tile_stack_id in tile_stack_ids
                for channel_index, img_id in enumerate(img_ids)
            ]
            tile_stack_member_ids = self.db_logger.log_tile_stack_member(tile_stack_members)

            n_channels = len(img_ids)
            channel_stats_rows = []
            for tile_idx, tile_info in enumerate(tiles):
                tile_tensor = tile_info["tile"]                      # (C, crop, crop)
                for channel_idx in range(n_channels):
                    member_id = tile_stack_member_ids[tile_idx * n_channels + channel_idx]
                    pixels = tile_tensor[channel_idx]
                    channel_stats_rows.append((member_id, *compute_stats(pixels)))
        
            _ = self.db_logger.log_tile_channel_stats(channel_stats_rows)

        # Predict per tile
        tile_predictions = self.predict_tiles(tiles)

        # Majority vote
        image_prediction = self.majority_vote(tile_predictions)

        if self.db_logger:
            # Log image prediction
            image_prediction_db = (
                image_metadata["plate"],
                image_metadata["well"],
                image_metadata["field"],
                self.model_info["run_id"],
                image_prediction["predicted_class"],
                image_metadata.get("label", None),
                image_prediction["total_tiles"],
                image_prediction["vote_fraction"],
                image_prediction["confidence"],
                image_metadata.get("is_reference", False),
                image_metadata.get("benchmark_id", None),
            )
            img_pred_id = self.db_logger.log_image_prediction(image_prediction_db)
            # Log tile predictions (one row per tile)
            tile_predictions_db = [
                (img_pred_id, tile_stack_ids[i], self.model_info["run_id"],
                 tile["predicted_class"], image_metadata.get("label", None), tile["confidence"],
                 image_metadata.get("is_reference", False), image_metadata.get("benchmark_id", None))
                for i, tile in enumerate(tile_predictions)
            ]
            self.db_logger.log_tile_prediction(tile_predictions_db)

        result = {
            "plate": image_metadata["plate"],
            "well": image_metadata["well"],
            "field": image_metadata["field"],
            "run_id": self.model_info["run_id"],
            "predicted_class": image_prediction["predicted_class"],
            "total_tiles": image_prediction["total_tiles"],
            "vote_fraction": image_prediction["vote_fraction"],
            "confidence": image_prediction["confidence"],
        }
        return result
