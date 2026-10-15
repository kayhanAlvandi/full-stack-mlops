"""Custom Dataset for multi-channel microscopy images (JXL/TIF).

Sample *selection* (which plate/well/field/channel files make up the dataset)
lives in ``src/sample_selection.py`` -- torch-free and shared with the standalone
dataset builder. The classes here are the access layer: loading, normalising,
tiling and transforming the samples they are given. Either pass a prebuilt
``samples`` list (the datamodule does, so the directory is scanned once for both
splits) or let them build their own from ``root_dir`` + ``labels_dict``.
"""
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.filename_parser import FILENAME_PATTERN

from .sample_selection import SUPPORTED_EXTENSIONS, build_samples, scan_directory

try:
    import pillow_jxl  # JXL support for PIL  # noqa: F401
except ImportError:
    print("Warning: pillow-jxl not installed. JXL files may not load.")


class LabelEncoder:
    """Encode string labels to integers and decode back.
    
    Maintains a consistent mapping between string labels and integer indices.
    """
    
    def __init__(self):
        self.label_to_idx: dict[str, int] = {}
        self.idx_to_label: dict[int, str] = {}
        self._fitted = False
    
    def fit(self, labels: list[str]) -> "LabelEncoder":
        """Fit encoder on a list of string labels."""
        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self._fitted = True
        return self
    
    def transform(self, labels: list[str]) -> list[int]:
        """Transform string labels to integers."""
        if not self._fitted:
            raise ValueError("LabelEncoder must be fitted before transform")
        return [self.label_to_idx[label] for label in labels]
    
    def fit_transform(self, labels: list[str]) -> list[int]:
        """Fit and transform in one step."""
        self.fit(labels)
        return self.transform(labels)
    
    def inverse_transform(self, indices: list[int]) -> list[str]:
        """Convert integer indices back to string labels."""
        return [self.idx_to_label[idx] for idx in indices]
    
    def encode(self, label: str) -> int:
        """Encode a single label."""
        return self.label_to_idx[label]
    
    def decode(self, idx: int) -> str:
        """Decode a single index."""
        return self.idx_to_label[idx]
    
    @property
    def num_classes(self) -> int:
        """Return number of unique classes."""
        return len(self.label_to_idx)
    
    @property
    def classes(self) -> list[str]:
        """Return list of class names in order."""
        return [self.idx_to_label[i] for i in range(len(self.idx_to_label))]


class MultiChannelImageDataset(Dataset):
    """Dataset for loading multi-channel microscopy images (no tiling).
    
    Each sample consists of multiple channels (C01-C05) from the same field (F).
    Labels are retrieved based on plate and well information.
    
    Filename pattern: {plate}_{well}_T{time}F{field}L{layer}A{action}Z{z}C{channel}.jxl
    """
    
    # Kept as class attributes for backwards compatibility; the canonical
    # definitions live in utils/filename_parser and src/sample_selection.
    FILENAME_PATTERN = FILENAME_PATTERN
    SUPPORTED_EXTENSIONS = SUPPORTED_EXTENSIONS
    
    def __init__(
        self,
        root_dir: str,
        channels: list[int],
        labels_dict: dict[tuple[str, str], str],
        label_encoder: LabelEncoder,
        transform: Callable | None = None,
        max_samples_per_label: int | None = None,
        seed: int = 42,
        verbose: bool = False,
        samples: list[dict] | None = None,
    ):
        """
        Args:
            root_dir: Root directory containing images.
            channels: List of channel numbers to use (1-5).
            labels_dict: Dictionary mapping (plate, well) to string label.
            label_encoder: Fitted LabelEncoder to convert string labels to int.
            transform: Optional transform to apply to samples.
            max_samples_per_label: Max samples per label for balanced dataset.
            seed: Seeds the balancing selection so it is reproducible.
            verbose: Print detailed sample selection info.
            samples: Prebuilt sample list (from src.sample_selection.build_samples).
                When given, the directory is not scanned again.
        """
        self.root_dir = Path(root_dir)
        self.channels = sorted(channels)
        self.labels_dict = labels_dict
        self.label_encoder = label_encoder
        self.transform = transform
        self.max_samples_per_label = max_samples_per_label
        self.seed = seed
        self.verbose = verbose
        
        # Build list of unique samples (plate, well, field combinations)
        self.samples = samples if samples is not None else build_samples(
            scan_directory(self.root_dir),
            self.labels_dict,
            self.channels,
            max_samples_per_label=self.max_samples_per_label,
            seed=self.seed,
            verbose=self.verbose,
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_image(self, file_path: Path) -> np.ndarray:
        """Load a single image file."""
        suffix = file_path.suffix.lower()
        if suffix == ".tif":
            image_source = cv2.imread(str(file_path), -1)
        elif suffix == ".jxl":
            image_source = Image.open(file_path)
        else:
            raise ValueError("image path should end with .tif or .jxl")
        return np.array(image_source, dtype=np.float32)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        # Load all channels for this sample
        channel_images = []
        for ch in self.channels:
            file_path = sample["channel_files"][ch]
            img = self._load_image(file_path)
            
            # Percentile-clip to [0, 1]: robust to outliers (hot/dead pixels)
            p_lo, p_hi = np.percentile(img, [1, 99.5])
            if p_hi - p_lo > 0:
                img = np.clip(img, p_lo, p_hi)
                img = ((img - p_lo) / (p_hi - p_lo)).astype(np.float32)
            else:
                img = np.zeros_like(img)
            
            channel_images.append(img)
        
        # Stack channels: (C, H, W)
        image = np.stack(channel_images, axis=0)
        
        # Convert to tensor
        image = torch.from_numpy(image)
        
        # Encode string label to integer
        label = self.label_encoder.encode(sample["label"])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


class LRUImageCache:
    """LRU cache for loaded images to avoid repeated disk reads."""
    
    def __init__(self, max_size: int = 16):
        self.max_size = max_size
        self.cache: dict[str, np.ndarray] = {}
        self.access_order: list[str] = []
    
    def get(self, key: str) -> np.ndarray | None:
        """Get image from cache, returns None if not found."""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, image: np.ndarray) -> None:
        """Add image to cache, evicting oldest if full."""
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Evict least recently used
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = image
        self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()


class TiledMultiChannelDataset(Dataset):
    """Dataset that extracts grid tiles from large multi-channel images.
    
    Dynamically calculates tile grid based on image size and crop size.
    Uses LRU cache to efficiently reuse loaded images across tile requests.
    """
    
    FILENAME_PATTERN = MultiChannelImageDataset.FILENAME_PATTERN
    SUPPORTED_EXTENSIONS = MultiChannelImageDataset.SUPPORTED_EXTENSIONS
    
    def __init__(
        self,
        root_dir: str,
        channels: list[int],
        labels_dict: dict[tuple[str, str], str],
        label_encoder: LabelEncoder,
        crop_size: int = 224,
        stride: int | None = None,
        transform: Callable | None = None,
        cache_size: int = 16,
        max_samples_per_label: int | None = None,
        seed: int = 42,
        verbose: bool = False,
        samples: list[dict] | None = None,
    ):
        """
        Args:
            root_dir: Root directory containing images.
            channels: List of channel numbers to use (1-5).
            labels_dict: Dictionary mapping (plate, well) to string label.
            label_encoder: Fitted LabelEncoder to convert string labels to int.
            crop_size: Size of each tile (crop_size x crop_size).
            stride: Step between tiles. If None, uses crop_size (non-overlapping).
            transform: Optional transform to apply (e.g., normalization, augmentation).
            cache_size: Number of images to keep in LRU cache.
            max_samples_per_label: Max samples per label for balanced dataset.
            seed: Seeds the balancing selection so it is reproducible.
            verbose: Print detailed sample selection info.
            samples: Prebuilt sample list (from src.sample_selection.build_samples,
                built with shape_mode set). When given, the directory is not
                scanned again.
        """
        self.root_dir = Path(root_dir)
        self.channels = sorted(channels)
        self.labels_dict = labels_dict
        self.label_encoder = label_encoder
        self.crop_size = crop_size
        self.stride = stride or crop_size
        self.transform = transform
        self.cache = LRUImageCache(max_size=cache_size)
        self.max_samples_per_label = max_samples_per_label
        self.seed = seed
        self.verbose = verbose
        
        # Build sample list (one per image/field), balanced if requested.
        # image_size is needed up front to lay out the tile grid.
        self.samples = samples if samples is not None else build_samples(
            scan_directory(self.root_dir),
            self.labels_dict,
            self.channels,
            max_samples_per_label=self.max_samples_per_label,
            seed=self.seed,
            shape_mode="first",
            verbose=self.verbose,
        )
        
        # Build tile index: list of (sample_idx, row, col) for all tiles
        self.tiles = self._build_tile_index()
    
    def _build_tile_index(self) -> list[tuple[int, int, int]]:
        """Build index of all tiles: (sample_idx, top, left)."""
        tiles = []
        
        for sample_idx, sample in enumerate(self.samples):
            h, w = sample["image_size"]
            
            # Calculate number of tiles in each dimension
            n_rows = max(1, (h - self.crop_size) // self.stride + 1)
            n_cols = max(1, (w - self.crop_size) // self.stride + 1)
            
            for row in range(n_rows):
                for col in range(n_cols):
                    top = min(row * self.stride, h - self.crop_size)
                    left = min(col * self.stride, w - self.crop_size)
                    tiles.append((sample_idx, top, left))
        
        return tiles
    
    def _load_single_image(self, file_path: Path) -> np.ndarray:
        """Load a single image file."""
        suffix = file_path.suffix.lower()
        if suffix == ".tif":
            image_source = cv2.imread(str(file_path), -1)
        elif suffix == ".jxl":
            image_source = Image.open(file_path)
        else:
            raise ValueError("image path should end with .tif or .jxl")
        return np.array(image_source, dtype=np.float32)
    
    def _load_sample_image(self, sample_idx: int) -> np.ndarray:
        """Load full multi-channel image for a sample, using cache."""
        sample = self.samples[sample_idx]
        cache_key = f"{sample['plate']}_{sample['well']}_{sample['field']}"
        
        # Check cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Load all channels
        channel_images = []
        for ch in self.channels:
            file_path = sample["channel_files"][ch]
            img = self._load_single_image(file_path)
            
            # Percentile-clip to [0, 1]: robust to outliers (hot/dead pixels)
            p_lo, p_hi = np.percentile(img, [1, 99.5])
            if p_hi - p_lo > 0:
                img = np.clip(img, p_lo, p_hi)
                img = ((img - p_lo) / (p_hi - p_lo)).astype(np.float32)
            else:
                img = np.zeros_like(img)
            
            channel_images.append(img)
        
        # Stack channels: (C, H, W)
        image = np.stack(channel_images, axis=0)
        
        # Store in cache
        self.cache.put(cache_key, image)
        
        return image
    
    def __len__(self) -> int:
        return len(self.tiles)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample_idx, top, left = self.tiles[idx]
        sample = self.samples[sample_idx]
        
        # Load full image (from cache if available)
        image = self._load_sample_image(sample_idx)
        
        # Extract tile
        tile = image[:, top:top + self.crop_size, left:left + self.crop_size]
        
        # Convert to tensor
        tile = torch.from_numpy(tile.copy())
        
        # Encode label
        label = self.label_encoder.encode(sample["label"])
        
        # Apply transforms (e.g., augmentation, normalization)
        if self.transform:
            tile = self.transform(tile)
        
        return tile, label
    
    @property
    def num_samples(self) -> int:
        """Number of original samples (images)."""
        return len(self.samples)
    
    @property
    def num_tiles(self) -> int:
        """Total number of tiles across all samples."""
        return len(self.tiles)
    
    def get_tiles_per_sample(self) -> list[int]:
        """Get number of tiles for each sample."""
        counts = [0] * len(self.samples)
        for sample_idx, _, _ in self.tiles:
            counts[sample_idx] += 1
        return counts
