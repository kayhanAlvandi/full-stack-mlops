"""PyTorch Lightning DataModule for multi-channel images."""

from collections import Counter

import pytorch_lightning as pl
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils.labels import limit_wells_per_label

from .dataset import LabelEncoder, TiledMultiChannelDataset
from .sample_selection import build_samples, scan_directory, wells_from_index
from .transforms import build_batch_transform, build_transforms


class MultiChannelDataModule(pl.LightningDataModule):
    """DataModule for multi-channel microscopy images.
    
    Handles data loading, splitting, and transformations.
    All components (dataset class, train/val transforms) are determined
    by Hydra config _target_ fields and created via hydra.utils.instantiate().
    
    Constructor params match the YAML keys exactly so that
    ``instantiate(cfg.datamodule, _recursive_=False)`` works directly.
    """
    
    def __init__(
        self,
        dataset: DictConfig,
        train_transform: DictConfig,
        val_transform: DictConfig,
        batch_transform: DictConfig | None = None,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_val_split: float = 0.66,
        label_provider: DictConfig | dict | None = None,
        max_wells_per_label: int | None = None,
        exclude_wells: list | None = None,
    ):
        """
        Args:
            dataset: Hydra config for the dataset (contains _target_ and dataset params).
            train_transform: List of transform dicts with "name" + params, or legacy
                            dict with _target_ pointing to a factory function.
            val_transform: Same format as train_transform.
            batch_transform: Optional batch-level transform config (Mixup/CutMix).
                            Dict with "name" + params. Applied in training_step.
            batch_size: Batch size for data loaders.
            num_workers: Number of workers for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            train_val_split: Fraction of data for training (rest for validation).
            label_provider: Hydra config for the label provider (contains
                _target_, e.g. ``utils.labels.MongoDBLabelsProvider`` /
                ``CSVLabelsProvider`` / ``DummyLabelsProvider``, plus that
                class's own constructor params). Instantiated with
                ``hydra.utils.instantiate()``, same pattern as ``dataset`` --
                a custom label provider just needs a new class with a
                ``get_labels_for_wells(wells) -> {(plate, well): label}``
                method and a YAML pointing ``_target_`` at it, no code change
                here. Training always needs real labels, so the provider must
                resolve every requested well (there is no "none" option here).
            max_wells_per_label: Max wells per label class (None = all).
            exclude_wells: List of [plate, well] pairs to exclude (corrupted images).
        """
        super().__init__()
        
        # Convert exclude_wells to plain Python types BEFORE save_hyperparameters
        # to prevent OmegaConf ListConfig from leaking into checkpoints
        if exclude_wells is not None:
            from omegaconf import OmegaConf
            raw = OmegaConf.to_container(exclude_wells) if hasattr(exclude_wells, '_metadata') else exclude_wells
            exclude_wells = [tuple(w) for w in raw]
        
        self.save_hyperparameters(
            ignore=["dataset", "train_transform", "val_transform", "batch_transform", "label_provider"]
        )
        
        self.dataset_cfg = dataset
        self.train_transform_cfg = train_transform
        self.val_transform_cfg = val_transform
        self.batch_transform_cfg = batch_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_val_split = train_val_split
        if label_provider is None:
            raise ValueError(
                "label_provider is required (a Hydra config with _target_ pointing at a "
                "label provider class, e.g. src.dataset.MongoDBLabelsProvider)."
            )
        self.label_provider_cfg = label_provider
        self.max_wells_per_label = max_wells_per_label
        self.exclude_wells = exclude_wells
        
        # Expose dataset-level params for external access
        self.root_dir = dataset.root_dir
        self.channels = list(dataset.channels)
        self.crop_size = dataset.crop_size
        
        # Build batch-level transform (Mixup/CutMix) — exposed for model to use
        self.batch_transform = build_batch_transform(
            OmegaConf.to_container(batch_transform, resolve=True) if batch_transform is not None else None
        )
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.label_encoder = None
        self.file_index = None
        
    def setup(self, stage: str | None = None):
        """Setup datasets for training, validation, and testing.
        
        Scans the image directory exactly once and reuses that index for well
        discovery and for both splits' sample lists.
        """
        # Scan the image directory once; every later step filters this in memory
        self.file_index = scan_directory(self.root_dir)
        wells = wells_from_index(self.file_index, exclude_wells=self.exclude_wells)
        
        # Get labels (string labels). The provider class and its own params
        # (csv_path, collection, class_names, ...) are fully determined by the
        # label_provider config's _target_ -- adding a new label source is a
        # new class + YAML, no change needed here.
        labels_provider = instantiate(self.label_provider_cfg)
        labels_dict = labels_provider.get_labels_for_wells(wells)
        
        if len(labels_dict) == 0:
            raise ValueError(f"No labels found for images in {self.root_dir}")
        
        # Limit wells per label if set
        labels_dict = limit_wells_per_label(
            labels_dict,
            self.max_wells_per_label,
            verbose=getattr(self.dataset_cfg, "verbose", False),
        )
        
        # Create and fit label encoder on all unique string labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(list(labels_dict.values()))
        
        print(f"Classes found: {self.label_encoder.classes}")
        print(f"Number of classes: {self.label_encoder.num_classes}")
        
        self._setup_datasets(labels_dict)
    
    def _setup_datasets(self, labels_dict: dict):
        """Setup train/val datasets using hydra.utils.instantiate().
        
        The dataset class and transforms are fully determined by config _target_ fields.
        No hardcoded class names — adding a new dataset type only requires a new YAML.
        """
        # Split at (plate, well) level to avoid data leakage
        # Stratified by label to ensure balanced representation
        unique_wells = list(labels_dict.keys())
        labels = [labels_dict[w] for w in unique_wells]
        
        if len(unique_wells) < 2:
            raise ValueError(f"Not enough wells to split: {len(unique_wells)} wells")
        
        # Stratified split by label; fall back to non-stratified when a class
        # has too few wells for stratification (sklearn requires >=2 per class)
        stratify = labels if min(Counter(labels).values()) >= 2 else None
        if stratify is None and len(set(labels)) > 1:
            print("[WARNING] Some labels have only 1 well; using non-stratified split.")
        train_wells, val_wells = train_test_split(
            unique_wells,
            train_size=self.train_val_split,
            stratify=stratify,
            random_state=42,
        )
        
        train_labels = {w: labels_dict[w] for w in train_wells}
        val_labels = {w: labels_dict[w] for w in val_wells}
        
        # Build transforms: list-based (new) or _target_-based (legacy)
        train_transform = self._build_transform(self.train_transform_cfg)
        val_transform = self._build_transform(self.val_transform_cfg)
        
        # Instantiate datasets: use get_class() + direct constructor call because
        # labels_dict has tuple keys and label_encoder is a runtime object,
        # neither of which can pass through OmegaConf.merge inside instantiate().
        dataset_class = get_class(self.dataset_cfg._target_)
        dataset_kwargs = OmegaConf.to_container(self.dataset_cfg, resolve=True)
        dataset_kwargs.pop("_target_")
        
        # Build each split's samples from the already-scanned index, so the
        # directory isn't walked again per dataset. max_samples_per_label stays
        # per-split (each split balanced independently), as before.
        # Tiled datasets need each sample's image size up front to lay out the
        # tile grid; the plain image dataset doesn't, so skip the extra read.
        selection_kwargs = {
            "channels": self.channels,
            "max_samples_per_label": dataset_kwargs.get("max_samples_per_label"),
            "seed": dataset_kwargs.get("seed", 42),
            "shape_mode": "first" if issubclass(dataset_class, TiledMultiChannelDataset) else None,
            "verbose": dataset_kwargs.get("verbose", False),
        }
        train_samples = build_samples(self.file_index, train_labels, **selection_kwargs)
        val_samples = build_samples(self.file_index, val_labels, **selection_kwargs)
        
        self.train_dataset = dataset_class(
            **dataset_kwargs,
            labels_dict=train_labels,
            label_encoder=self.label_encoder,
            transform=train_transform,
            samples=train_samples,
        )
        
        self.val_dataset = dataset_class(
            **dataset_kwargs,
            labels_dict=val_labels,
            label_encoder=self.label_encoder,
            transform=val_transform,
            samples=val_samples,
        )
        
        print(f"Train: {len(train_labels)} wells -> {len(self.train_dataset)} samples")
        print(f"Val: {len(val_labels)} wells -> {len(self.val_dataset)} samples")
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            return self.val_dataloader()
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    @staticmethod
    def _build_transform(cfg):
        """Build transforms from config.
        
        Supports two formats:
          1. List of dicts (new): [{name: RandomHorizontalFlip, p: 0.5}, ...]
             -> uses build_transforms() registry-based builder
          2. Dict with _target_ (legacy): {_target_: src.transforms.get_train_transforms, ...}
             -> uses hydra.utils.instantiate()
        """
        if isinstance(cfg, (list, ListConfig)):
            # New list-based format
            return build_transforms(OmegaConf.to_container(cfg, resolve=True)
                                    if isinstance(cfg, ListConfig) else cfg)
        else:
            # Legacy _target_ format
            return instantiate(cfg)
