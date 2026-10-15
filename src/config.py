"""Configuration management for CNN classifier using Hydra structured configs."""
from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


# ── Dataset configs (nested inside datamodule) ────────────────────────────
@dataclass
class TiledDatasetConfig:
    _target_: str = "src.dataset.TiledMultiChannelDataset"
    root_dir: str = MISSING
    channels: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    crop_size: int = 224
    stride: int | None = None
    cache_size: int = 16
    max_samples_per_label: int | None = None
    verbose: bool = False


@dataclass
class RandomCropDatasetConfig:
    _target_: str = "src.dataset.MultiChannelImageDataset"
    root_dir: str = MISSING
    channels: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    crop_size: int = 224


# ── Transform configs ─────────────────────────────────────────────────────
# Transforms are now configured as lists in YAML (see datamodule configs).
# Each entry: {name: TransformClass, param1: val1, ...}
# No structured config needed — plain lists work with OmegaConf.


# ── Label provider configs (nested inside datamodule) ─────────────────────
# Instantiated via hydra.utils.instantiate() in MultiChannelDataModule.setup(),
# same _target_ pattern as the dataset configs above -- a new label source is
# a new provider class (get_labels_for_wells(wells) -> {(plate, well): label})
# plus a YAML entry, no datamodule code change.
@dataclass
class MongoDBLabelProviderConfig:
    _target_: str = "utils.labels.MongoDBLabelsProvider"
    collection: str = "tags"


@dataclass
class CSVLabelProviderConfig:
    _target_: str = "utils.labels.CSVLabelsProvider"
    csv_path: str = MISSING
    plate_col: str = "Plate"
    well_col: str = "Well"
    label_col: str = "label"


@dataclass
class DummyLabelProviderConfig:
    _target_: str = "utils.labels.DummyLabelsProvider"
    class_names: list[str] | None = None
    seed: int = 42


# ── Datamodule (dataloader-level config + nested dataset + transforms) ────
@dataclass
class DataModuleConfig:
    _target_: str = "src.datamodule.MultiChannelDataModule"
    dataset: Any = field(default_factory=TiledDatasetConfig)
    train_transform: Any = field(default_factory=list)   # List of {name: ..., params}
    val_transform: Any = field(default_factory=list)      # List of {name: ..., params}
    batch_transform: Any | None = None                 # Batch-level (Mixup/CutMix)
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    label_provider: Any = field(default_factory=MongoDBLabelProviderConfig)
    train_val_split: float = 0.8
    max_wells_per_label: int | None = None
    exclude_wells: Any | None = None


# ── Model ──────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    _target_: str = "src.model.CNNClassifier"
    dropout: float = 0.5
    num_blocks: int = 4
    base_channels: int = 32
    channel_multiplier: float = 2.0
    hidden_dim: int = 128


# ── Optimizer and Scheduler (sibling keys) ────────────────────────────────
@dataclass
class OptimizerInnerConfig:
    _target_: str = "torch.optim.AdamW"
    lr: float = 0.001
    weight_decay: float = 0.0001


@dataclass
class SchedulerConfig:
    _target_: str = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    mode: str = "min"
    factor: float = 0.5
    patience: int = 5


@dataclass
class OptimizerConfig:
    optimizer: OptimizerInnerConfig = field(default_factory=OptimizerInnerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


# ── Callbacks (list of _target_ dicts, instantiated generically) ──────────
# No structured config needed — callbacks is a plain list in YAML.
# Each entry has a _target_ and its constructor args.


# ── Trainer (with nested logger) ──────────────────────────────────────────
@dataclass
class TensorBoardLoggerConfig:
    _target_: str = "pytorch_lightning.loggers.TensorBoardLogger"
    save_dir: str = "logs"
    name: str = "image_classifier"


@dataclass
class MLflowLoggerConfig:
    _target_: str = "pytorch_lightning.loggers.MLFlowLogger"
    experiment_name: str = "image_classifier"
    tracking_uri: str = "sqlite:///mlflow.db"


@dataclass
class LoggerConfig:
    tensorboard: TensorBoardLoggerConfig = field(default_factory=TensorBoardLoggerConfig)
    mlflow: MLflowLoggerConfig = field(default_factory=MLflowLoggerConfig)


@dataclass
class TrainerConfig:
    _target_: str = "pytorch_lightning.Trainer"
    max_epochs: int = 50
    accelerator: str = "cuda"
    devices: int = 1
    precision: str = "16-mixed"
    log_every_n_steps: int = 10
    deterministic: bool = True
    gradient_clip_val: float = 10000
    logger: LoggerConfig = field(default_factory=LoggerConfig)


# ── Top-level config ──────────────────────────────────────────────────────
@dataclass
class Config:
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    callbacks: list[Any] = field(default_factory=list)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    # Runtime parameters
    seed: int = 42
    ckpt: str | None = None
    resume_from_model: str | None = None
    resume_weights_only: bool = True  # If True, load only model weights (not optimizer/scheduler/epoch)
    run_name: str | None = None
    dataset_name: str | None = None
    tags: list[str] | None = None
