"""API configuration via environment variables or defaults."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """API settings loaded from environment or .env file.
    
    Model source (pick one):
        - model_name: MLflow registered model, e.g. "TransferLearning/20" or "TransferLearning/latest"
        - run_name:   MLflow run name, e.g. "Vits_finetune_cosine_warmup_..."
    """
    
    # ── MLflow settings ──────────────────────────────────────────────────
    tracking_uri: str = "http://mlflow:5000"
    experiment_name: str = "image_classifier"
    
    # ── Model source (priority: model_name > run_name) ──────────────────
    model_name: str = ""       # e.g. "TransferLearning/20" or "TransferLearning/latest"
    run_name: str = ""         # e.g. "Vits_finetune_cosine_warmup_..."
    
    # ── Tiling parameters ────────────────────────────────────────────────
    crop_size: int = 224
    stride: int | None = None  # None = non-overlapping (same as crop_size)
    
    # ── Device ───────────────────────────────────────────────────────────
    device: str = "cpu"

    # ── Artifact store ───────────────────────────────────────────────────
    # Local path to the tracking server's artifact store (its mlruns dir).
    # When set and the run's artifacts exist there, they're read straight
    # off disk instead of being HTTP-proxied through the tracking server.
    artifact_root: str | None = None

    # ── Database ─────────────────────────────────────────────────────────
    db_uri: str | None = None
    
    model_config = {"env_prefix": "API_", "env_file": ".env", "extra": "ignore"}
    
    @property
    def effective_stride(self) -> int:
        return self.stride if self.stride is not None else self.crop_size
    
    @property
    def has_model_source(self) -> bool:
        return bool(self.model_name or self.run_name)

    @property
    def has_db_uri(self) -> bool:
        return bool(self.db_uri)

