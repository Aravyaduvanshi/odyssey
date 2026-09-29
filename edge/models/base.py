from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ModelStatus(StrEnum):
    NOT_LOADED = "NOT_LOADED"
    LOADING = "LOADING"
    READY = "READY"
    ERROR = "ERROR"
    DISABLED = "DISABLED"


class YawPrediction(BaseModel):
    mounting_yaw_rad: float
    mounting_yaw_deg: float
    sin_yaw: float
    cos_yaw: float
    model_name: str
    timestamp_ns: int
    inference_latency_ms: float
    quality: Optional[float] = None
    is_stale: bool = False


class VelocityPrediction(BaseModel):
    velocity_mps: float
    velocity_kmh: float
    velocity_normalized: float
    log_variance: float
    sigma_mps: Optional[float] = None
    variance_mps2: Optional[float] = None
    model_name: str
    timestamp_ns: int
    inference_latency_ms: float
    quality: Optional[float] = None
    is_stale: bool = False


class BaseModelAdapter(ABC):
    name: str = "base_model"
    status: ModelStatus = ModelStatus.NOT_LOADED
    load_latency_ms: float = 0.0
    warmup_latency_ms: float = 0.0
    last_inference_latency_ms: float = 0.0
    last_inference_timestamp_ns: int = 0
    error_message: Optional[str] = None

    @abstractmethod
    def load(self) -> bool:
        """Load model weights and preprocessing artifacts once."""

    @abstractmethod
    def warmup(self) -> bool:
        """Perform warmup forward pass."""

    @abstractmethod
    def predict(self, window: Any) -> Any:
        """Execute inference on validated input window."""
