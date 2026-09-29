from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class ModelStatus(StrEnum):
    NOT_LOADED = "NOT_LOADED"
    LOADED = "LOADED"
    ERROR = "ERROR"


class ModelMetadata(BaseModel):
    name: str
    version: str
    runtime: str
    input_shape: list[int]
    output_shape: list[int]
    dtype: str
    sampling_rate: float
    window_size: int
    stride: int
    feature_names: list[str]
    normalization: dict[str, Any]
    output_units: str
    frame: str
    angle_convention: str | None = None
    description: str = ""


class ModelPrediction(BaseModel):
    value: float
    confidence: float
    timestamp_ns: int
    latency_ms: float
    model_version: str


class BaseModelAdapter(ABC):
    status: ModelStatus = ModelStatus.NOT_LOADED

    @abstractmethod
    def predict(self, window: Any) -> ModelPrediction | None:
        """Return a model prediction when a validated model is loaded."""


class VelocityModelAdapter(BaseModelAdapter):
    def predict(self, window: Any) -> ModelPrediction | None:
        return None


class YawModelAdapter(BaseModelAdapter):
    def predict(self, window: Any) -> ModelPrediction | None:
        return None


class ModelManager:
    def __init__(self, auto_load: bool = False) -> None:
        self.velocity = VelocityModelAdapter()
        self.yaw = YawModelAdapter()

    def statuses(self) -> dict[str, str]:
        return {"velocity": self.velocity.status.value, "yaw": self.yaw.status.value}

