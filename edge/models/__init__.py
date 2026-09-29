from edge.models.base import (
    BaseModelAdapter,
    ModelStatus,
    VelocityPrediction,
    YawPrediction,
)
from edge.models.manager import ModelManager
from edge.models.velocity.adapter import VelocityTCNAdapter
from edge.models.yaw.adapter import MountNetAdapter

# Backwards compatibility aliases
VelocityModelAdapter = VelocityTCNAdapter
YawModelAdapter = MountNetAdapter

__all__ = [
    "BaseModelAdapter",
    "ModelStatus",
    "YawPrediction",
    "VelocityPrediction",
    "ModelManager",
    "MountNetAdapter",
    "VelocityTCNAdapter",
    "VelocityModelAdapter",
    "YawModelAdapter",
]
