   
#DEFINES SENSORSOURCE AND SENSORSAMPLE


from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SensorSource(StrEnum):
    SYNTHETIC = "SYNTHETIC"
    SMARTPHONE = "SMARTPHONE"
    IO_VNBD = "IO_VNBD"
    CSV = "CSV"
    REPLAY = "REPLAY"
    EXTERNAL_IMU = "EXTERNAL_IMU"
    FOG_IMU = "FOG_IMU"


class SensorSample(BaseModel):
    timestamp_ns: int = Field(..., ge=0)
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    mx: Optional[float] = None
    my: Optional[float] = None
    mz: Optional[float] = None
    latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    altitude: Optional[float] = None
    gnss_velocity: Optional[float] = Field(default=None, ge=0.0)
    gnss_heading: Optional[float] = None
    gnss_accuracy: Optional[float] = Field(default=None, ge=0.0)
    gnss_available: bool = False
    gnss_fix_id: Optional[str] = None
    sensor_source: SensorSource
    truth_local_x: Optional[float] = None
    truth_local_y: Optional[float] = None
    truth_yaw_rad: Optional[float] = None
    scenario_phase: Optional[str] = None

    @field_validator("ax", "ay", "az", "gx", "gy", "gz")
    @classmethod
    def finite_required(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("required inertial values must be finite")
        return value
