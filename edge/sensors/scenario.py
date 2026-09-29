
#This is your current simulation engine.
#This is why your dashboard currently has a trajectory to display.

import math
import random
from dataclasses import dataclass

from edge.navigation.models import MotionState
from edge.sensors.models import SensorSample, SensorSource

EARTH_RADIUS_M = 6_378_137.0
GRAVITY_MPS2 = 9.80665


@dataclass(frozen=True)
class VehicleTruth:
    timestamp_ns: int
    phase: str
    local_x: float
    local_y: float
    yaw_rad: float
    velocity_mps: float
    acceleration_mps2: float
    yaw_rate_rps: float
    motion_state: MotionState


class MockVehicleScenario:
    """Deterministic seven-phase vehicle motion scenario for simulation only."""

    def __init__(
        self,
        rate_hz: float = 100.0,
        origin_latitude: float = 12.9716,
        origin_longitude: float = 77.5946,
        origin_altitude: float = 920.0,
        seed: int = 7,
        noise_std: float = 0.0,
    ) -> None:
        self.rate_hz = rate_hz
        self.origin_latitude = origin_latitude
        self.origin_longitude = origin_longitude
        self.origin_altitude = origin_altitude
        self.noise_std = noise_std
        self._rng = random.Random(seed)
        self._dt = 1.0 / rate_hz
        self._truth: list[VehicleTruth] = []
        self._build()

    @property
    def length(self) -> int:
        return len(self._truth)

    def truth_at(self, index: int, start_ns: int) -> VehicleTruth:
        base = self._truth[index % len(self._truth)]
        cycle = index // len(self._truth)
        cycle_offset_ns = int(cycle * len(self._truth) * self._dt * 1_000_000_000)
        return VehicleTruth(
            timestamp_ns=start_ns + base.timestamp_ns + cycle_offset_ns,
            phase=base.phase,
            local_x=base.local_x,
            local_y=base.local_y,
            yaw_rad=base.yaw_rad,
            velocity_mps=base.velocity_mps,
            acceleration_mps2=base.acceleration_mps2,
            yaw_rate_rps=base.yaw_rate_rps,
            motion_state=base.motion_state,
        )

    def sample_at(self, index: int, start_ns: int) -> SensorSample:
        truth = self.truth_at(index, start_ns)
        latitude, longitude = self._enu_to_geodetic(truth.local_x, truth.local_y)
        ax = truth.acceleration_mps2 + self._noise()
        ay = truth.velocity_mps * truth.yaw_rate_rps + self._noise()
        az = GRAVITY_MPS2 + self._noise()
        gx = self._noise(scale=0.02)
        gy = self._noise(scale=0.02)
        gz = truth.yaw_rate_rps + self._noise(scale=0.02)
        magnetic_heading = truth.yaw_rad
        return SensorSample(
            timestamp_ns=truth.timestamp_ns,
            ax=ax,
            ay=ay,
            az=az,
            gx=gx,
            gy=gy,
            gz=gz,
            mx=math.cos(magnetic_heading),
            my=math.sin(magnetic_heading),
            mz=0.18,
            latitude=latitude,
            longitude=longitude,
            altitude=self.origin_altitude,
            gnss_velocity=max(0.0, truth.velocity_mps),
            gnss_heading=truth.yaw_rad,
            gnss_accuracy=3.0,
            gnss_available=True,
            sensor_source=SensorSource.SYNTHETIC,
            truth_local_x=truth.local_x,
            truth_local_y=truth.local_y,
            truth_yaw_rad=truth.yaw_rad,
            scenario_phase=truth.phase,
        )

    def _build(self) -> None:
        x = 0.0
        y = 0.0
        yaw = math.radians(90.0)
        velocity = 0.0
        timestamp_ns = 0
        phases = [
            ("stationary", 4.0, 0.0, 0.0, MotionState.STATIONARY),
            ("acceleration", 6.0, 1.35, 0.0, MotionState.ACCELERATION),
            ("straight", 8.0, 0.0, 0.0, MotionState.NORMAL_DRIVING),
            ("gentle_left_turn", 8.0, 0.0, math.radians(4.0), MotionState.TURNING),
            ("constant_speed", 8.0, 0.0, 0.0, MotionState.NORMAL_DRIVING),
            ("gentle_right_turn", 8.0, 0.0, math.radians(-4.0), MotionState.TURNING),
            ("braking", 6.0, -1.35, 0.0, MotionState.BRAKING),
        ]
        for phase, duration_s, acceleration, yaw_rate, motion_state in phases:
            steps = int(duration_s * self.rate_hz)
            for _ in range(steps):
                self._truth.append(
                    VehicleTruth(
                        timestamp_ns=timestamp_ns,
                        phase=phase,
                        local_x=x,
                        local_y=y,
                        yaw_rad=yaw,
                        velocity_mps=max(0.0, velocity),
                        acceleration_mps2=acceleration if velocity > 0.0 or acceleration > 0.0 else 0.0,
                        yaw_rate_rps=yaw_rate if velocity > 0.5 else 0.0,
                        motion_state=motion_state,
                    )
                )
                yaw += yaw_rate * self._dt
                velocity = max(0.0, velocity + acceleration * self._dt)
                x += velocity * math.sin(yaw) * self._dt
                y += velocity * math.cos(yaw) * self._dt
                timestamp_ns += int(self._dt * 1_000_000_000)

    def _enu_to_geodetic(self, east_m: float, north_m: float) -> tuple[float, float]:
        lat = self.origin_latitude + math.degrees(north_m / EARTH_RADIUS_M)
        lon = self.origin_longitude + math.degrees(east_m / (EARTH_RADIUS_M * math.cos(math.radians(self.origin_latitude))))
        return lat, lon

    def _noise(self, scale: float = 1.0) -> float:
        if self.noise_std <= 0.0:
            return 0.0
        return self._rng.gauss(0.0, self.noise_std * scale)

