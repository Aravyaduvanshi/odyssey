
#This is your timestamp-quality layer.

from pydantic import BaseModel

from edge.sensors.models import SensorSample


class TimeSyncStatus(BaseModel):
    last_timestamp_ns: int | None = None
    duplicate_timestamps: int = 0
    timestamp_reversals: int = 0
    dropped_samples: int = 0
    estimated_rate_hz: float | None = None


class TimeSynchronizer:
    def __init__(self, target_rate_hz: float = 100.0, drop_gap_factor: float = 2.5) -> None:
        self.target_rate_hz = target_rate_hz
        self.drop_gap_factor = drop_gap_factor
        self.status = TimeSyncStatus()
        self._last_timestamp_ns: int | None = None

    def accept(self, sample: SensorSample) -> bool:
        if self._last_timestamp_ns is None:
            self._last_timestamp_ns = sample.timestamp_ns
            self.status.last_timestamp_ns = sample.timestamp_ns
            return True
        dt_ns = sample.timestamp_ns - self._last_timestamp_ns
        if dt_ns == 0:
            self.status.duplicate_timestamps += 1
            return False
        if dt_ns < 0:
            self.status.timestamp_reversals += 1
            return False
        expected_dt_ns = 1_000_000_000 / self.target_rate_hz
        if dt_ns > expected_dt_ns * self.drop_gap_factor:
            self.status.dropped_samples += max(1, round(dt_ns / expected_dt_ns) - 1)
        self.status.estimated_rate_hz = 1_000_000_000 / dt_ns
        self._last_timestamp_ns = sample.timestamp_ns
        self.status.last_timestamp_ns = sample.timestamp_ns
        return True

