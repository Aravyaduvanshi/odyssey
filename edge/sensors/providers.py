
#This implements the actual sensor sources.

from abc import ABC, abstractmethod
import asyncio
import time
from collections.abc import AsyncIterator
from typing import Optional

from edge.sensors.models import SensorSample, SensorSource
from edge.sensors.scenario import MockVehicleScenario


class SensorProvider(ABC):
    source: SensorSource

    @abstractmethod
    async def samples(self) -> AsyncIterator[SensorSample]:
        """Yield unified SI-unit sensor samples."""


class MockSensorProvider(SensorProvider):
    source = SensorSource.SYNTHETIC

    def __init__(self, rate_hz: float = 100.0, seed: int = 7, noise_std: float = 0.0) -> None:
        self.rate_hz = rate_hz
        self._start_ns = time.monotonic_ns()
        self.scenario = MockVehicleScenario(rate_hz=rate_hz, seed=seed, noise_std=noise_std)

    async def samples(self) -> AsyncIterator[SensorSample]:
        index = 0
        while True:
            yield self.scenario.sample_at(index, self._start_ns)
            index += 1
            await asyncio.sleep(1.0 / self.rate_hz)


class SmartphoneSensorProvider(SensorProvider):
    source = SensorSource.SMARTPHONE

    async def samples(self) -> AsyncIterator[SensorSample]:
        raise NotImplementedError("Smartphone sensor ingestion is scheduled for Phase 2.")


class IOVNBDProvider(SensorProvider):
    source = SensorSource.IO_VNBD

    def __init__(
        self,
        sequence_name: str = "Vfa01",
        dataset_root: Optional[object] = None,
        max_samples: Optional[int] = None,
        playback_rate_factor: float = 0.0,
    ) -> None:
        from edge.evaluation.replay import IOVNBDLoader
        self.sequence_name = sequence_name
        self.loader = IOVNBDLoader(dataset_root=dataset_root)
        self.max_samples = max_samples
        self.playback_rate_factor = playback_rate_factor
        self._samples, self.metadata = self.loader.load_sequence(sequence_name, max_samples=max_samples)
        self.rate_hz = 10.0

    async def samples(self) -> AsyncIterator[SensorSample]:
        for sample in self._samples:
            yield sample
            if self.playback_rate_factor > 0:
                await asyncio.sleep(0.1 / self.playback_rate_factor)


class CSVProvider(SensorProvider):
    source = SensorSource.CSV

    async def samples(self) -> AsyncIterator[SensorSample]:
        raise NotImplementedError("CSV replay is scheduled for Phase 2.")


class ReplayProvider(SensorProvider):
    source = SensorSource.REPLAY

    async def samples(self) -> AsyncIterator[SensorSample]:
        raise NotImplementedError("Replay provider is scheduled for Phase 14.")


class ExternalIMUProvider(SensorProvider):
    source = SensorSource.EXTERNAL_IMU

    async def samples(self) -> AsyncIterator[SensorSample]:
        raise NotImplementedError("External IMU support is scheduled for Phase 17.")
