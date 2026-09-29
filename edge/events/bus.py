import asyncio
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class EventType(StrEnum):
    SYSTEM_STARTED = "SYSTEM_STARTED"
    SENSOR_CONNECTED = "SENSOR_CONNECTED"
    CALIBRATION_STARTED = "CALIBRATION_STARTED"
    CALIBRATION_COMPLETE = "CALIBRATION_COMPLETE"
    GNSS_LOST = "GNSS_LOST"
    GNSS_DEGRADED = "GNSS_DEGRADED"
    GNSS_RECOVERED = "GNSS_RECOVERED"
    MODEL_LOADED = "MODEL_LOADED"
    MODEL_ERROR = "MODEL_ERROR"
    MODEL_FAILED = "MODEL_FAILED"
    POTHOLE_DETECTED = "POTHOLE_DETECTED"
    PHONE_MOVEMENT = "PHONE_MOVEMENT"
    MAP_MATCH_CHANGED = "MAP_MATCH_CHANGED"
    NAVIGATION_MODE_CHANGED = "NAVIGATION_MODE_CHANGED"
    SESSION_STARTED = "SESSION_STARTED"
    SESSION_STOPPED = "SESSION_STOPPED"
    ODOMETRY_FUSED = "ODOMETRY_FUSED"
    ANOMALY_DETECTED = "ANOMALY_DETECTED"


class Event(BaseModel):
    type: EventType
    message: str
    timestamp: str
    payload: dict[str, Any] = {}


class EventBus:
    def __init__(self) -> None:
        self._history: list[Event] = []
        self._subscribers: set[asyncio.Queue[Event]] = set()

    def publish(self, event_type: EventType, message: str, payload: dict[str, Any] | None = None) -> Event:
        event = Event(
            type=event_type,
            message=message,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload=payload or {},
        )
        self._history.append(event)
        for subscriber in self._subscribers:
            subscriber.put_nowait(event)
        return event

    def history(self) -> list[Event]:
        return list(self._history[-200:])

    async def subscribe(self) -> asyncio.Queue[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        self._subscribers.discard(queue)
