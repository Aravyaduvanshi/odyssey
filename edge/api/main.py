
#FASTAPI ENTRY POINT 

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from edge.events.bus import EventBus
from edge.navigation.engine import NavigationEngine


class CommandResult(BaseModel):
    ok: bool
    message: str


event_bus = EventBus()
engine = NavigationEngine(event_bus=event_bus)

app = FastAPI(title="OdysseyV1 Edge Engine", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
async def root() -> dict[str, str]:
    return {
        "application": "OdysseyV1",
        "service": "Edge Navigation Engine",
        "status": "online",
        "version": "1.0.0",
    }


@app.get("/health")
async def health() -> dict[str, object]:
    model_status = engine.models.statuses()
    return {
        "status": "healthy",
        "edge_engine": "online",
        "sensor_provider": "MOCK" if engine.provider.source.value == "SYNTHETIC" else engine.provider.source.value,
        "navigation": "READY",
        "models": model_status,
        "python": "3.12",
        "offline_capable": True,
    }


@app.get("/status")
async def status() -> dict[str, object]:
    return {
        "running": engine.running,
        "gnss_blackout": engine.gnss_blackout,
        "sensor_source": engine.provider.source.value,
        "phase": "PHASE_1",
        "phase_2_foundation": True,
        "calibration": engine.calibration_state(),
        "models": engine.models.statuses(),
        "last_state": engine.last_state().model_dump() if engine.last_state() else None,
    }


@app.get("/models")
async def models() -> dict[str, object]:
    return {
        "velocity": {
            "status": engine.models.velocity.status.value,
            "path": "models/velocity/S2_velocity_TCN_uncertainty_v2.keras",
            "latency_ms": round(engine.models.velocity.last_inference_latency_ms, 2),
        },
        "yaw": {
            "status": engine.models.mountnet.status.value,
            "path": "models/yaw/mountnet_v2_baseline.keras",
            "latency_ms": round(engine.models.mountnet.last_inference_latency_ms, 2),
        },
    }


@app.get("/configuration")
async def configuration() -> dict[str, object]:
    return {
        "sensor_rate_hz": 100,
        "publish_rate_hz": 10,
        "navigation_frame": "ENU",
        "data_label": "SYNTHETIC",
        "environment": "simulation",
        "gravity_mps2": 9.80665,
    }


@app.post("/models/upload")
async def model_upload() -> CommandResult:
    return CommandResult(ok=False, message="Model upload validation is reserved for Phase 7/8.")


@app.post("/models/activate")
async def model_activate() -> CommandResult:
    return CommandResult(ok=False, message="No model activated because no validated model is installed.")


@app.post("/session/start")
async def session_start() -> CommandResult:
    engine.start()
    return CommandResult(ok=True, message="Navigation session started.")


@app.post("/session/stop")
async def session_stop() -> CommandResult:
    engine.stop()
    return CommandResult(ok=True, message="Navigation session stopped.")


@app.post("/calibration/reset")
async def calibration_reset() -> CommandResult:
    engine.reset_calibration()
    return CommandResult(ok=True, message="Calibration reset to initial state.")


@app.post("/gnss/blackout")
async def gnss_blackout() -> CommandResult:
    engine.set_gnss_blackout(True)
    return CommandResult(ok=True, message="GNSS blackout simulation enabled.")


@app.post("/gnss/restore")
async def gnss_restore() -> CommandResult:
    engine.set_gnss_blackout(False)
    return CommandResult(ok=True, message="GNSS restored.")


@app.post("/replay/start")
async def replay_start() -> CommandResult:
    return CommandResult(ok=False, message="Replay is reserved for Phase 14.")


@app.post("/replay/pause")
async def replay_pause() -> CommandResult:
    return CommandResult(ok=False, message="Replay is reserved for Phase 14.")


@app.websocket("/ws/navigation")
async def ws_navigation(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        async for state in engine.states():
            await websocket.send_json(state.model_dump())
    except WebSocketDisconnect:
        return


@app.websocket("/ws/sensors")
async def ws_sensors(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        async for state in engine.states():
            await websocket.send_json(
                {
                    "timestamp": state.timestamp,
                    "sensor_source": state.sensor_source,
                    "scenario_phase": state.scenario_phase,
                    "raw_imu": state.raw_imu,
                    "calibrated_imu": state.calibrated_imu,
                    "vehicle_frame_imu": state.vehicle_frame_imu,
                    "rates": state.rates,
                }
            )
    except WebSocketDisconnect:
        return


@app.websocket("/ws/calibration")
async def ws_calibration(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        async for state in engine.states():
            await websocket.send_json(state.calibration)
    except WebSocketDisconnect:
        return


@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = await event_bus.subscribe()
    try:
        for event in event_bus.history():
            await websocket.send_json(event.model_dump())
        while True:
            event = await queue.get()
            await websocket.send_json(event.model_dump())
    except WebSocketDisconnect:
        return
    finally:
        event_bus.unsubscribe(queue)
