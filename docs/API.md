# API

## HTTP

- `GET /health`
- `GET /status`
- `GET /models`
- `GET /configuration`
- `POST /models/upload`
- `POST /models/activate`
- `POST /session/start`
- `POST /session/stop`
- `POST /gnss/blackout`
- `POST /gnss/restore`
- `POST /replay/start`
- `POST /replay/pause`

## WebSocket

- `/ws/navigation`: emits `NavigationState` at 10 Hz.
- `/ws/sensors`: reserved for Phase 2.
- `/ws/events`: emits backend events.

Frontend commands remain high-level. It must not manipulate EKF matrices, covariance, bias state, or map matcher internals.

