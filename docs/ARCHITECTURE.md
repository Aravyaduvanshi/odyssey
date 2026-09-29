# Architecture

OdysseyV1 is split into three hard boundaries:

- Frontend: visualization and high-level controls only.
- Edge API: FastAPI HTTP/WebSocket communication.
- Edge navigation engine: sensor abstraction, future calibration, INS, EKF, trust, motion, map, replay, logging, and evaluation modules.

Phase 1 implements the minimum live path:

```text
MockSensorProvider
  -> SensorSample
  -> NavigationEngine
  -> NavigationState
  -> FastAPI WebSocket
  -> Next.js dashboard
```

The synthetic provider is intentionally labeled `SYNTHETIC`. AI values are `null` and confidence is `0.0` until real model adapters are added.

