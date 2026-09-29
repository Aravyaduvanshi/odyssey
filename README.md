# OdysseyV1

OdysseyV1 is an uncertainty-aware AI-assisted edge navigation engine for GNSS-denied environments.

---

## Target Architecture

```text
                WEB UI
                  │
             WebSocket
                  │
                  ▼
            FASTAPI EDGE
                  │
                  ▼
          SensorProvider
                  │
                  ▼
          TimeSynchronizer
                  │
                  ▼
          CalibrationEngine
                  │
                  ▼
          VehicleFrameTransform
                  │
          ┌───────┴────────┐
          ▼                ▼
    VelocityAdapter   YawAdapter
      NOT LOADED       NOT LOADED
          │                │
          └───────┬────────┘
                  ▼
           NavigationState
                  │
                  ▼
               WEB UI
```

---

## Python Environment & Setup

- **Standard Python Version**: `3.12` (documented in `.python-version`)
- Install the repository in editable development mode from the root:

```bash
# Recommended Python 3.12
pip3.12 install -e .
```

This ensures `edge` is discoverable as a first-class Python module across the workspace, VS Code, and all tests without ad-hoc path hacks.

---

## Development Workflow

### Terminal 1: Edge Navigation Engine (FastAPI)
Run from the repository root:

```bash
python3 -m uvicorn edge.api.main:app --host 127.0.0.1 --port 8000
```

Verify endpoints:
- `GET http://127.0.0.1:8000/` → Health & status
- `GET http://127.0.0.1:8000/health` → Structured subsystem health
- `WS  ws://127.0.0.1:8000/ws/navigation` → 10 Hz NavigationState stream
- `WS  ws://127.0.0.1:8000/ws/events` → Edge lifecycle event stream

### Terminal 2: Web Navigation Console (Next.js)
Run from `frontend/` or repository root:

```bash
npm --prefix frontend run dev
```

Open `http://127.0.0.1:3000` to access:
- `/` — Automotive Tactical Mission Control Dashboard
- `/sensors` — High-Frequency Sensor Diagnostics & Strip Charts
- `/calibration` — Automatic Phone → Vehicle 6-Stage Alignment

Configuration:
- `frontend/.env.local` sets `NEXT_PUBLIC_EDGE_URL=http://127.0.0.1:8000`.

---

## Docker Deployment

To launch the unified edge engine and frontend via Docker Compose:

```bash
docker compose up --build
```

---

## Testing

### Backend & End-to-End Tests
```bash
python3 -m unittest discover tests
```

### Frontend Tests & Type Validation
```bash
npm --prefix frontend test
npm --prefix frontend run build
```
