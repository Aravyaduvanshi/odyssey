# Demo Guide

1. Start the backend with `python3 -m uvicorn edge.api.main:app --host 127.0.0.1 --port 8000`.
2. Connect a WebSocket client to `ws://127.0.0.1:8000/ws/navigation`.
3. Start the frontend with `npm run dev` inside `frontend`.
4. Use the satellite button to enable GNSS blackout.
5. Use the restore button to return to synthetic GNSS mode.

All displayed sensor values in Phase 1 are synthetic.

