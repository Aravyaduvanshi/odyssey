# Edge Deployment

The edge engine can run locally:

```bash
python3 -m uvicorn edge.api.main:app --host 0.0.0.0 --port 8000
```

Deployment modes:

- local development
- edge device on LAN
- remote browser connected to local edge device

The real-time navigation path is designed to run without cloud connectivity.

