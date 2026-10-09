# Screenshots

These files are captures of the local compose stack on 9 October 2026. They are not mockups.

| File | What it shows |
| --- | --- |
| `api_docs.png` | FastAPI docs at http://127.0.0.1:8000/docs |
| `api_health.png` | `GET /health`: logistic regression 1.0.0, threshold 0.18 |
| `metrics_endpoint.png` | `GET /metrics`, including 8 `disease_signal` / `very_high` predictions |
| `prometheus_targets.png` | Prometheus target `http://api:8000/metrics` is UP |
| `grafana_board.png` | Provisioned Grafana board after those prediction requests |

GitHub Actions has no screenshot yet. The workflow runs after the repository is pushed. Record that screen from the Actions tab and add it here.
