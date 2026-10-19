# Demo recording

A 50-second recording of the running pipeline is in `videos/pipeline_demo.webm`. It shows the green GitHub Actions run, the MLflow experiment, a live `POST /predict`, Prometheus, and Grafana. It has no narration. Record over it with your own voice if the course wants you on the recording.

The shot list below is what that file follows. `docker compose up -d` has to be healthy first.

1. **Data and training (40 seconds).** Show `data/raw` and run `python -m cardiascreen.train --skip-download`. Point at the printed winner, the threshold, and the holdout ROC-AUC.
2. **Tracking (30 seconds).** Open `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` and show the nested runs, the metrics, and one saved figure.
3. **API (50 seconds).** Open `http://127.0.0.1:8000/docs`. Send `deploy/sample_request.json`, then `deploy/sample_request_screening.json` with the invasive fields left null. Show the probability, the label, and the disclaimer.
4. **Monitoring (40 seconds).** Open `http://127.0.0.1:9090/targets` and show the CardiaScreen target as UP. Open Grafana at `http://127.0.0.1:3000` (admin / admin, or the anonymous viewer) and show the CardiaScreen API dashboard after a few predictions.
5. **CI (20 seconds).** Show `.github/workflows/ci.yml` and, after the GitHub push, the green lint, test, retrain, and image-smoke jobs.

Say explicitly that the response is a screening score for the assignment, not a diagnosis.
