# CardiaScreen

Binary screening API for the UCI Heart Disease cohorts. Coursework for **AIMLCZG523 Assignment 01**. The response is a risk score for this assignment. It is not a diagnosis and not a medical device.

The report cover names Ankit Kumar Srivastava, BITS ID 2025AE05293.

## Result

Logistic regression (`C=0.1`) won a three-model comparison on training-split ROC-AUC. The holdout was not used to choose it.

| | |
| --- | --- |
| Rows (Cleveland, Hungary, Switzerland, VA) | 920 |
| Positive rate (`num > 0`) | 0.553 |
| Training / holdout | 736 / 184, seed 42 |
| CV ROC-AUC | 0.895 ± 0.013 |
| Holdout ROC-AUC | 0.909 |
| Threshold | 0.18, max F2 on training out-of-fold scores |
| Holdout recall / precision at 0.18 | 0.980 / 0.694 |
| Holdout accuracy / precision / recall at 0.50 | 0.832 / 0.851 / 0.843 |

Random forest CV ROC-AUC was 0.881. HistGradientBoosting was 0.873. Full tables, including per-site slices and a Cleveland-only transport check, are in `reports/metrics.json`.

## Reproduce

Python 3.11. From this directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
$env:PYTHONPATH = "src"
python -m cardiascreen.data
python -m cardiascreen.eda --skip-download
python -m cardiascreen.train --skip-download
pytest -q
python -m cardiascreen.infer --input deploy/sample_request.json
```

`environment.yml` is the same stack for Conda. `requirements.txt` trains and tracks. `requirements-api.txt` is the smaller set installed in the serving image.

MLflow uses a local sqlite store because MLflow 3 does not treat a plain folder as a supported tracking backend:

```powershell
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

The home page is the GenAI view, which has no traces. The training runs and the Model Registry are here:

```powershell
start http://127.0.0.1:5000/#/experiments/1/runs?workflowType=machine_learning
start http://127.0.0.1:5000/#/models/cardiascreen-heart-risk
```

`mlruns/` is not in git. On a new machine, publish the committed model first. That creates the sqlite store, one run when the store is empty, and the registered model `cardiascreen-heart-risk` with the alias `champion`:

```powershell
$env:PYTHONPATH = "src"
python -m cardiascreen.tracking
```

## Prayogshala

The BITS virtual lab is either Windows or Red Hat. Book a slot from the elearn portal, My Virtual Labs, open the session, and sign in with the account that slot page shows. Clone this project and run the script for that operating system. Either script finds Python, a container engine, and Kubernetes. Anything missing is installed when the VM allows it. If a Kubernetes node is already Ready, the script loads `cardiascreen-api:local` into that cluster, applies `deploy/k8s`, waits for the rollout, and forwards the service to port 8080. On Windows it otherwise starts Docker Desktop Kubernetes when the VM has at least 8 GB, or a small minikube cluster from 4 GB up. On Red Hat it starts k3s, which does not need Docker Desktop, and falls back to minikube. Podman is used when Docker is not installed. The committed model is served. A new training run is not required.

```bat
git clone https://github.com/SriNeural/assignment-1-mlops.git
cd assignment-1-mlops
setup_prayogshala.bat
```

```bash
git clone https://github.com/SriNeural/assignment-1-mlops.git
cd assignment-1-mlops
chmod +x setup_prayogshala.sh
./setup_prayogshala.sh
```

`setup_prayogshala.bat --check` and `./setup_prayogshala.sh --check` only print what is already installed. The first full run can take a long time because the container engine and the Kubernetes images download. On Windows, approve the administrator prompt if it appears, and restart the lab machine if Docker says a restart is required. On Red Hat, the script asks for `sudo` when it installs packages or starts k3s.

The script also publishes the committed model into the local MLflow Model Registry before it opens the UI. The registry name is `cardiascreen-heart-risk` and the alias is `champion`. Running `setup_prayogshala.sh` again restarts the CardiaScreen pods, the port forward, Compose when that stack is in use, the Python API when that is the server, and MLflow. If Kubernetes is already up and you only need MLflow, skip the full script and run this from the project folder.

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m cardiascreen.tracking
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5000
```

```bash
PYTHONPATH=src .venv/bin/python -m cardiascreen.tracking
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5000
```

A VM under 8 GB that does get a cluster runs one replica, and Prometheus and Grafana stay off so the node keeps its memory. At 8 GB or more, Compose also starts the API, Prometheus, and Grafana. Under about 4 GB, with no Ready node, the local cluster is not started. If the container engine cannot start and no deployment is already Ready, the same file starts the API with Python instead: `scripts/run_lab.py`.

| Service | URL |
| --- | --- |
| Kubernetes API docs | http://127.0.0.1:8080/docs |
| Compose API docs, when Compose starts | http://127.0.0.1:8000/docs |
| Prometheus | http://127.0.0.1:9090 |
| Grafana | http://127.0.0.1:3000 (admin / admin) |
| MLflow training runs | http://127.0.0.1:5000/#/experiments/1/runs?workflowType=machine_learning |
| MLflow Model Registry | http://127.0.0.1:5000/#/models/cardiascreen-heart-risk |

The same API is already public at https://iamvastava-cardiascreen-api.hf.space if the lab cannot reach PyPI.

## API

```powershell
docker build -t cardiascreen-api:local .
docker run --rm -p 8000:8000 cardiascreen-api:local
```

Interactive docs: <http://127.0.0.1:8000/docs>

Chest-pain `cp` uses the original UCI codes 1–4 (typical, atypical, non-anginal, asymptomatic), not the 0–3 recoding from public CSV mirrors. `ca`, `thal`, `fbs`, and `chol` may be null. A zero cholesterol is treated as unknown.

```powershell
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data "@deploy/sample_request.json"
```

A local container run of the three samples returned:

- `deploy/sample_request.json`: disease signal, probability 0.988, band very high
- `deploy/sample_request_low_risk.json`: no disease signal, probability 0.031, band low
- `deploy/sample_request_screening.json` (no cholesterol, vessels, or thallium): disease signal, probability 0.949

`GET /health`, `GET /metadata`, and `GET /metrics` are the other routes. Request logs are JSON on stdout and in `logs/api.jsonl` when that directory is writable.

## Monitoring

```powershell
docker compose up --build
```

| Service | URL |
| --- | --- |
| API | http://127.0.0.1:8000 |
| Prometheus | http://127.0.0.1:9090 |
| Grafana | http://127.0.0.1:3000 (admin / admin, anonymous view is on) |

The Grafana board is provisioned from `monitoring/grafana/dashboards/api.json`. Prometheus scrapes `api:8000/metrics`.

## Deployment

Kubernetes manifests are in `deploy/k8s` (namespace, two-replica Deployment, probes, LoadBalancer Service, Ingress). Docker Desktop's kubectl context exists here, and the cluster API on port 6443 was not running, so the checked runtime is Compose. When the cluster is enabled:

```powershell
docker build -t cardiascreen-api:local .
kubectl apply -k deploy/k8s
kubectl port-forward svc/cardiascreen-api 8080:80 -n cardiascreen
```

The public API is https://iamvastava-cardiascreen-api.hf.space. `GET /health` and `POST /predict` return the same JSON as the local container. The Space page is https://huggingface.co/spaces/iamvastava/cardiascreen-api.

Hugging Face now requires a paid plan for a Docker Space on the free CPU tier. This account uses the free ZeroGPU allowance instead. `deploy/huggingface/gradio` is the Space that is live. `scripts/export_space.py` still builds the Docker folder for a machine that can host it. The Space sleeps when idle, so the first request after sleep is slow.

`.github/workflows/ci.yml` runs on every push and pull request. It lints, runs pytest, checks the pinned data version, retrains, and then decides whether that model replaces production. The image job copies the gated `heart_risk_model.joblib` into the build and calls `/health` and `/predict`. A failure stops the workflow, so a rejected model never becomes the image. The latest recorded run is https://github.com/SriNeural/assignment-1-mlops/actions/runs/39314946898. A short recording of that pipeline is `videos/pipeline_demo.webm`.

## Data versions and model handover

The four UCI files are pinned in `data/versions/1.0.0.json`. `data/versions/CURRENT` names the pin that training is allowed to use. `python -m cardiascreen.versioning check` hashes the raw files and fails if any byte differs. After you have reviewed a real change, `python -m cardiascreen.versioning stamp 1.1.0` writes the next pin. Do not train on an unpinned change.

`models/registry.json` is the production record. After training, `python -m cardiascreen.release promote` compares the new holdout fingerprint with that record:

| Result | What happens |
| --- | --- |
| Unchanged | Same data version, model, parameters, and threshold, with ROC-AUC and recall inside 0.005. The version stays put and nothing is committed. |
| Promote | The gate holds: ROC-AUC does not drop by more than 0.005, and recall stays at least 0.90. A new data version bumps the minor number. Any other accepted change bumps the patch. On a push to `main`, Actions commits the model, the registry, and `reports/metrics.json`. |
| Reject | The holdout missed the gate. The command exits non-zero, the quality job fails, and the image job does not run. The previous release stays in git. |

Pull requests run the same gate and do not push. The bot commit message contains `[skip ci]`, and a `GITHUB_TOKEN` push does not start another workflow.

## Layout

| Path | Role |
| --- | --- |
| `src/cardiascreen/` | Download, cleaning, features, training, scoring, API |
| `tests/` | Unit tests for data, features, metrics, and HTTP |
| `notebooks/01_eda_and_inference.ipynb` | EDA narrative and sample scores |
| `data/raw/`, `data/processed/` | UCI files and the cleaned combined CSV |
| `data/versions/` | SHA256 pin for the raw files and the `CURRENT` pointer |
| `models/heart_risk_model.joblib` | Pipeline, threshold, and metric snapshot |
| `models/registry.json` | Production version and the release history |
| `reports/` | Figures, `metrics.json`, and the written report |
| `.github/workflows/ci.yml` | Lint, test, data pin, train, promote or reject, image smoke |
| `deploy/k8s/`, `deploy/huggingface/` | Cluster manifests and the Space card |
| `setup_prayogshala.bat` | Windows lab setup: Python, Docker, Kubernetes, Compose, MLflow |
| `scripts/run_lab.py` | Serve the committed model when Docker is not available |
| `docs/DEMO_SCRIPT.md` | Shot list for the required screen recording |
| `screenshots/` | API, Prometheus, Grafana, MLflow, and Actions captures |
| `videos/pipeline_demo.webm` | Short recording of the running pipeline |

## Data

Janosi, A., Steinbrunn, W., Pfisterer, M., and Detrano, R. (1988). Heart Disease. UCI Machine Learning Repository. <https://doi.org/10.24432/C52P4X>.

The label is binary: `num == 0` is no disease, and `num` of 1–4 is disease present. Site name is not a model feature. Missing cholesterol is partly a proxy for the Switzerland file, which stored every cholesterol value as zero. That limit is discussed in the report.

## Recording

Follow `docs/DEMO_SCRIPT.md` and attach the video to the submission. The recording has to be captured on the machine that runs the stack.
