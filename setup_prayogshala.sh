#!/usr/bin/env bash
# CardiaScreen setup for a Prayogshala Red Hat machine.
# Same outcome as setup_prayogshala.bat: serve the committed model on Kubernetes.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CHECK_ONLY=0
DOCKER_OK=0
CONTAINER=""
KUBECTL_OK=0
CLUSTER_OK=0
COMPOSE_OK=0
PYTHON_OK=0
K8S_MODE=""
K8S_CONTEXT=""
KIND_NAME="kind"
K3D_NAME=""
RAM_GB=0
PY=""
WAIT_URL=""

if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
fi

say() { printf '%s\n' "$*"; }

as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    return 1
  fi
}

http_ok() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 5 "$url" >/dev/null 2>&1
    return
  fi
  "$ROOT/.venv/bin/python" -c "import urllib.request,sys; urllib.request.urlopen(sys.argv[1], timeout=5)" "$url" >/dev/null 2>&1
}

stop_port() {
  local port="$1"
  local pidfile="${2:-}"
  local pid listeners
  if [[ -n "$pidfile" && -f "$pidfile" ]]; then
    pid="$(tr -d '[:space:]' < "$pidfile" || true)"
    if [[ "$pid" =~ ^[0-9]+$ ]]; then
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi
  if command -v ss >/dev/null 2>&1; then
    listeners="$(ss -lptn "sport = :${port}" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true)"
    for pid in $listeners; do
      kill "$pid" 2>/dev/null || as_root kill "$pid" 2>/dev/null || true
    done
  elif command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" >/dev/null 2>&1 || as_root fuser -k "${port}/tcp" >/dev/null 2>&1 || true
  fi
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if ! command -v ss >/dev/null 2>&1; then
      return 0
    fi
    if ! ss -lnt "sport = :${port}" 2>/dev/null | grep -q ":${port}"; then
      return 0
    fi
    sleep 1
  done
}

node_ready() {
  kubectl get nodes --request-timeout=8s --no-headers 2>/dev/null | grep -E '[[:space:]]Ready[[:space:]]' >/dev/null
}

fail() {
  say ""
  say "Setup stopped. Read the message above."
  exit 1
}

python_ok() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1
}

ensure_python() {
  local candidate
  for candidate in python3.11 python3.12 python3.13 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && python_ok "$candidate"; then
      PY="$(command -v "$candidate")"
      say "Python: $PY"
      return 0
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    say "Creating the conda env cardiascreen with Python 3.11"
    if conda create -y -n cardiascreen python=3.11; then
      PY="$(conda run -n cardiascreen python -c 'import sys; print(sys.executable)')"
      if [[ -n "$PY" ]] && python_ok "$PY"; then
        say "Python: $PY"
        return 0
      fi
    fi
  fi
  if command -v dnf >/dev/null 2>&1; then
    say "Installing Python 3.11 from the Red Hat repositories"
    as_root dnf install -y python3.11 python3.11-pip || as_root dnf install -y python3.12 python3.12-pip || true
    for candidate in python3.11 python3.12; do
      if command -v "$candidate" >/dev/null 2>&1 && python_ok "$candidate"; then
        PY="$(command -v "$candidate")"
        say "Python: $PY"
        return 0
      fi
    done
  fi
  say "Python 3.11 is not installed, and this machine could not install it."
  say "On Red Hat, run: sudo dnf install -y python3.11 python3.11-pip"
  return 1
}

install_packages() {
  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    say "Creating .venv"
    "$PY" -m venv "$ROOT/.venv" || return 1
  fi
  say "Installing the training, test, and API packages. This can take several minutes."
  "$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements-dev.txt"
}

verify_program() {
  say "Checking the data pin and the committed model"
  PYTHONPATH=src "$ROOT/.venv/bin/python" scripts/run_lab.py --check || return 1
  say "Running tests"
  PYTHONPATH=src "$ROOT/.venv/bin/python" -m pytest -q
}

ensure_container() {
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    CONTAINER=docker
    DOCKER_OK=1
    say "Docker is already running."
    return 0
  fi
  if command -v systemctl >/dev/null 2>&1 && as_root systemctl start docker >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1; then
      CONTAINER=docker
      DOCKER_OK=1
      say "Started the Docker service."
      return 0
    fi
  fi
  if command -v podman >/dev/null 2>&1 && podman info >/dev/null 2>&1; then
    CONTAINER=podman
    DOCKER_OK=1
    say "Podman is available. Red Hat will use it instead of Docker."
    return 0
  fi
  if command -v dnf >/dev/null 2>&1; then
    say "Installing Podman. It is the container engine in the Red Hat repositories."
    if as_root dnf install -y podman && podman info >/dev/null 2>&1; then
      CONTAINER=podman
      DOCKER_OK=1
      say "Podman is ready."
      return 0
    fi
  fi
  say "Docker and Podman are not available. A new local cluster cannot be created."
  DOCKER_OK=0
  return 1
}

ensure_kubectl() {
  export PATH="$HOME/.local/bin:$PATH"
  if command -v kubectl >/dev/null 2>&1; then
    say "kubectl is already installed."
    KUBECTL_OK=1
    return 0
  fi
  say "Installing kubectl"
  local version dest
  version="$(curl -fsSL https://dl.k8s.io/release/stable.txt 2>/dev/null || true)"
  if [[ -z "$version" ]]; then
    say "kubectl was not found, and the Kubernetes release page could not be reached."
    KUBECTL_OK=0
    return 1
  fi
  mkdir -p "$HOME/.local/bin"
  dest="$HOME/.local/bin/kubectl"
  if ! curl -fsSL -o "$dest" "https://dl.k8s.io/release/${version}/bin/linux/amd64/kubectl"; then
    say "The kubectl download failed."
    KUBECTL_OK=0
    return 1
  fi
  chmod 755 "$dest"
  if ! command -v kubectl >/dev/null 2>&1; then
    say "kubectl installed, but it is not on PATH yet. Open a new shell and run this file again."
    KUBECTL_OK=0
    return 1
  fi
  KUBECTL_OK=1
}

ram_gb() {
  local kb
  kb="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
  RAM_GB=$((kb / 1024 / 1024))
  say "RAM: ${RAM_GB} GB"
}

mark_cluster() {
  K8S_CONTEXT="$(kubectl config current-context 2>/dev/null || echo unknown)"
  say "Kubernetes context: ${K8S_CONTEXT}"
  K8S_MODE="existing"
  case "$K8S_CONTEXT" in
    *minikube*) K8S_MODE="minikube" ;;
    *docker-desktop*) K8S_MODE="desktop" ;;
    kind-*) K8S_MODE="kind"; KIND_NAME="${K8S_CONTEXT#kind-}" ;;
    kind) K8S_MODE="kind"; KIND_NAME="kind" ;;
    k3d-*) K8S_MODE="k3d"; K3D_NAME="${K8S_CONTEXT#k3d-}" ;;
  esac
  if [[ "$K8S_MODE" == "existing" ]] && command -v k3s >/dev/null 2>&1; then
    K8S_MODE="k3s"
  fi
  if [[ "$K8S_MODE" == "existing" ]] && command -v microk8s >/dev/null 2>&1; then
    K8S_MODE="microk8s"
  fi
  say "Cluster mode: ${K8S_MODE}"
}

use_k3s_kubeconfig() {
  local src="/etc/rancher/k3s/k3s.yaml"
  [[ -f "$src" ]] || return 1
  mkdir -p "$HOME/.kube"
  if [[ ! -r "$src" ]]; then
    as_root cp "$src" "$HOME/.kube/cardiascreen-k3s.yaml" || return 1
    as_root chown "$(id -u):$(id -g)" "$HOME/.kube/cardiascreen-k3s.yaml" || return 1
  else
    cp "$src" "$HOME/.kube/cardiascreen-k3s.yaml"
  fi
  chmod 600 "$HOME/.kube/cardiascreen-k3s.yaml"
  export KUBECONFIG="$HOME/.kube/cardiascreen-k3s.yaml"
}

ensure_cluster() {
  ram_gb
  if node_ready; then
    say "Using the Kubernetes cluster that is already Ready."
    CLUSTER_OK=1
    mark_cluster
    return 0
  fi
  if [[ -f /etc/rancher/k3s/k3s.yaml ]] && use_k3s_kubeconfig && node_ready; then
    say "Using the Kubernetes cluster that is already Ready."
    CLUSTER_OK=1
    mark_cluster
    return 0
  fi
  if [[ "$DOCKER_OK" != "1" ]]; then
    say "No Kubernetes node is Ready, and no container engine is running."
    CLUSTER_OK=0
    return 0
  fi
  if [[ "$RAM_GB" -lt 4 ]]; then
    say "This machine has ${RAM_GB} GB of RAM. Starting a local cluster needs about 4 GB."
    say "No Ready node was found, so Kubernetes was not started."
    CLUSTER_OK=0
    return 0
  fi
  say "Starting a local k3s cluster. k3s is the Kubernetes build that runs on Red Hat."
  if start_k3s; then
    return 0
  fi
  say "k3s did not start. Trying minikube."
  start_minikube
}

start_k3s() {
  if ! command -v k3s >/dev/null 2>&1; then
    if ! command -v curl >/dev/null 2>&1; then
      as_root dnf install -y curl || return 1
    fi
    say "Installing k3s. The first start downloads the cluster images."
    curl -sfL https://get.k3s.io | as_root sh -s - --write-kubeconfig-mode 644 || return 1
  fi
  if command -v systemctl >/dev/null 2>&1; then
    as_root systemctl enable --now k3s >/dev/null 2>&1 || true
  fi
  use_k3s_kubeconfig || return 1
  local i
  for i in $(seq 1 40); do
    if node_ready; then
      CLUSTER_OK=1
      mark_cluster
      return 0
    fi
    say "Waiting for a Kubernetes node... ${i} of 40"
    sleep 10
  done
  say "k3s did not become Ready."
  return 1
}

start_minikube() {
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v minikube >/dev/null 2>&1; then
    say "Installing minikube"
    mkdir -p "$HOME/.local/bin"
    curl -fsSL -o "$HOME/.local/bin/minikube" https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 || {
      say "minikube is not available, and no Kubernetes node is Ready."
      CLUSTER_OK=0
      return 0
    }
    chmod 755 "$HOME/.local/bin/minikube"
  fi
  local mem=1900 cpus=1 driver="docker"
  if [[ "$RAM_GB" -ge 8 ]]; then
    mem=3072
    cpus=2
  fi
  if [[ "$RAM_GB" -ge 16 ]]; then
    mem=4096
    cpus=2
  fi
  if [[ "$CONTAINER" == "podman" ]]; then
    driver="podman"
  fi
  say "Starting minikube with ${cpus} CPU and ${mem} MB."
  if ! minikube start --driver="$driver" --cpus="$cpus" --memory="$mem"; then
    say "minikube did not start."
    CLUSTER_OK=0
    return 0
  fi
  kubectl config use-context minikube >/dev/null 2>&1 || true
  CLUSTER_OK=1
  mark_cluster
}

load_image() {
  case "$K8S_MODE" in
    minikube)
      say "Loading the image into minikube."
      minikube image load cardiascreen-api:local
      ;;
    kind)
      say "Loading the image into kind cluster ${KIND_NAME}."
      kind load docker-image cardiascreen-api:local --name "$KIND_NAME"
      ;;
    k3d)
      say "Loading the image into k3d cluster ${K3D_NAME}."
      k3d image import cardiascreen-api:local -c "$K3D_NAME"
      ;;
    k3s)
      say "Loading the image into k3s."
      if ! "$CONTAINER" save docker.io/library/cardiascreen-api:local | as_root k3s ctr images import -; then
        "$CONTAINER" save cardiascreen-api:local | as_root k3s ctr images import -
      fi
      ;;
    microk8s)
      say "Loading the image into microk8s."
      "$CONTAINER" save docker.io/library/cardiascreen-api:local | as_root microk8s ctr images import -
      ;;
    desktop)
      say "Docker Desktop Kubernetes uses the local image cardiascreen-api:local."
      ;;
    *)
      say "Context ${K8S_CONTEXT} is not minikube, kind, k3d, k3s, or microk8s."
      say "deploy/k8s expects cardiascreen-api:local on the node. If the node cannot see it, the pod will not start."
      ;;
  esac
}

expose_k8s() {
  WAIT_URL="http://127.0.0.1:8080/health"
  say "Restarting the port forward to http://127.0.0.1:8080"
  mkdir -p "$ROOT/logs"
  stop_port 8080 "$ROOT/logs/port-forward.pid"
  nohup kubectl port-forward --address 127.0.0.1 svc/cardiascreen-api 8080:80 -n cardiascreen \
    >"$ROOT/logs/port-forward.log" 2>&1 &
  echo $! >"$ROOT/logs/port-forward.pid"
}

reuse_if_running() {
  say "Restarting the CardiaScreen pods."
  kubectl rollout restart deployment/cardiascreen-api -n cardiascreen || {
    say "No Ready cardiascreen-api deployment was found."
    return 1
  }
  kubectl rollout status deployment/cardiascreen-api -n cardiascreen --timeout=240s || {
    say "The deployment did not become Ready."
    return 1
  }
  expose_k8s
  WAIT_URL="http://127.0.0.1:8080/health"
  wait_url
}

apply_k8s() {
  [[ "$CLUSTER_OK" == "1" ]] || return 0
  if [[ "$DOCKER_OK" != "1" ]]; then
    say "The container engine is down. Checking for a deployment that is already Ready."
    reuse_if_running
    return
  fi
  say "Building cardiascreen-api:local for Kubernetes"
  "$CONTAINER" build -t cardiascreen-api:local . || {
    say "The image build failed. Kubernetes was not changed."
    return 1
  }
  "$CONTAINER" tag cardiascreen-api:local docker.io/library/cardiascreen-api:local >/dev/null 2>&1 || true
  load_image || {
    say "The image did not load into the cluster."
    return 1
  }
  say "Applying the Kubernetes manifests in deploy/k8s"
  kubectl apply -k deploy/k8s || {
    say "kubectl apply failed."
    return 1
  }
  if [[ "$RAM_GB" -lt 8 ]]; then
    say "This VM has ${RAM_GB} GB. Running one replica so the pod can schedule."
    kubectl scale deployment/cardiascreen-api -n cardiascreen --replicas=1
    kubectl set resources deployment/cardiascreen-api -n cardiascreen -c api \
      --requests=cpu=50m,memory=128Mi --limits=cpu=500m,memory=512Mi
  fi
  say "Restarting the CardiaScreen pods."
  kubectl rollout restart deployment/cardiascreen-api -n cardiascreen || {
    say "The deployment did not restart."
    return 1
  }
  say "Waiting until the deployment is Ready."
  if ! kubectl rollout status deployment/cardiascreen-api -n cardiascreen --timeout=240s; then
    say "The deployment did not become Ready."
    kubectl get pods,svc -n cardiascreen -o wide || true
    kubectl describe deployment/cardiascreen-api -n cardiascreen || true
    return 1
  fi
  expose_k8s
  WAIT_URL="http://127.0.0.1:8080/health"
  if ! wait_url; then
    say "The Kubernetes service did not answer on http://127.0.0.1:8080/health"
    kubectl get pods,svc -n cardiascreen -o wide || true
    return 1
  fi
  say "Kubernetes is serving http://127.0.0.1:8080/docs"
}

compose_cmd() {
  if [[ "$CONTAINER" == "docker" ]] && docker compose version >/dev/null 2>&1; then
    docker compose "$@"
    return
  fi
  if [[ "$CONTAINER" == "podman" ]] && podman compose version >/dev/null 2>&1; then
    podman compose "$@"
    return
  fi
  if command -v podman-compose >/dev/null 2>&1; then
    podman-compose "$@"
    return
  fi
  return 1
}

start_compose() {
  [[ "$DOCKER_OK" == "1" ]] || return 1
  if [[ "$CLUSTER_OK" == "1" && "$RAM_GB" -lt 8 ]]; then
    say "Kubernetes is running on ${RAM_GB} GB. Prometheus and Grafana stay off so the cluster keeps its memory."
    compose_cmd down >/dev/null 2>&1 || true
    COMPOSE_OK=0
    return 0
  fi
  say "Restarting the API, Prometheus, and Grafana."
  compose_cmd down >/dev/null 2>&1 || true
  compose_cmd up -d --build --force-recreate || return 1
  WAIT_URL="http://127.0.0.1:8000/health"
  wait_url || return 1
  COMPOSE_OK=1
}

wait_url() {
  local i
  for i in $(seq 1 40); do
    if http_ok "$WAIT_URL"; then
      return 0
    fi
    say "Waiting for ${WAIT_URL} ... ${i} of 40"
    sleep 3
  done
  return 1
}

smoke_at() {
  say "Scoring the three sample requests at $1"
  "$ROOT/.venv/bin/python" scripts/smoke_predict.py "$1" || say "The API is up, but a sample request failed."
}

serve_python() {
  say "Restarting the API on http://127.0.0.1:8000 without a container"
  mkdir -p "$ROOT/logs"
  stop_port 8000 "$ROOT/logs/api.pid"
  PYTHONPATH=src nohup "$ROOT/.venv/bin/python" scripts/run_lab.py >"$ROOT/logs/api.log" 2>&1 &
  echo $! >"$ROOT/logs/api.pid"
  PYTHON_OK=1
}

publish_tracking() {
  say "Publishing the committed model into the MLflow Model Registry"
  PYTHONPATH=src "$ROOT/.venv/bin/python" -m cardiascreen.tracking \
    || say "The model was not added to the MLflow registry. The UI can still open."
}

start_mlflow() {
  [[ -x "$ROOT/.venv/bin/mlflow" ]] || return 0
  say "Restarting MLflow."
  mkdir -p "$ROOT/logs"
  stop_port 5000 "$ROOT/logs/mlflow.pid"
  nohup "$ROOT/.venv/bin/mlflow" ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5000 \
    >"$ROOT/logs/mlflow.log" 2>&1 &
  echo $! >"$ROOT/logs/mlflow.pid"
}

summary() {
  say ""
  say "----------------------------------------"
  if [[ "$CLUSTER_OK" == "1" ]]; then
    say "Kubernetes is serving the API."
    say "  API docs     http://127.0.0.1:8080/docs"
    say "  Health       http://127.0.0.1:8080/health"
    say "  Namespace    cardiascreen"
    kubectl get pods,svc -n cardiascreen || true
  fi
  if [[ "$COMPOSE_OK" == "1" ]]; then
    say "Compose is serving the API, Prometheus, and Grafana."
    say "  API docs     http://127.0.0.1:8000/docs"
    say "  Health       http://127.0.0.1:8000/health"
    say "  Prometheus   http://127.0.0.1:9090"
    say "  Grafana      http://127.0.0.1:3000"
    say "  Grafana login is admin / admin"
  fi
  if [[ "$PYTHON_OK" == "1" ]]; then
    say "Python is serving the API because the container engine and Kubernetes did not stay up."
    say "  API docs     http://127.0.0.1:8000/docs"
    say "  Health       http://127.0.0.1:8000/health"
  fi
  if [[ "$CLUSTER_OK" == "0" && "$COMPOSE_OK" == "0" && "$PYTHON_OK" == "0" ]]; then
    say "Nothing is serving. Read the messages above."
  fi
  say "  MLflow runs     http://127.0.0.1:5000/#/experiments/1/runs?workflowType=machine_learning"
  say "  Model Registry  http://127.0.0.1:5000/#/models/cardiascreen-heart-risk"
  say "The public copy is https://iamvastava-cardiascreen-api.hf.space"
  say "A new training run is optional. The model already in git is the one being served."
  say "----------------------------------------"
}

report_python() {
  local candidate
  for candidate in python3.11 python3.12 python3.13 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && python_ok "$candidate"; then
      "$candidate" -c 'import sys; print(sys.version)'
      return 0
    fi
  done
  say "Python 3.11 was not found."
  return 1
}

status_only() {
  say "--- check only, nothing will be installed ---"
  say ""
  say "Python 3.11:"
  report_python || true
  say ""
  say "Container engine:"
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    say "Docker is running."
  elif command -v podman >/dev/null 2>&1 && podman info >/dev/null 2>&1; then
    say "Podman is running."
  else
    say "Docker and Podman are not running."
  fi
  say ""
  say "Kubernetes:"
  if node_ready; then
    kubectl get nodes --request-timeout=8s
  else
    say "Kubernetes cluster is not reachable."
  fi
  say ""
  ram_gb
}

main() {
  say ""
  say "CardiaScreen setup for Prayogshala (Red Hat)"
  say "Directory: $ROOT"
  say ""
  if [[ ! -f src/cardiascreen/service.py ]]; then
    say "This file must stay in the project folder, next to the src directory."
    fail
  fi
  if [[ ! -f models/heart_risk_model.joblib ]]; then
    say "models/heart_risk_model.joblib is missing. Clone the full repository and run this file again."
    fail
  fi
  if [[ "$CHECK_ONLY" == "1" ]]; then
    status_only
    exit 0
  fi
  ensure_python || fail
  install_packages || fail
  verify_program || fail
  ensure_container || true
  if ensure_kubectl; then
    ensure_cluster
    if [[ "$CLUSTER_OK" == "1" ]]; then
      apply_k8s || CLUSTER_OK=0
    fi
  else
    say "kubectl is not available. Kubernetes will not be used."
  fi
  if start_compose; then
    if [[ "$COMPOSE_OK" == "1" ]]; then
      smoke_at "http://127.0.0.1:8000"
    fi
  elif [[ "$CLUSTER_OK" != "1" ]]; then
    say "Compose did not stay up. Starting the API with Python."
    serve_python
  else
    say "Compose did not stay up. The Kubernetes API is still running."
  fi
  if [[ "$CLUSTER_OK" == "1" ]]; then
    smoke_at "http://127.0.0.1:8080"
  fi
  publish_tracking
  start_mlflow
  summary
}

main
