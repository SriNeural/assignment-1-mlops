@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title CardiaScreen Prayogshala setup
set "CHECK_ONLY=0"
set "DOCKER_OK=0"
set "KUBECTL_OK=0"
set "CLUSTER_OK=0"
set "COMPOSE_OK=0"
set "PYTHON_OK=0"
set "K8S_MODE="
set "K8S_CONTEXT="
set "KIND_NAME=kind"
set "K3D_NAME="
set "RAM_GB=0"
set "PY_MODE="
set "PY_EXE="
set "DOCKER_DESKTOP="
set "WAIT_URL="
if /I "%~1"=="--check" set "CHECK_ONLY=1"

echo.
echo CardiaScreen setup for Prayogshala
echo Directory: %CD%
echo.

if not exist "src\cardiascreen\service.py" goto :not_repo
if not exist "models\heart_risk_model.joblib" goto :missing_model
if "%CHECK_ONLY%"=="1" goto :status

call :ensure_python
if errorlevel 1 goto :failed
call :install_packages
if errorlevel 1 goto :failed
call :verify_program
if errorlevel 1 goto :failed
call :ensure_docker
if errorlevel 1 (
    echo Docker is not available. A new local cluster cannot be created.
    set "DOCKER_OK=0"
) else (
    set "DOCKER_OK=1"
)
call :ensure_kubectl
if errorlevel 1 (
    echo kubectl is not available. Kubernetes will not be used.
    set "KUBECTL_OK=0"
) else (
    set "KUBECTL_OK=1"
)
if "!KUBECTL_OK!"=="0" goto :after_cluster
call :ensure_cluster
if not "!CLUSTER_OK!"=="1" goto :after_cluster
call :apply_k8s
if errorlevel 1 set "CLUSTER_OK=0"
:after_cluster
call :start_compose
if errorlevel 1 (
    if not "!CLUSTER_OK!"=="1" (
        echo Compose did not stay up. Starting the API with Python.
        call :serve_python
        set "PYTHON_OK=1"
    ) else (
        echo Compose did not stay up. The Kubernetes API is still running.
    )
) else (
    if "!COMPOSE_OK!"=="1" call :smoke_compose
)
if "!CLUSTER_OK!"=="1" call :smoke_k8s
call :start_mlflow
call :summary
echo.
echo Press any key to close this window.
pause >nul
exit /b 0

:status
echo --- check only, nothing will be installed ---
echo.
echo Python 3.11:
py -3.11 -c "import sys; print(sys.version)" 2>&1
if errorlevel 1 echo Python 3.11 launcher was not found.
echo.
echo Docker:
docker info >nul 2>&1
if errorlevel 1 (echo Docker is not running.) else (echo Docker is running.)
echo.
echo Kubernetes:
kubectl get nodes --request-timeout=8s 2>&1
if errorlevel 1 echo Kubernetes cluster is not reachable.
echo.
powershell -NoProfile -Command "Write-Output ('RAM_GB=' + [int]((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB))"
exit /b 0

:not_repo
echo This file must stay in the project folder, next to the src directory.
goto :failed

:missing_model
echo models\heart_risk_model.joblib is missing. Clone the full repository and run this file again.
goto :failed

:failed
echo.
echo Setup stopped. Read the message above.
pause
exit /b 1

:ensure_python
where py >nul 2>&1
if errorlevel 1 goto :try_python_cmd
py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 goto :try_python_cmd
set "PY_MODE=launcher"
echo Python: py -3.11
exit /b 0

:try_python_cmd
where python >nul 2>&1
if errorlevel 1 goto :try_conda
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 goto :try_conda
set "PY_MODE=path"
set "PY_EXE=python"
echo Python: python
exit /b 0

:try_conda
where conda >nul 2>&1
if errorlevel 1 goto :install_python
echo Creating the conda env cardiascreen with Python 3.11
call conda create -y -n cardiascreen python=3.11
if errorlevel 1 goto :install_python
for /f "delims=" %%I in ('conda run -n cardiascreen python -c "import sys; print(sys.executable)"') do set "PY_EXE=%%I"
if not defined PY_EXE goto :install_python
set "PY_MODE=path"
echo Python: !PY_EXE!
exit /b 0

:install_python
where winget >nul 2>&1
if errorlevel 1 goto :python_missing
echo Installing Python 3.11
winget install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
    set "PY_MODE=path"
    set "PY_EXE=%LocalAppData%\Programs\Python\Python311\python.exe"
    echo Python: !PY_EXE!
    exit /b 0
)
where py >nul 2>&1
if errorlevel 1 goto :python_missing
py -3.11 -c "import sys" >nul 2>&1
if errorlevel 1 goto :python_missing
set "PY_MODE=launcher"
echo Python: py -3.11
exit /b 0

:python_missing
echo Python 3.11 is not installed, and this machine could not install it.
echo From Anaconda Prompt run: conda create -n cardiascreen python=3.11 -y
echo Then run this file again.
exit /b 1

:install_packages
if exist ".venv\Scripts\python.exe" goto :pip_install
echo Creating .venv
if "%PY_MODE%"=="launcher" (
    py -3.11 -m venv .venv
) else (
    "!PY_EXE!" -m venv .venv
)
if errorlevel 1 exit /b 1
if not exist ".venv\Scripts\python.exe" exit /b 1

:pip_install
echo Installing the training, test, and API packages. This can take several minutes.
".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt
if errorlevel 1 exit /b 1
exit /b 0

:verify_program
echo Checking the data pin and the committed model
set "PYTHONPATH=src"
".venv\Scripts\python.exe" scripts\run_lab.py --check
if errorlevel 1 exit /b 1
echo Running tests
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 exit /b 1
exit /b 0

:ensure_docker
call :add_docker_path
docker info >nul 2>&1
if not errorlevel 1 (
    echo Docker is already running.
    exit /b 0
)
call :find_desktop
if not defined DOCKER_DESKTOP goto :download_docker
echo Docker Desktop is installed. Starting it.
call :start_docker_desktop
call :wait_docker
if not errorlevel 1 exit /b 0
echo Docker Desktop did not start. This VM may not allow virtualization.
exit /b 1

:download_docker
echo Docker is not installed. Downloading Docker Desktop.
curl -L -o "%TEMP%\DockerDesktopInstaller.exe" "https://desktop.docker.com/win/main/amd64/Docker%%20Desktop%%20Installer.exe"
if not exist "%TEMP%\DockerDesktopInstaller.exe" (
    echo The Docker Desktop download failed. Check the lab internet connection.
    exit /b 1
)
echo Installing Docker Desktop for the current Windows user.
start /wait "" "%TEMP%\DockerDesktopInstaller.exe" install --user --quiet --accept-license --backend=wsl-2
set "DOCKER_RC=!ERRORLEVEL!"
if "!DOCKER_RC!"=="3010" (
    echo Docker Desktop was installed and Windows must restart.
    echo Restart the lab machine, then double-click this file again.
    exit /b 1
)
if not "!DOCKER_RC!"=="0" (
    echo The user install did not finish. Approve the administrator prompt if Windows shows one.
    powershell -NoProfile -Command "Start-Process -FilePath ($env:TEMP + '\DockerDesktopInstaller.exe') -ArgumentList 'install','--quiet','--accept-license','--backend=wsl-2' -Verb RunAs -Wait"
    if errorlevel 1 exit /b 1
)
call :add_docker_path
call :find_desktop
if not defined DOCKER_DESKTOP (
    echo Docker Desktop installed, but Docker Desktop.exe was not found.
    exit /b 1
)
call :start_docker_desktop
call :wait_docker
if errorlevel 1 exit /b 1
exit /b 0

:add_docker_path
if exist "%ProgramFiles%\Docker\Docker\resources\bin\docker.exe" set "PATH=%ProgramFiles%\Docker\Docker\resources\bin;%PATH%"
if exist "%LocalAppData%\Programs\Docker\Docker\resources\bin\docker.exe" set "PATH=%LocalAppData%\Programs\Docker\Docker\resources\bin;%PATH%"
exit /b 0

:find_desktop
set "DOCKER_DESKTOP="
if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" set "DOCKER_DESKTOP=%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
if not defined DOCKER_DESKTOP if exist "%LocalAppData%\Programs\Docker\Docker\Docker Desktop.exe" set "DOCKER_DESKTOP=%LocalAppData%\Programs\Docker\Docker\Docker Desktop.exe"
exit /b 0

:start_docker_desktop
if not defined DOCKER_DESKTOP call :find_desktop
if not defined DOCKER_DESKTOP exit /b 1
echo Starting Docker Desktop
start "" "!DOCKER_DESKTOP!"
exit /b 0

:wait_docker
set /a N=0
:wait_docker_loop
docker info >nul 2>&1
if not errorlevel 1 exit /b 0
set /a N+=1
if !N! GEQ 36 exit /b 1
echo Waiting for Docker... !N! of 36
ping -n 6 127.0.0.1 >nul
goto :wait_docker_loop

:ensure_kubectl
where kubectl >nul 2>&1
if not errorlevel 1 (
    echo kubectl is already installed.
    exit /b 0
)
call :add_docker_path
where kubectl >nul 2>&1
if not errorlevel 1 (
    echo kubectl is already installed.
    exit /b 0
)
where winget >nul 2>&1
if errorlevel 1 (
    echo kubectl was not found, and winget is not available.
    exit /b 1
)
echo Installing kubectl
winget install -e --id Kubernetes.kubectl --accept-package-agreements --accept-source-agreements
set "PATH=%LocalAppData%\Microsoft\WinGet\Links;%PATH%"
where kubectl >nul 2>&1
if errorlevel 1 (
    echo kubectl installed, but it is not on PATH yet. Close this window and run the file again.
    exit /b 1
)
exit /b 0

:ensure_cluster
call :ram_gb
kubectl get nodes --request-timeout=8s --no-headers 2>nul | findstr /I /C:" Ready " >nul
if not errorlevel 1 (
    echo Using the Kubernetes cluster that is already Ready.
    set "CLUSTER_OK=1"
    call :mark_cluster
    exit /b 0
)
if not "%DOCKER_OK%"=="1" (
    echo No Kubernetes node is Ready, and Docker is not running.
    set "CLUSTER_OK=0"
    exit /b 0
)
if !RAM_GB! GEQ 8 (
    echo Turning on Kubernetes in Docker Desktop.
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\enable_docker_kubernetes.ps1"
    if errorlevel 2 (
        call :start_docker_desktop
        call :wait_docker
        powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\enable_docker_kubernetes.ps1"
    )
    echo Restarting Docker Desktop so Kubernetes can start.
    taskkill /IM "Docker Desktop.exe" /F >nul 2>&1
    ping -n 4 127.0.0.1 >nul
    call :start_docker_desktop
    call :wait_docker
    if not errorlevel 1 (
        kubectl config use-context docker-desktop >nul 2>&1
        call :wait_nodes
        if not errorlevel 1 (
            set "CLUSTER_OK=1"
            call :mark_cluster
            exit /b 0
        )
    )
    echo Docker Desktop Kubernetes did not become Ready.
)
if !RAM_GB! LSS 4 (
    echo This machine has !RAM_GB! GB of RAM. Starting a local cluster needs about 4 GB.
    echo No Ready node was found, so Kubernetes was not started.
    set "CLUSTER_OK=0"
    exit /b 0
)
echo Starting a local minikube cluster.
call :start_minikube
exit /b 0

:mark_cluster
set "K8S_CONTEXT="
for /f "delims=" %%C in ('kubectl config current-context 2^>nul') do set "K8S_CONTEXT=%%C"
if not defined K8S_CONTEXT set "K8S_CONTEXT=unknown"
echo Kubernetes context: !K8S_CONTEXT!
set "K8S_MODE=existing"
echo !K8S_CONTEXT! | findstr /I /C:"minikube" >nul && set "K8S_MODE=minikube"
echo !K8S_CONTEXT! | findstr /I /C:"docker-desktop" >nul && set "K8S_MODE=desktop"
echo !K8S_CONTEXT! | findstr /I /C:"kind" >nul && set "K8S_MODE=kind"
echo !K8S_CONTEXT! | findstr /I /C:"k3d" >nul && set "K8S_MODE=k3d"
set "KIND_NAME=kind"
set "K3D_NAME="
for /f "tokens=1* delims=-" %%A in ("!K8S_CONTEXT!") do (
    if /I "%%A"=="kind" if not "%%B"=="" set "KIND_NAME=%%B"
    if /I "%%A"=="k3d" if not "%%B"=="" set "K3D_NAME=%%B"
)
echo Cluster mode: !K8S_MODE!
exit /b 0

:ram_gb
set "RAM_GB=0"
for /f %%M in ('powershell -NoProfile -Command "[int]((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB)"') do set "RAM_GB=%%M"
echo RAM: !RAM_GB! GB
exit /b 0

:wait_nodes
set /a N=0
:wait_nodes_loop
kubectl get nodes --request-timeout=8s --no-headers 2>nul | findstr /I /C:" Ready " >nul
if not errorlevel 1 exit /b 0
set /a N+=1
if !N! GEQ 40 exit /b 1
echo Waiting for a Kubernetes node... !N! of 40
ping -n 11 127.0.0.1 >nul
goto :wait_nodes_loop

:start_minikube
where winget >nul 2>&1
if errorlevel 1 goto :minikube_missing
where minikube >nul 2>&1
if errorlevel 1 (
    echo Installing minikube
    winget install -e --id Kubernetes.minikube --accept-package-agreements --accept-source-agreements
    set "PATH=%LocalAppData%\Microsoft\WinGet\Links;%PATH%"
)
where minikube >nul 2>&1
if errorlevel 1 goto :minikube_missing
set "MK_MEM=1900"
set "MK_CPU=1"
if !RAM_GB! GEQ 8 (
    set "MK_MEM=3072"
    set "MK_CPU=2"
)
if !RAM_GB! GEQ 16 (
    set "MK_MEM=4096"
    set "MK_CPU=2"
)
echo Starting minikube with !MK_CPU! CPU and !MK_MEM! MB. The first start downloads a cluster image.
minikube start --driver=docker --cpus=!MK_CPU! --memory=!MK_MEM!
if errorlevel 1 (
    echo minikube did not start.
    set "CLUSTER_OK=0"
    exit /b 0
)
kubectl config use-context minikube >nul 2>&1
set "CLUSTER_OK=1"
call :mark_cluster
exit /b 0

:minikube_missing
echo minikube is not available, and no Kubernetes node is Ready.
set "CLUSTER_OK=0"
exit /b 0

:apply_k8s
if not "!CLUSTER_OK!"=="1" exit /b 0
if not "%DOCKER_OK%"=="1" (
    echo Docker is down. Checking for a deployment that is already Ready.
    call :reuse_if_running
    exit /b
)
echo Building cardiascreen-api:local for Kubernetes
docker build -t cardiascreen-api:local .
if errorlevel 1 (
    echo The image build failed. Kubernetes was not changed.
    exit /b 1
)
call :load_image
if errorlevel 1 (
    echo The image did not load into the cluster.
    exit /b 1
)
echo Applying the Kubernetes manifests in deploy\k8s
kubectl apply -k deploy\k8s
if errorlevel 1 (
    echo kubectl apply failed.
    exit /b 1
)
if !RAM_GB! LSS 8 (
    echo This VM has !RAM_GB! GB. Running one replica so the pod can schedule.
    kubectl scale deployment/cardiascreen-api -n cardiascreen --replicas=1
    kubectl set resources deployment/cardiascreen-api -n cardiascreen -c api --requests=cpu=50m,memory=128Mi --limits=cpu=500m,memory=512Mi
)
echo Waiting until the deployment is Ready.
kubectl rollout status deployment/cardiascreen-api -n cardiascreen --timeout=240s
if errorlevel 1 (
    echo The deployment did not become Ready.
    kubectl get pods,svc -n cardiascreen -o wide
    kubectl describe deployment/cardiascreen-api -n cardiascreen
    exit /b 1
)
call :expose_k8s
set "WAIT_URL=http://127.0.0.1:8080/health"
call :wait_url
if errorlevel 1 (
    echo The Kubernetes service did not answer on http://127.0.0.1:8080/health
    kubectl get pods,svc -n cardiascreen -o wide
    exit /b 1
)
echo Kubernetes is serving http://127.0.0.1:8080/docs
exit /b 0

:reuse_if_running
kubectl rollout status deployment/cardiascreen-api -n cardiascreen --timeout=30s
if errorlevel 1 (
    echo No Ready cardiascreen-api deployment was found.
    exit /b 1
)
call :expose_k8s
set "WAIT_URL=http://127.0.0.1:8080/health"
call :wait_url
exit /b

:load_image
if /I "!K8S_MODE!"=="minikube" (
    echo Loading the image into minikube.
    minikube image load cardiascreen-api:local
    if errorlevel 1 exit /b 1
    exit /b 0
)
if /I "!K8S_MODE!"=="kind" (
    echo Loading the image into kind cluster !KIND_NAME!.
    kind load docker-image cardiascreen-api:local --name "!KIND_NAME!"
    if errorlevel 1 exit /b 1
    exit /b 0
)
if /I "!K8S_MODE!"=="k3d" (
    echo Loading the image into k3d cluster !K3D_NAME!.
    k3d image import cardiascreen-api:local -c "!K3D_NAME!"
    if errorlevel 1 exit /b 1
    exit /b 0
)
if /I "!K8S_MODE!"=="desktop" (
    echo Docker Desktop Kubernetes uses the local image cardiascreen-api:local.
    exit /b 0
)
echo Context !K8S_CONTEXT! is not minikube, kind, k3d, or docker-desktop.
echo deploy\k8s expects cardiascreen-api:local on the node. If the node cannot see it, the pod will not start.
exit /b 0

:expose_k8s
set "WAIT_URL=http://127.0.0.1:8080/health"
powershell -NoProfile -Command "try { if ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 '!WAIT_URL!').StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if not errorlevel 1 (
    echo Kubernetes API is already answering on port 8080.
    exit /b 0
)
echo Forwarding the Kubernetes service to http://127.0.0.1:8080
start "CardiaScreen Kubernetes" cmd /k "kubectl port-forward --address 127.0.0.1 svc/cardiascreen-api 8080:80 -n cardiascreen"
exit /b 0

:start_compose
if not "%DOCKER_OK%"=="1" exit /b 1
if "!CLUSTER_OK!"=="1" if !RAM_GB! LSS 8 (
    echo Kubernetes is running on !RAM_GB! GB. Prometheus and Grafana stay off so the cluster keeps its memory.
    set "COMPOSE_OK=0"
    exit /b 0
)
echo Building and starting the API, Prometheus, and Grafana.
if "!CLUSTER_OK!"=="1" (
    docker compose up -d
) else (
    docker compose up -d --build
)
if errorlevel 1 exit /b 1
set "WAIT_URL=http://127.0.0.1:8000/health"
call :wait_url
if errorlevel 1 exit /b 1
set "COMPOSE_OK=1"
exit /b 0

:wait_url
set /a N=0
:wait_url_loop
powershell -NoProfile -Command "try { if ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 '!WAIT_URL!').StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if not errorlevel 1 exit /b 0
set /a N+=1
if !N! GEQ 40 exit /b 1
echo Waiting for !WAIT_URL! ... !N! of 40
ping -n 4 127.0.0.1 >nul
goto :wait_url_loop

:smoke_compose
echo Scoring the three sample requests on the Compose API
".venv\Scripts\python.exe" scripts\smoke_predict.py http://127.0.0.1:8000
if errorlevel 1 echo The Compose API is up, but a sample request failed.
exit /b 0

:smoke_k8s
echo Scoring the three sample requests through Kubernetes
".venv\Scripts\python.exe" scripts\smoke_predict.py http://127.0.0.1:8080
if errorlevel 1 echo The Kubernetes API is up, but a sample request failed.
exit /b 0

:serve_python
echo Starting the API on http://127.0.0.1:8000 without Docker
start "CardiaScreen API" cmd /k "cd /d ""%~dp0"" && set PYTHONPATH=src && .venv\Scripts\python.exe scripts\run_lab.py"
exit /b 0

:start_mlflow
netstat -ano | findstr ":5000" | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo MLflow is already using port 5000.
    exit /b 0
)
if not exist ".venv\Scripts\mlflow.exe" exit /b 0
start "MLflow" cmd /k "cd /d ""%~dp0"" && .venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5000"
exit /b 0

:summary
echo.
echo ----------------------------------------
if "!CLUSTER_OK!"=="1" (
    echo Kubernetes is serving the API.
    echo   API docs     http://127.0.0.1:8080/docs
    echo   Health       http://127.0.0.1:8080/health
    echo   Namespace    cardiascreen
    kubectl get pods,svc -n cardiascreen
)
if "!COMPOSE_OK!"=="1" (
    echo Compose is serving the API, Prometheus, and Grafana.
    echo   API docs     http://127.0.0.1:8000/docs
    echo   Health       http://127.0.0.1:8000/health
    echo   Prometheus   http://127.0.0.1:9090
    echo   Grafana      http://127.0.0.1:3000
    echo   Grafana login is admin / admin
)
if "!PYTHON_OK!"=="1" (
    echo Python is serving the API because Docker and Kubernetes did not stay up.
    echo   API docs     http://127.0.0.1:8000/docs
    echo   Health       http://127.0.0.1:8000/health
)
if "!CLUSTER_OK!"=="0" if "!COMPOSE_OK!"=="0" if "!PYTHON_OK!"=="0" (
    echo Nothing is serving. Read the messages above.
)
echo   MLflow       http://127.0.0.1:5000
echo The public copy is https://iamvastava-cardiascreen-api.hf.space
echo A new training run is optional. The model already in git is the one being served.
echo ----------------------------------------
exit /b 0
