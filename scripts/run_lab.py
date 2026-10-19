"""Score and, when asked, serve CardiaScreen without Docker.

setup_prayogshala.bat is the Windows entry point. It calls this script
with --check after installing packages, and uses it to serve the API
when Docker cannot start. This script does not retrain.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_VERSION = (3, 11)
SAMPLE = ROOT / "deploy" / "sample_request.json"


def venv_python() -> Path:
    if os.name == "nt":
        return ROOT / ".venv" / "Scripts" / "python.exe"
    return ROOT / ".venv" / "bin" / "python"


def in_project_venv() -> bool:
    return Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CardiaScreen on a lab VM without Docker.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check the data pin, score one sample, and exit without serving.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Install the training and test stack, then run pytest before serving.",
    )
    return parser.parse_args()


def require_python() -> None:
    if sys.version_info >= MIN_VERSION:
        return
    found = ".".join(str(part) for part in sys.version_info[:3])
    raise SystemExit(
        f"Python {found} is too old. pandas 3.0.6 needs Python 3.11 or newer.\n"
        "On the lab, run:\n"
        "  conda create -n cardiascreen python=3.11 -y\n"
        "  conda activate cardiascreen\n"
        "  python scripts/run_lab.py"
    )


def requirements_file(full: bool) -> Path:
    name = "requirements-dev.txt" if full else "requirements-api.txt"
    return ROOT / name


def ensure_packages(python: str, full: bool) -> None:
    requirement = requirements_file(full)
    print(f"Installing {requirement.name}")
    subprocess.check_call(
        [python, "-m", "pip", "install", "-r", str(requirement)],
        cwd=ROOT,
    )


def packages_ready(python: str, full: bool) -> bool:
    modules = ["fastapi", "uvicorn", "joblib", "sklearn", "pandas", "prometheus_client"]
    if full:
        modules.extend(["pytest", "mlflow"])
    completed = subprocess.run(
        [python, "-c", "import " + ", ".join(modules)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def switch_to_venv(full: bool) -> None:
    """Install into .venv and replace this process with that interpreter."""
    require_python()
    if in_project_venv():
        if not packages_ready(sys.executable, full):
            ensure_packages(sys.executable, full)
        return
    py = venv_python()
    if not py.exists():
        print("Creating .venv")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", str(ROOT / ".venv")], cwd=ROOT)
        except subprocess.CalledProcessError:
            print("python -m venv failed. Installing into the current interpreter.")
            if not packages_ready(sys.executable, full):
                ensure_packages(sys.executable, full)
            return
    if not packages_ready(str(py), full):
        ensure_packages(str(py), full)
    script = str(Path(__file__).resolve())
    print(f"Switching to {py}")
    # subprocess, not os.execv: on Windows execv detaches from the console
    # that launched the script, so the score and the API log disappear.
    completed = subprocess.run([str(py), script, *sys.argv[1:]], cwd=ROOT, check=False)
    raise SystemExit(completed.returncode)


def prepare_imports() -> None:
    src = str(ROOT / "src")
    os.environ["PYTHONPATH"] = src
    if src not in sys.path:
        sys.path.insert(0, src)
    os.chdir(ROOT)


def check_and_score() -> dict:
    from cardiascreen.scoring import score_records
    from cardiascreen.service import load_artifact
    from cardiascreen.versioning import check_data_version

    pin = check_data_version()
    print(f"Data pin {pin['data_version']} matches the raw files.")
    artifact = load_artifact(ROOT / "models" / "heart_risk_model.joblib")
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
    scored = score_records(artifact, [payload])[0]
    print(
        "Sample score:"
        f" prediction={scored['prediction']}"
        f" label={scored['label']}"
        f" probability={scored['probability']:.6f}"
        f" band={scored['risk_band']}"
    )
    return scored


def run_pytest() -> None:
    print("Running pytest")
    subprocess.check_call([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)


def serve(host: str, port: int) -> None:
    import uvicorn

    base = f"http://{host}:{port}"
    print(f"API docs: {base}/docs")
    print(f"Health:   {base}/health")
    print("Stop the server with Ctrl+C. This process does not retrain.")
    uvicorn.run(
        "cardiascreen.service:create_app",
        factory=True,
        host=host,
        port=port,
    )


def main() -> None:
    args = parse_args()
    switch_to_venv(args.full)
    prepare_imports()
    check_and_score()
    if args.full:
        run_pytest()
    if args.check:
        return
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
