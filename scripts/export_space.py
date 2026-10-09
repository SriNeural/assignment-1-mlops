"""Assemble a Hugging Face Docker Space folder under dist/huggingface-space."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "dist" / "huggingface-space"


def copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def main() -> None:
    model = ROOT / "models" / "heart_risk_model.joblib"
    if not model.exists():
        raise SystemExit(
            "Train the model before exporting a Space. Missing models/heart_risk_model.joblib"
        )
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    shutil.copy2(ROOT / "Dockerfile", DEST / "Dockerfile")
    shutil.copy2(ROOT / "requirements-api.txt", DEST / "requirements-api.txt")
    shutil.copy2(ROOT / "deploy" / "huggingface" / "README.md", DEST / "README.md")
    copy_tree(ROOT / "src", DEST / "src")
    copy_tree(ROOT / "models", DEST / "models")
    print(f"Space folder ready: {DEST}")
    print("Upload with: huggingface-cli upload <user>/cardiascreen-api . --repo-type space")
    print(f"Run that command from {DEST}")


if __name__ == "__main__":
    main()
