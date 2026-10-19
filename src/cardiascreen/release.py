"""Decide whether a freshly trained model replaces the one the API serves.

An unchanged fingerprint keeps the production version. A model that holds its
ROC-AUC and recall is promoted and becomes the file the container build copies.
A model that drops below the gate fails the run, so the previous image stays.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import joblib

from cardiascreen.config import METRICS_PATH, MODEL_PATH, MODELS_DIR
from cardiascreen.evaluate import to_builtin
from cardiascreen.versioning import check_data_version

REGISTRY_PATH = MODELS_DIR / "registry.json"
DECISION_PATH = MODELS_DIR / "release_decision.json"
AUC_TOLERANCE = 0.005
RECALL_TOLERANCE = 0.005
MIN_RECALL = 0.90


def _round(value: float, digits: int) -> float:
    return round(float(value), digits)


def fingerprint(metrics: dict, data_version: str) -> dict:
    holdout = metrics["holdout"]
    return {
        "data_version": data_version,
        "model_name": metrics["model_name"],
        "holdout_roc_auc": _round(holdout["roc_auc"], 4),
        "holdout_recall": _round(holdout["recall"], 4),
        "threshold": _round(metrics["threshold"], 2),
        "params": to_builtin(metrics["best_params"]),
    }


def production_release(registry: dict) -> dict | None:
    version = registry.get("production")
    for item in registry.get("releases", []):
        if item.get("model_version") == version:
            return item
    return None


def next_version(current: str, data_changed: bool) -> str:
    major, minor, patch = (int(part) for part in current.split("."))
    if data_changed:
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def _same_release(current: dict, candidate: dict) -> bool:
    if all(current.get(key) == candidate[key] for key in candidate):
        return True
    # Same data, model, and parameters can move in the fourth decimal between
    # Windows and Ubuntu. That shift is not a new release.
    if current.get("data_version") != candidate["data_version"]:
        return False
    if current.get("model_name") != candidate["model_name"]:
        return False
    if current.get("params") != candidate["params"]:
        return False
    if current.get("threshold") != candidate["threshold"]:
        return False
    auc_gap = abs(float(current["holdout_roc_auc"]) - candidate["holdout_roc_auc"])
    recall_gap = abs(float(current["holdout_recall"]) - candidate["holdout_recall"])
    return auc_gap <= AUC_TOLERANCE and recall_gap <= RECALL_TOLERANCE


def decide(metrics: dict, registry: dict, data_version: str) -> dict:
    """Return promote, unchanged, or reject. Rejection is a failed release."""
    candidate = fingerprint(metrics, data_version)
    current = production_release(registry)
    if current is None:
        return {
            "action": "promote",
            "model_version": "1.0.0",
            "reason": "No production model is registered.",
            **candidate,
        }
    if _same_release(current, candidate):
        return {
            "action": "unchanged",
            "model_version": current["model_version"],
            "reason": "Fingerprint matches the production release.",
            **candidate,
        }
    auc_drop = float(current["holdout_roc_auc"]) - candidate["holdout_roc_auc"]
    if auc_drop > AUC_TOLERANCE or candidate["holdout_recall"] < MIN_RECALL:
        return {
            "action": "reject",
            "model_version": current["model_version"],
            "reason": (
                "Candidate holdout ROC-AUC "
                f"{candidate['holdout_roc_auc']:.4f} or recall "
                f"{candidate['holdout_recall']:.4f} missed the production gate "
                f"(AUC drop limit {AUC_TOLERANCE}, minimum recall {MIN_RECALL})."
            ),
            **candidate,
        }
    data_changed = current["data_version"] != data_version
    version = next_version(current["model_version"], data_changed)
    reason = "Data version changed." if data_changed else "Gate passed."
    return {"action": "promote", "model_version": version, "reason": reason, **candidate}


def _write_github_output(decision: dict) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(f"action={decision['action']}\n")
        handle.write(f"version={decision['model_version']}\n")


def promote(
    metrics_path: Path = METRICS_PATH,
    model_path: Path = MODEL_PATH,
    registry_path: Path = REGISTRY_PATH,
) -> dict:
    pin = check_data_version()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {"production": None, "releases": []}
    decision = decide(metrics, registry, pin["data_version"])
    decision["decided_at"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DECISION_PATH.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    _write_github_output(decision)
    if decision["action"] == "reject":
        raise SystemExit(decision["reason"])

    artifact = joblib.load(model_path)
    metrics["model_version"] = decision["model_version"]
    metrics["data_version"] = pin["data_version"]
    metrics["release_action"] = decision["action"]
    artifact["model_version"] = decision["model_version"]
    artifact["data_version"] = pin["data_version"]
    artifact["metrics"] = metrics
    joblib.dump(artifact, model_path)
    metrics_path.write_text(json.dumps(to_builtin(metrics), indent=2) + "\n", encoding="utf-8")

    if decision["action"] == "promote":
        for item in registry["releases"]:
            if item.get("status") == "production":
                item["status"] = "archived"
        registry["production"] = decision["model_version"]
        registry["releases"].append(
            {
                "model_version": decision["model_version"],
                "data_version": decision["data_version"],
                "model_name": decision["model_name"],
                "holdout_roc_auc": decision["holdout_roc_auc"],
                "holdout_recall": decision["holdout_recall"],
                "threshold": decision["threshold"],
                "params": decision["params"],
                "status": "production",
                "promoted_at": decision["decided_at"],
                "reason": decision["reason"],
            }
        )
        registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(
        f"action={decision['action']} model_version={decision['model_version']} "
        f"data_version={pin['data_version']}"
    )
    print(decision["reason"])
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote or reject a trained CardiaScreen model.")
    parser.add_argument("command", choices=("promote",))
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    args = parser.parse_args()
    promote(args.metrics, args.model, args.registry)


if __name__ == "__main__":
    main()
