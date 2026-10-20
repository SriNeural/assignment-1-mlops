"""Publish the committed model into the local MLflow store.

``mlruns/`` is not in git. A Prayogshala clone therefore has no experiment and
no Model Registry entry until this module runs. It reuses a logged model when
one is already in the store. Otherwise it records one run from
``models/heart_risk_model.joblib`` and registers that. A second run leaves the
champion alias where it is.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode

from cardiascreen.config import (
    CHAMPION_ALIAS,
    EXPERIMENT_NAME,
    FIGURES_DIR,
    METRICS_PATH,
    MLRUNS_DIR,
    MODEL_INPUT_COLUMNS,
    MODEL_PATH,
    MODEL_VERSION,
    PROJECT_ROOT,
    REGISTERED_MODEL_NAME,
)

SKOPS_TRUSTED_TYPES = [
    "cardiascreen.features.ClinicalFeatureBuilder",
    "numpy.dtype",
]


def open_store(store: Path | None = None):
    root = Path(store) if store is not None else MLRUNS_DIR
    root.mkdir(parents=True, exist_ok=True)
    database = (root / "mlflow.db").resolve()
    mlflow.set_tracking_uri("sqlite:///" + database.as_posix())
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        artifact_location = (root / "artifacts").resolve().as_uri()
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=artifact_location
        )
        experiment = client.get_experiment(experiment_id)
    mlflow.set_experiment(EXPERIMENT_NAME)
    # MLflow 3 opens an experiment on the empty GenAI page unless the kind is set.
    mlflow.set_experiment_tag("mlflow.experimentKind", "classification")
    return client, experiment


def _champion(client: mlflow.tracking.MlflowClient):
    try:
        return client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException as exc:
        if exc.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return None
        raise


def _best_logged_model(client: mlflow.tracking.MlflowClient, experiment_id: str):
    found = []
    for model in client.search_logged_models(experiment_ids=[experiment_id]):
        if model.status != LoggedModelStatus.READY or not model.source_run_id:
            continue
        run = client.get_run(model.source_run_id)
        if run.info.status != "FINISHED":
            continue
        auc = float(run.data.metrics.get("holdout_roc_auc", -1.0))
        found.append((auc, run.info.end_time or 0, model))
    if not found:
        return None
    found.sort(key=lambda row: (row[0], row[1]))
    return found[-1][2]


def _description() -> str:
    if not METRICS_PATH.is_file():
        return f"CardiaScreen {MODEL_VERSION}."
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    holdout = metrics.get("holdout") or {}
    threshold = metrics.get("threshold")
    auc = holdout.get("roc_auc")
    if threshold is None or auc is None:
        return f"CardiaScreen {MODEL_VERSION}."
    name = metrics.get("model_name", "model")
    version = metrics.get("model_version", MODEL_VERSION)
    return (
        f"{name} {version}, threshold {float(threshold):.2f}, "
        f"holdout ROC-AUC {float(auc):.3f}."
    )


def _input_example() -> pd.DataFrame:
    sample = PROJECT_ROOT / "deploy" / "sample_request.json"
    row = json.loads(sample.read_text(encoding="utf-8"))
    filled = {column: row.get(column) for column in MODEL_INPUT_COLUMNS}
    # A second row with the nullable screening fields keeps the MLflow schema as float.
    missing = dict(filled)
    for column in ("chol", "fbs", "ca", "thal"):
        missing[column] = None
    return pd.DataFrame([filled, missing]).astype("float64")


def _seed_committed_run() -> str:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"{MODEL_PATH} is missing. Clone the full repository.")
    artifact = joblib.load(MODEL_PATH)
    metrics = {}
    if METRICS_PATH.is_file():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    params = {
        "model_name": artifact.get("model_name", metrics.get("model_name")),
        "model_version": artifact.get("model_version", MODEL_VERSION),
        "threshold": artifact.get("threshold", metrics.get("threshold")),
        "source": "committed-joblib",
    }
    for key, value in (metrics.get("best_params") or {}).items():
        params[key] = value
    logged: dict[str, float] = {}
    holdout = metrics.get("holdout") or {}
    for key in ("roc_auc", "recall", "precision", "f2", "accuracy"):
        if key in holdout:
            logged[f"holdout_{key}"] = float(holdout[key])
    if "oof_f2" in metrics:
        logged["oof_f2"] = float(metrics["oof_f2"])
    if metrics.get("threshold") is not None:
        logged["threshold"] = float(metrics["threshold"])
    for row in metrics.get("leaderboard") or []:
        logged[f"cv_roc_auc_{row['model_name']}"] = float(row["cv_roc_auc"])

    with mlflow.start_run(run_name="committed-model") as run:
        mlflow.set_tags(
            {
                "model_source": "models/heart_risk_model.joblib",
                "label": "num>0 angiographic narrowing",
            }
        )
        mlflow.log_params({key: value for key, value in params.items() if value is not None})
        if logged:
            mlflow.log_metrics(logged)
        if FIGURES_DIR.is_dir():
            for figure in sorted(FIGURES_DIR.glob("model_*.png")):
                mlflow.log_artifact(str(figure))
        if METRICS_PATH.is_file():
            mlflow.log_artifact(str(METRICS_PATH))
        info = mlflow.sklearn.log_model(
            artifact["pipeline"],
            name="model",
            input_example=_input_example(),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        if info.model_id:
            return f"models:/{info.model_id}"
        return f"runs:/{run.info.run_id}/{info.artifact_path}"


def register_model_uri(model_uri: str, description: str | None = None) -> str:
    """Register ``model_uri`` and move the champion alias to that version."""
    version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client = mlflow.tracking.MlflowClient()
    text = description or _description()
    client.update_registered_model(
        REGISTERED_MODEL_NAME,
        description="CardiaScreen screening model. The champion alias is the served model.",
    )
    client.update_model_version(REGISTERED_MODEL_NAME, version.version, description=text)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version)
    # The alias is what the UI shows. The stage remains for older registry views.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        client.transition_model_version_stage(
            REGISTERED_MODEL_NAME,
            version.version,
            "Production",
            archive_existing_versions=True,
        )
    return str(version.version)


def publish(store: Path | None = None) -> dict:
    """Ensure the local store has a champion model. Safe to call more than once."""
    client, experiment = open_store(store)
    current = _champion(client)
    if current is not None:
        return {
            "registered_model": REGISTERED_MODEL_NAME,
            "version": str(current.version),
            "alias": CHAMPION_ALIAS,
            "created": False,
            "experiment_id": experiment.experiment_id,
        }
    logged = _best_logged_model(client, experiment.experiment_id)
    if logged is None:
        model_uri = _seed_committed_run()
    else:
        model_uri = f"models:/{logged.model_id}"
    version = register_model_uri(model_uri, _description())
    return {
        "registered_model": REGISTERED_MODEL_NAME,
        "version": version,
        "alias": CHAMPION_ALIAS,
        "created": True,
        "experiment_id": experiment.experiment_id,
    }


def main() -> None:
    result = publish()
    experiment_id = result["experiment_id"]
    print(f"registered_model={result['registered_model']}")
    print(f"version={result['version']}")
    print(f"alias={result['alias']}")
    print(f"created={result['created']}")
    print(
        "runs_url="
        "http://127.0.0.1:5000/#/experiments/"
        f"{experiment_id}/runs?workflowType=machine_learning"
    )
    print(f"registry_url=http://127.0.0.1:5000/#/models/{REGISTERED_MODEL_NAME}")


if __name__ == "__main__":
    main()
