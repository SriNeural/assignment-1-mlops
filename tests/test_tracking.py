"""The local MLflow store gets one champion model, including on an empty clone."""

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression

from cardiascreen.config import CHAMPION_ALIAS, REGISTERED_MODEL_NAME
from cardiascreen.tracking import open_store, publish


def test_publish_registers_an_existing_logged_model(tmp_path) -> None:
    client, experiment = open_store(tmp_path)
    with mlflow.start_run(run_name="model-selection"):
        model = LogisticRegression().fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
        mlflow.log_metric("holdout_roc_auc", 0.91)
        mlflow.sklearn.log_model(model, name="model")

    result = publish(tmp_path)

    assert result["created"] is True
    assert result["version"] == "1"
    assert result["alias"] == CHAMPION_ALIAS
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    assert str(champion.version) == "1"
    assert champion.current_stage == "Production"


def test_publish_seeds_the_committed_model_once(tmp_path) -> None:
    first = publish(tmp_path)
    second = publish(tmp_path)

    assert first["created"] is True
    assert first["version"] == "1"
    assert second["created"] is False
    assert second["version"] == "1"
    client, experiment = open_store(tmp_path)
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    assert len(versions) == 1
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.run_name == "committed-model"
    assert client.get_experiment(experiment.experiment_id).tags["mlflow.experimentKind"] == (
        "classification"
    )
