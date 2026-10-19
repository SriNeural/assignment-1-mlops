import joblib
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from cardiascreen.config import MODEL_INPUT_COLUMNS, PROJECT_ROOT
from cardiascreen.data import load_modeling_table
from cardiascreen.features import build_pipeline
from cardiascreen.scoring import score_records
from cardiascreen.service import create_app


def _artifact_path(tmp_path):
    table = load_modeling_table()
    x = table[MODEL_INPUT_COLUMNS]
    y = table["target"]
    x_train, _, y_train, _ = train_test_split(x, y, train_size=240, stratify=y, random_state=0)
    pipeline = build_pipeline(
        LogisticRegression(max_iter=800, class_weight="balanced", random_state=0)
    )
    pipeline.fit(x_train, y_train)
    artifact = {
        "pipeline": pipeline,
        "model_name": "logistic_regression",
        "model_version": "test",
        "threshold": 0.40,
        "trained_at": "2026-01-01T00:00:00+00:00",
        "input_columns": MODEL_INPUT_COLUMNS,
        "metrics": {"holdout": {"roc_auc": 0.75, "recall": 0.8}},
        "disclaimer": "test artifact",
    }
    path = tmp_path / "model.joblib"
    joblib.dump(artifact, path)
    return path


def _body():
    return {
        "age": 67,
        "sex": 1,
        "cp": 4,
        "trestbps": 160,
        "chol": 286,
        "fbs": 0,
        "restecg": 2,
        "thalach": 108,
        "exang": 1,
        "oldpeak": 1.5,
        "slope": 2,
        "ca": 3,
        "thal": 3,
    }


def test_health_predict_and_metrics(tmp_path) -> None:
    client = TestClient(create_app(_artifact_path(tmp_path)))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["threshold"] == 0.40

    predicted = client.post("/predict", json=_body())
    assert predicted.status_code == 200
    body = predicted.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability"] <= 1
    assert body["label"] in {"disease_signal", "no_disease_signal"}
    assert body["risk_band"] in {"low", "moderate", "high", "very_high"}
    assert body["request_id"]
    assert "not a diagnosis" in body["disclaimer"] or body["disclaimer"] == "test artifact"

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    text = metrics.text
    assert "cardiascreen_requests_total" in text
    assert "cardiascreen_predictions_total" in text
    assert "cardiascreen_predicted_probability" in text


def test_kaggle_chest_pain_code_is_rejected(tmp_path) -> None:
    client = TestClient(create_app(_artifact_path(tmp_path)))
    payload = _body()
    payload["cp"] = 0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_client_cannot_submit_the_label(tmp_path) -> None:
    client = TestClient(create_app(_artifact_path(tmp_path)))
    payload = _body()
    payload["target"] = 1
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_optional_invasive_fields_score(tmp_path) -> None:
    path = _artifact_path(tmp_path)
    artifact = joblib.load(path)
    payload = _body()
    payload["ca"] = None
    payload["thal"] = None
    payload["fbs"] = None
    decision = score_records(artifact, [payload])[0]
    assert decision["model_version"] == "test"
    client = TestClient(create_app(path))
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_ci_and_container_contracts_exist() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "ruff check" in workflow
    assert "pytest" in workflow
    assert "cardiascreen.train" in workflow
    assert "cardiascreen.versioning check" in workflow
    assert "cardiascreen.release promote" in workflow
    assert "download-artifact" in workflow
    assert "continue-on-error" not in workflow
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "CARDIA_MODEL_PATH" in dockerfile
    assert "uvicorn" in dockerfile
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "prometheus" in compose
    assert "grafana" in compose
