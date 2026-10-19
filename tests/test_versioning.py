import json
import shutil

import joblib
import pytest

from cardiascreen.config import COHORT_FILES, DATA_RAW, METRICS_PATH, PROJECT_ROOT
from cardiascreen.release import decide, next_version, promote
from cardiascreen.versioning import DataVersionError, check_data_version, mismatches


def _metrics(auc: float = 0.9092539454806313, recall: float = 0.9803921568627451, **extra):
    document = {
        "model_name": "logistic_regression",
        "threshold": 0.18,
        "best_params": {"clf__C": 0.1},
        "holdout": {"roc_auc": auc, "recall": recall},
    }
    document.update(extra)
    return document


def _registry(**overrides):
    release = {
        "model_version": "1.0.0",
        "data_version": "1.0.0",
        "model_name": "logistic_regression",
        "holdout_roc_auc": 0.9093,
        "holdout_recall": 0.9804,
        "threshold": 0.18,
        "params": {"clf__C": 0.1},
        "status": "production",
    }
    release.update(overrides)
    return {"production": release["model_version"], "releases": [release]}


def test_pinned_files_match_the_raw_download() -> None:
    pin = check_data_version()
    assert pin["data_version"] == "1.0.0"
    assert [item["cohort"] for item in pin["files"]] == list(COHORT_FILES)


def test_check_rejects_a_changed_raw_file(tmp_path) -> None:
    for filename in COHORT_FILES.values():
        shutil.copy(DATA_RAW / filename, tmp_path / filename)
    cleveland = tmp_path / "cleveland.data"
    cleveland.write_bytes(cleveland.read_bytes() + b"\n")
    with pytest.raises(DataVersionError, match="cleveland"):
        check_data_version(tmp_path)


def test_mismatches_names_the_changed_cohort() -> None:
    pin = {"files": [{"cohort": "cleveland", "sha256": "aaa"}]}
    live = [{"cohort": "cleveland", "sha256": "bbb"}]
    assert mismatches(pin, live) == ["cleveland"]


def test_seeded_registry_matches_the_saved_metrics() -> None:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    registry = json.loads((PROJECT_ROOT / "models" / "registry.json").read_text(encoding="utf-8"))
    decision = decide(metrics, registry, "1.0.0")
    assert decision["action"] == "unchanged"
    assert decision["model_version"] == "1.0.0"


def test_small_metric_noise_stays_on_the_current_release() -> None:
    decision = decide(_metrics(auc=0.9070, recall=0.9780), _registry(), "1.0.0")
    assert decision["action"] == "unchanged"
    assert decision["model_version"] == "1.0.0"


def test_a_stronger_model_promotes_a_patch() -> None:
    decision = decide(_metrics(auc=0.9400), _registry(), "1.0.0")
    assert decision["action"] == "promote"
    assert decision["model_version"] == "1.0.1"
    assert decision["reason"] == "Gate passed."


def test_a_new_data_version_promotes_a_minor() -> None:
    decision = decide(_metrics(), _registry(), "1.1.0")
    assert decision["action"] == "promote"
    assert decision["model_version"] == "1.1.0"
    assert decision["reason"] == "Data version changed."


def test_a_weaker_model_is_rejected() -> None:
    decision = decide(_metrics(auc=0.80, recall=0.70), _registry(), "1.0.0")
    assert decision["action"] == "reject"
    assert decision["model_version"] == "1.0.0"


def test_low_recall_is_rejected_even_when_auc_holds() -> None:
    decision = decide(_metrics(recall=0.85), _registry(), "1.0.0")
    assert decision["action"] == "reject"


def test_first_release_starts_at_1_0_0() -> None:
    decision = decide(_metrics(), {"production": None, "releases": []}, "1.0.0")
    assert decision["action"] == "promote"
    assert decision["model_version"] == "1.0.0"


def test_next_version_bumps_patch_or_minor() -> None:
    assert next_version("1.0.0", data_changed=False) == "1.0.1"
    assert next_version("1.2.4", data_changed=True) == "1.3.0"


def test_promote_rejects_without_replacing_the_model(tmp_path, monkeypatch) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(_metrics(auc=0.70, recall=0.40)), encoding="utf-8")
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model_version": "1.0.0"}, model_path)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(_registry()), encoding="utf-8")
    output = tmp_path / "github.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr("cardiascreen.release.DECISION_PATH", tmp_path / "decision.json")

    with pytest.raises(SystemExit):
        promote(metrics_path, model_path, registry_path)

    assert joblib.load(model_path) == {"model_version": "1.0.0"}
    text = output.read_text(encoding="utf-8")
    assert "action=reject" in text
    assert "version=1.0.0" in text


def test_promote_writes_the_new_version_onto_a_passing_model(tmp_path, monkeypatch) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(_metrics(auc=0.95)), encoding="utf-8")
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model_version": "1.0.0", "pipeline": None}, model_path)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(_registry()), encoding="utf-8")
    monkeypatch.setattr("cardiascreen.release.DECISION_PATH", tmp_path / "decision.json")

    decision = promote(metrics_path, model_path, registry_path)

    assert decision["action"] == "promote"
    assert decision["model_version"] == "1.0.1"
    saved = joblib.load(model_path)
    assert saved["model_version"] == "1.0.1"
    assert saved["data_version"] == "1.0.0"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["production"] == "1.0.1"
    assert registry["releases"][0]["status"] == "archived"
    assert registry["releases"][1]["status"] == "production"
