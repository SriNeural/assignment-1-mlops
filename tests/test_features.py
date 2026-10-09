import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from cardiascreen.config import MODEL_INPUT_COLUMNS
from cardiascreen.data import load_modeling_table
from cardiascreen.features import ClinicalFeatureBuilder, build_pipeline


def _payload(**overrides) -> dict:
    row = {
        "age": 54,
        "sex": 1,
        "cp": 4,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 2,
        "ca": 0,
        "thal": 3,
    }
    row.update(overrides)
    return row


def test_zero_cholesterol_sets_missing_indicator() -> None:
    built = ClinicalFeatureBuilder().transform(pd.DataFrame([_payload(chol=0)]))
    assert built.loc[0, "missing_chol"] == 1.0
    assert pd.isna(built.loc[0, "chol"])
    assert built.loc[0, "cp"] == "4"
    assert built.loc[0, "thal"] == "3"


def test_invasive_tests_can_be_absent() -> None:
    built = ClinicalFeatureBuilder().transform(
        pd.DataFrame([_payload(ca=None, thal=None, fbs=None)])
    )
    assert built.loc[0, "missing_ca"] == 1.0
    assert built.loc[0, "missing_thal"] == 1.0
    assert built.loc[0, "missing_fbs"] == 1.0


def test_site_and_label_never_enter_the_matrix() -> None:
    table = load_modeling_table()
    x = table[MODEL_INPUT_COLUMNS].copy()
    x["cohort"] = table["cohort"]
    x["target"] = table["target"]
    y = table["target"]
    x_train, _, y_train, _ = train_test_split(x, y, train_size=180, stratify=y, random_state=0)
    pipeline = build_pipeline(LogisticRegression(max_iter=500, class_weight="balanced"))
    pipeline.fit(x_train, y_train)
    raw_names = [
        name.split("__", 1)[-1]
        for name in pipeline.named_steps["prep"].get_feature_names_out()
    ]
    assert "cohort" not in raw_names
    assert "target" not in raw_names
    assert "num" not in raw_names
    proba = pipeline.predict_proba(x_train.head(5))
    assert proba.shape == (5, 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_percent_predicted_heart_rate_is_bounded_for_normal_ages() -> None:
    built = ClinicalFeatureBuilder().transform(pd.DataFrame([_payload(age=50, thalach=170)]))
    assert 0.5 < built.loc[0, "hr_pct_predicted"] < 1.2
