"""Clinical feature construction and the sklearn training pipeline.

The serving process unpickles ``ClinicalFeatureBuilder``, so this module path
has to stay ``cardiascreen.features`` in every environment that loads the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cardiascreen.config import MODEL_INPUT_COLUMNS

NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca",
    "hr_pct_predicted",
    "chol_per_age",
    "bp_per_age",
    "st_pain_product",
]
BINARY_FEATURES = [
    "sex",
    "fbs",
    "exang",
    "missing_chol",
    "missing_ca",
    "missing_thal",
    "missing_fbs",
]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
ENGINEERED_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def _category_label(value: object) -> object:
    """Map UCI codes such as 3.0 and '3' onto one string, leaving missing as NaN."""
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except TypeError:
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(number):
        return np.nan
    if number.is_integer():
        return str(int(number))
    return str(number)


class ClinicalFeatureBuilder(BaseEstimator, TransformerMixin):
    """Build screening features from raw UCI fields.

    Site (``cohort``) is intentionally dropped. A random split that is allowed
    to see the hospital learns that hospital's base rate, then looks stronger
    than a model that has to score a new clinic. Missingness indicators stay,
    because "this test was not done" is observed at prediction time.
    """

    def fit(self, X, y=None):  # noqa: N803 - sklearn transformer signature
        return self

    def transform(self, X):  # noqa: N803
        frame = self._as_frame(X)
        for column in MODEL_INPUT_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        # Repeat the zero-sentinel rule so a raw API payload cannot bypass cleaning.
        for column in ("chol", "trestbps", "thalach"):
            frame.loc[frame[column] <= 0, column] = np.nan

        age = frame["age"].clip(lower=1)
        predicted_max_hr = (220 - age).clip(lower=1)
        frame["hr_pct_predicted"] = frame["thalach"] / predicted_max_hr
        frame["chol_per_age"] = frame["chol"] / age
        frame["bp_per_age"] = frame["trestbps"] / age
        exercise_pain = frame["exang"].fillna(0)
        frame["st_pain_product"] = frame["oldpeak"] * (1 + exercise_pain)

        frame["missing_chol"] = frame["chol"].isna().astype(float)
        frame["missing_ca"] = frame["ca"].isna().astype(float)
        frame["missing_thal"] = frame["thal"].isna().astype(float)
        frame["missing_fbs"] = frame["fbs"].isna().astype(float)

        for column in CATEGORICAL_FEATURES:
            frame[column] = frame[column].map(_category_label).astype("object")
        return frame[ENGINEERED_COLUMNS]

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:  # noqa: N803
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            frame = pd.DataFrame(X, columns=MODEL_INPUT_COLUMNS)
        missing = [column for column in MODEL_INPUT_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Input is missing columns: {missing}")
        return frame


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    binary = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("bin", binary, BINARY_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("clinical", ClinicalFeatureBuilder()),
            ("prep", build_preprocessor()),
            ("clf", estimator),
        ]
    )
