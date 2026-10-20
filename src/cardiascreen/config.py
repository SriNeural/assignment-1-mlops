"""Paths, column contracts, and training constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
LOGS_DIR = PROJECT_ROOT / "logs"

MODEL_FILENAME = "heart_risk_model.joblib"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
METRICS_PATH = REPORTS_DIR / "metrics.json"
PROCESSED_CSV = DATA_PROCESSED / "heart_disease_combined.csv"

# Column order of the UCI processed.*.data files. There is no header row.
FILE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

# Raw inputs the serving API and the feature builder both accept.
# `ca` and `thal` stay optional at score time: they come from fluoroscopy and a
# thallium scan, which a screening request often does not have.
MODEL_INPUT_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

COHORT_FILES = {
    "cleveland": "cleveland.data",
    "hungarian": "hungarian.data",
    "switzerland": "switzerland.data",
    "va": "va.data",
}

UCI_URLS = {
    "cleveland": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "heart-disease/processed.cleveland.data"
    ),
    "hungarian": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "heart-disease/processed.hungarian.data"
    ),
    "switzerland": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "heart-disease/processed.switzerland.data"
    ),
    "va": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data"
    ),
}

# UCI cp codes are 1-4, not the 0-3 recoding used by many Kaggle mirrors.
CP_LABELS = {
    "1": "typical angina",
    "2": "atypical angina",
    "3": "non-anginal pain",
    "4": "asymptomatic",
}

RANDOM_STATE = 42
HOLDOUT_SIZE = 0.20
MODEL_VERSION = "1.0.0"
# beta=2 weights recall above precision: a missed narrowing is costlier than an extra review.
F_BETA = 2.0
EXPERIMENT_NAME = "cardiascreen-heart-disease"
REGISTERED_MODEL_NAME = "cardiascreen-heart-risk"
CHAMPION_ALIAS = "champion"
