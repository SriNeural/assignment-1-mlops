"""Train, select, track, and package the screening model.

Selection uses cross-validated ROC-AUC on the training split only. The decision
threshold maximizes F2 on training out-of-fold probabilities. The holdout split
is scored once, after the model and the threshold are frozen.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from cardiascreen.config import (
    EXPERIMENT_NAME,
    FIGURES_DIR,
    HOLDOUT_SIZE,
    METRICS_PATH,
    MLRUNS_DIR,
    MODEL_INPUT_COLUMNS,
    MODEL_PATH,
    MODEL_VERSION,
    RANDOM_STATE,
)
from cardiascreen.data import prepare_data
from cardiascreen.evaluate import best_fbeta_threshold, classification_scores, to_builtin
from cardiascreen.features import build_pipeline
from cardiascreen.plots import (
    plot_calibration,
    plot_confusion,
    plot_importance,
    plot_leaderboard,
    plot_pr,
    plot_roc,
    plot_threshold_curve,
    readable_feature_names,
)
from cardiascreen.tracking import register_model_uri
from cardiascreen.versioning import check_data_version


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable[[], object]
    grid: dict


def _logistic() -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def _forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=250,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
    )


def _boosting() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def model_specs(fast: bool) -> list[ModelSpec]:
    if fast:
        return [ModelSpec("logistic_regression", _logistic, {"clf__C": [1.0]})]
    return [
        ModelSpec("logistic_regression", _logistic, {"clf__C": [0.1, 1.0, 10.0]}),
        ModelSpec(
            "random_forest",
            _forest,
            {"clf__max_depth": [6, 12, None], "clf__min_samples_leaf": [2, 8]},
        ),
        ModelSpec(
            "hist_gradient_boosting",
            _boosting,
            {"clf__max_depth": [3, None], "clf__learning_rate": [0.05, 0.1]},
        ),
    ]


def _stringify_params(params: dict) -> dict:
    rendered = {}
    for key, value in params.items():
        rendered[key] = "none" if value is None else value
    return rendered


def _fit_candidate(spec: ModelSpec, x_train, y_train, n_splits: int) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=build_pipeline(spec.builder()),
        param_grid=spec.grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(x_train, y_train)
    best_index = int(search.best_index_)
    return {
        "model_name": spec.name,
        "cv_roc_auc": float(search.best_score_),
        "cv_roc_auc_std": float(search.cv_results_["std_test_score"][best_index]),
        "best_params": _stringify_params(search.best_params_),
        "estimator": search.best_estimator_,
        "spec": spec,
    }


def _oof_probability(estimator, x_train, y_train, n_splits: int) -> np.ndarray:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(
        clone(estimator),
        x_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )
    classes = list(estimator.named_steps["clf"].classes_)
    return oof[:, classes.index(1)]


def _per_cohort(y_true, proba, cohorts, threshold: float) -> dict:
    scores = {}
    cohorts = np.asarray(cohorts)
    for name in sorted(set(cohorts)):
        mask = cohorts == name
        if int(mask.sum()) == 0:
            continue
        scores[name] = classification_scores(y_true[mask], proba[mask], threshold)
    return scores


def _transport_from_cleveland(spec: ModelSpec, params: dict, frame: pd.DataFrame) -> dict:
    """Fit the chosen algorithm on Cleveland only and score the other sites.

    This is a transport check. It is not the production model and it is not used
    to pick the winner.
    """
    pipeline = build_pipeline(spec.builder())
    pipeline.set_params(**_restore_params(params))
    cleveland = frame["cohort"] == "cleveland"
    pipeline.fit(frame.loc[cleveland, MODEL_INPUT_COLUMNS], frame.loc[cleveland, "target"])
    report = {"train_cohort": "cleveland", "train_rows": int(cleveland.sum()), "sites": {}}
    classes = list(pipeline.named_steps["clf"].classes_)
    class_index = classes.index(1)
    for cohort in ("hungarian", "switzerland", "va"):
        mask = frame["cohort"] == cohort
        proba = pipeline.predict_proba(frame.loc[mask, MODEL_INPUT_COLUMNS])[:, class_index]
        report["sites"][cohort] = classification_scores(
            frame.loc[mask, "target"],
            proba,
            threshold=0.5,
        )
    return report


def _restore_params(params: dict) -> dict:
    restored = {}
    for key, value in params.items():
        restored[key] = None if value == "none" else value
    return restored


def _log_mlflow_model(pipeline, x_example: pd.DataFrame):
    # Float example so MLflow's schema allows the nulls the API accepts.
    example = x_example.head(5).astype("float64")
    return mlflow.sklearn.log_model(
        pipeline,
        name="model",
        input_example=example,
        skops_trusted_types=[
            "cardiascreen.features.ClinicalFeatureBuilder",
            "numpy.dtype",
        ],
    )


def train(
    fast: bool = False,
    output_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    figure_dir: Path = FIGURES_DIR,
    tracking_uri: Path | None = None,
    download: bool = True,
) -> dict:
    frame = prepare_data(download=download)
    data_pin = check_data_version()
    x = frame[MODEL_INPUT_COLUMNS]
    y = frame["target"].to_numpy()
    cohort = frame["cohort"].to_numpy()
    x_train, x_hold, y_train, y_hold, _cohort_train, c_hold = train_test_split(
        x,
        y,
        cohort,
        test_size=HOLDOUT_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    n_splits = 3 if fast else 5
    store = Path(tracking_uri) if tracking_uri else MLRUNS_DIR
    store.mkdir(parents=True, exist_ok=True)
    database = (store / "mlflow.db").resolve()
    mlflow.set_tracking_uri("sqlite:///" + database.as_posix())
    artifact_location = (store / "artifacts").resolve().as_uri()
    client = mlflow.tracking.MlflowClient()
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_location)
    mlflow.set_experiment(EXPERIMENT_NAME)
    # MLflow 3 opens an experiment on the empty GenAI page unless the kind is set.
    mlflow.set_experiment_tag("mlflow.experimentKind", "classification")

    leaderboard = []
    with mlflow.start_run(run_name="model-selection"):
        mlflow.log_params(
            {
                "holdout_size": HOLDOUT_SIZE,
                "random_state": RANDOM_STATE,
                "n_rows": int(len(frame)),
                "n_train": int(len(x_train)),
                "n_holdout": int(len(x_hold)),
                "positive_rate": float(y.mean()),
                "fast": fast,
                "cv_folds": n_splits,
                "selection_metric": "roc_auc",
                "threshold_rule": "max_f2_on_training_oof",
                "data_version": data_pin["data_version"],
            }
        )
        mlflow.set_tags(
            {
                "site_feature": "excluded",
                "label": "num>0 angiographic narrowing",
                "data_version": data_pin["data_version"],
            }
        )
        for spec in model_specs(fast):
            with mlflow.start_run(run_name=spec.name, nested=True):
                record = _fit_candidate(spec, x_train, y_train, n_splits)
                mlflow.log_params(record["best_params"])
                mlflow.log_metrics(
                    {
                        "cv_roc_auc": record["cv_roc_auc"],
                        "cv_roc_auc_std": record["cv_roc_auc_std"],
                    }
                )
                leaderboard.append(record)
                print(
                    f"{record['model_name']:24} "
                    f"cv_auc={record['cv_roc_auc']:.4f} "
                    f"std={record['cv_roc_auc_std']:.4f} "
                    f"params={record['best_params']}"
                )

        winner = max(leaderboard, key=lambda row: row["cv_roc_auc"])
        oof = _oof_probability(winner["estimator"], x_train, y_train, n_splits)
        threshold, oof_f2 = best_fbeta_threshold(y_train, oof)
        holdout_proba = winner["estimator"].predict_proba(x_hold)
        classes = list(winner["estimator"].named_steps["clf"].classes_)
        holdout_proba = holdout_proba[:, classes.index(1)]
        holdout = classification_scores(y_hold, holdout_proba, threshold)
        holdout_default = classification_scores(y_hold, holdout_proba, 0.5)
        by_cohort = _per_cohort(y_hold, holdout_proba, c_hold, threshold)
        transport = _transport_from_cleveland(winner["spec"], winner["best_params"], frame)

        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_leaderboard(leaderboard, figure_dir / "model_leaderboard.png")
        plot_roc(
            y_hold,
            holdout_proba,
            figure_dir / "model_roc.png",
            f"{winner['model_name']} holdout ROC",
        )
        plot_pr(
            y_hold,
            holdout_proba,
            figure_dir / "model_pr.png",
            f"{winner['model_name']} holdout precision-recall",
        )
        plot_confusion(y_hold, holdout_proba, threshold, figure_dir / "model_confusion.png")
        plot_threshold_curve(y_train, oof, threshold, figure_dir / "model_threshold.png")
        plot_calibration(y_hold, holdout_proba, figure_dir / "model_calibration.png")
        importance = readable_feature_names(winner["estimator"])
        if importance is not None:
            plot_importance(
                importance[0],
                importance[1],
                figure_dir / "model_importance.png",
                f"{winner['model_name']} top features",
            )
        mlflow.log_artifacts(str(figure_dir))
        mlflow.log_metrics(
            {
                "threshold": threshold,
                "oof_f2": oof_f2,
                "holdout_roc_auc": holdout["roc_auc"],
                "holdout_recall": holdout["recall"],
                "holdout_precision": holdout["precision"],
                "holdout_f2": holdout["f2"],
                "holdout_accuracy": holdout["accuracy"],
            }
        )
        mlflow.log_params({"winner": winner["model_name"], **winner["best_params"]})
        info = _log_mlflow_model(winner["estimator"], x_train)
        if info.model_id:
            model_uri = f"models:/{info.model_id}"
        else:
            model_uri = f"runs:/{info.run_id}/{info.artifact_path}"
        register_model_uri(
            model_uri,
            description=(
                f"{winner['model_name']} {MODEL_VERSION}, threshold {threshold:.2f}, "
                f"holdout ROC-AUC {holdout['roc_auc']:.3f}."
            ),
        )

        trained_at = datetime.now(UTC).replace(microsecond=0).isoformat()
        metrics = {
            "model_name": winner["model_name"],
            "model_version": MODEL_VERSION,
            "data_version": data_pin["data_version"],
            "trained_at": trained_at,
            "selection_metric": "roc_auc",
            "threshold_rule": "max_f2_on_training_oof",
            "threshold": threshold,
            "oof_f2": oof_f2,
            "rows": int(len(frame)),
            "train_rows": int(len(x_train)),
            "holdout_rows": int(len(x_hold)),
            "positive_rate": float(y.mean()),
            "best_params": winner["best_params"],
            "leaderboard": [
                {
                    "model_name": row["model_name"],
                    "cv_roc_auc": row["cv_roc_auc"],
                    "cv_roc_auc_std": row["cv_roc_auc_std"],
                    "best_params": row["best_params"],
                }
                for row in leaderboard
            ],
            "holdout": holdout,
            "holdout_at_0_5": holdout_default,
            "holdout_by_cohort": by_cohort,
            "cleveland_transport": transport,
            "site_feature": "excluded",
            "random_state": RANDOM_STATE,
        }
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(to_builtin(metrics), indent=2), encoding="utf-8")
        leaderboard_path = metrics_path.parent / "leaderboard.csv"
        pd.DataFrame(metrics["leaderboard"]).to_csv(leaderboard_path, index=False)

        artifact = {
            "pipeline": winner["estimator"],
            "model_name": winner["model_name"],
            "model_version": MODEL_VERSION,
            "data_version": data_pin["data_version"],
            "threshold": threshold,
            "trained_at": trained_at,
            "metrics": metrics,
            "input_columns": MODEL_INPUT_COLUMNS,
            "disclaimer": (
                "Screening score for coursework. Not a diagnosis and not a medical device."
            ),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, output_path)
        mlflow.log_artifact(str(output_path))
        mlflow.log_artifact(str(metrics_path))

    print(
        f"winner={winner['model_name']} threshold={threshold:.2f} "
        f"holdout_auc={holdout['roc_auc']:.4f} holdout_recall={holdout['recall']:.4f}"
    )
    print(f"saved {output_path}")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CardiaScreen model.")
    parser.add_argument("--fast", action="store_true", help="One model, for smoke tests.")
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    train(fast=args.fast, output_path=args.output, download=not args.skip_download)


if __name__ == "__main__":
    main()
