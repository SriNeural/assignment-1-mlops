"""Figures for EDA, the model report, and the architecture diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    fbeta_score,
    precision_score,
    recall_score,
)

from cardiascreen.config import CP_LABELS, F_BETA

plt.switch_backend("Agg")

DISEASE = "#9C2F2F"
HEALTHY = "#1F6F5B"
INK = "#1C2833"
COHORT_COLORS = {
    "cleveland": "#1F4E79",
    "hungarian": "#B9770E",
    "switzerland": "#148F77",
    "va": "#6C3483",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFBF9",
            "axes.edgecolor": "#D5D8DC",
            "axes.labelcolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.grid": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cohort_balance(frame: pd.DataFrame, path: Path) -> Path:
    apply_style()
    summary = (
        frame.groupby("cohort")
        .agg(patients=("target", "size"), disease_rate=("target", "mean"))
        .reindex(["cleveland", "hungarian", "switzerland", "va"])
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    colors = [COHORT_COLORS[name] for name in summary.index]
    axes[0].bar(summary.index, summary["patients"], color=colors)
    axes[0].set_title("Patients by source cohort")
    axes[0].set_ylabel("Rows")
    axes[1].bar(summary.index, summary["disease_rate"], color=colors)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Disease prevalence")
    axes[1].set_ylabel("Share labeled disease")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle("The four UCI sites are not one population", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_missingness(frame: pd.DataFrame, path: Path) -> Path:
    apply_style()
    columns = [
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
    rates = (
        frame.groupby("cohort")[columns]
        .apply(lambda part: part.isna().mean())
        .reindex(["cleveland", "hungarian", "switzerland", "va"])
    )
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    sns.heatmap(
        rates,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Fraction missing"},
    )
    ax.set_title("Missingness is a site property, not a random hole")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return _save(fig, path)


def plot_correlation(frame: pd.DataFrame, path: Path) -> Path:
    """Correlation on Cleveland only. The other sites would correlate missingness, not biology."""
    apply_style()
    columns = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "exang", "target"]
    cleveland = frame.loc[frame["cohort"] == "cleveland", columns]
    corr = cleveland.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        square=True,
    )
    ax.set_title("Cleveland correlations (the only nearly complete cohort)")
    fig.tight_layout()
    return _save(fig, path)


def plot_distributions(frame: pd.DataFrame, path: Path) -> Path:
    apply_style()
    specs = [
        ("age", "Age"),
        ("trestbps", "Resting blood pressure"),
        ("chol", "Serum cholesterol"),
        ("thalach", "Max heart rate"),
        ("oldpeak", "Exercise ST depression"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4))
    flat = list(axes.ravel())
    for axis, (column, title) in zip(flat, specs):
        for label, color, name in ((0, HEALTHY, "no disease"), (1, DISEASE, "disease")):
            values = frame.loc[frame["target"] == label, column].dropna()
            axis.hist(values, bins=18, alpha=0.62, color=color, label=name, density=True)
        axis.set_title(title)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    flat[0].legend(frameon=False, fontsize=8)
    flat[-1].axis("off")
    note = (
        "Cholesterol is plotted only where it was measured. "
        "Switzerland records 0 for every cholesterol value; those zeros are treated as missing."
    )
    flat[-1].text(0.0, 0.6, note, fontsize=9, color=INK, wrap=True, va="center")
    fig.suptitle("Observed clinical distributions by angiographic label", fontsize=14)
    fig.tight_layout()
    return _save(fig, path)


def plot_chest_pain(frame: pd.DataFrame, path: Path) -> Path:
    apply_style()
    work = frame.dropna(subset=["cp"]).copy()
    work["cp_code"] = work["cp"].round().astype(int).astype(str)
    grouped = work.groupby("cp_code")["target"].agg(["mean", "count"]).reindex(["1", "2", "3", "4"])
    labels = [f"{code}\n{CP_LABELS[code]}" for code in grouped.index]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bars = ax.bar(labels, grouped["mean"], color=[HEALTHY, "#2E86AB", "#B9770E", DISEASE])
    for bar, count, rate in zip(bars, grouped["count"], grouped["mean"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.03,
            f"n={int(count)}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Disease rate")
    ax.set_title("Asymptomatic chest-pain code carries the highest disease rate")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def plot_engineered_scatter(features: pd.DataFrame, target: pd.Series, path: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for label, color, name in ((0, HEALTHY, "no disease"), (1, DISEASE, "disease")):
        mask = target.to_numpy() == label
        ax.scatter(
            features.loc[mask, "hr_pct_predicted"],
            features.loc[mask, "oldpeak"],
            s=16,
            alpha=0.55,
            c=color,
            label=name,
            linewidths=0,
        )
    ax.set_xlabel("Heart rate as a fraction of predicted maximum (220 − age)")
    ax.set_ylabel("ST depression (oldpeak)")
    ax.set_title("Engineered effort feature against exercise ST change")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def plot_roc(y_true, proba, path: Path, title: str) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    RocCurveDisplay.from_predictions(
        y_true,
        proba,
        ax=ax,
        name="Holdout",
        curve_kwargs={"color": DISEASE},
    )
    ax.plot([0, 1], [0, 1], color="#ABB2B9", linestyle="--", linewidth=1)
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, path)


def plot_pr(y_true, proba, path: Path, title: str) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    PrecisionRecallDisplay.from_predictions(
        y_true,
        proba,
        ax=ax,
        name="Holdout",
        curve_kwargs={"color": "#1F4E79"},
    )
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion(y_true, proba, threshold: float, path: Path) -> Path:
    apply_style()
    predicted = (np.asarray(proba) >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predicted,
        display_labels=["no disease", "disease"],
        cmap="Reds",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Holdout decisions at threshold {threshold:.2f}")
    fig.tight_layout()
    return _save(fig, path)


def plot_threshold_curve(y_true, proba, threshold: float, path: Path) -> Path:
    apply_style()
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    grid = np.linspace(0.05, 0.95, 37)
    precision, recall, f2 = [], [], []
    for value in grid:
        predicted = (proba >= value).astype(int)
        precision.append(precision_score(y_true, predicted, zero_division=0))
        recall.append(recall_score(y_true, predicted, zero_division=0))
        f2.append(fbeta_score(y_true, predicted, beta=F_BETA, zero_division=0))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(grid, precision, label="precision", color="#1F4E79")
    ax.plot(grid, recall, label="recall", color=DISEASE)
    ax.plot(grid, f2, label="F2", color="#B9770E")
    ax.axvline(threshold, color=INK, linestyle="--", linewidth=1, label=f"chosen {threshold:.2f}")
    ax.set_xlabel("Threshold on P(disease)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Operating point is the F2 maximum on training out-of-fold scores")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def plot_calibration(y_true, proba, path: Path) -> Path:
    apply_style()
    fraction, mean_predicted = calibration_curve(y_true, proba, n_bins=6, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#ABB2B9", label="perfect")
    ax.plot(mean_predicted, fraction, marker="o", color=DISEASE, label="holdout")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed disease rate")
    ax.set_title("Holdout calibration")
    ax.legend(frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return _save(fig, path)


def plot_importance(names: np.ndarray, values: np.ndarray, path: Path, title: str) -> Path:
    apply_style()
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.barh(np.asarray(names)[order], np.asarray(values)[order], color="#1F4E79")
    ax.set_title(title)
    ax.set_xlabel("Absolute coefficient or impurity importance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def plot_leaderboard(rows: list[dict], path: Path) -> Path:
    apply_style()
    names = [row["model_name"] for row in rows]
    means = [row["cv_roc_auc"] for row in rows]
    stds = [row["cv_roc_auc_std"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.barh(names, means, xerr=stds, color="#1F4E79", alpha=0.9, capsize=4)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Cross-validated ROC-AUC on the training split")
    ax.set_title("Model selection metric (holdout was not used)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def plot_architecture(path: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(11.4, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    boxes = [
        (0.3, 4.6, 2.3, 1.3, "UCI cohorts\nCleveland, Hungary,\nSwitzerland, VA"),
        (3.1, 4.6, 2.2, 1.3, "Clean + features\nzero-sentinel chol\nHR% of predicted max"),
        (5.8, 4.6, 2.3, 1.3, "Train / select\nLR, RF, HGB\nMLflow + F2 threshold"),
        (8.6, 4.6, 2.8, 1.3, "Joblib artifact\npipeline + threshold\nbaked into the image"),
        (3.1, 2.2, 2.4, 1.3, "FastAPI\n/predict /health\n/metrics"),
        (6.1, 2.2, 2.5, 1.3, "Docker image\nCompose or Kubernetes\nHugging Face Space"),
        (9.1, 2.2, 2.4, 1.3, "Prometheus\n+ Grafana\nrequest and score drift"),
        (0.3, 2.2, 2.3, 1.3, "GitHub Actions\nruff, pytest,\nretrain, image smoke"),
    ]
    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor="#F4F7FB", edgecolor="#1F4E79", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color=INK)
    for start, end in (
        ((2.6, 5.25), (3.1, 5.25)),
        ((5.3, 5.25), (5.8, 5.25)),
        ((8.1, 5.25), (8.6, 5.25)),
        ((10.0, 4.6), (10.0, 3.5)),
        ((4.3, 4.6), (4.3, 3.5)),
        ((7.3, 2.85), (9.1, 2.85)),
        ((2.6, 2.85), (3.1, 2.85)),
    ):
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#1F4E79", "lw": 1.1},
        )
    ax.set_title("CardiaScreen production path", loc="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def readable_feature_names(pipeline) -> tuple[np.ndarray, np.ndarray] | None:
    model = pipeline.named_steps["clf"]
    names = pipeline.named_steps["prep"].get_feature_names_out()
    cleaned = [name.split("__", 1)[-1] for name in names]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:
        return None
    order = np.argsort(values)[::-1][:12]
    return np.asarray(cleaned)[order], values[order]
