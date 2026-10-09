"""Exploratory figures for the combined UCI heart-disease table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cardiascreen.config import FIGURES_DIR, MODEL_INPUT_COLUMNS
from cardiascreen.data import prepare_data
from cardiascreen.features import ClinicalFeatureBuilder
from cardiascreen.plots import (
    plot_architecture,
    plot_chest_pain,
    plot_cohort_balance,
    plot_correlation,
    plot_distributions,
    plot_engineered_scatter,
    plot_missingness,
)


def run_eda(frame: pd.DataFrame | None = None, figure_dir: Path = FIGURES_DIR) -> dict[str, Path]:
    if frame is None:
        frame = prepare_data()
    figure_dir.mkdir(parents=True, exist_ok=True)
    features = ClinicalFeatureBuilder().transform(frame[MODEL_INPUT_COLUMNS]).reset_index(drop=True)
    target = frame["target"].reset_index(drop=True)
    written = {
        "cohort_balance": plot_cohort_balance(frame, figure_dir / "eda_cohort_balance.png"),
        "missingness": plot_missingness(frame, figure_dir / "eda_missingness.png"),
        "correlation": plot_correlation(frame, figure_dir / "eda_correlation.png"),
        "distributions": plot_distributions(frame, figure_dir / "eda_distributions.png"),
        "chest_pain": plot_chest_pain(frame, figure_dir / "eda_chest_pain.png"),
        "engineered": plot_engineered_scatter(
            features, target, figure_dir / "eda_engineered_scatter.png"
        ),
        "architecture": plot_architecture(figure_dir / "architecture.png"),
    }
    missing = frame[MODEL_INPUT_COLUMNS].isna().mean().sort_values(ascending=False)
    print(f"rows={len(frame)} prevalence={frame['target'].mean():.3f}")
    print("missing fraction:")
    print(missing.round(3).to_string())
    for name, path in written.items():
        print(f"{name}: {path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Write CardiaScreen EDA figures.")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    frame = prepare_data(download=not args.skip_download)
    run_eda(frame)


if __name__ == "__main__":
    main()
