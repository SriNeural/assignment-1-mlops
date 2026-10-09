"""Download and clean the four UCI Heart Disease cohorts."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from cardiascreen.config import (
    COHORT_FILES,
    DATA_RAW,
    FILE_COLUMNS,
    PROCESSED_CSV,
    PROJECT_ROOT,
    UCI_URLS,
)

# Some cohorts store "unknown" cholesterol or blood pressure as 0, not as "?".
SENTINEL_ZERO_COLUMNS = ("chol", "trestbps", "thalach")
CHECKSUM_PATH = PROJECT_ROOT / "data" / "CHECKSUMS.txt"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def download_raw(dest: Path = DATA_RAW, force: bool = False) -> list[Path]:
    """Download the official processed UCI files. Existing files are kept."""
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for cohort, url in UCI_URLS.items():
        target = dest / COHORT_FILES[cohort]
        if target.exists() and target.stat().st_size > 0 and not force:
            written.append(target)
            continue
        urllib.request.urlretrieve(url, target)
        written.append(target)
    _write_checksums(dest)
    return written


def _write_checksums(dest: Path) -> None:
    lines = []
    for cohort, filename in COHORT_FILES.items():
        path = dest / filename
        lines.append(f"{file_sha256(path)}  {filename}  {cohort}")
    CHECKSUM_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKSUM_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_cohort_file(path: Path, cohort: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        header=None,
        names=FILE_COLUMNS,
        na_values="?",
        skipinitialspace=True,
    )
    if len(frame.columns) != len(FILE_COLUMNS):
        raise ValueError(f"{path} does not have the 14 UCI columns.")
    frame["cohort"] = cohort
    return frame


def clean_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, drop unlabeled rows, and turn zero-sentinels into missing values.

    The binary label follows the UCI definition: ``num == 0`` means less than 50%
    diameter narrowing, and ``num`` of 1-4 means disease is present. Severity is
    not modeled; the assignment asks for a binary risk classifier.
    """
    out = frame.copy()
    for column in FILE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    before = len(out)
    out = out.dropna(subset=["num"]).reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        out.attrs["dropped_unlabeled"] = dropped
    out["target"] = (out["num"] > 0).astype(int)
    for column in SENTINEL_ZERO_COLUMNS:
        out.loc[out[column] <= 0, column] = np.nan
    out["cohort"] = out["cohort"].astype(str)
    return out


def load_modeling_table(raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    frames = []
    for cohort, filename in COHORT_FILES.items():
        path = Path(raw_dir) / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `python -m cardiascreen.data` to download the UCI files."
            )
        frames.append(load_cohort_file(path, cohort))
    combined = pd.concat(frames, ignore_index=True)
    return clean_table(combined)


def save_processed(frame: pd.DataFrame, path: Path = PROCESSED_CSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def prepare_data(raw_dir: Path = DATA_RAW, download: bool = True) -> pd.DataFrame:
    if download:
        download_raw(raw_dir)
    table = load_modeling_table(raw_dir)
    save_processed(table)
    return table


def main() -> None:
    table = prepare_data()
    print(f"rows={len(table)} positive_rate={table['target'].mean():.3f}")
    print(f"wrote {PROCESSED_CSV}")
    print(table.groupby("cohort")["target"].agg(["count", "mean"]).to_string())


if __name__ == "__main__":
    main()
