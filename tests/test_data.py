import pandas as pd

from cardiascreen.config import FILE_COLUMNS
from cardiascreen.data import clean_table, load_modeling_table


def _row(num: int, **overrides) -> dict:
    row = {column: 1 for column in FILE_COLUMNS}
    row["num"] = num
    row["cohort"] = "cleveland"
    row.update(overrides)
    return row


def test_binary_label_collapses_severity() -> None:
    frame = pd.DataFrame([_row(num) for num in (0, 1, 2, 3, 4)])
    cleaned = clean_table(frame)
    assert cleaned["target"].tolist() == [0, 1, 1, 1, 1]


def test_zero_sentinels_become_missing() -> None:
    frame = pd.DataFrame([_row(0, chol=0, trestbps=0, thalach=0)])
    cleaned = clean_table(frame)
    assert pd.isna(cleaned.loc[0, "chol"])
    assert pd.isna(cleaned.loc[0, "trestbps"])
    assert pd.isna(cleaned.loc[0, "thalach"])


def test_unlabeled_rows_are_dropped() -> None:
    unlabeled = _row(1)
    unlabeled["num"] = None
    frame = pd.DataFrame([_row(0), unlabeled])
    cleaned = clean_table(frame)
    assert len(cleaned) == 1
    assert cleaned.attrs["dropped_unlabeled"] == 1


def test_real_cohorts_keep_site_specific_gaps() -> None:
    table = load_modeling_table()
    assert set(table["cohort"]) == {"cleveland", "hungarian", "switzerland", "va"}
    assert len(table) > 800
    assert (table["cohort"] == "cleveland").sum() == 303
    switzerland = table.loc[table["cohort"] == "switzerland", "chol"]
    assert switzerland.isna().all()
    assert table["oldpeak"].notna().any()
    assert set(table["target"].unique()) == {0, 1}
