"""Write notebooks/01_eda_and_inference.ipynb with executed outputs."""

from __future__ import annotations

import base64
import io
import sys
import traceback
import types
from contextlib import redirect_stdout
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_eda_and_inference.ipynb"

MARKDOWN = [
    """# CardiaScreen — exploratory review and scoring

This notebook reads the cleaned UCI table and the packaged model.
Training itself is `python -m cardiascreen.train`, which writes
`reports/metrics.json` and `models/heart_risk_model.joblib`.
Re-run that command before a full refresh.
The score is a coursework screening signal, not a diagnosis.
""",
    """## Cohort balance

The four processed UCI files are one table. Site is kept for description and is not a model feature.
""",
    """## Figures

The same files are embedded in the written report.
""",
    """## Frozen experiment

Model choice used training-split ROC-AUC only.
The threshold maximizes F2 on training out-of-fold scores.
""",
    """## Sample scores

`ca` and `thal` can be omitted. Chest pain uses UCI codes 1 through 4.
""",
]

CODE = [
    """from pathlib import Path
import json
import pandas as pd

root = Path.cwd()
if not (root / "reports" / "metrics.json").exists():
    root = Path.cwd().parent
table = pd.read_csv(root / "data" / "processed" / "heart_disease_combined.csv")
summary = table.groupby("cohort")["target"].agg(rows="size", disease_rate="mean").round(3)
print(f"rows={len(table)} prevalence={table['target'].mean():.3f}")
print(summary.to_string())
print("missing fraction")
missing = (
    table.drop(columns=["target", "num"])
    .isna()
    .mean()
    .round(3)
    .sort_values(ascending=False)
    .head(8)
)
print(missing.to_string())
""",
    """from IPython.display import Image, display
from pathlib import Path

root = Path.cwd()
if not (root / "reports" / "figures").exists():
    root = Path.cwd().parent
names = [
    "eda_cohort_balance.png",
    "eda_missingness.png",
    "eda_chest_pain.png",
    "eda_correlation.png",
    "model_roc.png",
    "model_confusion.png",
    "architecture.png",
]
for name in names:
    print(name)
    display(Image(filename=str(root / "reports" / "figures" / name)))
""",
    """import json
from pathlib import Path

root = Path.cwd()
if not (root / "reports" / "metrics.json").exists():
    root = Path.cwd().parent
metrics = json.loads((root / "reports" / "metrics.json").read_text(encoding="utf-8"))
print("winner", metrics["model_name"], "threshold", round(metrics["threshold"], 2))
for row in metrics["leaderboard"]:
    print(f"{row['model_name']:24} cv_auc={row['cv_roc_auc']:.3f}  sd={row['cv_roc_auc_std']:.3f}")
holdout = metrics["holdout"]
print(
    "holdout",
    f"auc={holdout['roc_auc']:.3f}",
    f"recall={holdout['recall']:.3f}",
    f"precision={holdout['precision']:.3f}",
    f"accuracy={holdout['accuracy']:.3f}",
)
""",
    """import json
from pathlib import Path
import joblib

from cardiascreen.scoring import score_records

root = Path.cwd()
if not (root / "models" / "heart_risk_model.joblib").exists():
    root = Path.cwd().parent
artifact = joblib.load(root / "models" / "heart_risk_model.joblib")
for name in (
    "sample_request.json",
    "sample_request_low_risk.json",
    "sample_request_screening.json",
):
    payload = json.loads((root / "deploy" / name).read_text(encoding="utf-8"))
    decision = score_records(artifact, [payload])[0]
    print(name, decision["label"], decision["probability"], decision["risk_band"])
""",
]


class ImageDisplay:
    def __init__(self, bucket: list) -> None:
        self.bucket = bucket

    def __call__(self, obj) -> None:
        data = getattr(obj, "data", None)
        if data is None and hasattr(obj, "filename"):
            data = Path(obj.filename).read_bytes()
        if data is None:
            return
        encoded = base64.b64encode(data).decode("ascii")
        self.bucket.append(
            new_output(
                output_type="display_data",
                data={"image/png": encoded, "text/plain": "figure"},
                metadata={},
            )
        )


def execute(source: str) -> list:
    outputs = []
    buffer = io.StringIO()
    display_mod = types.ModuleType("IPython.display")
    display_mod.Image = _Image
    display_mod.display = ImageDisplay(outputs)
    ipython_mod = types.ModuleType("IPython")
    ipython_mod.display = display_mod
    previous = {name: sys.modules.get(name) for name in ("IPython", "IPython.display")}
    sys.modules["IPython"] = ipython_mod
    sys.modules["IPython.display"] = display_mod
    namespace: dict = {}
    try:
        with redirect_stdout(buffer):
            exec(compile(source, "<notebook>", "exec"), namespace, namespace)
    except Exception:
        outputs.append(
            new_output(
                output_type="error",
                ename="ExecutionError",
                evalue="cell failed",
                traceback=traceback.format_exc().splitlines(),
            )
        )
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    text = buffer.getvalue()
    if text:
        outputs.insert(0, new_output(output_type="stream", name="stdout", text=text))
    return outputs


class _Image:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.data = Path(filename).read_bytes()


def main() -> None:
    cells = [new_markdown_cell(MARKDOWN[0])]
    for index, source in enumerate(CODE):
        cells.append(new_markdown_cell(MARKDOWN[index + 1]))
        cell = new_code_cell(source)
        cell.outputs = execute(source)
        cell.execution_count = index + 1
        cells.append(cell)
    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, NOTEBOOK)
    failed = [
        cell.execution_count
        for cell in notebook.cells
        if cell.cell_type == "code" and any(item.output_type == "error" for item in cell.outputs)
    ]
    print(NOTEBOOK)
    print("failed_cells", failed)


if __name__ == "__main__":
    main()
