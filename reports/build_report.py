"""Build the assignment report from the saved metrics, figures, and screenshots."""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
METRICS_PATH = ROOT / "reports" / "metrics.json"
MODEL_PATH = ROOT / "models" / "heart_risk_model.joblib"
OUTPUT = ROOT / "reports" / "AIMLCZG523_Assignment_01_Report.pdf"

NAVY = colors.HexColor("#1F4E79")
RULE = colors.HexColor("#D5D8DC")
ZEBRA = colors.HexColor("#F4F7FB")
INK = colors.HexColor("#1C2833")


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def styles() -> dict[str, ParagraphStyle]:
    return {
        "cover": ParagraphStyle(
            "cover",
            fontName="Times-Bold",
            fontSize=18,
            leading=22,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "sub": ParagraphStyle(
            "sub",
            fontName="Times-Bold",
            fontSize=12,
            leading=15,
            textColor=INK,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName="Times-Bold",
            fontSize=13,
            leading=16,
            textColor=NAVY,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="Times-Bold",
            fontSize=11,
            leading=14,
            textColor=INK,
            spaceBefore=8,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Times-Roman",
            fontSize=10,
            leading=13,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small",
            fontName="Times-Roman",
            fontSize=8,
            leading=10,
            textColor=INK,
        ),
        "head": ParagraphStyle(
            "head",
            fontName="Times-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Times-Italic",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#34495E"),
            spaceBefore=1,
            spaceAfter=6,
        ),
    }


def fitted(path: Path, width: float) -> Image:
    with PILImage.open(path) as image:
        pixel_w, pixel_h = image.size
    height = width * pixel_h / pixel_w
    return Image(str(path), width=width, height=height)


def crop_top(path: Path, width: float, fraction: float) -> Image:
    """Keep the top of a tall browser capture so empty page space is not printed."""
    with PILImage.open(path) as image:
        pixel_w, pixel_h = image.size
        kept = max(1, int(pixel_h * fraction))
        cropped = image.crop((0, 0, pixel_w, kept)).convert("RGB")
        buffer = io.BytesIO()
        cropped.save(buffer, format="PNG")
    buffer.seek(0)
    return Image(buffer, width=width, height=width * kept / pixel_w)


def side_by_side(left: Image, right: Image, width: float) -> Table:
    gap = 0.18 * inch
    half = (width - gap) / 2
    table = Table([[left, "", right]], colWidths=[half, gap, half])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return table


def grid(headers: list[str], rows: list[list[str]], widths: list[float], style: dict) -> Table:
    data = [[Paragraph(cell, style["head"]) for cell in headers]]
    for row in rows:
        data.append([Paragraph(cell, style["small"]) for cell in row])
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("BACKGROUND", (0, 1), (-1, -1), ZEBRA),
                ("GRID", (0, 0), (-1, -1), 0.3, RULE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def signed_coefficients(limit: int = 8) -> list[tuple[str, float]]:
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    model = pipeline.named_steps["clf"]
    names = [
        name.split("__", 1)[-1] for name in pipeline.named_steps["prep"].get_feature_names_out()
    ]
    coefs = model.coef_.ravel()
    order = sorted(range(len(coefs)), key=lambda index: abs(coefs[index]), reverse=True)
    return [(names[index], float(coefs[index])) for index in order[:limit]]


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(NAVY)
    canvas.setLineWidth(0.6)
    canvas.line(0.7 * inch, 0.55 * inch, letter[0] - 0.7 * inch, 0.55 * inch)
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(NAVY)
    canvas.drawString(0.7 * inch, 0.38 * inch, "AIMLCZG523 Assignment 01  |  CardiaScreen")
    canvas.drawRightString(letter[0] - 0.7 * inch, 0.38 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build() -> Path:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    holdout = metrics["holdout"]
    default = metrics["holdout_at_0_5"]
    style = styles()
    story: list = []
    usable = 7.1 * inch

    story.append(Paragraph("Machine Learning Operations (MLOps)", style["sub"]))
    story.append(Paragraph("AIMLCZG523 Assignment 01", style["cover"]))
    story.append(
        Paragraph(
            "CardiaScreen: an end-to-end screening model, tracked experiment, and monitored API",
            style["sub"],
        )
    )
    story.append(
        Paragraph(
            "Student name: ________________________ &nbsp;&nbsp; BITS ID: ________________________",
            style["body"],
        )
    )
    story.append(
        Paragraph(
            "Write your name and ID above before you submit. The repository, figures, and model "
            "artifact were produced for this assignment from the official UCI Heart Disease files.",
            style["body"],
        )
    )
    story.append(Paragraph("1. Problem and result", style["h1"]))
    story.append(
        Paragraph(
            "The task is a binary classifier for angiographic heart disease. A row is positive when "
            "the UCI <i>num</i> field is greater than zero, which means at least 50 percent diameter "
            "narrowing. The delivered system trains on the four public cohorts, freezes one model "
            "after a training-only comparison, packages the preprocessing with the estimator, and "
            "serves it from a container that logs every request and exposes Prometheus metrics.",
            style["body"],
        )
    )
    story.append(
        Paragraph(
            f"The selected model is L2 logistic regression with C = 0.1. Five-fold ROC-AUC on the "
            f"training split is {fmt(metrics['leaderboard'][0]['cv_roc_auc'])} "
            f"(standard deviation {fmt(metrics['leaderboard'][0]['cv_roc_auc_std'])}). "
            f"On the untouched 20 percent holdout the ROC-AUC is {fmt(holdout['roc_auc'])} and "
            f"average precision is {fmt(holdout['average_precision'])}. The operating threshold "
            f"{fmt(metrics['threshold'], 2)} was chosen to maximize F2 on training out-of-fold "
            f"scores, so holdout recall is {fmt(holdout['recall'])} and precision is "
            f"{fmt(holdout['precision'])}. At the default threshold of 0.50, holdout accuracy is "
            f"{fmt(default['accuracy'])} with precision {fmt(default['precision'])} and recall "
            f"{fmt(default['recall'])}. This score is a coursework screening signal. It is not a "
            "diagnosis and it is not a medical device.",
            style["body"],
        )
    )
    story.append(Paragraph("2. How to reproduce the run", style["h1"]))
    story.append(
        Paragraph(
            "From a clean checkout, create a Python 3.11 environment and install the pinned files. "
            "<font face='Courier'>requirements.txt</font> trains and tracks. "
            "<font face='Courier'>requirements-api.txt</font> is the smaller set baked into the "
            "serving image. <font face='Courier'>environment.yml</font> is the same stack for Conda. "
            "PowerShell:",
            style["body"],
        )
    )
    story.append(
        Paragraph(
            "python -m venv .venv<br/>"
            ".\\.venv\\Scripts\\Activate.ps1<br/>"
            "pip install -r requirements.txt -r requirements-dev.txt<br/>"
            "$env:PYTHONPATH = \"src\"<br/>"
            "python -m cardiascreen.data<br/>"
            "python -m cardiascreen.eda --skip-download<br/>"
            "python -m cardiascreen.train --skip-download<br/>"
            "pytest -q<br/>"
            "docker compose up --build",
            style["small"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Training writes <font face='Courier'>models/heart_risk_model.joblib</font>, "
            "<font face='Courier'>reports/metrics.json</font>, the figures in "
            "<font face='Courier'>reports/figures</font>, and an MLflow sqlite store at "
            "<font face='Courier'>mlruns/mlflow.db</font>. Open the tracking UI with "
            "<font face='Courier'>mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db</font>. "
            "MLflow 3 no longer treats a plain folder as a supported tracking backend, so the run "
            "uses sqlite and still keeps artifacts on disk. The random seed is 42.",
            style["body"],
        )
    )
    story.append(Paragraph("3. Data acquisition and cleaning", style["h1"]))
    story.append(
        Paragraph(
            "The source is the UCI Heart Disease collection (Janosi, Steinbrunn, Pfisterer, and "
            "Detrano, 1988; DOI 10.24432/C52P4X). "
            "<font face='Courier'>python -m cardiascreen.data</font> downloads the four processed "
            "files from the UCI repository: Cleveland (303 rows), Hungary (294), Switzerland (123), "
            "and the VA Long Beach file (200). The combined table has "
            f"{metrics['rows']} labeled rows and a positive rate of {fmt(metrics['positive_rate'])}. "
            "Question marks are missing values. Cholesterol, resting blood pressure, and maximum "
            "heart rate stored as zero are also treated as missing. Switzerland's cholesterol column "
            "is entirely zero in the source file, so after cleaning it is entirely missing. That "
            "choice matters: a model that treated those zeros as real measurements would learn a "
            "false biological rule.",
            style["body"],
        )
    )
    story.append(
        Paragraph(
            "The label is binary. Severity codes 1 through 4 are collapsed to disease present, and "
            "code 0 stays disease absent. Site identity is kept for evaluation and is not given to "
            "the model. A random split that can see the hospital learns the hospital's base rate. "
            "Cleveland and Hungary sit near 45 and 36 percent positive. Switzerland is above 90 "
            "percent and the VA file is about 75 percent. Those are different populations.",
            style["body"],
        )
    )
    half = (usable - 0.18 * inch) / 2
    story.append(
        side_by_side(
            fitted(FIGURES / "eda_cohort_balance.png", half),
            fitted(FIGURES / "eda_missingness.png", half),
            usable,
        )
    )
    story.append(
        Paragraph(
            "Figure 1. Cohort size and disease prevalence. The sites do not share a base rate. "
            "Figure 2. Missingness by site. Vessel count (ca) is almost absent outside Cleveland, "
            "and cholesterol is missing for every Switzerland row.",
            style["caption"],
        )
    )
    story.append(Paragraph("4. Exploratory findings", style["h1"]))
    story.append(
        Paragraph(
            "Exploratory plots describe the full assembled table. They were not used to pick the "
            "winning hyperparameters. Those were chosen on the training split only. The correlation "
            "heatmap is restricted to Cleveland, because correlating the pooled table would mostly "
            "measure which site forgot which test. In Cleveland, disease moves with lower maximum "
            "heart rate, higher ST depression, more colored vessels, and exercise-induced angina. "
            "Age and resting blood pressure move together, and cholesterol is only weakly tied to "
            "the label. That is why the later model is allowed to use cholesterol, but the report "
            "does not treat it as the main signal.",
            style["body"],
        )
    )
    story.append(fitted(FIGURES / "eda_correlation.png", 4.5 * inch))
    story.append(
        Paragraph(
            "Figure 3. Cleveland-only correlations. The other sites are left out so that missingness "
            "patterns do not masquerade as clinical correlation.",
            style["caption"],
        )
    )
    story.append(fitted(FIGURES / "eda_chest_pain.png", usable * 0.9))
    story.append(
        Paragraph(
            "Figure 4. UCI chest-pain codes. Code 4 (asymptomatic) is the largest group and the "
            "highest disease rate. The API rejects code 0 so a Kaggle-style payload cannot be scored.",
            style["caption"],
        )
    )
    story.append(fitted(FIGURES / "eda_distributions.png", usable * 0.9))
    story.append(
        Paragraph(
            "Figure 5. Disease rows are older, reach a lower peak heart rate, and show more ST "
            "depression. Cholesterol overlaps heavily once the zero sentinels are removed.",
            style["caption"],
        )
    )
    story.append(Paragraph("5. Feature engineering", style["h1"]))
    story.append(
        Paragraph(
            "The serving path and the training path share one sklearn transformer, "
            "<font face='Courier'>ClinicalFeatureBuilder</font>. It reads the thirteen raw UCI "
            "fields, repeats the zero-sentinel rule, and adds four clinical fields: heart rate as a "
            "fraction of the predicted maximum (220 minus age), cholesterol per year of age, blood "
            "pressure per year of age, and the product of ST depression and exercise angina. "
            "Missingness indicators are added for cholesterol, fasting blood sugar, vessel count, "
            "and the thallium code, because 'this test was not done' is known at request time.",
            style["body"],
        )
    )
    story.append(
        Paragraph(
            "Numeric fields are median-imputed and standardized. Binary fields use the most frequent "
            "value. Chest pain, resting ECG, ST slope, and thallium code are one-hot encoded with "
            "unknown codes ignored. Thallium is a nominal code (3 normal, 6 fixed defect, 7 "
            "reversible defect), so it is not scaled as if it were a measurement. Vessel count stays "
            "numeric because it is a count, and it keeps its own missingness indicator. The "
            "transformer drops cohort, the raw severity code, and the label if a caller passes them. "
            "A unit test locks that rule. Fluoroscopy and thallium are optional on "
            "<font face='Courier'>POST /predict</font>, since a screening form often will not have "
            "those invasive results.",
            style["body"],
        )
    )
    story.append(Paragraph("6. Model selection and the operating point", style["h1"]))
    story.append(
        Paragraph(
            "The table is split 80/20, stratified on the label, with seed 42 "
            f"({metrics['train_rows']} train rows, {metrics['holdout_rows']} holdout rows). "
            "Three families are compared with a grid search and five stratified folds, scored by "
            "ROC-AUC. The holdout is not used for that choice. Logistic regression uses C in "
            "{0.1, 1, 10} with balanced class weights. Random forest varies depth and leaf size "
            "around 250 trees. HistGradientBoosting varies depth and learning rate. Logistic "
            "regression wins. The forest and the boosting model are close behind and less stable "
            "across folds, which is a reason to prefer the linear model on a table of this size.",
            style["body"],
        )
    )
    board = []
    for row in metrics["leaderboard"]:
        board.append(
            [
                row["model_name"].replace("_", " "),
                fmt(row["cv_roc_auc"]),
                fmt(row["cv_roc_auc_std"]),
                ", ".join(f"{key.split('__')[-1]}={value}" for key, value in row["best_params"].items()),
            ]
        )
    story.append(
        grid(
            ["Model", "CV ROC-AUC", "Fold SD", "Selected parameters"],
            board,
            [1.8 * inch, 1.2 * inch, 1.0 * inch, 3.1 * inch],
            style,
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 1. Training-split comparison. The winner is the first row. Parameters were not "
            "retuned after looking at the holdout.",
            style["caption"],
        )
    )
    story.append(
        Paragraph(
            "After the winner is frozen, out-of-fold probabilities on the training split are used to "
            f"pick a threshold that maximizes F2 (beta = 2). The chosen threshold is "
            f"{fmt(metrics['threshold'], 2)}. F2 is the selection rule because a missed narrowing "
            "is more costly, in this screening setup, than an extra review. The same holdout "
            "probabilities are also reported at 0.50 so the accuracy cost of that choice is visible. "
            "Moving from 0.50 to "
            f"{fmt(metrics['threshold'], 2)} raises recall from {fmt(default['recall'])} to "
            f"{fmt(holdout['recall'])} and lowers precision from {fmt(default['precision'])} to "
            f"{fmt(holdout['precision'])}. Accuracy falls from {fmt(default['accuracy'])} to "
            f"{fmt(holdout['accuracy'])}. The confusion matrix at the operating point has 100 true "
            "positives and 2 false negatives on 102 holdout disease rows, with 44 false positives.",
            style["body"],
        )
    )
    pair = Table(
        [
            [
                fitted(FIGURES / "model_roc.png", 3.35 * inch),
                fitted(FIGURES / "model_confusion.png", 3.35 * inch),
            ]
        ],
        colWidths=[3.55 * inch, 3.55 * inch],
    )
    pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(pair)
    story.append(
        Paragraph(
            "Figure 6. Holdout ROC for the frozen logistic model, and holdout decisions at the F2 threshold.",
            style["caption"],
        )
    )
    story.append(fitted(FIGURES / "model_threshold.png", usable))
    story.append(
        Paragraph(
            "Figure 7. Precision, recall, and F2 against the threshold. The vertical line is the "
            "training out-of-fold choice. It is not fit on the holdout.",
            style["caption"],
        )
    )
    story.append(Paragraph("7. Holdout results, including the sites", style["h1"]))
    story.append(
        grid(
            ["Threshold", "Accuracy", "Precision", "Recall", "F1", "F2", "ROC-AUC", "Brier"],
            [
                [
                    "0.18 (F2)",
                    fmt(holdout["accuracy"]),
                    fmt(holdout["precision"]),
                    fmt(holdout["recall"]),
                    fmt(holdout["f1"]),
                    fmt(holdout["f2"]),
                    fmt(holdout["roc_auc"]),
                    fmt(holdout["brier"]),
                ],
                [
                    "0.50",
                    fmt(default["accuracy"]),
                    fmt(default["precision"]),
                    fmt(default["recall"]),
                    fmt(default["f1"]),
                    fmt(default["f2"]),
                    fmt(default["roc_auc"]),
                    fmt(default["brier"]),
                ],
            ],
            [0.9 * inch, 0.85 * inch, 0.85 * inch, 0.75 * inch, 0.7 * inch, 0.7 * inch, 0.9 * inch, 0.7 * inch],
            style,
        )
    )
    story.append(Spacer(1, 3))
    story.append(
        Paragraph(
            "Table 2. One holdout, two operating points. ROC-AUC and Brier do not depend on the threshold.",
            style["caption"],
        )
    )
    cohort_rows = []
    for name, row in metrics["holdout_by_cohort"].items():
        cohort_rows.append(
            [
                name,
                str(row["support"]),
                str(row["positives"]),
                fmt(row["recall"]),
                fmt(row["precision"]),
                fmt(row["roc_auc"]),
            ]
        )
    story.append(
        grid(
            ["Holdout site", "Rows", "Positives", "Recall", "Precision", "ROC-AUC"],
            cohort_rows,
            [1.4 * inch, 0.8 * inch, 0.9 * inch, 1.0 * inch, 1.1 * inch, 1.1 * inch],
            style,
        )
    )
    story.append(Spacer(1, 3))
    story.append(
        Paragraph(
            "Table 3. The same frozen model and the same threshold, sliced by source site. "
            "ROC-AUC is omitted when either class has fewer than five rows. The Switzerland holdout "
            "slice is almost entirely positive, so an AUC there would be noise. High accuracy on "
            "that slice mostly repeats the base rate. Hungary, where disease is less common, is "
            "where the low threshold spends precision.",
            style["caption"],
        )
    )
    story.append(
        Paragraph(
            "A separate transport check fits the same algorithm and the same hyperparameters on "
            "Cleveland only, then scores the other three sites in full. This model is not the "
            "production artifact and it did not vote in model selection. Ranking still travels: "
            "Hungary's ROC-AUC stays near 0.89. Recall at 0.50 does not travel. Hungary recall is "
            "about 0.60, Switzerland about 0.60, and the VA file about 0.50. The production model "
            "is therefore the pooled model, because it has seen the missingness patterns a new "
            "request will actually have. The threshold would still need a local review before use "
            "in a clinic whose prevalence differs from this pool.",
            style["body"],
        )
    )
    transport_rows = []
    for name, row in metrics["cleveland_transport"]["sites"].items():
        transport_rows.append(
            [
                name,
                str(row["support"]),
                fmt(row["roc_auc"]),
                fmt(row["recall"]),
                fmt(row["precision"]),
            ]
        )
    story.append(
        grid(
            ["External site", "Rows", "ROC-AUC", "Recall at 0.50", "Precision at 0.50"],
            transport_rows,
            [1.5 * inch, 1.0 * inch, 1.2 * inch, 1.6 * inch, 1.6 * inch],
            style,
        )
    )
    story.append(Spacer(1, 3))
    story.append(
        Paragraph(
            "Table 4. Cleveland-only model scored on the other UCI sites. Transport of ranking is "
            "better than transport of a fixed 0.50 cutoff.",
            style["caption"],
        )
    )
    story.append(fitted(FIGURES / "model_importance.png", usable * 0.86))
    story.append(
        Paragraph(
            "Figure 8. Largest absolute coefficients. Asymptomatic chest pain and sex lead. "
            "A positive sign raises the log-odds of disease.",
            style["caption"],
        )
    )
    coef_rows = [[name, f"{value:+.3f}"] for name, value in signed_coefficients()]
    story.append(
        grid(
            ["Feature", "Signed coefficient"],
            coef_rows,
            [3.2 * inch, 2.2 * inch],
            style,
        )
    )
    story.append(Spacer(1, 3))
    story.append(
        Paragraph(
            "Table 5. Signs for the largest coefficients of the packaged logistic model. "
            "missing_chol has a large coefficient because Switzerland, a high-prevalence site, "
            "contributes an entirely missing cholesterol column. The indicator is partly a stand-in "
            "for that site. A clinic that always measures cholesterol will not activate it. This is "
            "the main leakage risk left after dropping the cohort column, and it is why the API "
            "metrics include the predicted-probability histogram: a shift in missingness shows up "
            "as a shift in scores.",
            style["caption"],
        )
    )
    story.append(fitted(FIGURES / "model_calibration.png", 4.4 * inch))
    story.append(
        Paragraph(
            "Figure 9. Holdout calibration. Ranking can be good while a probability is still a "
            "poor frequency, which is why the API returns both a probability and a confidence.",
            style["caption"],
        )
    )
    story.append(Paragraph("8. Experiment tracking and packaging", style["h1"]))
    story.append(
        Paragraph(
            "Each candidate is a nested MLflow run under the experiment "
            "<font face='Courier'>cardiascreen-heart-disease</font>. The parent run records the "
            "split sizes, the seed, the selection metric, the threshold rule, the winner, the "
            "holdout metrics, the figure directory, the metrics file, and the sklearn model. "
            "The custom transformer is declared in the skops trusted-type list so the pyfunc save "
            "can record it. The file that the API actually loads is a joblib dictionary: the fitted "
            "pipeline, the threshold, the version, the training time, the metric snapshot, and a "
            "disclaimer. Loading that file requires the package path "
            "<font face='Courier'>cardiascreen.features</font>, which the image provides through "
            "<font face='Courier'>PYTHONPATH=/app/src</font>. Code and model ship in one image, so "
            "a release is one digest rather than a model file that can drift away from its transformer.",
            style["body"],
        )
    )
    story.append(
        Paragraph(
            "Inference without HTTP is <font face='Courier'>python -m cardiascreen.infer "
            "--input deploy/sample_request.json</font>. The three sample files cover a high-risk "
            "Cleveland-style row, a lower-risk row, and a screening row that omits cholesterol, "
            "fasting sugar, vessel count, and thallium.",
            style["body"],
        )
    )
    story.append(Paragraph("9. API, tests, and the CI pipeline", style["h1"]))
    story.append(
        Paragraph(
            "FastAPI exposes <font face='Courier'>GET /health</font>, "
            "<font face='Courier'>GET /metadata</font>, <font face='Courier'>GET /metrics</font>, "
            "and <font face='Courier'>POST /predict</font>. The body is validated with Pydantic. "
            "Extra fields are rejected, including a client-supplied label. Chest-pain values outside "
            "1-4 are rejected. The response returns the class, the probability of disease, the "
            "confidence in the emitted class, the threshold, a risk band (low, moderate, high, "
            "very high, cut at 0.25, 0.50, and 0.75), a request id, and the disclaimer. Each "
            "decision is a JSON log line on stdout and, when the directory is writable, in "
            "<font face='Courier'>logs/api.jsonl</font>. The log records age, sex, chest-pain code, "
            "the decision, and latency. A production deployment would redact even those fields.",
            style["body"],
        )
    )
    story.append(
        Paragraph(
            "Pytest covers label construction, zero sentinels, the four-site load, the feature "
            "contract, the threshold helper, the small-class AUC guard, and the HTTP contract "
            "including the Kaggle chest-pain rejection. GitHub Actions in "
            "<font face='Courier'>.github/workflows/ci.yml</font> runs ruff, then pytest, then a "
            "full retrain, and uploads the model, metrics, and figures. A second job builds the "
            "image and calls <font face='Courier'>/health</font> and <font face='Courier'>/predict</font>. "
            "The workflow does not set continue-on-error. A lint failure, a test failure, or a "
            "training exception fails the job and the logs stay in the Actions run. The container "
            "job uses the model committed in the repository. The retrain job proves that a clean "
            "runner can rebuild it.",
            style["body"],
        )
    )
    story.append(Paragraph("10. Container, deployment, and monitoring", style["h1"]))
    story.append(
        Paragraph(
            "The Dockerfile is a Python 3.11 slim image running as a non-root user. It installs "
            "only the API requirements, copies the package and the joblib artifact, and starts "
            "uvicorn on port 8000 with a factory so the model is loaded at process start. A failed "
            "load stops the process, which is what a Kubernetes liveness probe should see. "
            "<font face='Courier'>docker compose up --build</font> starts the API, Prometheus, and "
            "Grafana. Prometheus scrapes <font face='Courier'>api:8000/metrics</font> every five "
            "seconds. Grafana is provisioned with a Prometheus datasource and a dashboard for "
            "request rate, latency, decision counts, and the probability histogram. Anonymous "
            "viewing is enabled. The admin password in the compose file is <font face='Courier'>admin</font> "
            "and is only for this local lab.",
            style["body"],
        )
    )
    story.append(
        Paragraph(
            "Kubernetes manifests under <font face='Courier'>deploy/k8s</font> define a namespace, "
            "a two-replica Deployment with requests and limits, readiness and liveness probes on "
            "<font face='Courier'>/health</font>, a LoadBalancer Service on port 80, and an nginx "
            "Ingress for <font face='Courier'>cardiascreen.local</font>. Pods carry Prometheus "
            "scrape annotations. Apply them with <font face='Courier'>kubectl apply -k deploy/k8s</font> "
            "after <font face='Courier'>docker build -t cardiascreen-api:local .</font>. "
            "Docker Desktop's kubectl context is present on the development machine, and the API "
            "server on port 6443 was not running when this report was written, so the verified "
            "runtime is the compose stack rather than a live cluster. Enabling Kubernetes in "
            "Docker Desktop, or loading the image into minikube, uses these same manifests. "
            "If the LoadBalancer address stays pending, <font face='Courier'>kubectl port-forward "
            "svc/cardiascreen-api 8080:80 -n cardiascreen</font> is the check.",
            style["body"],
        )
    )
    story.append(
        Paragraph(
            "<font face='Courier'>scripts/export_space.py</font> copies the Dockerfile, the API "
            "requirements, the package, the model, and a Space README into "
            "<font face='Courier'>dist/huggingface-space</font>. That folder is a Hugging Face "
            "Docker Space on the free CPU tier. Upload it with the Hugging Face CLI after "
            "<font face='Courier'>huggingface-cli login</font>. The Space sleeps when idle; the "
            "first request after sleep is slow. The public URL, once created, is the deployed API "
            "link for the submission.",
            style["body"],
        )
    )
    story.append(crop_top(FIGURES / "architecture.png", usable * 0.92, 0.74))
    story.append(
        Paragraph(
            "Figure 10. Production path. GitHub Actions gates the image. The image pins code and "
            "model together. Prometheus watches the running API.",
            style["caption"],
        )
    )
    story.append(Paragraph("11. Evidence from the running stack", style["h1"]))
    story.append(
        Paragraph(
            "The figures below are from the local compose stack on 9 October 2026, after the image "
            "was built and the API had served prediction requests. "
            "<font face='Courier'>screenshots/</font> also holds the health response and the raw "
            "metrics page. GitHub Actions checks the data pin, then hands a model over only "
            "when the holdout gate passes (<font face='Courier'>screenshots/ci_actions.png</font>). "
            "The workflow is <font face='Courier'>.github/workflows/ci.yml</font>.",
            style["body"],
        )
    )
    shots = ROOT / "screenshots"
    story.append(crop_top(shots / "api_docs.png", usable * 0.86, 0.82))
    story.append(
        Paragraph(
            "Figure 11. Interactive API documentation at http://127.0.0.1:8000/docs, including POST /predict.",
            style["caption"],
        )
    )
    story.append(fitted(shots / "grafana_board.png", usable * 0.72))
    story.append(
        Paragraph(
            "Figure 12. Grafana board provisioned for the API. The prediction counter reads 8 "
            "after the smoke requests used to check the stack.",
            style["caption"],
        )
    )
    story.append(crop_top(shots / "prometheus_targets.png", usable * 0.92, 0.34))
    story.append(
        Paragraph(
            "Figure 13. Prometheus target http://api:8000/metrics is UP for job cardiascreen.",
            style["caption"],
        )
    )
    story.append(Paragraph("12. What this submission does not claim", style["h1"]))
    story.append(
        Paragraph(
            "The holdout is a random split of a 920-row public table, not a prospective clinic. "
            "Switzerland and the VA file are small, and their labels are heavily imbalanced, so "
            "site-level accuracy is easy to over-read. The missing-cholesterol coefficient is a "
            "site proxy and should be watched if the measurement process changes. Probabilities are "
            "good enough to rank (holdout Brier "
            f"{fmt(holdout['brier'])}) but the calibration plot in "
            "<font face='Courier'>reports/figures/model_calibration.png</font> should be read before "
            "anyone treats a probability as a frequency. No fairness audit by sex was used as a "
            "release gate, even though sex is one of the largest coefficients. That would be the "
            "next experiment, not a claim already made.",
            style["body"],
        )
    )
    story.append(
        Paragraph(
            "A short screen recording is part of the deliverable and has to be captured on the "
            "machine that runs the pipeline. <font face='Courier'>docs/DEMO_SCRIPT.md</font> is a "
            "three-minute shot list: the training command, the MLflow UI, a prediction, the "
            "Prometheus target, and the Grafana board. Record it after the compose stack is up.",
            style["body"],
        )
    )
    story.append(Paragraph("13. Where each graded part lives", style["h1"]))
    story.append(
        grid(
            ["Task", "Where to look"],
            [
                ["1. Data and EDA", "cardiascreen.data, cardiascreen.eda, reports/figures/eda_*.png, notebooks/"],
                ["2. Features and models", "cardiascreen.features, cardiascreen.train, reports/metrics.json"],
                ["3. Experiment tracking", "mlruns/mlflow.db, nested runs under cardiascreen-heart-disease"],
                ["4. Packaging", "models/heart_risk_model.joblib, requirements.txt, environment.yml"],
                ["5. Tests and CI", "tests/, .github/workflows/ci.yml (ruff, pytest, retrain, image smoke)"],
                ["6. Container API", "Dockerfile, POST /predict, scripts/smoke_predict.py"],
                ["7. Deployment", "docker-compose.yml, deploy/k8s, scripts/export_space.py"],
                ["8. Monitoring", "GET /metrics, Prometheus, Grafana, screenshots/"],
                ["9. This report", "README.md, docs/DEMO_SCRIPT.md, and this PDF"],
            ],
            [1.7 * inch, 5.4 * inch],
            style,
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Table 6. Map from the assignment tasks to the files a reviewer can open. "
            "The screen recording is the one item that has to be made after this PDF: follow "
            "docs/DEMO_SCRIPT.md once the compose stack is running.",
            style["caption"],
        )
    )
    story.append(
        Paragraph(
            "Repository layout: <font face='Courier'>src/cardiascreen</font> (data, features, "
            "training, API), <font face='Courier'>tests</font>, <font face='Courier'>notebooks</font>, "
            "<font face='Courier'>data/raw</font> and <font face='Courier'>data/processed</font>, "
            "<font face='Courier'>.github/workflows/ci.yml</font>, <font face='Courier'>Dockerfile</font>, "
            "<font face='Courier'>docker-compose.yml</font>, <font face='Courier'>deploy/k8s</font>, "
            "and <font face='Courier'>monitoring</font>. Dataset citation: Janosi, A., Steinbrunn, W., "
            "Pfisterer, M., and Detrano, R. (1988). Heart Disease. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C52P4X.",
            style["body"],
        )
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.7 * inch,
        title="AIMLCZG523 Assignment 01 — CardiaScreen",
        author="AIMLCZG523 Assignment 01",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT


if __name__ == "__main__":
    path = build()
    print(path)
