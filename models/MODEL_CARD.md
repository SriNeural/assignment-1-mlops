# CardiaScreen model card

| Field | Value |
| --- | --- |
| Model | L2 logistic regression, `C=0.1`, `class_weight=balanced` |
| Version | 1.0.0 |
| Data version | 1.0.0, pinned in `data/versions/1.0.0.json` |
| Label | UCI `num > 0` (angiographic narrowing of at least 50%) |
| Training rows / holdout | 736 / 184, stratified, seed 42 |
| Selection metric | 5-fold ROC-AUC on the training split only |
| Operating threshold | 0.18, maximizing F2 on training out-of-fold scores |
| Holdout ROC-AUC | 0.909 |
| Holdout recall / precision at 0.18 | 0.980 / 0.694 |
| Holdout accuracy / precision / recall at 0.50 | 0.832 / 0.851 / 0.843 |
| Artifact | `models/heart_risk_model.joblib` |
| Inputs | 13 UCI fields. `ca`, `thal`, `fbs`, and `chol` may be null. Chest pain `cp` is 1-4. |
| Not for | Diagnosis, triage, or any clinical decision |

Site name is not a feature. The missing-cholesterol indicator is partly a proxy for the Switzerland cohort, which recorded cholesterol as zero. Read `reports/metrics.json` for the per-site and Cleveland-transport tables.

The packaged object is a dictionary. Scoring must use `cardiascreen.scoring.score_records` or the HTTP API so the threshold stays attached to the pipeline.

A replacement is promoted only when `python -m cardiascreen.release promote` accepts it. The gate keeps the current release when the fingerprint matches, including a holdout ROC-AUC or recall move of at most 0.005 on the same data version and parameters. A larger ROC-AUC drop, or recall under 0.90, rejects the candidate and leaves this file in place. The registry is `models/registry.json`.
