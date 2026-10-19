---
title: CardiaScreen Heart Disease API
emoji: "❤️"
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# CardiaScreen

Binary screening API trained on the four UCI Heart Disease cohorts (Cleveland, Hungary, Switzerland, and the VA Long Beach file).

This Space is the public host for the coursework API. `POST /predict` accepts the original UCI feature codes. Chest pain `cp` is 1–4, not the 0–3 recoding used by many public CSV mirrors.

The response is not a diagnosis and not a medical device. The first request after the Space sleeps can be slow.

## Sample

```bash
curl -X POST "$SPACE_HOST/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}'
```

`GET /health`, `GET /metadata`, and `GET /metrics` are the other routes. Interactive Gradio fields are on `/`.
