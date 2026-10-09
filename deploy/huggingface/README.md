---
title: CardiaScreen Heart Disease API
emoji: "❤️"
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
---

# CardiaScreen

Binary screening API trained on the four UCI Heart Disease cohorts (Cleveland, Hungary, Switzerland, and the VA Long Beach file).

`POST /predict` accepts the original UCI feature codes. Chest pain `cp` is 1–4, not the 0–3 recoding used by many public CSV mirrors.

This Space is a coursework deployment. The response is not a diagnosis and not a medical device.

## Sample

```bash
curl -X POST "$SPACE_HOST/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}'
```

Interactive documentation is at `/docs`. Prometheus metrics are at `/metrics`.
