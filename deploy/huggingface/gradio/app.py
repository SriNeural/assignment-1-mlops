"""Public CardiaScreen host for a Hugging Face Gradio Space.

The Docker Space path needs a paid Hugging Face plan. This file keeps the same
JSON routes on the free ZeroGPU runtime and still shows a small form.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# The Space image turns on Gradio's Node frontend, which swallows /health.
# The screening routes have to be served by the Python app.
os.environ["GRADIO_SSR_MODE"] = "false"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["CARDIA_MODEL_PATH"] = str(ROOT / "models" / "heart_risk_model.joblib")

try:
    import spaces
except ImportError:  # local smoke tests do not have the Space runtime

    class _Spaces:
        @staticmethod
        def GPU(*_args, **_kwargs):
            def wrap(function):
                return function

            return wrap

    spaces = _Spaces()

import gradio as gr

from cardiascreen.scoring import score_records
from cardiascreen.service import create_app

API = create_app()
ARTIFACT = API.state.artifact
API_PATHS = {"/health", "/metadata", "/metrics", "/predict"}


def mount_screening_routes(target) -> None:
    """Expose the FastAPI contract on the Space port, ahead of Gradio's routes.

    Gradio rebuilds its FastAPI app inside launch(), so this must run on that
    new app, immediately before the server starts listening.
    """
    selected = [route for route in API.router.routes if getattr(route, "path", None) in API_PATHS]
    current = target.router.routes
    for route in reversed(selected):
        if route not in current:
            current.insert(0, route)


def score_form(
    age: float,
    sex: int,
    cp: int,
    trestbps: float,
    chol: float | None,
    fbs: int | None,
    restecg: int,
    thalach: float,
    exang: int,
    oldpeak: float,
    slope: int,
    ca: float | None,
    thal: float | None,
) -> str:
    record = {
        "age": age,
        "sex": int(sex),
        "cp": int(cp),
        "trestbps": trestbps,
        "chol": chol,
        "fbs": None if fbs is None else int(fbs),
        "restecg": int(restecg),
        "thalach": thalach,
        "exang": int(exang),
        "oldpeak": oldpeak,
        "slope": int(slope),
        "ca": ca,
        "thal": thal,
    }
    decision = score_records(ARTIFACT, [record])[0]
    payload = {
        **decision,
        "model_name": ARTIFACT["model_name"],
        "model_version": ARTIFACT["model_version"],
        "disclaimer": ARTIFACT.get("disclaimer", ""),
    }
    return json.dumps(payload, indent=2)


@spaces.GPU(duration=10)
def runtime_probe() -> str:
    """Register one GPU function so the ZeroGPU runtime will start.

    Screening itself stays on CPU. This probe is not part of the score.
    """
    return "Runtime is up. POST /predict scores on CPU and does not use the GPU."


with gr.Blocks(title="CardiaScreen") as demo:
    gr.Markdown(
        """
# CardiaScreen
Coursework screening score for the UCI Heart Disease files. This is not a diagnosis
and not a medical device. Chest pain `cp` uses the original codes 1–4.

`POST /predict` on this host returns `prediction`, `probability`, and `confidence`.
`GET /health` reports the model version.
        """
    )
    with gr.Row():
        age = gr.Number(label="age", value=67)
        sex = gr.Radio(choices=[("female", 0), ("male", 1)], value=1, label="sex")
        cp = gr.Radio(
            choices=[("typical", 1), ("atypical", 2), ("non-anginal", 3), ("asymptomatic", 4)],
            value=4,
            label="cp (UCI 1-4)",
        )
    with gr.Row():
        trestbps = gr.Number(label="trestbps", value=160)
        chol = gr.Number(label="chol (blank if unknown)", value=286)
        fbs = gr.Radio(choices=[("no", 0), ("yes", 1)], value=0, label="fbs")
        restecg = gr.Radio(choices=[0, 1, 2], value=2, label="restecg")
    with gr.Row():
        thalach = gr.Number(label="thalach", value=108)
        exang = gr.Radio(choices=[("no", 0), ("yes", 1)], value=1, label="exang")
        oldpeak = gr.Number(label="oldpeak", value=1.5)
        slope = gr.Radio(choices=[1, 2, 3], value=2, label="slope")
    with gr.Row():
        ca = gr.Number(label="ca (blank if unknown)", value=3)
        thal = gr.Number(label="thal 3, 6, or 7", value=3)
    score_button = gr.Button("Score this record", variant="primary")
    result = gr.Code(label="Decision", language="json")
    score_button.click(
        score_form,
        inputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal],
        outputs=result,
    )
    probe_button = gr.Button("Runtime check")
    probe_result = gr.Textbox(label="Runtime")
    probe_button.click(runtime_probe, outputs=probe_result)

import gradio.http_server as http_server

_start_server = http_server.start_server


def _start_server_with_api(app, *args, **kwargs):
    mount_screening_routes(app)
    return _start_server(app, *args, **kwargs)


http_server.start_server = _start_server_with_api
mount_screening_routes(demo.app)

_launch = demo.launch


def _launch_without_ssr(*args, **kwargs):
    kwargs["ssr_mode"] = False
    return _launch(*args, **kwargs)


demo.launch = _launch_without_ssr

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
