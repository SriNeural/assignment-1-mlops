"""Record a short browser walkthrough of the running CardiaScreen pipeline.

Requires the compose stack on ports 8000, 9090, and 3000, plus MLflow on port 5000.
Playwright and its Chromium build are local recording tools. They are not part of
the training or serving requirements.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "videos"
SHOTS = ROOT / "screenshots"
SAMPLE = json.loads((ROOT / "deploy" / "sample_request.json").read_text(encoding="utf-8"))
ACTIONS = "https://github.com/SriNeural/assignment-1-mlops/actions/runs/39314946898"


def pause(page, milliseconds: int) -> None:
    page.wait_for_timeout(milliseconds)


def open_page(page, url: str, ready: str | None, settle_ms: int) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    if ready:
        try:
            page.wait_for_selector(ready, timeout=20000)
        except PlaywrightTimeout:
            print(f"selector not found on {url}: {ready}")
    pause(page, settle_ms)


def dismiss(page, name: str) -> None:
    button = page.get_by_role("button", name=name)
    if button.count():
        button.first.click()
        pause(page, 400)


def show_mlflow(page) -> None:
    open_page(page, "http://127.0.0.1:5000/", "text=Model training", 1500)
    dismiss(page, "Got it")
    page.get_by_text("Model training", exact=True).click()
    pause(page, 1200)
    page.get_by_role("link", name="cardiascreen-heart-disease").click()
    page.wait_for_selector("text=model-selection", timeout=20000)
    pause(page, 5500)
    page.screenshot(path=str(SHOTS / "mlflow_runs.png"))


def show_prediction(page) -> None:
    open_page(page, "http://127.0.0.1:8000/docs", "text=POST", 1500)
    page.locator(".opblock-post").first.click()
    pause(page, 700)
    page.get_by_role("button", name="Try it out").click()
    pause(page, 500)
    page.locator("textarea").first.fill(json.dumps(SAMPLE, indent=2))
    pause(page, 700)
    page.get_by_role("button", name="Execute").click()
    response = page.locator("text=disease_signal").last
    response.wait_for(timeout=20000)
    response.scroll_into_view_if_needed()
    pause(page, 6000)
    page.screenshot(path=str(SHOTS / "api_predict.png"))


def main() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    SHOTS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            record_video_dir=str(VIDEO_DIR),
            record_video_size={"width": 1280, "height": 720},
        )
        page = context.new_page()
        video = page.video
        page.set_content(
            """
            <body style="margin:0;background:#10243e;color:white;font-family:Georgia,serif;
                         display:flex;align-items:center;justify-content:center;height:100vh">
              <div style="max-width:860px;padding:40px">
                <p style="letter-spacing:.12em;font-size:14px">AIMLCZG523 ASSIGNMENT 01</p>
                <h1 style="font-size:42px;margin:8px 0">CardiaScreen pipeline</h1>
                <p style="font-size:20px;line-height:1.45">
                  GitHub Actions, MLflow, the prediction API, Prometheus, and Grafana.
                  The score is coursework evidence, not a diagnosis.
                </p>
              </div>
            </body>
            """
        )
        pause(page, 3500)

        open_page(page, ACTIONS, "text=lint test train", 7000)
        page.screenshot(path=str(SHOTS / "ci_actions.png"), full_page=False)
        page.mouse.wheel(0, 500)
        pause(page, 2500)

        show_mlflow(page)
        show_prediction(page)

        open_page(page, "http://127.0.0.1:9090/targets", "text=UP", 4000)
        open_page(
            page,
            "http://127.0.0.1:3000/d/cardiascreen-api/cardiascreen-api?orgId=1&from=now-15m&to=now&kiosk",
            "text=Prediction responses",
            7000,
        )

        context.close()
        browser.close()
        if video is not None:
            saved = Path(video.path())
            target = VIDEO_DIR / "pipeline_demo.webm"
            if target.exists():
                target.unlink()
            shutil.move(saved, target)
            print(target, target.stat().st_size)


if __name__ == "__main__":
    main()
