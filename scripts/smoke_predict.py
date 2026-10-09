"""Call a running CardiaScreen API with the three sample payloads."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = [
    ROOT / "deploy" / "sample_request.json",
    ROOT / "deploy" / "sample_request_low_risk.json",
    ROOT / "deploy" / "sample_request_screening.json",
]


def call(base: str, path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    print(json.dumps(call(base, "/health"), indent=2))
    for sample in SAMPLES:
        body = json.loads(sample.read_text(encoding="utf-8"))
        try:
            result = call(base, "/predict", body)
        except urllib.error.HTTPError as exc:
            print(sample.name, exc.read().decode("utf-8"))
            raise
        print(sample.name, json.dumps(result))


if __name__ == "__main__":
    main()
