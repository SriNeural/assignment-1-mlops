"""Score one JSON object or a JSON list with the packaged model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cardiascreen.config import MODEL_PATH
from cardiascreen.scoring import score_records
from cardiascreen.service import load_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a local JSON payload.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else [payload]
    artifact = load_artifact(args.model)
    print(json.dumps(score_records(artifact, records), indent=2))


if __name__ == "__main__":
    main()
