"""Pin the UCI source files so a changed download cannot train quietly.

The current pin is ``data/versions/CURRENT``. ``check`` fails when a raw file
hash differs. ``stamp`` writes a new pin after you have decided the change is real.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cardiascreen.config import COHORT_FILES, DATA_RAW, PROJECT_ROOT
from cardiascreen.data import file_sha256

VERSIONS_DIR = PROJECT_ROOT / "data" / "versions"
CURRENT_PATH = VERSIONS_DIR / "CURRENT"


class DataVersionError(RuntimeError):
    """Raised when the files on disk are not the pinned data version."""


def current_name() -> str:
    if not CURRENT_PATH.exists():
        raise DataVersionError(f"Missing data pin: {CURRENT_PATH}")
    return CURRENT_PATH.read_text(encoding="utf-8").strip()


def load_version(name: str | None = None) -> dict:
    version = name or current_name()
    path = VERSIONS_DIR / f"{version}.json"
    if not path.exists():
        raise DataVersionError(f"Missing data version file: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("data_version") != version:
        raise DataVersionError(f"{path.name} does not declare data_version {version}")
    return document


def snapshot_files(raw_dir: Path = DATA_RAW) -> list[dict]:
    records = []
    for cohort, filename in COHORT_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            raise DataVersionError(f"Missing raw file for {cohort}: {path}")
        records.append(
            {
                "cohort": cohort,
                "filename": filename,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return records


def mismatches(pin: dict, live: list[dict]) -> list[str]:
    expected = {item["cohort"]: item["sha256"] for item in pin["files"]}
    actual = {item["cohort"]: item["sha256"] for item in live}
    problems = []
    for cohort, digest in expected.items():
        if actual.get(cohort) != digest:
            problems.append(cohort)
    extra = sorted(set(actual) - set(expected))
    problems.extend(extra)
    return problems


def check_data_version(raw_dir: Path = DATA_RAW) -> dict:
    pin = load_version()
    problems = mismatches(pin, snapshot_files(raw_dir))
    if problems:
        names = ", ".join(problems)
        raise DataVersionError(
            f"Data version {pin['data_version']} does not match: {names}. "
            "Inspect the file, then stamp a new version. Do not train on an unpinned change."
        )
    return pin


def stamp(version: str, raw_dir: Path = DATA_RAW) -> Path:
    document = {
        "data_version": version,
        "source": "UCI Heart Disease processed files",
        "files": snapshot_files(raw_dir),
    }
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = VERSIONS_DIR / f"{version}.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    CURRENT_PATH.write_text(version + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check or stamp the CardiaScreen data version.")
    parser.add_argument("command", choices=("check", "stamp"))
    parser.add_argument("version", nargs="?", help="Version name for stamp, for example 1.1.0")
    args = parser.parse_args()
    if args.command == "check":
        pin = check_data_version()
        print(f"data_version={pin['data_version']} files={len(pin['files'])}")
        return
    if not args.version:
        raise SystemExit("stamp requires a version, for example 1.1.0")
    path = stamp(args.version)
    print(f"stamped {path}")


if __name__ == "__main__":
    main()
