"""Download the IEEE-CIS Fraud Detection data into data/raw/ and verify it.

Requires a Kaggle API token at ~/.kaggle/kaggle.json (or KAGGLE_USERNAME /
KAGGLE_KEY in the environment) and prior acceptance of the competition rules.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "ieee-fraud-detection"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Row counts (excluding header) as published on the competition data page.
EXPECTED_ROWS: dict[str, int] = {
    "train_transaction.csv": 590_540,
    "train_identity.csv": 144_233,
    "test_transaction.csv": 506_691,
    "test_identity.csv": 141_907,
    "sample_submission.csv": 506_691,
}


def count_rows(path: Path) -> int:
    """Count newline-terminated data rows (header excluded) without loading the file."""
    with path.open("rb") as fh:
        return sum(1 for _ in fh) - 1


def download(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / f"{COMPETITION}.zip"
    if archive.exists():
        print(f"archive already present: {archive}")
        return archive
    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(raw_dir)]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not archive.exists():
        raise FileNotFoundError(f"expected {archive} after download")
    return archive


def extract(archive: Path, raw_dir: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        missing = set(EXPECTED_ROWS) - set(names)
        if missing:
            raise ValueError(f"archive is missing {sorted(missing)}; contains {names}")
        for name in EXPECTED_ROWS:
            target = raw_dir / name
            if target.exists():
                print(f"exists, skipping extract: {target.name}")
                continue
            print(f"extracting {name}")
            zf.extract(name, raw_dir)


def verify(raw_dir: Path) -> None:
    bad: list[str] = []
    for name, expected in EXPECTED_ROWS.items():
        actual = count_rows(raw_dir / name)
        status = "ok" if actual == expected else "MISMATCH"
        print(f"{name:26s} rows={actual:>9,d} expected={expected:>9,d} {status}")
        if actual != expected:
            bad.append(name)
    if bad:
        raise SystemExit(f"row-count mismatch in {bad}")


def main() -> None:
    archive = download(RAW_DIR)
    extract(archive, RAW_DIR)
    verify(RAW_DIR)
    print("done")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(f"kaggle CLI failed with exit code {exc.returncode}")
