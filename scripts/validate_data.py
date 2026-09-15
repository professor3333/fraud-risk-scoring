"""Run the raw-data contract checks and print a report.

Example:
    uv run python scripts/validate_data.py        # data/raw, expects the published row counts
    uv run python scripts/validate_data.py --raw-dir tests/fixtures/raw --no-row-count
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from fraud.data import schema
from fraud.data.load import join_transaction_identity
from fraud.data.validate import (
    IDENTITY_FILE,
    TRANSACTION_FILE,
    SchemaError,
    check_row_count,
    validate_identity,
    validate_join,
    validate_transactions,
)

ROOT = Path(__file__).resolve().parents[1]


def run(name: str, fn: Callable[[], None]) -> bool:
    try:
        fn()
    except SchemaError as exc:
        print(f"FAIL  {name}: {exc}")
        return False
    print(f"ok    {name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--no-row-count", action="store_true", help="skip the published counts")
    args = parser.parse_args()

    # Read without the loader's built-in validation so every check is reported, not the first.
    tx_dtypes = dict.fromkeys(schema.TRANSACTION_STR_COLS, "str")
    id_dtypes = dict.fromkeys(schema.IDENTITY_STR_COLS, "str")
    tx = pd.read_csv(args.raw_dir / TRANSACTION_FILE, dtype=tx_dtypes)
    idn = pd.read_csv(args.raw_dir / IDENTITY_FILE, dtype=id_dtypes)
    print(f"{TRANSACTION_FILE}: {tx.shape[0]:,} rows x {tx.shape[1]} cols")
    print(f"{IDENTITY_FILE}:    {idn.shape[0]:,} rows x {idn.shape[1]} cols")

    def row_counts() -> None:
        check_row_count(tx, TRANSACTION_FILE)
        check_row_count(idn, IDENTITY_FILE)

    def join_checks() -> None:
        merged = join_transaction_identity(tx, idn)
        validate_join(tx, idn, merged)
        print(
            f"      joined: {merged.shape[0]:,} rows x {merged.shape[1]} cols; "
            f"identity coverage {merged[schema.HAS_IDENTITY_COL].mean():.1%}; "
            f"label rate {merged[schema.TARGET_COL].mean():.2%}"
        )

    checks: list[tuple[str, Callable[[], None]]] = [
        (
            "transactions: columns, dtypes, non-null, unique id, label in {0,1}, rate, DT > 0",
            lambda: validate_transactions(tx),
        ),
        ("identity: columns, dtypes, unique id", lambda: validate_identity(idn)),
    ]
    if not args.no_row_count:
        checks.append(("row counts match the published data page", row_counts))
    checks.append(
        ("left join keeps every transaction, duplicates none, flags identity", join_checks)
    )

    results = [run(name, fn) for name, fn in checks]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
