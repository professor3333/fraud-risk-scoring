"""Data contract for the raw IEEE-CIS files.

Column names are enumerated explicitly so that a renamed or missing column fails
loudly at load time rather than silently downstream.
"""

from __future__ import annotations

from typing import Final

ID_COL: Final = "TransactionID"
TARGET_COL: Final = "isFraud"
TIME_COL: Final = "TransactionDT"
HAS_IDENTITY_COL: Final = "has_identity"

# --- train_transaction.csv ----------------------------------------------------

TRANSACTION_HEAD: Final[tuple[str, ...]] = (
    ID_COL,
    TARGET_COL,
    TIME_COL,
    "TransactionAmt",
    "ProductCD",
    *[f"card{i}" for i in range(1, 7)],
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
)
C_COLS: Final[tuple[str, ...]] = tuple(f"C{i}" for i in range(1, 15))
D_COLS: Final[tuple[str, ...]] = tuple(f"D{i}" for i in range(1, 16))
M_COLS: Final[tuple[str, ...]] = tuple(f"M{i}" for i in range(1, 10))
V_COLS: Final[tuple[str, ...]] = tuple(f"V{i}" for i in range(1, 340))

TRANSACTION_COLS: Final[tuple[str, ...]] = (
    *TRANSACTION_HEAD,
    *C_COLS,
    *D_COLS,
    *M_COLS,
    *V_COLS,
)

TRANSACTION_STR_COLS: Final[frozenset[str]] = frozenset(
    {"ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", *M_COLS}
)
# Columns that are never null in the raw file. Everything else may be missing.
TRANSACTION_NON_NULL: Final[frozenset[str]] = frozenset(
    {ID_COL, TARGET_COL, TIME_COL, "TransactionAmt", "ProductCD", "card1", *C_COLS}
)

# --- train_identity.csv -------------------------------------------------------

IDENTITY_COLS: Final[tuple[str, ...]] = (
    ID_COL,
    *[f"id_{i:02d}" for i in range(1, 39)],
    "DeviceType",
    "DeviceInfo",
)
IDENTITY_STR_COLS: Final[frozenset[str]] = frozenset(
    {
        "id_12",
        "id_15",
        "id_16",
        "id_23",
        "id_27",
        "id_28",
        "id_29",
        "id_30",
        "id_31",
        "id_33",
        "id_34",
        "id_35",
        "id_36",
        "id_37",
        "id_38",
        "DeviceType",
        "DeviceInfo",
    }
)
IDENTITY_NON_NULL: Final[frozenset[str]] = frozenset({ID_COL})

# Sanity bounds for the label rate; the published positive rate is ~3.5 %.
LABEL_RATE_BOUNDS: Final[tuple[float, float]] = (0.02, 0.06)
# Identity rows exist for roughly a quarter of transactions.
IDENTITY_COVERAGE_BOUNDS: Final[tuple[float, float]] = (0.15, 0.40)

assert len(TRANSACTION_COLS) == 394, len(TRANSACTION_COLS)
assert len(IDENTITY_COLS) == 41, len(IDENTITY_COLS)
