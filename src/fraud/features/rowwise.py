"""Row-local derived features (feature sets F1, F2, F4, F5).

Every function is pure — frame in, frame with extra columns out — and reads only
the row's own fields, so each is safe at serving time without any state. They
are registered in :mod:`fraud.features.derive` and run as the pipeline's first
step, so training and the API compute them identically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud.data import schema

# --- F1: missingness counts -------------------------------------------------------

_TX_COUNTED = tuple(
    c for c in schema.TRANSACTION_COLS if c not in (schema.ID_COL, schema.TARGET_COL)
)
_ID_COUNTED = tuple(c for c in schema.IDENTITY_COLS if c != schema.ID_COL)
MISSINGNESS_FEATURES: tuple[str, ...] = (
    "n_missing_transaction",
    "n_missing_identity",
    "n_missing_total",
)


def add_missingness_counts(df: pd.DataFrame) -> pd.DataFrame:
    """How many transaction / identity fields are missing on this row."""
    tx = df[list(_TX_COUNTED)].isna().sum(axis=1).astype("int64")
    idn = df[list(_ID_COUNTED)].isna().sum(axis=1).astype("int64")
    return df.assign(n_missing_transaction=tx, n_missing_identity=idn, n_missing_total=tx + idn)


# --- F2: amount structure ---------------------------------------------------------

AMOUNT_FEATURES: tuple[str, ...] = (
    "amt_log",
    "amt_integer_part",
    "amt_fraction",
    "amt_is_round",
    "amt_cents_digits",
)


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose the amount: log, integer part, cents, whether it is a round number."""
    amt = df["TransactionAmt"].astype("float64")
    cents = np.round(amt * 100).astype("int64")
    integer = cents // 100
    fraction = (cents % 100).astype("float64") / 100.0
    # number of significant decimal digits in the cents: 0 (x.00), 1 (x.50), 2 (x.99)
    digits = np.where(cents % 100 == 0, 0, np.where(cents % 10 == 0, 1, 2))
    return df.assign(
        amt_log=np.log1p(amt),
        amt_integer_part=integer.astype("float64"),
        amt_fraction=fraction,
        amt_is_round=((cents % 100 == 0) & (integer % 5 == 0)).astype("int64"),
        amt_cents_digits=digits.astype("int64"),
    )


# --- F4: e-mail structure ----------------------------------------------------------

EMAIL_FEATURES: tuple[str, ...] = ("p_email_missing", "r_email_missing", "same_email_domain")
EMAIL_FAMILY_FEATURES: tuple[str, ...] = ("p_email_family", "r_email_family")

_EMAIL_FAMILIES: dict[str, str] = {
    "gmail.com": "gmail",
    "googlemail.com": "gmail",
    "yahoo.com": "yahoo",
    "yahoo.com.mx": "yahoo",
    "yahoo.co.uk": "yahoo",
    "yahoo.fr": "yahoo",
    "yahoo.de": "yahoo",
    "yahoo.es": "yahoo",
    "yahoo.co.jp": "yahoo",
    "ymail.com": "yahoo",
    "rocketmail.com": "yahoo",
    "hotmail.com": "microsoft",
    "hotmail.fr": "microsoft",
    "hotmail.de": "microsoft",
    "hotmail.es": "microsoft",
    "hotmail.co.uk": "microsoft",
    "outlook.com": "microsoft",
    "outlook.es": "microsoft",
    "live.com": "microsoft",
    "live.com.mx": "microsoft",
    "live.fr": "microsoft",
    "msn.com": "microsoft",
    "icloud.com": "apple",
    "me.com": "apple",
    "mac.com": "apple",
    "aol.com": "aol",
    "aim.com": "aol",
    "anonymous.com": "anonymous",
    "protonmail.com": "privacy",
}


def email_family(domain: pd.Series) -> pd.Series:
    fam = domain.map(_EMAIL_FAMILIES)
    fam = fam.where(fam.notna() | domain.isna(), other="other")
    return fam.where(domain.notna(), other="missing").astype("str")


def add_email_features(df: pd.DataFrame) -> pd.DataFrame:
    p, r = df["P_emaildomain"], df["R_emaildomain"]
    return df.assign(
        p_email_missing=p.isna().astype("int64"),
        r_email_missing=r.isna().astype("int64"),
        same_email_domain=(p.notna() & r.notna() & (p == r)).astype("int64"),
        p_email_family=email_family(p),
        r_email_family=email_family(r),
    )


# --- F5: card / address identifiers ----------------------------------------------

INTERACTION_FEATURES: tuple[str, ...] = (
    "card1_addr1",
    "card1_addr1_pemail",
    "card1_card4",
    "card1_card6",
)


def _key(*parts: pd.Series) -> pd.Series:
    """Join columns into one string key; any missing component makes the key missing."""
    complete = np.logical_and.reduce([p.notna().to_numpy() for p in parts])
    joined = parts[0].astype("str")
    for p in parts[1:]:
        joined = joined + "|" + p.astype("str")
    return joined.where(complete, other=None).astype("str")


def _numeric_key(s: pd.Series) -> pd.Series:
    """Integer-valued numbers as ``"1000"``, others as their float repr; NaN stays NaN."""
    out = s.map(
        lambda v: None if pd.isna(v) else (str(int(v)) if float(v).is_integer() else repr(float(v)))
    )
    return pd.Series(out, index=s.index, dtype="object")


def add_interaction_keys(df: pd.DataFrame) -> pd.DataFrame:
    """String identifiers for card × address / e-mail / card-type combinations.

    They are never used raw; the feature spec lists them under ``frequency`` so
    they enter as training-window shares (ADR 0005).
    """
    c1 = _numeric_key(df["card1"])
    a1 = _numeric_key(df["addr1"])
    return df.assign(
        card1_addr1=_key(c1, a1),
        card1_addr1_pemail=_key(c1, a1, df["P_emaildomain"]),
        card1_card4=_key(c1, df["card4"]),
        card1_card6=_key(c1, df["card6"]),
    )
