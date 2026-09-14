# Data

Source: Kaggle competition **IEEE-CIS Fraud Detection**
(<https://www.kaggle.com/c/ieee-fraud-detection>). The data is under the
competition's rules and is **not committed** to this repository.

## Layout

```
data/
├── raw/          # the unmodified Kaggle CSVs
│   ├── train_transaction.csv   (~590k rows × 394 cols)
│   ├── train_identity.csv      (~144k rows × 41 cols)
│   ├── test_transaction.csv    (unlabelled; not used for training)
│   ├── test_identity.csv
│   └── sample_submission.csv
└── processed/    # Parquet caches written by scripts/; safe to delete
```

Only the **labelled `train_*` files** are used. Kaggle's `test_*` files have no
labels, so all train/validation/test windows are carved from `train_*` by time.

## Download

1. Accept the competition rules on the Kaggle page (required once, in a browser).
2. Put a Kaggle API token at `~/.kaggle/kaggle.json` (Kaggle → Settings → API →
   *Create New Token*), `chmod 600` it.
3. Run:

```bash
uv run python scripts/download_data.py
```

which fetches and unzips the files into `data/raw/` and verifies the expected
row counts.
