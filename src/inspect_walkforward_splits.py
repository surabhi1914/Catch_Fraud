from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TIMESTAMP_COL = "transaction_timestamp"
TARGET_COL = "is_laundering"

DEV_START = pd.Timestamp("2022-09-01")
TEST_START = pd.Timestamp("2022-09-08")


def load_split(path: Path) -> pd.DataFrame:
    """Load only timestamp and target columns from a CSV or Parquet split."""

    columns = [TIMESTAMP_COL, TARGET_COL]

    return pd.read_parquet(path, columns=columns)


def summarize_period(
    df: pd.DataFrame,
    start: str,
    end: str,
) -> dict[str, float | int]:
    """Summarize transactions and frauds in [start, end)."""

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    period = df[(df[TIMESTAMP_COL] >= start_ts) & (df[TIMESTAMP_COL] < end_ts)]

    transactions = len(period)
    frauds = int(period[TARGET_COL].sum())

    fraud_rate = frauds / transactions * 100 if transactions > 0 else 0.0

    return {
        "transactions": transactions,
        "frauds": frauds,
        "fraud_rate_pct": fraud_rate,
    }


def main(train_path: Path) -> None:
    dev_df = load_split(train_path)

    # Parse timestamp safely.
    dev_df[TIMESTAMP_COL] = pd.to_datetime(
        dev_df[TIMESTAMP_COL],
        errors="raise",
    )

    # Ensure the target is numeric/binary.
    dev_df[TARGET_COL] = pd.to_numeric(
        dev_df[TARGET_COL],
        errors="raise",
    ).astype(int)

    unique_targets = set(dev_df[TARGET_COL].unique())

    # safe error handling choice
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"{TARGET_COL} must contain only 0/1. Found: {sorted(unique_targets)}"
        )

    # Critical leakage guard check - this script must never contain Sept 8+ rows.
    if (dev_df[TIMESTAMP_COL] >= TEST_START).any():
        raise ValueError(
            "September 8+ data detected. The final test set must remain untouched."
        )

    # a check to make sure there is no record before sept 1
    if (dev_df[TIMESTAMP_COL] < DEV_START).any():
        raise ValueError("Rows before September 1 were detected.")

    # ---------------------------------------------------------
    # 1. DAILY DEVELOPMENT DISTRIBUTION
    # ---------------------------------------------------------

    dev_df["date"] = dev_df[TIMESTAMP_COL].dt.floor("D")

    daily = dev_df.groupby("date", as_index=False).agg(
        transactions=(TARGET_COL, "size"),
        frauds=(TARGET_COL, "sum"),
    )

    daily["fraud_rate_pct"] = daily["frauds"] / daily["transactions"] * 100

    print("\nDAILY DEVELOPMENT DISTRIBUTION")
    print("=" * 65)

    print(
        daily.to_string(
            index=False,
            formatters={"fraud_rate_pct": lambda x: f"{x:.6f}%"},
        )
    )

    # ---------------------------------------------------------
    # 2. VERIFY THE FROZEN SEP 6-7 BASELINE WINDOW
    # ---------------------------------------------------------

    baseline_window = summarize_period(
        dev_df,
        "2022-09-06",
        "2022-09-08",
    )

    print("\nSEP 6-7 BASELINE VALIDATION CHECK")
    print("=" * 65)

    print(f"Transactions : {baseline_window['transactions']:,}")
    print(f"Frauds       : {baseline_window['frauds']:,}")

    expected_transactions = 964_840
    expected_frauds = 1_028

    matches_baseline = (
        baseline_window["transactions"] == expected_transactions
        and baseline_window["frauds"] == expected_frauds
    )

    print(f"Matches frozen baseline: {matches_baseline}")

    # ---------------------------------------------------------
    # 3. PROPOSED WALK-FORWARD FOLDS
    # ---------------------------------------------------------

    folds = [
        {
            "fold": "Fold 1",
            "train_start": "2022-09-01",
            "train_end": "2022-09-04",
            "val_start": "2022-09-04",
            "val_end": "2022-09-05",
        },
        {
            "fold": "Fold 2",
            "train_start": "2022-09-01",
            "train_end": "2022-09-05",
            "val_start": "2022-09-05",
            "val_end": "2022-09-06",
        },
        {
            "fold": "Fold 3",
            "train_start": "2022-09-01",
            "train_end": "2022-09-06",
            "val_start": "2022-09-06",
            "val_end": "2022-09-08",
        },
    ]

    fold_rows = []

    for fold in folds:
        train_stats = summarize_period(
            dev_df,
            fold["train_start"],
            fold["train_end"],
        )

        val_stats = summarize_period(
            dev_df,
            fold["val_start"],
            fold["val_end"],
        )

        fold_rows.append(
            {
                "fold": fold["fold"],
                "train_period": (f"{fold['train_start']} -> {fold['train_end']}"),
                "train_tx": train_stats["transactions"],
                "train_fraud": train_stats["frauds"],
                "validation_period": (f"{fold['val_start']} -> {fold['val_end']}"),
                "val_tx": val_stats["transactions"],
                "val_fraud": val_stats["frauds"],
                "val_fraud_rate_pct": val_stats["fraud_rate_pct"],
            }
        )

    fold_summary = pd.DataFrame(fold_rows)

    print("\nPROPOSED WALK-FORWARD FOLDS")
    print("=" * 65)

    print(
        fold_summary.to_string(
            index=False,
            formatters={"val_fraud_rate_pct": lambda x: f"{x:.6f}%"},
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Inspect September 1-7 development data "
            "before freezing walk-forward CV folds."
        )
    )

    parser.add_argument(
        "--train",
        type=Path,
        required=True,
        help="Existing September 1-5 training split.",
    )

    args = parser.parse_args()

    main(train_path=args.train)
