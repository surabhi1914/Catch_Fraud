from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TIMESTAMP_COL = "transaction_timestamp"


@dataclass(frozen=True)
class WalkForwardFold:
    """Defines one chronological train/validation fold."""

    name: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str


FROZEN_FOLDS = (
    WalkForwardFold(
        name="fold_1",
        train_start="2022-09-01",
        train_end="2022-09-04",
        val_start="2022-09-04",
        val_end="2022-09-05",
    ),
    WalkForwardFold(
        name="fold_2",
        train_start="2022-09-01",
        train_end="2022-09-05",
        val_start="2022-09-05",
        val_end="2022-09-06",
    ),
    WalkForwardFold(
        name="fold_3",
        train_start="2022-09-01",
        train_end="2022-09-06",
        val_start="2022-09-06",
        val_end="2022-09-08",
    ),
)


def build_walkforward_splits(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
) -> list[tuple[np.ndarray, np.ndarray]]:

    if timestamp_col not in df.columns:
        raise ValueError(f"Missing required timestamp column: {timestamp_col}")

    timestamps = pd.to_datetime(
        df[timestamp_col],
        errors="raise",
    )

    # Final test-set guard.
    if (timestamps >= pd.Timestamp("2022-09-08")).any():
        raise ValueError(
            "September 8+ rows detected. "
            "The final test set must not be used during development."
        )

    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for fold in FROZEN_FOLDS:
        train_start = pd.Timestamp(fold.train_start)
        train_end = pd.Timestamp(fold.train_end)

        val_start = pd.Timestamp(fold.val_start)
        val_end = pd.Timestamp(fold.val_end)

        train_mask = (timestamps >= train_start) & (timestamps < train_end)

        val_mask = (timestamps >= val_start) & (timestamps < val_end)

        train_indices = np.flatnonzero(train_mask.to_numpy())
        val_indices = np.flatnonzero(val_mask.to_numpy())

        if len(train_indices) == 0:
            raise ValueError(f"{fold.name} contains no training rows.")

        if len(val_indices) == 0:
            raise ValueError(f"{fold.name} contains no validation rows.")

        # Important temporal sanity check.
        latest_train_time = timestamps.iloc[train_indices].max()
        earliest_val_time = timestamps.iloc[val_indices].min()

        if latest_train_time >= earliest_val_time:
            raise ValueError(
                f"Temporal leakage detected in {fold.name}: "
                f"latest training timestamp={latest_train_time}, "
                f"earliest validation timestamp={earliest_val_time}"
            )

        splits.append(
            (
                train_indices,
                val_indices,
            )
        )

    return splits


def describe_walkforward_splits(
    df: pd.DataFrame,
    target_col: str = "is_laundering",
) -> pd.DataFrame:

    # Return a human-readable summary of the frozen folds. This is primarily for verification/debugging.

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    splits = build_walkforward_splits(df)

    rows = []

    for fold, (train_idx, val_idx) in zip(
        FROZEN_FOLDS,
        splits,
        strict=True,
    ):
        train = df.iloc[train_idx]
        val = df.iloc[val_idx]

        train_fraud = int(train[target_col].sum())
        val_fraud = int(val[target_col].sum())

        rows.append(
            {
                "fold": fold.name,
                "train_period": (f"{fold.train_start} -> {fold.train_end}"),
                "train_rows": len(train),
                "train_fraud": train_fraud,
                "validation_period": (f"{fold.val_start} -> {fold.val_end}"),
                "val_rows": len(val),
                "val_fraud": val_fraud,
                "val_fraud_rate_pct": (val_fraud / len(val) * 100),
            }
        )

    return pd.DataFrame(rows)
