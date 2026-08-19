from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def validate_score_inputs(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate binary labels and continuous model scores.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true and y_score must both be 1-dimensional.")

    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must contain the same number of rows.")

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty validation set.")

    unique_labels = set(np.unique(y_true))

    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"y_true must contain only 0 and 1. Found: {sorted(unique_labels)}"
        )

    if len(unique_labels) != 2:
        raise ValueError("Validation data must contain both legitimate and fraud rows.")

    if not np.isfinite(y_score).all():
        raise ValueError("y_score contains NaN or infinite values.")

    return y_true.astype(np.int8), y_score


def operating_point_at_recall_floor(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_recall: float = 0.80,
) -> dict[str, float | int]:
    """
    Find the highest score cutoff that still achieves at least
    `min_recall`.

    This is a diagnostic operating point for comparing models.

    It is NOT the final production threshold.
    """

    y_true, y_score = validate_score_inputs(
        y_true,
        y_score,
    )

    if not 0 < min_recall <= 1:
        raise ValueError("min_recall must be greater than 0 and at most 1.")

    # Sort highest-risk transactions first.
    order = np.argsort(-y_score, kind="stable")

    sorted_labels = y_true[order]
    sorted_scores = y_score[order]

    total_fraud = int(y_true.sum())

    cumulative_true_positives = np.cumsum(sorted_labels)

    cumulative_recall = cumulative_true_positives / total_fraud

    # First ranked position where the recall floor is reached.
    crossing_positions = np.flatnonzero(cumulative_recall >= min_recall)

    if len(crossing_positions) == 0:
        raise ValueError(f"Could not reach recall >= {min_recall:.2f}.")

    crossing_index = crossing_positions[0]

    diagnostic_cutoff = float(sorted_scores[crossing_index])

    # Include all rows tied at the cutoff.
    predictions = (y_score >= diagnostic_cutoff).astype(np.int8)

    true_positive = int(((predictions == 1) & (y_true == 1)).sum())

    false_positive = int(((predictions == 1) & (y_true == 0)).sum())

    false_negative = int(((predictions == 0) & (y_true == 1)).sum())

    true_negative = int(((predictions == 0) & (y_true == 0)).sum())

    alerts = true_positive + false_positive

    precision = true_positive / alerts if alerts > 0 else 0.0

    recall = true_positive / (true_positive + false_negative)

    alert_rate = alerts / len(y_true)

    return {
        "diagnostic_cutoff": diagnostic_cutoff,
        "precision_at_recall_floor": precision,
        "recall_at_recall_floor": recall,
        "false_positives_at_recall_floor": false_positive,
        "true_positives_at_recall_floor": true_positive,
        "false_negatives_at_recall_floor": false_negative,
        "true_negatives_at_recall_floor": true_negative,
        "alerts_at_recall_floor": alerts,
        "alert_rate_at_recall_floor": alert_rate,
    }


def evaluate_fold_scores(
    fold_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_recall: float = 0.80,
) -> dict[str, float | int | str]:
    """
    Evaluate continuous risk scores from one chronological
    validation fold.
    """

    y_true, y_score = validate_score_inputs(
        y_true,
        y_score,
    )

    rows = len(y_true)
    frauds = int(y_true.sum())
    prevalence = frauds / rows

    average_precision = average_precision_score(
        y_true,
        y_score,
    )

    roc_auc = roc_auc_score(
        y_true,
        y_score,
    )

    pr_lift = average_precision / prevalence

    operating_point = operating_point_at_recall_floor(
        y_true=y_true,
        y_score=y_score,
        min_recall=min_recall,
    )

    return {
        "fold": fold_name,
        "rows": rows,
        "frauds": frauds,
        "prevalence": prevalence,
        "average_precision": average_precision,
        "pr_lift": pr_lift,
        "roc_auc": roc_auc,
        "recall_floor": min_recall,
        **operating_point,
    }


def build_fold_metrics_table(
    fold_results: Sequence[dict[str, float | int | str]],
) -> pd.DataFrame:
    """
    Combine multiple fold-level result dictionaries into one table.
    """

    if not fold_results:
        raise ValueError("At least one fold result is required.")

    table = pd.DataFrame(fold_results)

    columns = [
        "fold",
        "rows",
        "frauds",
        "prevalence",
        "average_precision",
        "pr_lift",
        "roc_auc",
        "recall_floor",
        "diagnostic_cutoff",
        "precision_at_recall_floor",
        "recall_at_recall_floor",
        "false_positives_at_recall_floor",
        "true_positives_at_recall_floor",
        "alerts_at_recall_floor",
        "alert_rate_at_recall_floor",
    ]

    return table[columns]


def summarize_temporal_stability(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize ranking stability across chronological folds.
    """

    if fold_metrics.empty:
        raise ValueError("fold_metrics cannot be empty.")

    ap = fold_metrics["average_precision"]

    summary = {
        "mean_average_precision": ap.mean(),
        "std_average_precision": ap.std(ddof=0),
        "min_average_precision": ap.min(),
        "max_average_precision": ap.max(),
        "range_average_precision": (ap.max() - ap.min()),
        "latest_fold_average_precision": ap.iloc[-1],
        "mean_pr_lift": fold_metrics["pr_lift"].mean(),
        "latest_fold_pr_lift": (fold_metrics["pr_lift"].iloc[-1]),
        "mean_precision_at_recall_floor": (
            fold_metrics["precision_at_recall_floor"].mean()
        ),
        "mean_alert_rate_at_recall_floor": (
            fold_metrics["alert_rate_at_recall_floor"].mean()
        ),
        "latest_fold_precision_at_recall_floor": (
            fold_metrics["precision_at_recall_floor"].iloc[-1]
        ),
        "latest_fold_alert_rate_at_recall_floor": (
            fold_metrics["alert_rate_at_recall_floor"].iloc[-1]
        ),
    }

    return pd.DataFrame([summary])
