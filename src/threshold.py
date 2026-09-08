import numpy as np

RECALL_FLOOR = 0.80


def find_threshold_for_recall_floor(
    y_true,
    y_score,
    recall_floor=0.80,
):
    """
    Find the highest score threshold that reaches at least the requested recall floor.
    Predictions tied at the selected threshold are all  included as alerts.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    total_positives = y_true.sum()

    if total_positives == 0:
        raise ValueError("No positive observations available.")

    # Highest scores first.
    order = np.argsort(-y_score)

    y_sorted = y_true[order]
    score_sorted = y_score[order]

    cumulative_tp = np.cumsum(y_sorted)

    cumulative_recall = cumulative_tp / total_positives

    valid_positions = np.flatnonzero(cumulative_recall >= recall_floor)

    if len(valid_positions) == 0:
        raise ValueError("Requested recall floor cannot be reached.")

    first_valid_position = valid_positions[0]

    threshold = float(score_sorted[first_valid_position])

    return threshold


def evaluate_fixed_threshold(
    y_true,
    y_score,
    threshold,
):
    """
    Evaluate one fixed score threshold.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    predicted_alert = y_score >= threshold

    tp = np.sum(predicted_alert & (y_true == 1))

    fp = np.sum(predicted_alert & (y_true == 0))

    tn = np.sum((~predicted_alert) & (y_true == 0))

    fn = np.sum((~predicted_alert) & (y_true == 1))

    alerts = int(predicted_alert.sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    alert_rate = alerts / len(y_true)

    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "threshold": threshold,
        "transactions": len(y_true),
        "frauds": int(y_true.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": precision,
        "recall": recall,
        "alerts": alerts,
        "alert_rate": alert_rate,
        "false_positive_rate": (false_positive_rate),
    }


def threshold_for_alert_rate(
    y_score,
    target_alert_rate,
):
    """
    Find a score threshold corresponding approximately
    to the requested fraction of highest-scoring rows.

    Ties at the threshold may make the actual alert rate
    slightly larger than requested.
    """

    y_score = np.asarray(y_score)

    if not 0 < target_alert_rate <= 1:
        raise ValueError("target_alert_rate must be between 0 and 1.")

    n_alerts = int(np.ceil(len(y_score) * target_alert_rate))

    sorted_scores = np.sort(y_score)[::-1]

    threshold = float(sorted_scores[n_alerts - 1])

    return threshold
