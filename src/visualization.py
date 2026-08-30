import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------
# v3a3 results across the 3 chronological validation folds
# ---------------------------------------------------------

results = pd.DataFrame(
    {
        "fold": ["Fold 1", "Fold 2", "Fold 3"],
        "average_precision": [0.111973, 0.075083, 0.118588],
        "precision_at_80_recall": [0.030399, 0.010131, 0.023508],
        "false_positives": [10398, 36836, 34186],
        "alert_rate": [0.051699, 0.077101, 0.036285],
    }
)


# ---------------------------------------------------------
# Create figure
# ---------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

folds = results["fold"]


# =========================================================
# 1. Average Precision
# =========================================================

ax = axes[0, 0]

bars = ax.bar(folds, results["average_precision"])

ax.set_title("Fraud Ranking Quality")
ax.set_ylabel("Average Precision (AP)")
ax.set_ylim(0, 0.14)

for bar, value in zip(bars, results["average_precision"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{value:.3f}",
        ha="center",
        fontsize=9,
    )

ax.text(0.02, 0.93, "Higher is better ↑", transform=ax.transAxes, fontsize=9)


# =========================================================
# 2. Precision at >=80% recall
# =========================================================

ax = axes[0, 1]

precision_pct = results["precision_at_80_recall"] * 100

bars = ax.bar(folds, precision_pct)

ax.set_title("Alert Precision at ≥80% Recall")
ax.set_ylabel("Precision (%)")
ax.set_ylim(0, 3.6)

for bar, value in zip(bars, precision_pct):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.08,
        f"{value:.2f}%",
        ha="center",
        fontsize=9,
    )

ax.text(0.02, 0.93, "Higher is better ↑", transform=ax.transAxes, fontsize=9)


# =========================================================
# 3. False positives
# =========================================================

ax = axes[1, 0]

bars = ax.bar(folds, results["false_positives"])

ax.set_title("False Positive Burden")
ax.set_ylabel("False Positives")

for bar, value in zip(bars, results["false_positives"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 700,
        f"{value:,}",
        ha="center",
        fontsize=9,
    )

ax.text(0.02, 0.93, "Lower is better ↓", transform=ax.transAxes, fontsize=9)


# =========================================================
# 4. Alert rate
# =========================================================

ax = axes[1, 1]

alert_pct = results["alert_rate"] * 100

bars = ax.bar(folds, alert_pct)

ax.set_title("Analyst Alert Workload")
ax.set_ylabel("Alert Rate (%)")
ax.set_ylim(0, 9)

for bar, value in zip(bars, alert_pct):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.18,
        f"{value:.2f}%",
        ha="center",
        fontsize=9,
    )

ax.text(0.02, 0.93, "Lower is better ↓", transform=ax.transAxes, fontsize=9)


# ---------------------------------------------------------
# Overall formatting
# ---------------------------------------------------------

fig.suptitle(
    "LightGBM Fraud Detection — Performance Across Chronological Folds",
    fontsize=15,
    fontweight="bold",
)

fig.text(
    0.5,
    0.02,
    "v3a3 evaluated at an operating point maintaining ≥80% fraud recall",
    ha="center",
    fontsize=10,
)

for ax in axes.flat:
    ax.grid(axis="y", alpha=0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


plt.tight_layout(rect=[0, 0.05, 1, 0.93])

plt.show()
