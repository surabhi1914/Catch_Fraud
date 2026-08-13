# SGD Baseline — Evaluation Findings and Modeling Implications

## 1. Baseline purpose

The SGD classifier is used as a **fixed linear benchmark**, not as the final fraud detector.

The model was trained chronologically:

* **Training:** September 1–5
* **Validation:** September 6–7
* **Final test:** September 8+ remains untouched

The baseline uses logistic loss, L2 regularization, balanced class weights, and a preprocessing pipeline containing numerical scaling and categorical one-hot encoding.

The model converged successfully after **14 epochs**, well below the configured maximum of 100, with no convergence warning.

---

## 2. Validation class imbalance

The validation fraud prevalence is extremely low:

* **Positive rate:** `0.106546%`
* Roughly **1 fraud transaction per 939 transactions**

This makes accuracy inappropriate as a primary metric.

Even a relatively small false-positive rate can generate a very large number of alerts because legitimate transactions overwhelmingly dominate the dataset.

The evaluation therefore focuses primarily on:

* PR-AUC
* precision
* recall
* false-positive burden
* alert rate
* threshold behavior

ROC-AUC is retained as a secondary ranking metric.

---

## 3. Ranking performance

| Metric                      |     Result |
| --------------------------- | ---------: |
| PR-AUC                      | `0.010280` |
| ROC-AUC                     | `0.898973` |
| Naive PR-AUC / prevalence   | `0.001065` |
| PR-AUC lift over prevalence |    `9.65x` |

### Interpretation

The ROC-AUC of approximately `0.899` shows that the model has learned meaningful global ranking information and generally assigns higher scores to fraud than to legitimate transactions.

However, ROC-AUC is not sufficient for evaluating this highly imbalanced problem. A model can rank the majority of legitimate transactions correctly while still producing many false positives in the small high-risk region that matters operationally.

PR-AUC provides a more useful view of performance on the rare positive class.

The model PR-AUC of `0.010280` is approximately `9.65x` the positive-class prevalence, showing that SGD has learned genuine fraud signal beyond random ranking.

The `9.65x` value should **not** be interpreted as the detector being 9.65 times better overall. The denominator is extremely small, so the relative lift appears large even though absolute precision-recall performance remains limited.

**Conclusion:** the baseline learns meaningful signal, but high-risk class separation remains weak.

---

## 4. Default threshold diagnostic

A probability threshold of `0.50` was used only as an **initial diagnostic threshold**.

It was not selected from validation data and should not be interpreted as the final operational fraud threshold.

At threshold `0.50`:

| Metric          |    Result |
| --------------- | --------: |
| Precision       |  `0.0059` |
| Recall          |  `0.8735` |
| F1              |  `0.0116` |
| True positives  |     `898` |
| False positives | `152,375` |
| False negatives |     `130` |
| True negatives  | `811,437` |
| Alert rate      |  `~15.9%` |

### Interpretation

The model detects approximately **87.35% of fraud**, which is encouraging.

However, this recall is achieved by flagging approximately **15.9% of all validation transactions**.

For every detected fraud, the system produces roughly:

* `171` total alerts
* `170` false alerts

Therefore, recall alone is not an adequate objective.

A model that simply labels more transactions as fraud will naturally increase recall while potentially overwhelming investigators.

The improved detector must preserve useful recall while substantially reducing false-positive burden.

---

## 5. Threshold sensitivity

Increasing the classification threshold reduced false positives, confirming that `0.50` is not an appropriate operational cutoff.

| Threshold | Precision |   Recall |      F1 | False Positives |    Alerts | Alert Rate |
| --------: | --------: | -------: | ------: | --------------: | --------: | ---------: |
|    `0.50` |   `0.59%` | `87.35%` | `1.16%` |       `152,375` | `153,273` |   `15.89%` |
|    `0.60` |   `0.83%` | `86.48%` | `1.63%` |       `106,849` | `107,738` |   `11.17%` |
|    `0.80` |   `0.84%` | `85.41%` | `1.66%` |       `103,832` | `104,710` |   `10.85%` |
|    `0.95` |   `0.89%` | `84.44%` | `1.76%` |        `96,795` |  `97,663` |   `10.12%` |
|    `0.99` |   `1.28%` | `55.35%` | `2.49%` |        `44,036` |  `44,605` |    `4.62%` |

### Interpretation

Raising the threshold from `0.50` to `0.95`:

* reduced false positives by `55,580`
* reduced recall only from `87.35%` to `84.44%`

This indicates that the model contains useful ranking information and that `0.50` is unnecessarily aggressive.

However, moving from `0.95` to `0.99`:

* reduced false positives from `96,795` to `44,036`
* but reduced recall from `84.44%` to `55.35%`

At this point, reducing false positives requires sacrificing a large fraction of detected fraud.

**Conclusion:** threshold adjustment improves the operating point but cannot fully solve the baseline's class-separation problem.

The final operational threshold will therefore not be selected from this experiment. Threshold selection will occur only after the winning improved model is identified using chronological out-of-fold predictions.

---

## 6. Score-distribution analysis

Fraud and legitimate transactions show very different central score distributions.

### Fraud decision scores

| Quantile |   Score |
| -------: | ------: |
|      25% | `4.120` |
|      50% | `4.732` |
|      75% | `5.288` |
|      90% | `5.716` |

### Legitimate decision scores

| Quantile |    Score |
| -------: | -------: |
|      25% | `-4.065` |
|      50% | `-3.296` |
|      75% | `-2.437` |
|      90% |  `2.974` |
|      95% |  `4.498` |
|      97% |  `4.992` |

### Interpretation

The typical fraud transaction receives a substantially higher score than the typical legitimate transaction.

This explains the relatively strong ROC-AUC.

However, the upper tail of legitimate transactions overlaps heavily with the main fraud distribution.

For example:

* Fraud median: `4.732`
* Legitimate 95th percentile: `4.498`
* Legitimate 97th percentile: `4.992`

Because legitimate transactions outnumber fraud by several hundred to one, even a small high-risk legitimate tail produces tens of thousands of false positives.

This overlap explains why global ranking can appear strong while precision remains poor.

---

## 7. Risk scores are not calibrated fraud probabilities

A large number of transactions receive scores close to `1.0`.

For example:

* At score threshold `0.95`, observed precision is only `0.89%`.
* At score threshold `0.99`, observed precision is only `1.28%`.

Therefore, a model output such as `0.99` must **not** be interpreted as a literal 99% probability that the transaction is fraudulent.

The current scores should be treated primarily as **risk-ranking scores**.

Balanced class weighting and extreme linear decision values likely contribute to this behavior.

Formal probability calibration has not yet been performed.

---

## 8. Error analysis — extreme false positives

The highest-scoring legitimate transactions are dominated by extremely large monetary values.

Examples include transactions on the order of:

* hundreds of billions
* more than one trillion currency units

These transactions receive extremely large positive decision scores, including values above `100,000`.

The SGD coefficient analysis supports the hypothesis that raw monetary amounts strongly influence the decision function:

| Feature               | Coefficient |
| --------------------- | ----------: |
| `amount_paid`         |   `+65.293` |
| `amount_received`     |   `+40.501` |
| `log_amount_paid`     |    `+0.869` |
| `log_amount_received` |    `+0.029` |

Raw amount and log-transformed amount are both present in the feature set.

Because the raw amount distribution is extremely heavy-tailed, StandardScaler does not remove the effect of extreme observations. Very large standardized values multiplied by large positive coefficients can produce extremely large decision scores.

This strongly suggests that the baseline has learned an overly strong approximately linear relationship between transaction magnitude and fraud risk.

The raw and log amount representations are also strongly related, creating redundant information that may reduce coefficient stability.

This should be treated as a hypothesis supported by the observed coefficients and error patterns rather than a formally isolated causal effect.

---

## 9. Error analysis — missed fraud

The lowest-scoring fraud transactions reveal a different failure pattern.

Many false-negative examples are:

* low or moderate monetary values
* Credit Card transactions
* same-currency transfers
* otherwise ordinary-looking transactions

Several fraud cases receive strongly negative decision scores around `-4` to `-6`, meaning they are not merely borderline cases—the linear model places them confidently in the legitimate region.

This demonstrates that fraud cannot be modeled only as unusually large transactions.

Some laundering behavior may resemble ordinary transactions when considered individually.

This exposes an important limitation of the current transaction-level feature set: it lacks behavioral context such as recent transaction frequency, new counterparties, account-specific amount deviations, and other temporal patterns.

---

## 10. Payment-format coefficients

Payment-format coefficients are strongly negative:

| Payment Format | Coefficient |
| -------------- | ----------: |
| ACH            |    `-3.788` |
| Bitcoin        |    `-6.055` |
| Cash           |   `-10.732` |
| Cheque         |   `-11.394` |
| Credit Card    |   `-12.077` |
| Wire           |   `-13.124` |
| Reinvestment   |   `-14.467` |

Credit Card appears considerably more negative than ACH, which is consistent with many low-scoring fraud examples being Credit Card transactions.

However, these coefficients should not be interpreted independently as absolute fraud probabilities or causal effects.

All one-hot categories remain in the model and the classifier also contains an intercept. Their values are therefore most useful for understanding **relative linear contributions within this fitted model**.

The broader error analysis is more important than any individual coefficient.

---

## 11. Main baseline failure modes

### Failure mode 1 — Extreme legitimate transactions

Very large legitimate transactions receive extreme fraud scores.

Likely contributors:

* heavy-tailed raw amount features
* strong positive amount coefficients
* linear extrapolation
* insufficient contextual features

### Failure mode 2 — Ordinary-looking fraud

Low/moderate-value fraud, particularly Credit Card activity, can receive strongly negative scores.

Likely contributors:

* additive linear decision boundary
* weak representation of feature interactions
* absence of account-level behavioral history

### Failure mode 3 — High-risk class overlap

The model separates most ordinary legitimate transactions from fraud well, but the high-score legitimate tail overlaps substantially with fraud.

This is the main reason precision remains low despite strong recall and ROC-AUC.

---

## 12. What the improved model must solve

The objective for the improved model is **not simply higher recall**.

The SGD baseline already achieves high recall by generating too many alerts.

The improved model should primarily:

1. improve high-risk fraud-vs-legitimate separation;
2. materially increase PR-AUC;
3. reduce false positives while preserving useful recall;
4. handle nonlinear interactions between transaction characteristics;
5. become less sensitive to extreme raw transaction values;
6. eventually incorporate leakage-safe temporal/behavioral information.

Primary model-selection metric:

**PR-AUC**

Important secondary operational comparisons:

* precision at a fixed recall target
* false positives at a fixed recall target
* alert rate at a fixed recall target
* Precision@k
* Recall@k
* temporal stability across chronological validation folds
* ROC-AUC as a secondary global ranking metric

A future model should be considered meaningfully better only if it improves the quality of the high-risk region, rather than merely producing a slightly higher ROC-AUC.

---

## 13. Baseline conclusion

The SGD baseline successfully established that the current feature set contains meaningful fraud signal.

It:

* converged normally;
* achieved strong broad ranking performance;
* detected most fraud at permissive thresholds;
* substantially exceeded random PR-AUC.

However, it is **not operationally adequate**.

Its main limitation is the inability to cleanly separate true fraud from a relatively small but numerically very large high-risk legitimate population.

Threshold tuning improves the trade-off but cannot fully correct this overlap.

The baseline therefore serves its intended purpose: it provides a fixed benchmark and reveals concrete weaknesses that the improved modeling stage must address.

**Baseline status: frozen as `SGD Baseline v1`.**
