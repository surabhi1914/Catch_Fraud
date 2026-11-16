# Catch_Fraud: Final LightGBM Holdout Evaluation

**Model:** V4-T2-B-02  
**Feature set:** V3-E1 (17 engineered features)  
**Evaluation period:** September 8, 2022, 00:00 through September 18, 2022, 16:18  
**Evaluation status:** First and final frozen-model holdout evaluation  
**Frozen decision threshold:** `0.105592425529`  
**Dataset:** IBM AML HI-Small synthetic transactions

## 1. Objective and evaluation protocol

I developed a transaction-level classifier for identifying potentially laundering-related transactions in the synthetic IBM AML dataset. My primary modeling objective was to detect at least 80% of laundering transactions while measuring the resulting false-positive and analyst-review burden.

I reserved all transactions from September 8 onward as a chronological holdout. I did not use this period to fit the model, engineer new feature definitions, tune hyperparameters, select the candidate model, or choose its decision threshold.

I selected LightGBM configuration **V4-T2-B-02** using development-period chronological validation. The final model was refitted on **3,731,672 transactions from September 1–7**, comprising **3,027 labeled laundering transactions** (approximately **0.0811%** prevalence). Its complete preprocessing-and-model pipeline was saved, reloaded, and verified before the holdout evaluation.

To build the holdout features, I applied the frozen V3-E1 feature definitions to transaction history in chronological order. Historical features only referenced strictly earlier timestamps; same-timestamp transactions did not provide history for one another. I verified each of the seven historical features against the saved development features across all 3,731,672 rows, with zero mismatches. I then isolated the September 8+ transactions and attached their labels only after feature construction.

The decision threshold, **0.105592425529**, was fixed before opening the holdout labels. It was chosen as a conservative development-period threshold that achieved at least 80% recall in each of three historical validation folds. This was a **recall-oriented experimental policy**, not an optimized or proven production alert policy. The class-weighted model scores are treated as ranking/risk scores rather than calibrated laundering probabilities.

## 2. Holdout population

| Property | Value |
|---|---:|
| Transactions | 1,346,673 |
| Labeled laundering transactions | 2,150 |
| Labeled legitimate transactions | 1,344,523 |
| Laundering prevalence | 0.1597% |
| Start | 2022-09-08 00:00:00 |
| End | 2022-09-18 16:18:00 |

The holdout prevalence was approximately **1.97 times** the full development prevalence (0.1597% vs. 0.0811%). This matters when comparing average precision between development and test periods: AP is sensitive to positive-class prevalence, so an increase in AP should not automatically be interpreted as an increase in underlying discrimination.

## 3. Frozen-model results

| Metric | Final holdout result |
|---|---:|
| Average precision (AP / PR-AUC summary) | **0.550584** |
| ROC-AUC | **0.985645** |
| Precision at frozen threshold | **0.016542 (1.6542%)** |
| Recall at frozen threshold | **0.941860 (94.1860%)** |
| F1 score | **0.032512** |
| Total alerts | **122,418** |
| Alert rate | **0.090904 (9.0904%)** |
| False positives per true positive | **59.453333** |
| Met preselected 80% recall floor | **Yes** |

### Confusion matrix

Rows represent actual labels; columns represent predicted labels.

| | Predicted legitimate | Predicted laundering |
|---|---:|---:|
| **Actually legitimate** | **1,224,130** (TN) | **120,393** (FP) |
| **Actually laundering** | **125** (FN) | **2,025** (TP) |

The totals reconcile: TN + FP + FN + TP = **1,346,673**; TP + FN = **2,150**; and TP + FP = **122,418**.

## 4. What the metrics mean

### Recall: most labeled laundering transactions were detected

Recall is the share of actual laundering transactions that were flagged:

`2,025 / (2,025 + 125) = 94.19%`.

I detected **2,025 of 2,150** laundering transactions and missed **125**. The frozen threshold therefore achieved the previously defined 80% recall goal on the holdout set. This is a meaningful result because it was obtained without changing the model or threshold after inspecting the test labels.

### Precision: most flagged transactions were legitimate

Precision measures the share of alerts that correspond to labeled laundering:

`2,025 / (2,025 + 120,393) = 1.65%`.

Only about **1.65 of every 100 alerts** corresponded to labeled laundering. The remaining **98.35%** were false positives. This demonstrates that high recall does not necessarily produce an operationally manageable investigation queue.

### Alert rate: the threshold imposes substantial review work

The model flagged **122,418 of 1,346,673 transactions**, or **9.09%** of all holdout transactions. It generated approximately **59.45 false positives for each true positive**. In a human-review workflow, this volume could be costly even though the recall target was met.

The chosen threshold is intentionally permissive. I used the least restrictive historical-fold constraint necessary to keep each fold above 80% recall, accepting additional alerts to reduce the risk of missed laundering. The test result reveals the operational cost of that decision.

### Average precision and ROC-AUC: strong ranking, not calibrated probability

The holdout **average precision was 0.550584** and **ROC-AUC was 0.985645**. These are threshold-independent ranking metrics. They indicate that the model ranked laundering transactions above legitimate transactions effectively across possible operating points.

Neither statistic implies that a flagged transaction has a 55% or 98.6% probability of laundering. In particular, class-balanced training scores are not assumed to be calibrated probabilities.

The holdout prevalence baseline for average precision was approximately **0.001597** (0.1597%); the observed AP was roughly **345 times** that baseline. This is a ranking-lift comparison, **not** an estimate of actual alert precision, business return, or improvement over another model on the same holdout.

## 5. Comparison with development-period evidence

| Metric | Development-period evidence | Final holdout |
|---|---:|---:|
| Average precision | 0.400486 mean across three validation folds | 0.550584 |
| Recall goal | At least 80% in each historical fold at frozen threshold | 94.186% |
| Alert rate at frozen threshold | 5.82% pooled across OOF predictions | 9.09% |
| Laundering prevalence | 0.0811% across development transactions | 0.1597% |

The holdout AP exceeds the mean validation AP by about **0.1501**. This is encouraging but requires careful interpretation. Fold models were trained on smaller, earlier training windows, whereas the final artifact used all September 1–7 development data. The holdout also had a higher laundering prevalence. These conditions prevent a clean claim that the model's ranking ability improved by exactly 0.1501 because of training alone.

At the operational threshold, the model achieved **94.19% recall**, but the holdout alert rate rose to **9.09%**, compared with approximately **5.82%** in pooled historical OOF results. Thus, the model generalized well under the recall criterion, while the false-positive burden remained a weakness and varied over time.

## 6. Conclusions supported by the evidence

1. **The frozen model met the primary recall requirement.** It detected 94.19% of labeled laundering transactions in a previously withheld chronological period.
2. **The risk-ranking model showed strong discrimination on this holdout.** Its AP and ROC-AUC are favorable, but they do not measure workload at a particular threshold.
3. **The current binary-alert policy is operationally expensive.** The 9.09% alert rate and 1.65% precision translate to 120,393 false alerts during the holdout period.
4. **The holdout results support generalization to this specific synthetic period, not real-world deployment readiness.** The population, synthetic data-generating process, and analyst capacity may differ sharply from an actual financial institution.
5. **This evaluation is the final frozen-model test report.** I should not revise the model or its decision threshold based on the observed holdout labels and then present a second evaluation on the same holdout as independent evidence.

## 7. Limitations and interpretation boundaries

- **Synthetic data:** The IBM AML dataset is simulated. The model has not been validated using real financial-institution transactions.
- **High false-positive burden:** A 9.09% alert rate may exceed realistic investigator capacity. No actual staffing or cost assumptions were supplied, so the threshold cannot be called business-optimal.
- **Temporal changes:** The holdout fraud prevalence differs from development. Precision, AP, and alert volume should be monitored by time period rather than assumed stable.
- **Uncalibrated scores:** Because of the modeling/weighting strategy, risk scores should not be presented as probabilities without a separately validated calibration process.
- **Historical feature availability:** Deployment will require strict event-time feature generation, consistent entity keys, state/history management, same-timestamp handling, and leakage-safe online or batch inference.
- **Research iteration:** The chosen configuration and threshold were selected after repeated development-fold experiments. The single untouched holdout is the strongest independent check for this version.


## 9. Project summary for GitHub and resumes

**Evidence-backed project description:**

> Built a LightGBM-based AML transaction risk detector using 5.08M synthetic transactions and 17 transaction/behavioral features. Evaluated a frozen model on a 1.35M-row chronological holdout, achieving **0.551 average precision, 0.986 ROC-AUC, and 94.2% recall** at a **9.09% alert rate**.

This wording describes what has actually been completed. API, website, monitoring, and deployment achievements should be added to the project summary **only after those components are implemented and tested**.

---

**Source:** Frozen model evaluation output from `08_final_lightgbm_test_evaluation.ipynb`, shared November 16, 2026. Numeric results are reported as output by the evaluation cell. Historical validation and feature-parity details reflect the preceding project experiments.
