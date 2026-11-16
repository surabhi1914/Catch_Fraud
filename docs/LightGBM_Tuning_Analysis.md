# Catch_Fraud: LightGBM Model Selection and Threshold Analysis

**Project:** Catch_Fraud, Fraud Intelligence Platform  
**Dataset:** IBM AML `HI-Small_Trans.csv` (synthetic financial transactions)  
**Model version:** V4 LightGBM, using frozen V3-E1 features  
**Analysis date:** November 16, 2026  
**Current stage:** Model selection and chronological out-of-fold evaluation completed; final refit and untouched test evaluation pending

## Abstract

I developed Catch_Fraud to investigate a central challenge in anti-money-laundering (AML) detection: identifying rare laundering transactions without flooding investigators with false alerts. The work progressed from a linear stochastic gradient descent (SGD) baseline toward a LightGBM classifier informed by a transaction's amount, payment characteristics, and strictly historical account behavior.

Using the same 17-feature representation and three chronological folds, the untuned LightGBM model achieved **0.360759 mean average precision (AP)**. After evaluating 20 Stage 1 configurations and 12 Stage 2 configurations, I selected **V4-T2-B-02**, which achieved **0.400486 mean fold AP**, an improvement of **0.039727 AP points, approximately 11.0% relative to the untuned model**. Although another configuration produced a higher mean AP of 0.409537, the selected model exhibited less variation between folds and a lower average false-positive count at individually optimized fold-level recall thresholds.

The operating-threshold analysis revealed a more difficult question than model selection. A pooled cutoff of **0.368623** recovered **80.01%** of laundering cases in combined chronological validation predictions with **32,946 alerts**, but recall fell to **68.15% in Fold 2**. A more conservative common cutoff of **0.105592** achieved at least 80% recall in each observed validation fold, while increasing the combined alert count to **96,247**, of which **94,557 were false positives**. The evidence therefore supports the value of the improved model while exposing a substantial tradeoff between recall consistency and investigation workload.

This report documents the experimental evidence, its interpretation, the model and threshold decisions, their limitations, and how the work contributes to a reproducible and eventually deployable ML system. It does not present final holdout performance or production deployment as completed work.

---

## 1. Problem definition and modeling objective

The problem is not to label every transaction correctly in aggregate. Most transactions are legitimate, so a model that almost never predicts laundering could achieve very high accuracy while missing the cases that matter. Conversely, flagging almost every transaction would catch nearly all laundering, but would make an analyst's review queue unusable.

I therefore treated the task as **rare-event risk ranking with an operational alerting decision**. The model assigns a continuous score to a transaction. A separately chosen threshold converts that score into an alert. This separation matters: model quality describes how well cases are ranked, while threshold quality describes the consequences of acting on those rankings.

The dataset labels **`is_laundering`**, so the measured outcome is synthetic money-laundering detection rather than every possible form of financial fraud. Any applicability to real banking data remains unvalidated.

### 1.1 Evaluation vocabulary

| Measure | Definition | Interpretation in this project |
|---|---|---|
| **True positive (TP)** | Laundering transaction correctly flagged | A laundering case surfaced for review |
| **False positive (FP)** | Legitimate transaction incorrectly flagged | An unnecessary investigation |
| **False negative (FN)** | Laundering transaction not flagged | A laundering case the system misses |
| **True negative (TN)** | Legitimate transaction correctly not flagged | A legitimate transaction kept out of the alert queue |
| **Precision** | `TP / (TP + FP)` | The proportion of alerts that are truly laundering |
| **Recall** | `TP / (TP + FN)` | The proportion of actual laundering transactions caught |
| **Alert rate** | `(TP + FP) / total transactions` | The portion of all transactions requiring review |
| **Average precision (AP)** | Summary of precision along the ranked recall curve | How effectively high-risk ranking concentrates laundering cases |
| **ROC-AUC** | Area under the true-positive-rate vs false-positive-rate curve | Broad separation of laundering from legitimate transactions |
| **AP lift over prevalence** | `AP / class prevalence` | Improvement over a no-skill ranking reference for this class distribution |

AP is the implementation's ranking metric and is often discussed as PR-AUC. Its numeric value is not a precision percentage at a particular threshold. For example, **AP of 0.40 does not mean 40% of alerts are laundering**. Precision at a chosen cutoff must be calculated separately.

A useful mental model is that the ranking model decides *which cases appear nearer the top of the review list*, while the threshold decides *how far down that list an investigator would go*. Improving one does not automatically solve the other.

---

## 2. Dataset, chronology, and leakage prevention

The source dataset contains approximately **5.08 million synthetic transactions**. My model-development period covers **September 1–7, 2022**, totaling **3,731,672 rows** with a laundering prevalence of approximately **0.081116%**. I reserved **September 8 onward** for a final holdout evaluation, which is not present in the reviewed tuning notebook's evaluation outputs.

Chronological separation is important because a deployed detector cannot learn from tomorrow's transactions when scoring today's transactions. I used three fixed, expanding-window folds rather than random shuffled cross-validation.

| Fold | Training transactions | Validation transactions | Actual laundering in validation | Validation prevalence |
|---|---:|---:|---:|---:|
| 1 | 2,076,752 | 207,430 | 407 | 0.1962% |
| 2 | 2,284,182 | 482,650 | 471 | 0.0976% |
| 3 | 2,766,832 | 964,840 | 1,028 | 0.1065% |
| **Combined validation** | Different historical training windows | **1,654,920** | **1,906** | **0.1152%** |

Each fold evaluates transactions occurring after the transactions used to train that fold's model. The combined out-of-fold (OOF) output contains **1,654,920 unique validation-row positions**, with **no duplicate positions** in the notebook's saved checks. The OOF predictions come from three independently fitted models, not a single model trained on all development data.

### 2.1 Why historical feature correctness matters

Several features summarize past behavior. Their intended semantics require that a transaction at time `t` can only use prior activity with timestamps strictly earlier than `t`. Transactions with the same timestamp cannot provide history for one another. Target labels must never enter the feature calculations.

This condition is as important as the chronological split itself. A correct time split would still give misleading results if historical aggregates were accidentally computed using future transactions. The tuning notebook loads already-engineered V3-E1 data, so the outputs establish the performance of those columns but do not independently prove that their upstream construction is leakage-free.

For eventual September 8+ scoring, historical calculations may use earlier transactions from September 1–7. They must not use later test-period transactions or same-timestamp transactions that would not have been known at prediction time.

---

## 3. Features and the reasoning behind the selected representation

The V3-E1 representation contains **17 features before categorical one-hot expansion**. I retained the same features throughout the tuning experiments so that differences between candidate models could be attributed to the tested model configurations rather than changing inputs.

### 3.1 Transaction-level features

The inputs describe the present transaction's time, relative banking/currency relationships, and monetary amount:

`hour_of_day`, `day_of_week`, `is_weekend`, `same_currency_flag`, `same_bank_flag`, `log_amount_received`, `log_amount_paid`, `receiving_currency`, `payment_currency`, and `payment_format`.

Log-transformed amounts preserve magnitude information while limiting the dominance of extreme raw monetary values. This choice followed earlier SGD error analysis, in which very large legitimate transfers frequently received extreme scores. A log transformation does not make outliers harmless, but it provides a more manageable numeric representation.

### 3.2 Historical sender and counterparty features

| Feature | Meaning and hypothesis |
|---|---|
| `sender_tx_count_1h` | Number of earlier sender transactions in the previous hour; frequent short-term activity may indicate unusual velocity |
| `log_sender_amount_paid_sum_24h_same_currency` | Log-transformed historical 24-hour sender spending in the same payment currency; measures recent monetary volume without directly mixing unlike currencies |
| `log_sender_hours_since_prev_tx` | Time since the sender's prior transaction, transformed; captures unusually short or long gaps |
| `log_amount_paid_vs_sender_median` | Current log amount relative to the sender's historical same-currency median; distinguishes unusual transfers from routine ones |
| `receiver_seen_before` | Whether the sender has previously transacted with this receiver; captures relationship novelty |
| `sender_receiver_prior_tx_count` | Number of strictly earlier sender-receiver transactions; measures relationship familiarity |
| `sender_distinct_receivers_24h` | Number of different receivers the sender contacted in the preceding 24 hours; captures short-term counterparty dispersion |

The core reasoning was that an amount of $500 can be ordinary for one account and unusual for another. An isolated transaction record cannot express that distinction. Historical features supply context, allowing the classifier to learn conditional patterns rather than relying on raw transaction characteristics alone.

### 3.3 Preprocessing and imbalanced learning

The fitted sklearn pipeline uses median imputation and `StandardScaler` for numeric columns, plus most-frequent imputation and `OneHotEncoder(handle_unknown="ignore", sparse_output=True)` for categorical columns. The same preprocessing procedure is fitted independently within each chronological training fold.

To account for rare laundering labels, training used `compute_sample_weight(class_weight="balanced", y=y_train)`, with weights calculated from each training fold only. I did not additionally set LightGBM's class weighting to `balanced`, avoiding duplicate balancing mechanisms. Weighting allows the rare class to influence learning more strongly, but the resulting `predict_proba` output should be interpreted as an **uncalibrated risk score** until probability calibration is assessed separately.

---

## 4. Original SGD baseline and the motivation for a nonlinear model

The initial SGD Baseline v1 established that useful discrimination was possible, but also exposed an extreme false-positive burden. On the original **September 6–7 validation period**, its results included:

| Measure | SGD baseline result |
|---|---:|
| Average precision | 0.010280 |
| ROC-AUC | 0.898973 |
| Laundering prevalence | 0.106546% |
| Precision at diagnostic 0.50 cutoff | ~0.59% |
| Recall at diagnostic 0.50 cutoff | ~87.35% |
| Alerts at diagnostic 0.50 cutoff | 153,273 |
| False positives at diagnostic 0.50 cutoff | 152,375 |
| Alert rate at diagnostic 0.50 cutoff | ~15.89% |

The SGD model could recover many laundering cases, but doing so generated an enormous number of incorrect alerts. Its high ROC-AUC did not contradict this result. ROC-AUC reflected broad ranking ability, while precision exposed the difficulty of distinguishing the rare laundering class from a much larger pool of legitimate transactions in the actionable high-risk region.

Earlier error inspection showed two important patterns: extremely large legitimate transactions could be assigned very high risk scores, while smaller, ordinary-looking laundering transactions could be missed. These findings motivated two changes: **nonlinear model learning** and **behavioral, history-aware features**.

LightGBM can learn interactions such as a large transfer being more or less suspicious depending on prior sender behavior, transaction type, and recipient history. That flexibility is a hypothesis-driven response to the linear baseline's shortcomings, not proof that tree models are inherently better for all fraud problems.

**Comparison boundary:** The SGD baseline and the final tuned LightGBM result use different feature/evaluation arrangements. I therefore do not present their AP ratio as an isolated measure of tuning improvement. The controlled tuning comparison is between LightGBM models using the **same V3-E1 features and same three folds**.

---

## 5. Hyperparameter tuning experiments

I treated model tuning as a constrained experiment rather than an unlimited search. Each configuration was evaluated using the same frozen chronological folds, training-only preprocessing and sample weights, and AP as the primary ranking metric. I also tracked temporal stability and alert burden because a marginal gain in AP could be less useful if it substantially increased the review workload.

### 5.1 Untuned reference

The untuned V3-E1 LightGBM benchmark achieved fold AP values of **0.377397**, **0.302327**, and **0.402554**, for a mean of **0.360759**. This benchmark provides the controlled reference for measuring the value added by hyperparameter tuning.

### 5.2 Stage 1: broad candidate exploration

Stage 1 evaluated **20 candidate configurations**, labeled `V4-T1-01` through `V4-T1-20`, across all three chronological folds. The implementation used `ParameterSampler` and an explicit evaluation loop rather than a direct `RandomizedSearchCV.fit()` call. The strongest Stage 1 candidate, **V4-T1-03**, achieved:

- Fold APs: **0.404564**, **0.342166**, **0.439840**.
- Mean AP: **0.395523**.

This was evidence that tuning improved the same-feature LightGBM benchmark, although the fold scores remained uneven.

### 5.3 Stage 2: targeted model comparison

Stage 2 evaluated **12 explicitly specified configurations**, divided between ranking-oriented (`R`) and operational-balance-oriented (`B`) variants. The notebook contains an earlier sampled list that was subsequently overwritten, so the evaluated Stage 2 candidates were the 12 explicitly defined variants, not those earlier sampled values.

| Candidate | Mean fold AP | Fold 1 AP | Fold 2 AP | Fold 3 AP | AP SD across folds | Mean fold FP at separate ≥80% recall cutoffs | Mean fold alert rate at those cutoffs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Untuned V3-E1 | 0.360759 | 0.377397 | 0.302327 | 0.402554 | — | 13,570 | ~2.91% |
| V4-T1-03 | 0.395523 | 0.404564 | 0.342166 | 0.439840 | 0.049461 | 16,825 | 3.41% |
| **V4-T2-R-02** | **0.409537** | 0.416741 | 0.354462 | **0.457407** | 0.051849 | 17,297 | 3.25% |
| **V4-T2-B-02** | 0.400486 | **0.420275** | **0.355299** | 0.425884 | **0.039233** | **12,669** | **2.67%** |

The false-positive and alert-rate columns refer to **different cutoffs selected separately within each fold** to reach approximately 80% recall. They are diagnostic measures of operating potential, **not results from a single fixed threshold**. Mean false-positive counts are simple averages across folds of unequal sizes, so they should not be interpreted as pooled counts.

### 5.4 Selection of V4-T2-B-02

The highest mean-AP model was **V4-T2-R-02** at **0.409537**. I selected **V4-T2-B-02** instead, at **0.400486**, because it provided a more balanced historical profile:

1. Its mean AP was only **0.009051 lower**, while remaining materially above the untuned benchmark.
2. Its across-fold AP standard deviation was **0.039233**, lower than R-02's **0.051849**.
3. At fold-specific cutoffs calibrated to roughly 80% recall, its mean false-positive count was **12,669**, compared with **17,297** for R-02. This represents **4,628 fewer false positives per fold on average**, under those diagnostic conditions.
4. Its mean diagnostic alert rate was **2.67%**, versus **3.25%** for R-02.

These observations led me to prioritize **stable ranking and manageable historical alert volume** over maximizing average AP alone. This is a model-selection judgment based on multiple validation metrics; it does not demonstrate that B-02 would outperform R-02 at every shared cutoff or in future data.

The selected model improved mean AP from **0.360759 to 0.400486**, an absolute increase of **0.039727** and a relative gain of about **11.0%**, with the feature set and chronological folds held fixed.

### 5.5 Selected configuration

| Hyperparameter | Value | Role in the model |
|---|---:|---|
| `num_leaves` | 63 | Controls the maximum number of leaf groups a tree may learn |
| `max_depth` | 10 | Limits tree depth and unrestricted branching |
| `min_child_samples` | 500 | Discourages rules built on very small leaf populations |
| `learning_rate` | 0.03 | Limits the contribution of each new boosting step |
| `n_estimators` | 500 | Number of boosting rounds |
| `reg_alpha` | 0.1 | L1 regularization of leaf values |
| `reg_lambda` | 2.0 | L2 regularization of leaf values |
| `subsample` | 0.9 | Samples rows when bagging is enabled |
| `subsample_freq` | 1 | Enables periodic row subsampling |
| `colsample_bytree` | 0.75 | Samples features per tree |

The configuration balances model capacity with constraints intended to discourage overfitting. I have not isolated the effect of each hyperparameter independently, so I interpret the observed improvement as the outcome of the configuration and evaluation process, not as proof about the causal contribution of any one setting.

---

## 6. Chronological out-of-fold evaluation

After model selection, I trained V4-T2-B-02 independently on each frozen training window and generated scores for each subsequent validation window. The resulting **1,654,920 OOF predictions** represent out-of-training-window evaluations rather than in-sample predictions.

| Fold | Transactions | Laundering cases | Positive prevalence | AP |
|---|---:|---:|---:|---:|
| 1 | 207,430 | 407 | 0.1962% | **0.420275** |
| 2 | 482,650 | 471 | 0.0976% | **0.355299** |
| 3 | 964,840 | 1,028 | 0.1065% | **0.425884** |
| **Equal-weight mean AP** | — | — | — | **0.400486** |

The mean AP is the **average of the three separately computed fold AP values**, not necessarily the AP that would result from ranking all 1.65 million OOF scores together. That distinction matters because scores came from three differently trained models.

### 6.1 Interpretation of Fold 2

Fold 2 was consistently more difficult. Its AP of **0.355299** was lower than Fold 1's **0.420275** and Fold 3's **0.425884**. One relevant difference is positive prevalence: Fold 2's laundering rate was approximately **0.0976%**, compared with Fold 1's **0.1962%**. AP depends partly on class prevalence, but that alone does not establish the cause of the performance gap.

Possible explanations include changing transaction composition, laundering patterns, sender histories, account populations, or training-data coverage. The current experiments do not isolate any of these mechanisms. The supported conclusion is narrower: **historical model performance varies over time, and the average conceals a weaker period**.

This finding shapes how I evaluate operational risk. A model with strong mean AP can still underperform during particular periods, so period-level recall and alert rates are necessary alongside ranking metrics.

---

## 7. Threshold experiments and operational interpretation

Ranking performance alone does not determine which transactions become alerts. For a score `s` and a threshold `τ`, the decision rule is:

`alert = 1 if s >= τ, otherwise 0`.

The threshold can be changed without retraining the model. Doing so changes the confusion matrix, precision, recall, and review burden, but not AP or ROC-AUC computed from the same underlying continuous scores.

My target in the threshold analysis was to assess the feasibility of **at least 80% recall**, while observing how many legitimate transactions that requirement would send to review.

### 7.1 Experiment A: pooled OOF threshold

I first selected a cutoff that reached at least 80% recall **across all combined OOF predictions**. The resulting cutoff was **0.36862335760028636**.

| Pooled outcome | Value |
|---|---:|
| Transactions scored | 1,654,920 |
| Actual laundering transactions | 1,906 |
| True positives | **1,525** |
| False negatives | **381** |
| False positives | **31,421** |
| True negatives | **1,621,593** |
| Alerts | **32,946** |
| Recall | **80.01%** |
| Precision | **4.63%** |
| Alert rate | **1.99%** |

At this cutoff, approximately **eight of every ten laundering transactions** were found across the combined validation population. Yet only **4.63 out of every 100 alerts** represented laundering. The remaining alerts would require attention despite being legitimate in the labeled dataset.

The aggregate outcome initially appears compatible with an 80% recall objective. The fold-level breakdown changes that interpretation:

| Fold | Recall at common 0.368623 cutoff | TP | FN | FP | Alerts | Alert rate |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 89.93% | 366 | 41 | 9,167 | 9,533 | 4.60% |
| **2** | **68.15%** | **321** | **150** | 4,340 | 4,661 | 0.97% |
| 3 | 81.52% | 838 | 190 | 17,914 | 18,752 | 1.94% |

**The main finding was that pooled success concealed a substantial period-level failure.** Fold 2 had a relatively small alert queue, but missed 150 of 471 laundering cases. That low alert rate was not evidence of a better operating point, because its recall was well below the intended floor.

I therefore did not regard 0.368623 as satisfying a **per-fold** 80% recall policy, even though it satisfied the pooled criterion.

### 7.2 Experiment B: common cutoff based on the weakest historical fold

I then determined the highest threshold within each fold that could retain at least approximately 80% recall:

| Fold | Fold-specific 80%-recall cutoff |
|---|---:|
| 1 | 0.740680 |
| **2** | **0.105592** |
| 3 | 0.422822 |

Fold 2 required a much lower threshold than the other periods. A single threshold that met the historical recall floor in all three folds was therefore based on the **minimum** of those values: **0.10559242552997583**.

| Fold | Recall | Precision | TP | FN | FP | Alerts | Alert rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 95.33% | 2.36% | 388 | 19 | 16,067 | 16,455 | 7.93% |
| 2 | 80.04% | 2.09% | 377 | 94 | 17,650 | 18,027 | 3.74% |
| 3 | 89.98% | 1.50% | 925 | 103 | 60,840 | 61,765 | 6.40% |
| **Combined** | **88.67%** | **1.76%** | **1,690** | **216** | **94,557** | **96,247** | **5.82%** |

The lower cutoff recovered **1,690 of 1,906 laundering transactions**, compared with 1,525 at the pooled cutoff. That is **165 additional detected laundering cases** and **165 fewer misses**.

However, the additional recall came at a substantial cost: **63,301 more alerts**, including **63,136 additional false positives**. Viewed another way, recovering those 165 extra cases in the combined historical validation data required flagging over sixty-three thousand additional legitimate transactions.

This is the most important operational finding of the study: **a model can satisfy a recall requirement while still imposing an impractical investigation burden**. A threshold cannot be described as preferable without specifying the relative costs of missed laundering and false investigations.

### 7.3 Experiment C: sensitivity to a fixed overall alert budget

I also examined cutoffs that flagged approximately 1% to 6% of all combined OOF transactions.

| Pooled alert budget | Score cutoff | Combined recall | Lowest fold recall | Combined precision | Combined false positives |
|---|---:|---:|---:|---:|---:|
| 1% | 0.690252 | 71.62% | 57.96% | 8.25% | 15,185 |
| 2% | 0.366827 | 80.01% | 68.15% | 4.61% | 31,574 |
| 3% | 0.239908 | 83.00% | 73.46% | 3.19% | 48,066 |
| 4% | 0.174046 | 85.41% | 75.80% | 2.46% | 64,569 |
| 5% | 0.131248 | 87.36% | 77.71% | 2.01% | 81,081 |
| 6% | 0.100711 | **89.03%** | **80.47%** | 1.71% | 97,599 |

The 2% pooled alert-budget setting appeared attractive because it recovered approximately 80% of laundering overall, but its worst-fold recall was only **68.15%**. At the 6% pooled budget, worst-fold recall exceeded 80%, but individual fold alert rates were not constrained to 6%. For example, Fold 1's alert rate was approximately **8.09%** at that setting.

The table illustrates why **pooled alert capacity, per-period alert capacity, pooled recall, and minimum period recall are separate objectives**. Achieving one does not imply the others.

---

## 8. Model and threshold decision record

### 8.1 Model selection

I selected **V4-T2-B-02** because it balanced ranking performance, variation across validation periods, and diagnostic false-positive burden more favorably than the highest-AP candidate. The selected model is not the maximum-AP configuration. Its advantage is better described as **historical multi-metric balance**, not universal superiority.

### 8.2 Threshold interpretation

The threshold **0.10559242552997583** is the **proposed recall-prioritized operating threshold** because it met at least 80% recall in each of the three observed chronological validation folds. The pooled threshold **0.368623** remains a useful alternative in the analysis: it substantially reduces alerts but fails the per-fold recall floor in Fold 2.

I treat the 0.105592 threshold as a **historical policy choice**, not a guaranteed future recall level or a demonstrated cost optimum. It was derived after selecting the model using the same overall development environment. The observed performance may therefore be optimistic. Moreover, the final model, once trained on all September 1–7 data, may produce a different score distribution from the three fold models that generated the OOF data.

The results do not define a real institution's analyst capacity, investigation cost, or relative penalty for missed cases. Consequently, the proposed policy prioritizes recall consistency rather than proving that a 5.82% historical pooled alert rate is operationally acceptable.

### 8.3 Evidence-based decision summary

| Decision dimension | Recorded conclusion |
|---|---|
| Frozen feature set | V3-E1, 17 inputs before one-hot expansion |
| Selected model | V4-T2-B-02 |
| Model-selection basis | Mean AP, lower fold-to-fold AP variation, and lower per-fold diagnostic alert burden |
| Mean fold AP | 0.400486 |
| Controlled tuning improvement | +0.039727 AP, about +11.0% relative to untuned V3-E1 |
| Historical recall priority | At least 80% recall in each observed fold |
| Proposed common threshold | 0.10559242552997583 |
| Historical pooled recall at threshold | 88.67% |
| Historical pooled precision at threshold | 1.76% |
| Historical pooled alert rate at threshold | 5.82% |
| Probability calibration | Not established |
| September 8+ final evaluation | Not yet performed in the reviewed notebook |

---

## 9. What the results establish, and what remains uncertain

### 9.1 Supported findings

**First, the selected model improves ranking under controlled comparison.** Keeping the feature set and folds fixed, tuning raised mean AP from 0.360759 to 0.400486. This is real evidence of a development-stage improvement, not merely a change in validation population.

**Second, temporal stability matters.** Fold 2 has weaker AP and much worse recall at the pooled 0.368623 cutoff. Reporting only the pooled or mean result would conceal a practical weakness.

**Third, threshold selection is a separate optimization problem.** A high-quality ranking model can still require a large number of alerts to satisfy a stringent historical recall target. No threshold removes the fundamental tradeoff between missed cases and false investigations.

**Fourth, historical account behavior adds a meaningful representation of risk.** Earlier feature-ablation experiments supported retaining sender velocity, recency, relative amount, previous receiver relationships, and distinct-receiver activity. These features express context that the original transaction-only baseline lacked. Their exact importance in individual predictions remains to be evaluated with dedicated explainability methods.

**Fifth, the results support an ML engineering story beyond algorithm selection.** The work requires data contracts, chronological evaluation, reproducible transformations, model serialization, historical feature consistency, and operational score interpretation. Those are central to turning notebook findings into a usable application.

### 9.2 Unresolved limitations

- The development dataset is synthetic; performance on real banking activity has not been established.
- The notebook does not independently audit the original V3-E1 historical feature-construction implementation.
- Sample-weighted LightGBM scores have not been calibrated as real event probabilities.
- The threshold's historical recall is not a guarantee for future periods.
- Model selection and threshold derivation both use development-fold evidence; further performance claims require the untouched holdout.
- A single refitted model may score differently from the three fold-specific OOF models.
- The selected B-02 candidate has not been shown to beat R-02 at every possible single common cutoff.
- The observed review volume has not been assessed against an actual staffing or investigation-cost constraint.
- No final September 8+ holdout results or deployed-system reliability metrics are available yet.

These limitations are part of the substantive conclusion. They prevent the analysis from overstating what a strong experimental result proves.

---