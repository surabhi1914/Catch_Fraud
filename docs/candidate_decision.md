# Improved Model Candidate Selection — Why LightGBM Was Chosen

## Decision

**LightGBM is selected as the primary improved model for the next phase of the Fraud Intelligence Platform.**

This decision is based on the three frozen chronological walk-forward folds, the frozen SGD Baseline v1 benchmark, operational false-positive behavior, temporal performance, and compute cost on a multi-million-row dataset.

CatBoost remains a useful comparison model, but LightGBM is the better model to carry forward into feature engineering and hyperparameter tuning.

---

## 1. Evaluation setup used for the decision

Both LightGBM and CatBoost were evaluated under the same development policy:

- September 8+ remained untouched as the final test set.
- September 1–7 was used only as development data.
- The same three expanding chronological folds were used:

| Fold | Training period | Validation period |
|---|---|---|
| Fold 1 | Sep 1–3 | Sep 4 |
| Fold 2 | Sep 1–4 | Sep 5 |
| Fold 3 | Sep 1–5 | Sep 6–7 |

The candidates were compared using:

- Average Precision / PR-AUC as the primary ranking metric;
- PR lift over prevalence;
- ROC-AUC as a secondary ranking metric;
- precision at approximately 80% recall;
- false positives at approximately 80% recall;
- alert rate at approximately 80% recall;
- temporal stability across folds;
- training cost.

No operational threshold was selected during this stage.

---

## 2. LightGBM produced stronger overall fraud ranking

The clearest reason to prefer LightGBM is its stronger Average Precision across all three chronological folds.

| Fold | LightGBM AP | CatBoost AP |
|---|---:|---:|
| Fold 1 | **0.063079** | 0.056461 |
| Fold 2 | **0.032902** | 0.023961 |
| Fold 3 | **0.047368** | 0.034632 |
| Mean | **0.047783** | 0.038351 |

LightGBM achieved higher AP on **every fold**.

Its mean AP was approximately **24.6% higher** than CatBoost's:

```text
LightGBM mean AP = 0.047783
CatBoost mean AP = 0.038351
```

Because Average Precision is the primary optimization metric for this highly imbalanced fraud problem, this is the strongest piece of evidence in favor of LightGBM.

The result also suggests that LightGBM provides a better overall ordering of fraudulent versus legitimate transactions across the full precision-recall curve.

---

## 3. LightGBM produced stronger PR lift across time

The validation folds have different fraud prevalence, so raw AP values should not be interpreted in isolation.

PR lift was therefore also recorded:

```text
PR lift = Average Precision / fraud prevalence
```

Results:

| Fold | LightGBM PR Lift | CatBoost PR Lift |
|---|---:|---:|
| Fold 1 | **32.15×** | 28.78× |
| Fold 2 | **33.72×** | 24.55× |
| Fold 3 | **44.46×** | 32.50× |
| Mean | **36.77×** | 28.61× |

LightGBM again outperformed CatBoost on every fold.

This matters because the folds have materially different positive rates. The improvement is therefore not simply explained by LightGBM receiving an easier validation period.

LightGBM consistently concentrated fraud toward the top of the ranking more strongly than would be expected from the underlying fraud prevalence.

---

## 4. LightGBM gave the strongest result on the most important comparison fold

Fold 3 is particularly important because it exactly matches the validation period used for SGD Baseline v1:

```text
Training:   Sep 1–5
Validation: Sep 6–7
```

This gives a direct apples-to-apples comparison.

### Fold 3 Average Precision

```text
SGD Baseline v1 = 0.010280
CatBoost        = 0.034632
LightGBM        = 0.047368
```

Relative to SGD:

```text
CatBoost  ≈ 3.37× baseline AP
LightGBM  ≈ 4.61× baseline AP
```

LightGBM therefore produced the strongest improvement over the frozen baseline on the exact same temporal validation window.

This is important because the project is not attempting to replace SGD merely with a more complicated model. The new model should demonstrate a material improvement over the frozen benchmark.

LightGBM clearly does so.

---

## 5. LightGBM better matches the primary model-selection objective

The main evaluation objective for candidate selection is **ranking quality**, measured primarily with Average Precision.

Operational threshold selection is intentionally postponed until after:

1. feature development;
2. hyperparameter tuning;
3. final candidate selection;
4. generation of chronological out-of-fold predictions.

For that reason, the candidate-selection stage should not choose a model only because one temporary diagnostic cutoff happens to produce fewer alerts.

LightGBM provides the strongest overall ranking model before operational threshold selection.

That gives the next stages a better ranking foundation from which to:

- improve features;
- tune the model;
- evaluate Precision@k and Recall@k;
- evaluate fixed analyst alert budgets;
- eventually select a threshold from chronological OOF predictions.

---

## 6. LightGBM is substantially more computationally practical

The dataset contains millions of transactions, and each model configuration must eventually be evaluated across three chronological folds.

This makes training cost an important modeling constraint.

Observed fit times:

| Fold | LightGBM | CatBoost |
|---|---:|---:|
| Fold 2 | **16.65 s** | 101.97 s |
| Fold 3 | **19.61 s** | 119.41 s |

CatBoost took roughly **six times longer** on these folds.

This difference becomes much more important during hyperparameter tuning.

For example:

```text
10 parameter configurations
× 3 chronological folds
= 30 model fits
```

A model that takes approximately six times longer per fit can make iterative experimentation substantially more expensive.

Since the development dataset contains several million rows, LightGBM provides a better balance between:

- model strength;
- training speed;
- repeated walk-forward evaluation;
- feature experimentation;
- future randomized hyperparameter search.

This is a major practical reason for selecting LightGBM.

---

## 7. LightGBM directly addresses important SGD limitations

SGD Baseline v1 is a linear additive classifier.

The baseline analysis showed several important limitations:

- extreme legitimate transactions could receive extremely high fraud scores;
- the legitimate high-risk tail overlapped substantially with fraud;
- raw transaction amounts appeared to have excessive influence;
- low/moderate-value fraud could look ordinary using only transaction-level features;
- the linear model could not naturally learn complex feature interactions.

A tree-based boosting model is structurally better suited to several of these problems.

LightGBM can learn conditional relationships such as:

```text
large amount
AND payment format = ACH
AND cross-bank transaction
AND particular currency relationship
```

rather than representing risk only as a linear sum of independent feature effects.

It can also model nonlinear thresholds in numerical features rather than allowing a numerical contribution to grow linearly without bound.

The large improvement in AP after switching from SGD to LightGBM while keeping essentially the same underlying transaction information supports the hypothesis that **nonlinear interactions were an important limitation of the baseline**.

However, this experiment does not prove that any single feature—such as raw transaction amount—was the sole cause of SGD's failure.

---

## 8. CatBoost revealed an important tradeoff

LightGBM did not dominate every operational metric.

At approximately 80% recall, CatBoost produced fewer false positives in Fold 2 and Fold 3.

### Fold 2

```text
LightGBM
Precision = 0.567%
False positives = 66,063
Alert rate = 13.77%

CatBoost
Precision = 1.128%
False positives = 33,043
Alert rate = 6.92%
```

CatBoost reduced the Fold 2 false-positive burden by roughly half.

### Fold 3

```text
LightGBM
Precision = 1.573%
False positives = 51,490
Alert rate = 5.42%

CatBoost
Precision = 1.648%
False positives = 49,116
Alert rate = 5.18%
```

CatBoost was again slightly better at this specific high-recall operating point.

Therefore, the conclusion is **not** that LightGBM is universally better than CatBoost.

Instead:

> LightGBM is the stronger overall ranking model and is much more computationally efficient, while CatBoost showed potentially better local behavior around the approximately 80% recall operating region.

This tradeoff should be remembered when evaluating the final LightGBM model after feature engineering.

---

## 9. Why LightGBM is still preferred despite CatBoost's lower alert burden

The decision favors LightGBM because candidate selection at this stage should prioritize the strongest combination of:

1. **overall ranking quality**;
2. **performance across chronological folds**;
3. **performance on the latest development fold**;
4. **material improvement over SGD Baseline v1**;
5. **scalability to millions of rows**;
6. **practicality for repeated feature experiments and hyperparameter tuning**.

LightGBM wins strongly on these criteria.

CatBoost's lower alert burden is important, but the current values come from a single untuned configuration and one diagnostic recall floor.

Operational behavior will be evaluated again after feature engineering and tuning using:

- precision at comparable recall;
- false positives at comparable recall;
- alert rate;
- Precision@k;
- Recall@k;
- recall under fixed analyst alert budgets.

The final operational threshold will not be chosen until after chronological OOF predictions are generated for the final winning configuration.

---

## 10. Temporal behavior

Neither model was perfectly stable across all three folds.

LightGBM AP:

```text
Fold 1 = 0.063079
Fold 2 = 0.032902
Fold 3 = 0.047368
```

CatBoost AP:

```text
Fold 1 = 0.056461
Fold 2 = 0.023961
Fold 3 = 0.034632
```

Both models experienced weaker performance on Fold 2.

This suggests that Sep 5 represents a more difficult temporal regime rather than a problem unique to LightGBM.

Importantly, LightGBM recovered strongly on Fold 3 instead of continuing to deteriorate.

The model should therefore continue to be evaluated fold-by-fold during feature engineering rather than using only mean performance.

---

## 11. Final model-selection rationale

LightGBM is selected because it currently offers the strongest overall balance of predictive performance and practical scalability.

### Main reasons

- Higher Average Precision on **all three chronological folds**.
- Mean AP approximately **24.6% higher than CatBoost**.
- Higher PR lift on all three folds.
- Best latest-fold Average Precision.
- Approximately **4.61× the SGD baseline AP** on the exact Sep 6–7 comparison window.
- Stronger overall precision-recall ranking.
- Natural ability to model nonlinear relationships and feature interactions that SGD cannot represent directly.
- Better fit for heavy-tailed and interaction-driven tabular patterns than a linear additive model.
- Approximately **6× faster training** than CatBoost in the larger observed folds.
- More practical for repeated walk-forward feature experiments and randomized hyperparameter tuning on a multi-million-row dataset.

### Important caveat

CatBoost achieved fewer false positives at approximately 80% recall in Fold 2 and Fold 3.

Therefore, future LightGBM improvements must not optimize AP alone.

The next phase should continue evaluating:

- Average Precision;
- temporal stability;
- precision at high recall;
- false-positive burden;
- alert rate;
- eventually Precision@k / Recall@k and fixed alert-budget performance.

---

## 12. Decision

**Proceed with LightGBM as the primary improved-model family.**

The next development phase will focus on **feature engineering**, using the same frozen chronological folds and common evaluation framework.

The objective is not merely to increase PR-AUC.

The main modeling goal remains:

> Improve fraud ranking while reducing the number of legitimate transactions occupying the high-risk tail, especially at operationally useful recall levels.

September 8+ remains completely untouched until all feature, model, hyperparameter, and threshold decisions are finalized.
