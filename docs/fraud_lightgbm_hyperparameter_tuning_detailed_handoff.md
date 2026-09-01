# Fraud Intelligence Platform — LightGBM Hyperparameter Tuning

**Detailed Reasoning & Handoff**

Feature version: V3-E1 \| Tuning phase: V4 \| Status: Hyperparameters frozen

| **Item**                            | **Status**                            |
|-------------------------------------|---------------------------------------|
| **Features**                        | Frozen at V3-E1                       |
| **Chronological folds**             | Frozen; same 3 walk-forward folds     |
| **Preprocessing**                   | Frozen                                |
| **Class imbalance handling**        | Fold-specific balanced sample weights |
| **Primary metric**                  | Average Precision / PR-AUC            |
| **Selected LightGBM configuration** | V4-T2-B-02                            |
| **Final threshold**                 | NOT selected yet                      |
| **September 8+ test set**           | UNTOUCHED                             |

*Primary source of truth: “Fraud Detection — LightGBM Hyperparameter Tuning Handoff.” Empirical Stage 1 and Stage 2 results are the notebook outputs generated during this tuning work.*

# 1. Executive summary

This document records the full LightGBM hyperparameter-tuning phase after feature engineering was frozen at V3-E1. The purpose of tuning was not to reopen feature selection or chase the highest single-fold score. The goal was to find a LightGBM configuration that improved fraud ranking quality across the same three chronological validation folds while remaining temporally stable and operationally reasonable at a diagnostic recall floor of at least 80%.

> **Final tuning decision**
>
> Stop hyperparameter tuning after Stage 2 and freeze V4-T2-B-02 as the winning LightGBM configuration. It does not have the absolute highest mean PR-AUC, but it delivers a strong PR-AUC improvement over V3-E1 while materially improving temporal stability and reducing false-positive and alert burden relative to the ranking-only winner.

| **Configuration**               | **Mean AP** | **AP std** | **Mean FP @80% R** | **Mean alert rate** | **Mean precision @80% R** |
|---------------------------------|-------------|------------|--------------------|---------------------|---------------------------|
| **Untuned V3-E1**               | 0.360759    | 0.052144   | 13,570             | 2.905%              | 4.712%                    |
| **V4-T2-R-02 (ranking winner)** | 0.409537    | 0.051849   | 17,297             | 3.247%              | 3.674%                    |
| **V4-T2-B-02 (selected)**       | 0.400486    | 0.039233   | 12,669             | 2.672%              | 4.407%                    |

- V4-T2-R-02 increased mean AP by about 13.5% versus V3-E1, but increased mean false positives by about 27.5%.

- V4-T2-B-02 increased mean AP by about 11.0% versus V3-E1, reduced mean false positives by about 6.6%, reduced mean alert rate by about 8.0%, and reduced AP standard deviation by about 24.8%.

- Relative to R-02, B-02 gives up only about 2.2% relative mean AP while producing about 26.8% fewer false positives, about 17.7% lower alert rate, about 20.0% higher precision at the 80% recall floor, and about 24.3% lower AP standard deviation.

# 2. Starting point and non-negotiable constraints

The tuning phase began only after feature engineering had been completed and frozen. This was important because tuning model hyperparameters while simultaneously changing features would make it impossible to tell whether performance changes came from the model or from the representation.

| **Constraint**                  | **Frozen decision**                                                                                                       |
|---------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| **Development period**          | September 1-7 only                                                                                                        |
| **Final test period**           | September 8+; completely untouched                                                                                        |
| **Validation design**           | Same three chronological walk-forward folds                                                                               |
| **Feature version**             | V3-E1; 17 selected features before one-hot expansion                                                                      |
| **Preprocessing**               | Numeric median imputation + StandardScaler; categorical most-frequent imputation + OneHotEncoder(handle_unknown="ignore") |
| **Imbalance handling**          | Balanced sample weights recomputed from each fold’s training labels                                                       |
| **Primary optimization metric** | Average Precision / PR-AUC                                                                                                |
| **Operational diagnostics**     | Precision, recall, false positives, alerts, alert rate at recall >=80%                                                   |
| **Final threshold**             | Deferred until after model selection and chronological OOF prediction generation                                          |

> **Why September 8+ remained locked**
>
> The final test period is meant to simulate truly unseen future data. Using it for feature decisions, tuning, model selection, or threshold selection would contaminate the final estimate of generalization.

# 3. Chronological validation design

The exact same expanding walk-forward folds used during feature evaluation were reused for tuning. The model therefore receives more historical data as time advances, while validation always occurs on later data.

| **Fold**   | **Training rows** | **Validation rows** |
|------------|-------------------|---------------------|
| **Fold 1** | 2,076,752         | 207,430             |
| **Fold 2** | 2,284,182         | 482,650             |
| **Fold 3** | 2,766,832         | 964,840             |

Fold 1: train [0 : 2,076,752) validate [2,076,752 : 2,284,182)  
Fold 2: train [0 : 2,284,182) validate [2,284,182 : 2,766,832)  
Fold 3: train [0 : 2,766,832) validate [2,766,832 : 3,731,672)

## 3.1 Why each fold starts with a fresh pipeline

A chronological split does not mean the same fitted model object should be carried from Fold 1 into Fold 2 and then Fold 3. What accumulates through time is the training data, not the fitted model state. Each fold is intended to independently simulate: “If I trained a model using only the history available up to this point, how well would it perform on the next future period?”

- Fold 2 can contain earlier Fold 1 validation dates in its training history because those observations are historical by the time Fold 2 is evaluated.

- However, the Fold 2 preprocessing statistics and LightGBM model are refit from scratch using Fold 2 training data.

- The same is true for Fold 3. This keeps each fold a clean historical simulation and prevents fitted state from one fold from leaking into another.

fresh clone -> fit Fold 1 training -> score Fold 1 validation  
fresh clone -> fit Fold 2 training -> score Fold 2 validation  
fresh clone -> fit Fold 3 training -> score Fold 3 validation

# 4. Frozen V3-E1 representation and untuned benchmark

The following 17 features were held fixed throughout tuning. Dropped features were not reintroduced.

| **Numeric features**                         | **Categorical features** |
|----------------------------------------------|--------------------------|
| hour_of_day                                  | receiving_currency       |
| day_of_week                                  | payment_currency         |
| is_weekend                                   | payment_format           |
| same_currency_flag                           |                          |
| same_bank_flag                               |                          |
| log_amount_received                          |                          |
| log_amount_paid                              |                          |
| sender_tx_count_1h                           |                          |
| log_sender_amount_paid_sum_24h_same_currency |                          |
| log_sender_hours_since_prev_tx               |                          |
| log_amount_paid_vs_sender_median             |                          |
| receiver_seen_before                         |                          |
| sender_receiver_prior_tx_count               |                          |
| sender_distinct_receivers_24h                |                          |

## 4.1 Untuned LightGBM benchmark

LGBMClassifier(  
objective="binary",  
n_estimators=300,  
learning_rate=0.05,  
num_leaves=31,  
max_depth=8,  
min_child_samples=100,  
reg_lambda=1.0,  
random_state=42,  
n_jobs=-1,  
verbosity=-1,  
)

| **Fold** | **AP**   | **Precision @ >=80% recall** | **False positives** | **Alert rate** |
|----------|----------|-------------------------------|---------------------|----------------|
| **1**    | 0.377397 | 6.5044%                       | 4,686               | 2.4162%        |
| **2**    | 0.302327 | 1.5986%                       | 23,206              | 4.8861%        |
| **3**    | 0.402554 | 6.0333%                       | 12,818              | 1.4138%        |

*Derived benchmark aggregates used during tuning: mean AP = 0.360759; AP standard deviation = 0.052144; mean FP = 13,570; mean alert rate = 2.905%; mean precision at the >=80% recall diagnostic point = 4.712%.*

# 5. What “better” meant during tuning

PR-AUC remained the primary objective because fraud is extremely rare and ranking positives well is more informative than plain accuracy. However, the project had already established that a model can have useful ranking performance and still create an impractical analyst workload. Therefore, model selection was intentionally multi-dimensional.

| **Criterion**                        | **Role in selection**   | **Reason**                                                                          |
|--------------------------------------|-------------------------|-------------------------------------------------------------------------------------|
| **Mean PR-AUC**                      | Primary                 | Overall fraud-ranking quality across chronological folds.                           |
| **Fold-to-fold stability**           | Secondary but important | Avoid selecting a model whose average is propped up by one unusually strong period. |
| **Precision at recall >=80%**       | Operational diagnostic  | Measures how clean the alert queue is when maintaining high fraud capture.          |
| **False positives at recall >=80%** | Operational diagnostic  | Direct proxy for analyst burden.                                                    |
| **Alert rate at recall >=80%**      | Operational diagnostic  | Shows what fraction of all transactions would require review at that recall level.  |
| **Training cost / complexity**       | Tie-breaker             | Avoid unnecessary complexity when gains are small.                                  |

## 5.1 Why the >=80% recall threshold was diagnostic only

For each validation fold, transactions were ranked by model score and the smallest alert set reaching at least 80% recall was identified. The resulting threshold was used only to compute precision, false positives, alerts, and alert rate for that fold. It was discarded afterward.

> **Important distinction**
>
> A fold-level threshold used to measure operational burden during tuning is not the final production threshold. The final threshold must be selected later from chronological out-of-fold predictions produced by the frozen winning configuration.

# 6. Why a constrained randomized search was used

The dataset contains millions of transactions. A brute-force Cartesian grid across nine LightGBM hyperparameters would create an excessive number of fits. Instead, Stage 1 used a constrained randomized search: 20 candidate configurations, each evaluated on the same three folds, for 60 model fits.

## 6.1 Why a manual randomized loop was preferred over plain RandomizedSearchCV

The project required fold-specific balanced sample weights and custom operational metrics at the recall floor. A small manual loop using ParameterSampler made both requirements explicit and easy to audit. The approach remained a randomized hyperparameter search; it simply used a custom evaluation harness rather than delegating all logic to a single scikit-learn search object.

train_weights = compute_sample_weight(  
class_weight="balanced",  
y=y_train_fold,  
)  
  
pipeline.fit(  
X_train_fold,  
y_train_fold,  
model\_\_sample_weight=train_weights,  
)

# 7. Stage 1 search space and reasoning

Stage 1 explored a deliberately bounded neighborhood around the existing benchmark rather than an extremely broad space.

| **Hyperparameter**    | **Stage 1 values** | **Reasoning**                                                                                                                                            |
|-----------------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **num_leaves**        | 15, 31, 63         | Controls tree complexity. 31 was the benchmark; 15 tested a simpler model and 63 tested more expressive trees without jumping to very large leaf counts. |
| **max_depth**         | 6, 8, 10, -1       | Tests whether the benchmark depth of 8 is too restrictive or too permissive. -1 removes the explicit depth cap.                                          |
| **min_child_samples** | 50, 100, 250, 500  | Controls how much data must support a leaf. Larger values can reduce fragile patterns in a very large, imbalanced dataset.                               |
| **learning_rate**     | 0.03, 0.05, 0.08   | Tests slower, benchmark, and moderately faster boosting.                                                                                                 |
| **n_estimators**      | 200, 300, 500      | Tests shorter and longer boosting sequences around the 300-tree benchmark.                                                                               |
| **reg_alpha**         | 0, 0.1, 0.5, 1.0   | Explores L1 regularization from none to moderately strong.                                                                                               |
| **reg_lambda**        | 0, 0.5, 1, 2, 5    | Explores L2 regularization around the benchmark value of 1.0.                                                                                            |
| **subsample**         | 0.8, 0.9, 1.0      | Tests row subsampling as a regularization mechanism.                                                                                                     |
| **colsample_bytree**  | 0.75, 0.90, 1.0    | Tests feature subsampling per tree versus using all encoded features.                                                                                    |

*Implementation detail: subsample_freq was fixed to 1 in the tuning pipeline so subsample values below 1.0 were actually used. This was not itself a tuned hyperparameter.*

# 8. Rebuilding the tuning notebook safely

The tuning notebook was new, so the frozen experiment had to be reconstructed from source-of-truth settings rather than relying on in-memory objects from the feature-engineering notebook.

**1.** Load only the September 1-7 V3-E1 development dataset.

**2.** Check the expected development row count: 3,731,672.

**3.** Check that the maximum timestamp is before September 8, 2022.

**4.** Freeze the exact 17 V3-E1 feature names.

**5.** Recreate the exact three chronological fold slices from the known fold sizes.

**6.** Recreate the frozen numeric and categorical preprocessing pipelines.

**7.** Recreate the untuned V3-E1 LightGBM pipeline.

**8.** Build the fold evaluator and the >=80% recall diagnostic evaluator.

**9.** Verify the harness against the known V3-E1 benchmark before tuning.

> **Why the benchmark reproduction check mattered**
>
> If the new tuning notebook could not reproduce the frozen V3-E1 benchmark, any later candidate ranking would be untrustworthy. The check was designed to catch fold-definition mistakes, preprocessing drift, weighting mistakes, or metric mismatches before spending compute on tuning.

# 9. Stage 1 execution

## 9.1 Candidate 1 smoke test

Before launching all 20 candidates, Candidate 1 was evaluated across all three folds to confirm that the candidate-setting mechanism, fold cloning, sample weighting, scoring, timing, and checkpoint structure behaved correctly.

| **Fold** | **AP**   | **Precision @80% R** | **False positives** | **Alert rate** |
|----------|----------|----------------------|---------------------|----------------|
| **1**    | 0.391576 | 6.6982%              | 4,541               | 2.3463%        |
| **2**    | 0.299530 | 1.9032%              | 19,432              | 4.1042%        |
| **3**    | 0.417984 | 5.0478%              | 15,481              | 1.6898%        |

*Candidate 1 mean AP = 0.369696; AP std = 0.062184; mean FP = 13,151; mean alert rate = 2.7135%. It improved mean AP versus V3-E1 but was less temporally stable and worsened Fold 3 false positives, demonstrating why one candidate was not enough to select a model.*

## 9.2 Full Stage 1 search

The remaining candidates were evaluated using the same function. Checkpoints were written after every completed candidate so that a kernel failure would not force a restart of earlier expensive fits.

| **Rank** | **Candidate** | **Mean AP** | **AP std** | **Fold 2 AP** | **Mean FP** | **Mean alert rate** |
|----------|---------------|-------------|------------|---------------|-------------|---------------------|
| 1        | V4-T1-03      | 0.395523    | 0.049461   | 0.342166      | 16,825      | 3.4099%             |
| 2        | V4-T1-16      | 0.390713    | 0.051529   | 0.331760      | 16,677      | 3.2817%             |
| 3        | V4-T1-14      | 0.386372    | 0.062320   | 0.316693      | 15,075      | 3.0882%             |
| 4        | V4-T1-05      | 0.385280    | 0.052351   | 0.325205      | 11,796      | 2.5263%             |
| 5        | V4-T1-17      | 0.379487    | 0.038968   | 0.338758      | 11,361      | 2.3720%             |
| 6        | V4-T1-08      | 0.378606    | 0.054548   | 0.317170      | 14,958      | 3.0932%             |
| 7        | V4-T1-01      | 0.369696    | 0.062184   | 0.299530      | 13,151      | 2.7135%             |
| 8        | V4-T1-06      | 0.367479    | 0.032287   | 0.341862      | 10,621      | 2.2503%             |
| 9        | V4-T1-11      | 0.366732    | 0.043797   | 0.317328      | 11,632      | 2.5138%             |
| 10       | V4-T1-04      | 0.361917    | 0.070817   | 0.280324      | 12,278      | 2.6140%             |

## 9.3 What Stage 1 revealed

- V4-T1-03 became the ranking anchor: highest mean AP at 0.395523, with improved Fold 2 AP but substantially higher false-positive burden.

- V4-T1-05 became a balanced anchor: mean AP 0.385280 with mean FP reduced below the V3-E1 benchmark.

- V4-T1-17 showed another useful trade-off: lower mean AP than the top ranking candidates, but stronger stability and lower alert burden.

- V4-T1-06 showed that operational metrics could improve further, but with a smaller PR-AUC gain.

# 10. Stage 1 parameter-effect review

After Stage 1, results were grouped by individual parameter value to identify promising regions for a narrower second stage. These summaries were used only for search-space guidance. They were not interpreted as causal effects because each value appeared inside different multi-parameter combinations.

> **Interpretation caution**
>
> A grouped result such as “reg_lambda=2.0 had higher mean AP” does not prove that reg_lambda=2.0 caused the gain. LightGBM hyperparameters interact. The summaries answer “where should we search more densely?”, not “which single parameter scientifically caused the improvement?”

| **Parameter**         | **Best grouped value** | **Grouped mean AP** | **Stage 1 interpretation**                                                                     |
|-----------------------|------------------------|---------------------|------------------------------------------------------------------------------------------------|
| **num_leaves**        | 63                     | 0.370174            | Clear Stage 1 signal; 4 of top 5 used 63 and 63 was the upper boundary.                        |
| **max_depth**         | -1                     | 0.359909            | Unrestricted depth led the grouped mean; depth 6 was clearly weaker.                           |
| **min_child_samples** | 250                    | 0.390947            | Strong grouped result but based on only two candidates, so retained neighboring values.        |
| **learning_rate**     | 0.08                   | 0.366022            | Highest grouped mean but only two candidates; strong interaction with number of trees.         |
| **n_estimators**      | 500                    | 0.353958            | Best grouped mean and best operational burden, but not uniformly best among top AP candidates. |
| **reg_alpha**         | 0.5                    | 0.359531            | Moderate grouped lead; stronger alpha values often helped operational burden.                  |
| **reg_lambda**        | 2.0                    | 0.376379            | Best grouped mean, but 0.5 appeared in 4 of the top-5 AP candidates.                           |
| **subsample**         | 0.8                    | 0.356716            | Best grouped mean and low FP; however top ranking candidates often used 1.0.                   |
| **colsample_bytree**  | 1.0                    | 0.360659            | Strongest grouped AP and 4 of top 5 used all features.                                         |

## 10.1 Key Stage 1 pattern: two performance regions

| **Region**               | **Typical characteristics**                                                            | **Observed behavior**                                                                                |
|--------------------------|----------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **Ranking-focused**      | 63 leaves; depth 10/-1; weak/moderate regularization; often full row and feature usage | Higher overall AP, but frequently more false positives at the 80% recall diagnostic point.           |
| **Balanced/operational** | 31-63 leaves; stronger regularization and/or partial sampling; often more trees        | Slightly lower AP, but better precision, lower FP, lower alert rate, and sometimes better stability. |

# 11. Why the first Stage 2 random candidate set was rejected

An initial Stage 2 plan attempted to randomly sample separate “ranking” and “balanced” neighborhoods. Before training, the generated candidate table was inspected. The coverage was lopsided in ways that contradicted Stage 1 evidence.

- The ranking group contained zero candidates with reg_lambda=0.5 even though 4 of the Stage 1 top 5 used 0.5.

- Most ranking candidates used subsample=0.9 even though the strongest Stage 1 AP candidates commonly used subsample=1.0.

- Because this was already a refinement stage, leaving important local regions to random chance would make Stage 2 less informative than a controlled local experiment.

> **Decision**
>
> Discard the first random Stage 2 candidate set before training it. Replace it with a controlled local search around two known anchors, changing one or two parameters at a time.

# 12. Controlled Stage 2 local search

Stage 2 used 12 new candidates (36 fits): six ranking-focused variants around V4-T1-03 and six balanced variants around V4-T1-05. This made each comparison easier to reason about and reduced unnecessary compute.

## 12.1 Ranking anchor: V4-T1-03

num_leaves=63  
max_depth=-1  
min_child_samples=250  
learning_rate=0.08  
n_estimators=200  
reg_alpha=0.1  
reg_lambda=0.5  
subsample=1.0  
colsample_bytree=1.0

*Anchor performance: mean AP = 0.395523; mean FP = 16,825; mean alert rate = 3.4099%.*

| **Candidate**  | **Local change from ranking anchor**    | **Purpose**                                      |
|----------------|-----------------------------------------|--------------------------------------------------|
| **V4-T2-R-01** | num_leaves 63 -> 95                    | Test whether Stage 1 upper boundary was too low. |
| **V4-T2-R-02** | num_leaves 63 -> 127                   | Test a larger increase in tree expressiveness.   |
| **V4-T2-R-03** | max_depth -1 -> 10                     | Test explicit depth restriction.                 |
| **V4-T2-R-04** | min_child_samples 250 -> 100           | Allow more specialized leaves.                   |
| **V4-T2-R-05** | min_child_samples 250 -> 500           | Force more conservative leaves.                  |
| **V4-T2-R-06** | 0.08x200 -> 0.05x300 boosting schedule | Test slower, more gradual boosting.              |

## 12.2 Balanced anchor: V4-T1-05

num_leaves=63  
max_depth=10  
min_child_samples=500  
learning_rate=0.03  
n_estimators=300  
reg_alpha=0.1  
reg_lambda=2.0  
subsample=0.9  
colsample_bytree=0.75

*Anchor performance: mean AP = 0.385280; mean FP = 11,796; mean alert rate = 2.5263%.*

| **Candidate**  | **Local change from balanced anchor**         | **Purpose**                                                                                           |
|----------------|-----------------------------------------------|-------------------------------------------------------------------------------------------------------|
| **V4-T2-B-01** | learning rate 0.03 -> 0.05, trees remain 300 | See whether a moderately faster boosting schedule improves AP without losing the low-burden behavior. |
| **V4-T2-B-02** | 0.03x300 -> 0.03x500                         | Add boosting capacity while keeping the conservative learning rate.                                   |
| **V4-T2-B-03** | reg_alpha 0.1 -> 0.5                         | Test stronger L1 regularization.                                                                      |
| **V4-T2-B-04** | subsample 0.9 -> 0.8                         | Test stronger row subsampling.                                                                        |
| **V4-T2-B-05** | colsample 0.75 -> 0.90                       | Give each tree more encoded features.                                                                 |
| **V4-T2-B-06** | num_leaves 63 -> 95                          | Test whether the balanced design can support more expressive trees.                                   |

# 13. Stage 2 results

| **Rank** | **Candidate** | **Mean AP** | **AP std** | **Fold 2 AP** | **Mean precision @80% R** | **Mean FP** | **Mean alert** |
|----------|---------------|-------------|------------|---------------|---------------------------|-------------|----------------|
| 1        | V4-T2-R-02    | 0.409537    | 0.051849   | 0.354462      | 3.6740%                   | 17,297      | 3.2472%        |
| 2        | V4-T2-R-04    | 0.402899    | 0.041615   | 0.359232      | 3.8180%                   | 16,449      | 3.2461%        |
| 3        | V4-T2-B-02    | 0.400486    | 0.039233   | 0.355299      | 4.4073%                   | 12,669      | 2.6718%        |
| 4        | V4-T2-R-06    | 0.398296    | 0.045090   | 0.351434      | 3.9000%                   | 15,623      | 3.1574%        |
| 5        | V4-T2-B-01    | 0.396486    | 0.041164   | 0.349944      | 4.6099%                   | 12,157      | 2.5698%        |
| 6        | V4-T2-B-06    | 0.395277    | 0.047401   | 0.341065      | 4.4940%                   | 12,377      | 2.6398%        |
| 7        | V4-T2-R-01    | 0.393750    | 0.058015   | 0.336432      | 3.8193%                   | 16,850      | 3.2251%        |
| 8        | V4-T2-R-05    | 0.392003    | 0.044902   | 0.348653      | 3.7342%                   | 16,364      | 3.1494%        |
| 9        | V4-T2-B-05    | 0.385957    | 0.048501   | 0.332349      | 4.6802%                   | 12,588      | 2.6797%        |
| 10       | V4-T2-B-04    | 0.383702    | 0.045034   | 0.331750      | 4.6106%                   | 12,100      | 2.5899%        |
| 11       | V4-T2-B-03    | 0.379947    | 0.058818   | 0.312039      | 4.4946%                   | 12,259      | 2.6043%        |
| 12       | V4-T2-R-03    | 0.375484    | 0.049592   | 0.323998      | 3.9992%                   | 15,181      | 3.0870%        |

## 13.1 Main Stage 2 lessons

- Increasing num_leaves from 63 to 127 within the ranking anchor produced the highest mean AP: V4-T2-R-02 = 0.409537.

- The 95-leaf change (R-01) did not improve the anchor, showing that tree complexity was not monotonically beneficial.

- Reducing min_child_samples from 250 to 100 (R-04) improved AP and stability relative to the Stage 1 ranking anchor, but still carried high false-positive burden.

- Increasing the balanced anchor from 300 to 500 trees at learning_rate=0.03 (B-02) produced the strongest overall compromise: mean AP 0.400486 with substantially lower FP and alert rate than the ranking-focused candidates.

- B-01 produced slightly cleaner operational metrics than B-02, but B-02 gained additional AP while remaining operationally better than the untuned benchmark.

# 14. Final model-selection reasoning

The final choice was not the configuration with the single highest mean AP. Instead, the top ranking candidate and the strongest balanced candidates were compared against the frozen benchmark and against each other.

| **Metric**                | **V3-E1** | **R-02** | **B-02** | **B-01** |
|---------------------------|-----------|----------|----------|----------|
| **Mean AP**               | 0.360759  | 0.409537 | 0.400486 | 0.396486 |
| **AP std**                | 0.052144  | 0.051849 | 0.039233 | 0.041164 |
| **Mean precision @80% R** | 4.712%    | 3.674%   | 4.407%   | 4.610%   |
| **Mean false positives**  | 13,570    | 17,297   | 12,669   | 12,157   |
| **Mean alert rate**       | 2.905%    | 3.247%   | 2.672%   | 2.570%   |

## 14.1 Why R-02 was not selected despite the highest PR-AUC

- R-02 improved mean AP by about 13.5% versus V3-E1.

- However, it increased mean false positives by about 27.5% and mean alert rate by about 11.8%.

- Its mean precision at the >=80% recall point fell by about 22.0% relative to V3-E1.

- Therefore, R-02 was excellent as a ranking model but materially worse in the operational region that matters to this AML workflow.

## 14.2 Why B-02 was selected

- Mean AP improved from 0.360759 to 0.400486: about +11.0%.

- AP standard deviation fell from about 0.052144 to 0.039233: about 24.8% lower temporal variability.

- Mean false positives fell from 13,570 to about 12,669: about 6.6% lower.

- Mean alert rate fell from about 2.905% to 2.672%: about 8.0% lower.

- Mean precision at the >=80% recall diagnostic point was 4.407%, slightly below the V3-E1 average but materially better than R-02.

- Relative to R-02, B-02 sacrificed only about 2.2% relative mean AP while producing about 26.8% fewer false positives and about 17.7% lower alert rate.

> **Selection principle**
>
> PR-AUC remained primary, but model selection did not ignore temporal stability or analyst burden. B-02 was selected because it remained near the ranking frontier while avoiding the severe operational cost of the pure PR-AUC winner.

## 14.3 Why B-01 was not selected

- B-01 had slightly better precision, lower FP, and lower alert rate than B-02.

- B-02 increased mean AP from 0.396486 to 0.400486, about +1.0% relative, while still remaining below the V3-E1 FP and alert burden.

- Therefore B-02 was chosen as the stronger middle ground between ranking quality and operational burden.

# 15. Frozen winning configuration

```python
LGBMClassifier(
objective="binary",
n_estimators=500,
learning_rate=0.03,
num_leaves=63,
max_depth=10,
min_child_samples=500,
reg_alpha=0.1,
reg_lambda=2.0,
subsample=0.9,
subsample_freq=1,
colsample_bytree=0.75,
random_state=42,
n_jobs=-1,
verbosity=-1,
)
```

| **Metadata field**         | **Frozen value**                                                                                                         |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------|
| **Model ID**               | V4-T2-B-02                                                                                                               |
| **Feature version**        | V3-E1                                                                                                                    |
| **Selection stage**        | V4-T2                                                                                                                    |
| **Threshold selected**     | False                                                                                                                    |
| **Probability calibrated** | Not performed                                                                                                            |
| **September 8+ evaluated** | False                                                                                                                    |
| **Selection rationale**    | Strong PR-AUC improvement with better temporal stability and lower FP/alert burden than the PR-AUC-maximizing candidate. |

# 16. Why tuning stopped after Stage 2

A third tuning stage was not justified. Stage 1 broadly explored the search space, and Stage 2 performed controlled local refinement around both ranking and balanced anchors. Stage 2 produced a clear ranking frontier and a clear balanced configuration. Continuing to tune would increase the risk of optimizing noise in the three development folds for progressively smaller gains.

- Stage 1: 20 candidates x 3 folds = 60 fits.

- Stage 2: 12 candidates x 3 folds = 36 fits.

- Total targeted tuning evaluations: 96 fold-level model fits, excluding benchmark verification/smoke checks.

- The selected configuration already delivered a double-digit relative mean AP improvement versus the untuned V3-E1 benchmark while improving stability and burden.

- The next source of improvement should come from threshold selection on OOF predictions and later final validation, not endless hyperparameter micro-adjustment.

# 17. What was intentionally NOT done

| **Not done**                            | **Why**                                                                                                                    |
|-----------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| **No feature changes**                  | Feature engineering was frozen at V3-E1 to preserve comparability and prevent tuning/feature-selection entanglement.       |
| **No random KFold / shuffled CV**       | Would break the chronological simulation and risk future leakage.                                                          |
| **No LightGBM class_weight="balanced"** | Balanced sample weights were already supplied per fold; using both would double-handle imbalance.                          |
| **No final threshold selection**        | Threshold must be chosen only after the winning configuration is frozen and chronological OOF predictions are generated.   |
| **No calibration**                      | Calibration comes later, after ranking/model selection is stable.                                                          |
| **No September 8+ evaluation**          | The final test period remains a one-time untouched evaluation set.                                                         |
| **No Stage 3 tuning**                   | Stage 2 produced diminishing-return territory and a defensible winner; further tuning risks overfitting development folds. |

# 18. Next phase after this handoff

The hyperparameter tuning phase is complete. The next phase should proceed in this order:

**1.** Generate chronological out-of-fold risk scores using the frozen V4-T2-B-02 configuration and the same three chronological folds.

**2.** Verify the OOF dataset: each score must come from a model that did not train on that row.

**3.** Use the combined chronological OOF predictions to select the operational threshold, targeting recall >=80% while considering precision, FP burden, alert rate, and analyst capacity.

**4.** Freeze the selected threshold.

**5.** Refit the full preprocessing + V4-T2-B-02 LightGBM pipeline on all September 1-7 development data using the established sample-weight approach.

**6.** Save the full artifact, feature metadata, hyperparameters, threshold, and version information.

**7.** Reload the artifact and evaluate exactly once on September 8+.

> **Current project state**
>
> Features: frozen. Hyperparameters: frozen. Final threshold: not yet selected. September 8+ test set: untouched.

# 19. Reproducibility checklist

| **Check**                                            | **Status** |
|------------------------------------------------------|------------|
| Development data limited to September 1-7            | Yes        |
| September 8+ excluded from notebook                  | Yes        |
| Expected development rows = 3,731,672                | Checked    |
| Feature set = V3-E1 (17 pre-OHE features)            | Frozen     |
| Preprocessing unchanged                              | Yes        |
| Same three chronological folds                       | Yes        |
| Fresh pipeline cloned per fold                       | Yes        |
| Balanced sample weights recomputed per training fold | Yes        |
| No LightGBM class_weight balancing added             | Yes        |
| PR-AUC primary metric                                | Yes        |
| Operational metrics at recall >=80% tracked         | Yes        |
| Diagnostic fold thresholds discarded after scoring   | Yes        |
| Stage 1 candidate definitions checkpointed           | Yes        |
| Stage 2 candidate definitions checkpointed           | Yes        |
| Winning model ID frozen                              | V4-T2-B-02 |
| Final threshold selected                             | No         |
| Final test evaluated                                 | No         |

# Appendix A. Stage 1 grouped parameter summaries

## num_leaves

| **Value** | **n** | **Mean AP** | **Mean AP std** | **Mean FP** | **Mean alert** |
|-----------|-------|-------------|-----------------|-------------|----------------|
| 63        | 9     | 0.370174    | 0.057195        | 14,255      | 2.946%         |
| 31        | 6     | 0.352118    | 0.045449        | 11,516      | 2.462%         |
| 15        | 5     | 0.300663    | 0.042768        | 12,199      | 2.654%         |

## max_depth

| **Value** | **n** | **Mean AP** | **Mean AP std** | **Mean FP** | **Mean alert** |
|-----------|-------|-------------|-----------------|-------------|----------------|
| -1        | 8     | 0.359909    | 0.048379        | 13,364      | 2.794%         |
| 10        | 4     | 0.351199    | 0.050611        | 13,178      | 2.789%         |
| 8         | 4     | 0.349177    | 0.052514        | 12,227      | 2.569%         |
| 6         | 4     | 0.316704    | 0.050441        | 12,462      | 2.694%         |

## min_child_samples

| **Value** | **n** | **Mean AP** | **Mean AP std** | **Mean FP** | **Mean alert** |
|-----------|-------|-------------|-----------------|-------------|----------------|
| 250       | 2     | 0.390947    | 0.055890        | 15,950      | 3.249%         |
| 500       | 3     | 0.362984    | 0.056470        | 12,799      | 2.707%         |
| 100       | 9     | 0.339711    | 0.047292        | 12,182      | 2.593%         |
| 50        | 6     | 0.336558    | 0.049078        | 13,074      | 2.767%         |

## learning_rate

| **Value** | **n** | **Mean AP** | **Mean AP std** | **Mean FP** | **Mean alert** |
|-----------|-------|-------------|-----------------|-------------|----------------|
| 0.08      | 2     | 0.366022    | 0.048958        | 14,165      | 2.927%         |
| 0.03      | 9     | 0.346268    | 0.054275        | 12,921      | 2.754%         |
| 0.05      | 9     | 0.344348    | 0.046100        | 12,640      | 2.657%         |

## n_estimators

| **Value** | **n** | **Mean AP** | **Mean AP std** | **Mean FP** | **Mean alert** |
|-----------|-------|-------------|-----------------|-------------|----------------|
| 500       | 5     | 0.353958    | 0.038124        | 11,103      | 2.347%         |
| 300       | 9     | 0.350919    | 0.055621        | 13,053      | 2.765%         |
| 200       | 6     | 0.336589    | 0.051679        | 14,231      | 2.990%         |

## reg_alpha

| **Value** | **n** | **Mean AP** | **Mean AP std** | **Mean FP** | **Mean alert** |
|-----------|-------|-------------|-----------------|-------------|----------------|
| 0.5       | 3     | 0.359531    | 0.054136        | 12,601      | 2.658%         |
| 0.1       | 8     | 0.354050    | 0.052602        | 13,613      | 2.869%         |
| 0.0       | 4     | 0.341157    | 0.044204        | 12,934      | 2.676%         |
| 1.0       | 5     | 0.334394    | 0.048250        | 11,986      | 2.586%         |

## reg_lambda

| **Value** | **n** | **Mean AP** | **Mean AP std** | **Mean FP** | **Mean alert** |
|-----------|-------|-------------|-----------------|-------------|----------------|
| 2.0       | 2     | 0.376379    | 0.042319        | 11,208      | 2.388%         |
| 1.0       | 3     | 0.356802    | 0.053618        | 12,744      | 2.703%         |
| 0.0       | 5     | 0.354581    | 0.059309        | 12,311      | 2.586%         |
| 0.5       | 9     | 0.337031    | 0.046195        | 13,729      | 2.895%         |
| 5.0       | 1     | 0.318246    | 0.043499        | 12,612      | 2.692%         |

## subsample

| **Value** | **n** | **Mean AP** | **Mean AP std** | **Mean FP** | **Mean alert** |
|-----------|-------|-------------|-----------------|-------------|----------------|
| 0.8       | 5     | 0.356716    | 0.044200        | 11,921      | 2.485%         |
| 1.0       | 7     | 0.344422    | 0.052765        | 14,827      | 3.113%         |
| 0.9       | 8     | 0.344132    | 0.051367        | 11,873      | 2.544%         |

## colsample_bytree

| **Value** | **n** | **Mean AP** | **Mean AP std** | **Mean FP** | **Mean alert** |
|-----------|-------|-------------|-----------------|-------------|----------------|
| 1.00      | 10    | 0.360659    | 0.054433        | 13,686      | 2.850%         |
| 0.75      | 6     | 0.341085    | 0.046041        | 12,279      | 2.623%         |
| 0.90      | 4     | 0.323624    | 0.045179        | 11,961      | 2.580%         |

# Appendix B. Final Stage 2 candidate configurations

| **Candidate** | **Leaves** | **Depth** | **Min child** | **LR** | **Trees** | **Alpha** | **Lambda** | **Subsample** | **Colsample** |
|---------------|------------|-----------|---------------|--------|-----------|-----------|------------|---------------|---------------|
| R-01          | 95         | -1        | 250           | 0.08   | 200       | 0.1       | 0.5        | 1.0           | 1.0           |
| R-02          | 127        | -1        | 250           | 0.08   | 200       | 0.1       | 0.5        | 1.0           | 1.0           |
| R-03          | 63         | 10        | 250           | 0.08   | 200       | 0.1       | 0.5        | 1.0           | 1.0           |
| R-04          | 63         | -1        | 100           | 0.08   | 200       | 0.1       | 0.5        | 1.0           | 1.0           |
| R-05          | 63         | -1        | 500           | 0.08   | 200       | 0.1       | 0.5        | 1.0           | 1.0           |
| R-06          | 63         | -1        | 250           | 0.05   | 300       | 0.1       | 0.5        | 1.0           | 1.0           |

| **Candidate** | **Leaves** | **Depth** | **Min child** | **LR** | **Trees** | **Alpha** | **Lambda** | **Subsample** | **Colsample** |
|---------------|------------|-----------|---------------|--------|-----------|-----------|------------|---------------|---------------|
| B-01          | 63         | 10        | 500           | 0.05   | 300       | 0.1       | 2.0        | 0.9           | 0.75          |
| B-02          | 63         | 10        | 500           | 0.03   | 500       | 0.1       | 2.0        | 0.9           | 0.75          |
| B-03          | 63         | 10        | 500           | 0.03   | 300       | 0.5       | 2.0        | 0.9           | 0.75          |
| B-04          | 63         | 10        | 500           | 0.03   | 300       | 0.1       | 2.0        | 0.8           | 0.75          |
| B-05          | 63         | 10        | 500           | 0.03   | 300       | 0.1       | 2.0        | 0.9           | 0.90          |
| B-06          | 95         | 10        | 500           | 0.03   | 300       | 0.1       | 2.0        | 0.9           | 0.75          |

# Appendix C. Sources and provenance

Primary project source of truth:

- Fraud Detection — LightGBM Hyperparameter Tuning Handoff (uploaded project handoff).

Additional empirical evidence recorded in this document:

- Stage 1 candidate outputs, grouped parameter summaries, and Stage 2 candidate outputs generated in the tuning notebook and shared during this tuning conversation.

- Derived aggregate comparisons and percentage changes computed from those reported metrics.

*No September 8+ test results are present in this document because the final test set remained untouched during tuning.*
