# Catch_Fraud - Fraud Intelligence Platform
![PyPI - Version](https://img.shields.io/badge/Python-3.12.10-blue.svg) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
![Git Commits](https://img.shields.io/github/commit-activity/t/surabhi1914/Catch_Fraud?logo=git&logoColor=black&label=Commits)


An AML / fraud detection project built step by step using the dataset `HI-Small_Trans.csv`(IBM HuggingFace).

## Project Goal

This project aims to build a transaction-level fraud / AML detection pipeline that can eventually answer:

- How risky is this transaction?
- Why was it flagged?
- What should an analyst check next?

> Note: The workflow is intentionally incremental so that each step is understandable, reusable, and easy to validate before moving on.

---

## Dataset

**Source dataset:** `HI-Small_Trans.csv`  
**Dataset type:** Synthetic AML transaction data

The raw dataset includes transaction-level fields such as:

- `Timestamp`
- `From Bank`
- `Account`
- `To Bank`
- `Account.1`
- `Amount Received`
- `Receiving Currency`
- `Amount Paid`
- `Payment Currency`
- `Payment Format`
- `Is Laundering`

In the pipeline, these columns are standardized into a cleaner canonical schema for downstream processing.

> Note: raw data files and generated parquet outputs are not tracked in Git.
---

## Current Progress

### Completed
- Project setup
- Validation notebook
- Raw schema inspection
- Canonical column renaming
- Required-column checks
- Missing-value inspection
- Type validation for timestamps, numeric fields, and target column
- Stable `transaction_id` generation
- Validated parquet output creation
- Chronological train / validation / test split
- Split-level class distribution checks
---

## Next step
The next stage is to build the first baseline model using a minimal feature set, such as:
- amount fields
- currencies
- payment format
- timestamp-derived features
- same-bank vs cross-bank behavior

The first model will be intentionally simple and interpretable before moving on to stronger models.

---

## Structure
- notebooks/ → step-by-step learning notebooks
- src/ → reusable Python code as the project matures
- data/raw/ → raw input dataset (not tracked in git)
- data/interim/ → validated intermediate outputs (not tracked in git)
- data/processed/ → train/val/test split files (not tracked in git)

### Current Notebooks

- `notebooks/01_validate_data.ipynb`  
  Loads the raw dataset, validates schema and types, inspects missingness, creates a stable transaction ID, and saves a validated parquet file.

- `notebooks/02_split_data.ipynb`  
  Loads the validated dataset, sorts transactions chronologically, creates time-aware train / validation / test splits, and saves processed parquet files.

---
## Key observations so far
- Raw dataset shape: **5,078,345 rows × 11 columns**
- Target label is highly imbalanced:
  - `0`: 5,073,168
  - `1`: 5,177
- The time-based split revealed a higher positive rate in the later test window, suggesting temporal distribution shift.

**Why the Split Matters? ** Instead of using a random split, this project uses a chronological split so that earlier transactions are used for training and later transactions are used for validation and testing. This is important because fraud / AML detection is time-sensitive, and random splitting can hide temporal drift or create unrealistic evaluation conditions.

---

## Project Structure

```text
Catch_Fraud/
├── data/
│   ├── raw/
│   │   └── dev/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_validate_data.ipynb
│   └── 02_split_data.ipynb
├── src/
│   └── data/
├── .gitignore
└── README.md
```
---

## Outputs Created So Far
- **Validation stage**
  - data/interim/transactions_validated.parquet
- **Split stage** 
  - data/processed/train.parquet
  - data/processed/val.parquet
  - data/processed/test.parquet


