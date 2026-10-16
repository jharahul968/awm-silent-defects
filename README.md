# Mining Silent Data Defects in LLM-Synthesized Agent Environments

COSC 50303: Data Mining, Fall 2026 · Rahul Jha, Texas Christian University

[Proposal](Project_Proposal.pdf) · [Midterm Report](midterm_report_ieee.pdf)

## Overview

This project investigates whether tabular data mining methods can detect SQL-valid injected value defects in synthetic databases synthesized for LLM agents. 

The dataset is derived from [AgentWorldModel-1K](https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K) (CC-BY-4.0), which contains 1,000 relational database environments. While SQLite automatically catches structural and constraint errors during database execution, silent semantic defects—such as negative balances, extreme ages, or corrupted dates—remain executable without database rejection. The objective is to evaluate pre-execution static statement features for identifying such anomalies across previously unseen database environments.

Upstream environment initialization routines in `awm/` are adapted from [Snowflake-Labs/agent-world-model](https://github.com/Snowflake-Labs/agent-world-model). All defect injection mechanisms, feature extractors, tabular benchmarks, SHAP explanations, test suites, and demo interfaces reside in `analysis/`, `tests/`, and `demo_app.py`.

## Setup and Data

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are used for environment and dependency management.

Dependencies are locked and installed via:
```bash
uv sync --locked
```

The underlying environment schemas and sample data are retrieved from Hugging Face into `data/`:
```bash
uv run --with huggingface-hub hf download Snowflake/AgentWorldModel-1K \
    gen_db.jsonl gen_sample.jsonl README.md --repo-type dataset --local-dir data
```

## Reproducing Experiments

The experimental workflow consists of an initialization audit, benchmark generation and model evaluation, data quality profiling, model interpretation, and figure generation.

### 1. Initialization Audit
The execution behavior of 129,328 original INSERT statements across 1,000 environments is audited under SQLite with foreign key constraints enabled:
```bash
uv run python -m analysis.build_insert_audit --data-dir data --output-dir analysis/results
uv run python -m analysis.evaluate_insert_predictability
uv run python -m analysis.run_final_checks
```

### 2. Benchmark Evaluation
Eight detection methods are evaluated across disjoint environment splits (Rules, Mean/Std Z-Score, Isolation Forest, Local Outlier Factor, Logistic Regression, Random Forest, HistGradientBoosting, and XGBoost):
```bash
uv run python -m analysis.detect_silent_defects --no-wandb
uv run python -m analysis.profile_data_quality
uv run python -m analysis.interpret_and_analyze_errors --no-wandb --sample-per-class 200 --shap-jobs 4
uv run python -m analysis.generate_figures
```
Validation-tuned decision thresholds are applied to anomaly estimators. Fast verification or lightweight test runs can be performed using `--skip-build` or `--n-envs 25`.

## Artifacts and Testing

Key experimental results and trained models are stored in `analysis/results/`:

| Artifact | Description |
|---|---|
| `analysis/results/tier2_dataset.parquet` | Benchmark dataset containing 26 extracted static features |
| `analysis/results/dataset_stats.json` | Dataset build metadata, SHA-256 hashes, and fingerprint |
| `analysis/results/tier2_results.json` | Evaluation metrics across held-out environment partitions |
| `analysis/results/detector.joblib` | Saved Random Forest model bundle and threshold configuration |
| `analysis/results/error_cases.json` | Catalog of false-positive and false-negative mutations |
| `analysis/results/shap_summary.json` | TreeExplainer feature attribution values |
| `analysis/results/midterm_numbers.tex` | Generated LaTeX macros for empirical figures in the report |
| `figures/` | Generated plots for model comparisons, severity decay, and SHAP attributions |

Automated unit tests covering SQL parsing, mutation generation, feature extraction, and split validation are executed via:
```bash
uv run python -m pytest tests/
```

## Experiment Tracking

Experiment tracking is integrated via Weights & Biases:
- **Main benchmark run:** [yj2l19av](https://wandb.ai/rahul-jha-tcu/awm-silent-defects/runs/yj2l19av)
- **Model interpretation run:** [vnawnwr7](https://wandb.ai/rahul-jha-tcu/awm-silent-defects/runs/vnawnwr7)
- **Project workspace:** [awm-silent-defects](https://wandb.ai/rahul-jha-tcu/awm-silent-defects)

Offline tracking is supported without external credentials:
```bash
WANDB_MODE=offline uv run python -m analysis.detect_silent_defects --skip-build
```

Synchronization to a remote workspace is initiated with:
```bash
uv run python -m analysis.detect_silent_defects --skip-build \
    --wandb-project awm-silent-defects --wandb-entity rahul-jha-tcu
```

## Interactive Demonstration

An interactive demonstration is provided via Streamlit for inspecting SQL statements and comparing detector scores against live SQLite execution:
```bash
uv run streamlit run demo_app.py
```
Pre-computed mutation presets are available, and custom INSERT statements can be evaluated against the saved Random Forest model.

## Report

The project progress report is available as [`midterm_report_ieee.pdf`](midterm_report_ieee.pdf).

## Limitations and Next Steps

- **Synthetic vs. Natural Defects:** Benchmark labels reflect controlled synthetic mutations; evaluation against naturally occurring semantic defects requires ongoing manual domain review.
- **Lexical Column Filtering:** Column candidate eligibility relies on keyword matching, which can occasionally select tangential attributes.
- **Planned Extensions:** Future experiments will evaluate multi-table relational consistency, temporal sequence ordering, and plausible positive-value anomalies.
