# Mining Silent Data Defects in LLM-Synthesized Agent Environments

COSC 50303: Data Mining, Fall 2026 · Rahul Jha, Texas Christian University

[Proposal](Project_Proposal.pdf) · [Midterm source](midterm_report_ieee.tex)

## Research question

Can tabular data-mining methods identify SQL-valid injected value defects in previously unseen AWM environments?

[Agent World Model](https://arxiv.org/abs/2602.10090) supplies 1,000 synthetic database-backed environments. This project first reproduces initialization failures, then builds a labeled benchmark by modifying individual values in statements that SQLite accepts. SQL acceptance establishes executable validity; it does not establish that an unmodified statement is semantically correct or that every injected change is invalid in its domain.

The proposal selected an in-place subset of 129,328 statements. The implemented benchmark keeps unmodified statements and separate corruption variants. Variants are dependent examples, not additional independent source observations. Every version of a source statement stays in the same environment partition.

## Data and setup

Use Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). Run commands from the repository root.

```bash
uv sync --locked
uv run --with huggingface-hub hf download Snowflake/AgentWorldModel-1K \
    gen_db.jsonl gen_sample.jsonl README.md --repo-type dataset --local-dir data
```

The dataset is [Snowflake/AgentWorldModel-1K](https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K), released under **CC-BY-4.0**. Its card is preserved in [data/README.md](data/README.md). Only `gen_db.jsonl` and `gen_sample.jsonl` are required. Downloaded data, generated databases, the benchmark parquet file, and the fitted model are excluded from Git. Source hashes and a benchmark fingerprint are recorded in `analysis/results/dataset_stats.json`; use them to check whether a later download matches the reported experiment.

The upstream database initialization code in `awm/` is adapted from [Snowflake-Labs/agent-world-model](https://github.com/Snowflake-Labs/agent-world-model). All defect injection, feature extraction, tabular benchmarks, analysis pipelines, tests, and the demo application live in `analysis/`, `tests/`, and `demo_app.py`.

## Reproduce the experiments

The audit reproduces SQLite initialization with foreign keys enabled. It records 129,328 statements, 1,374 execution failures, 217 affected environments, and one malformed sample payload. These are observed runtime failures, separate from the injected labels.

```bash
uv run python -m analysis.build_insert_audit --data-dir data --output-dir analysis/results
uv run python -m analysis.evaluate_insert_predictability
uv run python -m analysis.run_final_checks
```

The main benchmark compares rules, mean/std Z-score, Isolation Forest, Local Outlier Factor, logistic regression, Random Forest, HistGradientBoosting, and XGBoost. It retains only variants accepted by SQLite in the source database state. Money/count mutations change sign or magnitude; ratings, ages, and dates use specified values or year offsets. See `analysis/detect_silent_defects.py` for transformations and eligibility rules.

```bash
uv run python -m analysis.detect_silent_defects --no-wandb
uv run python -m analysis.profile_data_quality
uv run python -m analysis.interpret_and_analyze_errors --no-wandb --sample-per-class 200 --shap-jobs 4
uv run python -m analysis.validate_artifacts
uv run python -m analysis.generate_figures
```

The detector rebuilds the benchmark and fits the models. After a successful build, `--skip-build` verifies the cached dataset version, source hashes, and dataset fingerprint. Rebuild after changing the generator; the cache check does not establish that an edited generator would produce identical examples. `--n-envs 25` provides a development run; its outputs are not full-benchmark results. Restore the full build before generating submission figures or launching the submitted demo.

The submitted SHAP summary uses a balanced 400-example test sample (200 per label), seed 42, and four workers. Each worker computes exact TreeExplainer values; larger samples can use `--sample-per-class`, and `--shap-jobs 1` runs serially. Sampling affects attribution summaries, not model training or the benchmark metrics.

The outer split holds out 25% of environments; a second grouped split reserves 20% of the remainder for validation. Seed 42 fixes the main split. Models fit on training environments only, and learned thresholds maximize validation F1; Boolean rules keep their fixed cutoff. Test average precision (AP) uses `average_precision_score`, rather than trapezoidal precision–recall integration. Compare AP with positive prevalence, since this synthetic benchmark has many more injected variants than unmodified statements. Experiments also report severity subsets, omitted-type generalization, feature ablation, three grouped seeds, and sensitivity to 1%/5% positive prevalence.

Isolation Forest, Z-score, and LOF fit on unmodified training examples. LOF standardizes features using that subset and caps its fitting sample for runtime. Supervised models use both labels. Partition counts, configuration, thresholds, and fingerprint are saved with results; do not infer them from old tables.

## Outputs and checks

| Artifact | Purpose |
|---|---|
| `analysis/results/audit_summary.json` | Source initialization counts |
| `analysis/results/tier2_dataset.parquet` | Features and exact source/mutation provenance |
| `analysis/results/dataset_stats.json` | Build counts, hashes, and fingerprint |
| `analysis/results/tier2_results.json` | Metrics, protocol, configuration |
| `analysis/results/data_quality_report.json` | Source and feature quality/missingness |
| `analysis/results/detector.joblib` | Shared RF, feature order, threshold, provenance |
| `analysis/results/error_cases.json` | Original/mutated SQL and predictions |
| `analysis/results/shap_summary.json` | SHAP for the shared RF |
| `analysis/results/verification.json` | SQL replay, feature/prediction and metric reconciliation |
| `analysis/results/manual_review_pairs.csv` | Stratified pairs for human semantic review; judgments initially blank |
| `figures/` | Plots generated from recorded results |

```bash
uv run python -m pytest tests/
```

The final check passed 47 tests in a clean environment. Tests cover parsing, corruption eligibility, features, SQL validation, grouped partitions, and thresholds. The artifact audit passes five reconciliation checks, including replay/feature checks for 25 stratified pairs. These pairs retain blank semantic judgments for human review. Passing tests does not replace executing the reproduction commands from a clean environment. Earlier evaluation outputs are superseded by refreshed result files.

## Experiment tracking

W&B is optional. To record a local run without cloud access:

```bash
WANDB_MODE=offline uv run python -m analysis.detect_silent_defects --skip-build
WANDB_MODE=offline uv run python -m analysis.interpret_and_analyze_errors --sample-per-class 200 --shap-jobs 4
```

These log configurations, metrics, tables, and plots beneath `wandb/`. An offline run is local evidence, not proof of a public workspace. To upload after authenticating:

```bash
uv run wandb login
uv run python -m analysis.detect_silent_defects --skip-build \
    --wandb-project awm-silent-defects --wandb-entity rahul-jha-tcu
uv run python -m analysis.interpret_and_analyze_errors --sample-per-class 200 --shap-jobs 4 \
    --wandb-project awm-silent-defects --wandb-entity rahul-jha-tcu
```

Existing offline runs can instead be uploaded with `uv run wandb sync <offline-run-directory>`. The refreshed [main run](https://wandb.ai/rahul-jha-tcu/awm-silent-defects/runs/yj2l19av) and [interpretation run](https://wandb.ai/rahul-jha-tcu/awm-silent-defects/runs/vnawnwr7) have finished and synchronized. Check the [workspace](https://wandb.ai/rahul-jha-tcu/awm-silent-defects) for reviewer access before submission. Credentials belong in the local login store or environment, never source files.

## Demonstration

After generating the benchmark, saved model, and error cases:

```bash
uv run streamlit run demo_app.py
```

Choose a source example, edit an INSERT, and inspect the synthetic-corruption score. The app separately checks SQLite acceptance using the actual schema and preceding source statements. It uses the same saved RF and validation threshold as the report's RF analysis, without fitting another model at launch. This supports manual inspection, not a guarantee of semantic correctness.

## Report and reproduction

`midterm_report_ieee.tex` is the submission source. The copy in `analysis/results/` is retained for compatibility; compile from the root so paths resolve. In the course-required official IEEE Conference Template on Overleaf, upload this source, `analysis/results/midterm_numbers.tex`, and the referenced figure while preserving their paths, and select the root source as the main document. With a local LaTeX installation containing IEEEtran:

```bash
pdflatex -interaction=nonstopmode -halt-on-error midterm_report_ieee.tex
pdflatex -interaction=nonstopmode -halt-on-error midterm_report_ieee.tex
```

Verify at most three main-text pages, excluding references, readable tables, and unchanged IEEE margins/font sizes. The course requires the compiled PDF, a working code link or requested snapshot, and explanation of proposal changes. The [repository](https://github.com/jharahul968/awm-silent-defects) must contain the reviewed version before its link serves as submission evidence.

The proposal's positive-value errors, temporal ordering violations, and semantically incorrect references remain future work. Current labels mainly evaluate controlled-mutation recognition; natural defect detection requires manual domain review. Keyword matching can overreach: `count` matches `is_bank_account`, so a Boolean flag is categorized as an inventory-family candidate. Family names describe injection recipes, not verified domain types.
