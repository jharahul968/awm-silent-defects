# AWM-1K Seed-SQL Mining Feasibility Pilot

## Reproduction

Command:

```bash
uv run awm env reset_db \
  --input_db data/gen_db.jsonl \
  --input_sample data/gen_sample.jsonl \
  2>&1 | tee analysis/results/official_reset.log
```

The official AWM reset completed successfully for all 1,000 environments. It
uses SQLite with `PRAGMA foreign_keys = ON` and continues after individual
statement errors.

| Measure | Result |
|---|---:|
| Environments | 1,000 |
| INSERT statements | 129,328 |
| Successful statements | 127,954 |
| Failed statements | 1,374 |
| Environments with errors | 217 |
| Environments without errors | 783 |
| Malformed sample payloads | 1 (`dating_1`) |

The result reproduces the approximate 22% figure reported in GitHub Issue #10.
The malformed `dating_1` record has no `sample_data.tables` field and is not
counted as an SQL statement failure by the official reset.

## Statement-Level Dataset

Generated files:

- `insert_audit.jsonl`: one row per INSERT, including the SQL text, success/failure label, and pre-execution features.
- `environment_audit.json`: one summary row per environment.
- `audit_summary.json`: aggregate counts and SQLite error classes.

The audit has 129,328 rows and 1,374 positive failure labels. Error messages are
retained for post-hoc analysis only and are not used as predictors.

## Predictability Experiment

The experiment uses 999 environments with valid sample payloads. Statements are
split by environment, preventing statements from one environment leaking across
sets:

```text
train:      699 environments
validation: 150 environments
test:       150 environments
```

Features are available before execution: SQL length, column/value counts,
unknown-column count, literal counts, quote characteristics, statement
position, and static table constraint metadata.

| Model | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Majority baseline | 0.0184 | 0.000 | 0.000 | 0.000 |
| Static rules | 0.0251 | 0.126 | 0.063 | 0.084 |
| Logistic regression | 0.0615 | 0.033 | 0.557 | 0.062 |
| Random forest | 0.0527 | 0.246 | 0.046 | 0.077 |

These are feasibility results, not final tuned model results. The preliminary
signal is promising: both learned models exceed the majority baseline in
PR-AUC, and logistic regression captures over half of the failures at the
default threshold. Precision is low because the failure class is rare.

## Failure Concentration

- `IntegrityError`: 1,165 failures.
- Other SQLite errors: 209 failures.
- The top 20 environments account for 48.9% of failed statements.

This supports analyzing both statement-level prediction and environment-level
heterogeneity. The existence of defects is not the novel claim; the proposed
contribution is pre-execution detection and characterization.

## Final Baseline and Stability Checks

A deterministic static checker was run without using execution labels. It used
the declared schema, SQL parsing, earlier literal `INSERT` statements, and
primary/unique/foreign-key metadata.

| Static checker measure | Result |
|---|---:|
| Statements checked | 129,328 |
| Failures caught | 1,316 / 1,374 |
| Recall | 95.8% |
| Statements flagged | 56,176 |
| Precision | 2.34% |
| Missed failures | 58 |

This checker catches nearly all failures but produces many false positives. It
is therefore a strong screening baseline, not a complete replacement for
prediction. The ML question remains meaningful as a prioritization problem:
can a model rank likely failures better than a noisy deterministic screen?

Five repeated grouped 80/20 splits produced these test-set results (mean +/-
standard deviation):

| Model | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Majority baseline | 0.0110 +/- 0.0021 | 0.0000 | 0.0000 | 0.0000 |
| Logistic regression | 0.0382 +/- 0.0090 | 0.0217 | 0.5559 | 0.0416 |
| Random forest | 0.0383 +/- 0.0037 | 0.1038 | 0.0406 | 0.0571 |

The direction is stable across splits, but performance varies materially across
environment samples. The report should therefore present mean, standard
deviation, and per-split results rather than one favorable split.

## Error Categories

The 1,374 failures divide into these post-hoc categories:

| Category | Count |
|---|---:|
| Foreign-key violation or mismatch | 1,054 |
| Uniqueness or primary-key violation | 68 |
| Column/value or schema mismatch | 84 |
| Unknown column | 48 |
| Syntax or quoting | 77 |
| Other SQLite error | 43 |

These categories are used for interpretation only, not as model inputs.

## Reproducibility Commands

```bash
uv run python analysis/build_insert_audit.py \
  --data-dir data \
  --output-dir analysis/results

uv run python analysis/evaluate_insert_predictability.py

uv run python analysis/run_final_checks.py
```

Results are written to `analysis/results/predictability_results.json`.
The final checker, category, concentration, and repeated-split results are in
`analysis/results/final_checks.json`.

## Questions to Send for Feedback

1. Is it acceptable to frame the ML task as prioritizing failures beyond a
   high-recall, low-precision deterministic checker?
2. Is the environment-grouped split and statement-level label definition
   appropriate for the proposed data-mining study?
3. Should malformed payloads such as `dating_1` be a separate data-integrity
   class, or excluded from the classifier and reported separately?
4. Does the repeated PR-AUC improvement over the majority baseline justify
   larger feature-engineering and model-selection experiments?
5. Should the main contribution emphasize statement-level detection, or also
   include an environment-level risk score aggregated from statements?