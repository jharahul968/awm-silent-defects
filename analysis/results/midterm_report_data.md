# Midterm evidence record

This is the supporting record for `midterm_report_ieee.tex`. Numbers come from the refreshed result JSON files, not the earlier pilot. The compiled IEEE report is the submission artifact.

## Scope and proposal changes

The task evaluates SQL-valid controlled value mutations in held-out AWM environments. Unmodified executable statements are reference negatives, not manually verified semantic ground truth. The proposal's replacement-only dataset became a source-plus-variants benchmark; the usable unmodified subset still exceeds 100,000 instances independently of the variants. Eight methods replace the original three, with explicit validation thresholds, three grouped seeds, and prevalence sensitivity in addition to severity/type/feature comparisons.

Positive-value shifts, temporal-order violations, and semantically incorrect references remain future work. High scores on current mutations do not establish natural defect detection.

## Source and working data

- Source: Snowflake/AgentWorldModel-1K, CC-BY-4.0; 1,000 environments and 129,328 original INSERT statements.
- Initialization audit: 1,374 failures, 217 affected environments, 1,054 foreign-key failures, and one malformed sample payload.
- Benchmark: 619,619 instances in 999 environments; 127,954 unmodified statements and 491,665 variants.
- Candidates rejected by SQLite: 38,680; unchanged candidates skipped: 4.
- Source exclusions: 1,231 failed execution, 112 unsupported INSERTs, and 31 unparsed statements.
- Source NULLs: 45,982/1,046,921 parsed literals (4.39%); 19,634 omitted slots are separate and may have defaults.
- Dynamic date expressions: 60,789 statements; literal-year features do not expand these functions.
- CHECK coverage: 3,931/18,465 declared tables (21.3%).

| Mutation type | Accepted variants |
|---|---:|
| `negative_money` | 81,897 |
| `negative_inventory` | 35,346 |
| `impossible_age` | 555 |
| `out_of_bounds_rating` | 3,887 |
| `anomalous_date` | 369,980 |

## Protocol

AP means scikit-learn average precision, not trapezoidal PR-AUC. The legacy `pr_auc` JSON field aliases AP. For learned scores, the highest validation threshold maximizing F1 is selected; Boolean rules retain 0.5. Constant-score reference predicts all examples positive. Parameters, class weights, and feature scaling use training data. LOF uses training-only scaling and a deterministic 25,000-example cap.

| Split | Environments | Rows | Positives | Positive prevalence |
|---|---:|---:|---:|---:|
| train | 599 | 376,201 | 298,898 | 0.794517 |
| validation | 150 | 91,948 | 72,630 | 0.789903 |
| test | 250 | 151,470 | 120,137 | 0.793141 |

Main split seed: 42. Environment overlap: 0; source-ID overlap: 0. Grouping does not guarantee unique schemas across worlds.

## Main benchmark

| Method | AP | Precision | Recall | F1 | P@100 | R@100 | Decision threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rules | 0.888668 | 0.979836 | 0.511674 | 0.672281 | 1.0000 | 0.000832 | 0.5 |
| Mean/std Z-score | 0.934150 | 0.793141 | 1.000000 | 0.884638 | 0.7200 | 0.000599 | 0.50698795 |
| Isolation Forest | 0.896365 | 0.799506 | 0.994090 | 0.886243 | 1.0000 | 0.000832 | -0.14089043 |
| Local Outlier Factor | 0.950928 | 0.840195 | 0.949716 | 0.891605 | 0.8300 | 0.000691 | -0.22415159 |
| Logistic regression | 0.997662 | 0.990475 | 0.985025 | 0.987743 | 0.9900 | 0.000824 | 0.27978041 |
| Random Forest | 0.999683 | 0.994253 | 0.995056 | 0.994654 | 1.0000 | 0.000832 | 0.28705483 |
| Histogram boosting | 0.999213 | 0.994574 | 0.993299 | 0.993936 | 1.0000 | 0.000832 | 0.13810942 |
| XGBoost | 0.999750 | 0.995498 | 0.995697 | 0.995597 | 1.0000 | 0.000832 | 0.11511908 |
| Constant reference | 0.793141 | 0.793141 | 1.000000 | 0.884638 | 0.8600 | 0.000716 | 0 |

The Z-score threshold predicts every test example positive, so its F1 equals the always-positive reference even though AP differs. Interpret alert quality alongside ranking. Strong results mostly reflect the injected transformations.

## Severity subsets

Each subset reuses the same main test negatives, plus one severity of positive variants. Counts below are total subset counts and positive counts.

| Severity | Total | Positive | Prevalence/random AP | HGB AP | Isolation Forest AP |
|---|---:|---:|---:|---:|---:|
| obvious | 71,376 | 40,043 | 0.561015 | 0.998253 | 0.777051 |
| medium | 71,384 | 40,051 | 0.561064 | 0.998155 | 0.761220 |
| subtle | 71,376 | 40,043 | 0.561015 | 0.997736 | 0.720658 |

## Omitted-type transfer

Training and validation exclude one corruption type. Test includes that type and the same unmodified negatives. Different prevalence makes raw AP across types incomparable without the baseline.

| Omitted type | Total | Positive | Prevalence/random AP | HGB AP | HGB F1 |
|---|---:|---:|---:|---:|---:|
| `negative_money` | 50,125 | 18,792 | 0.374903 | 0.886067 | 0.521381 |
| `negative_inventory` | 39,589 | 8,256 | 0.208543 | 0.881785 | 0.594001 |
| `impossible_age` | 31,435 | 102 | 0.003245 | 0.005348 | 0.000000 |
| `out_of_bounds_rating` | 32,586 | 1,253 | 0.038452 | 0.094447 | 0.137359 |
| `anomalous_date` | 123,067 | 91,734 | 0.745399 | 0.741004 | 0.022015 |

Date AP falls below its prevalence baseline, despite appearing numerically large. Age/rating transfer is weak with limited support. Numeric types transfer more strongly under this benchmark.

## Cumulative feature ablation

| Features | Dimensions | HGB AP | HGB F1 |
|---|---:|---:|---:|
| value only | 11 | 0.999211 | 0.992377 |
| value + column | 19 | 0.999716 | 0.995505 |
| value + column + structural | 26 | 0.999213 | 0.993936 |

The one-split changes describe observed sensitivity; they do not prove necessity or causality.

## Repeated grouped splits

Seeds: 42, 43, 44. Standard deviation describes variability across these three runs, not a confidence interval.

| Method | AP mean | AP SD | F1 mean | F1 SD |
|---|---:|---:|---:|---:|
| Logistic regression | 0.997277 | 0.000351 | 0.987935 | 0.000208 |
| Histogram boosting | 0.999509 | 0.000289 | 0.995125 | 0.001170 |
| Isolation Forest | 0.898375 | 0.004971 | 0.889831 | 0.003111 |

## Simulated prevalence

Models and validation thresholds remain fixed. All unmodified test negatives are retained, with a reproducibly sampled positive subset. This does not estimate natural semantic-defect prevalence.

| Target prevalence | Method | Total | Positive | AP | Precision | Recall | F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1% | Rules | 31,649 | 316 | 0.065019 | 0.115385 | 0.522152 | 0.189003 |
| 1% | Mean/std Z-score | 31,649 | 316 | 0.232109 | 0.009985 | 1.000000 | 0.019772 |
| 1% | Isolation Forest | 31,649 | 316 | 0.047441 | 0.010343 | 0.990506 | 0.020472 |
| 1% | Local Outlier Factor | 31,649 | 316 | 0.195817 | 0.013501 | 0.939873 | 0.026620 |
| 1% | Logistic regression | 31,649 | 316 | 0.690285 | 0.215713 | 0.990506 | 0.354273 |
| 1% | Random Forest | 31,649 | 316 | 0.920047 | 0.313121 | 0.996835 | 0.476551 |
| 1% | Histogram boosting | 31,649 | 316 | 0.943685 | 0.326087 | 0.996835 | 0.491420 |
| 1% | XGBoost | 31,649 | 316 | 0.924619 | 0.368728 | 1.000000 | 0.538789 |
| 1% | Constant reference | 31,649 | 316 | 0.009985 | 0.009985 | 1.000000 | 0.019772 |
| 5% | Rules | 32,982 | 1,649 | 0.235349 | 0.404986 | 0.522135 | 0.456159 |
| 5% | Mean/std Z-score | 32,982 | 1,649 | 0.417906 | 0.049997 | 1.000000 | 0.095233 |
| 5% | Isolation Forest | 32,982 | 1,649 | 0.157358 | 0.051977 | 0.995755 | 0.098797 |
| 5% | Local Outlier Factor | 32,982 | 1,649 | 0.420800 | 0.067306 | 0.949666 | 0.125702 |
| 5% | Logistic regression | 32,982 | 1,649 | 0.895660 | 0.587980 | 0.984839 | 0.736341 |
| 5% | Random Forest | 32,982 | 1,649 | 0.978075 | 0.702796 | 0.990904 | 0.822345 |
| 5% | Histogram boosting | 32,982 | 1,649 | 0.985675 | 0.715721 | 0.993936 | 0.832191 |
| 5% | XGBoost | 32,982 | 1,649 | 0.982393 | 0.751949 | 0.994542 | 0.856397 |
| 5% | Constant reference | 32,982 | 1,649 | 0.049997 | 0.049997 | 1.000000 | 0.095233 |

## Interpretation and reproducibility

SHAP, error examples, and the demonstration consume the same saved 200-tree RF and validation threshold. Exact TreeExplainer uses 400 balanced test examples (200 per label), seed 42, and 4 workers. Leading mean absolute SHAP values: `min_year`=0.347354, `log_min_numeric`=0.048587, `has_negative_value`=0.047600, `has_absurd_year`=0.044596, `num_negative_values`=0.041867, `max_string_length`=0.021133. RF threshold is 0.28705483. False positives: 691 (2.2053%); false negatives: 594 (0.4944%). Exact source/mutated SQL and model metadata are preserved in error_cases.json.

The examples include a points-redemption debit of -1500 (`hr_employee_management_1`, score 1.0) and a rating change 4→10 whose row identifier 15 masks its maximum numeric value (`ats_applicant_tracking_system_3`, score 0.005). The error catalog and verification audit retain full statements rather than abbreviated fabricated SQL.

- Repository: https://github.com/jharahul968/awm-silent-defects
- Synchronized main run: https://wandb.ai/rahul-jha-tcu/awm-silent-defects/runs/yj2l19av
- Synchronized interpretation run: https://wandb.ai/rahul-jha-tcu/awm-silent-defects/runs/vnawnwr7
- Dataset fingerprint: `7c32b0a358f3257b1cbb3c00e1aea2bdf1a5ac47ea4fff02cdcabedcf9a8fd98`
- Source hashes: `gen_db.jsonl`: `ae8acb3c23765ca4866b35799ffb980fbb15831240fdc35c046e8a7d27a2c0e8`, `gen_sample.jsonl`: `39c40969ad76d52a3ea51384752a639cf55dce15bc1a8f0f022cfb6bbd25db3c`
- Reproduction and tests: README.md; exact model settings: tier2_results.json; pending final bonus evidence: docs/bonus_evidence.md.

## Remaining work

Manually review source/mutation pairs across domains; test plausible in-range and temporal/reference errors; use column-aligned features; reserve independent final evaluation after feature revisions. Verify accessible code/tracking evidence and required AI acknowledgment before submission.

## Artifact checks and annotation preparation

The final check passed 47 tests in a clean environment. `analysis/results/verification.json` records five passed audit checks, including saved RF metric/confusion reconciliation, hashes, and SQLite/feature replay for 25 stratified pairs. `manual_review_pairs.csv` leaves human judgments blank. These are reproducibility checks, not completed semantic validation. Real headless demo checks passed for an accepted injection, an unmodified example, a genuine schema CHECK rejection, and valid age/rating values with large identifiers.

A concrete lexical targeting limitation is `count` matching `is_bank_account`: this Boolean field can become an inventory-family candidate. Mutation-family names therefore identify keyword recipes rather than independently verified domain categories. Human reviewers should assess the actual changed column and original/mutated values; they should not accept a family label as evidence of defect validity.
