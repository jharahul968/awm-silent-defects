# Midterm Progress Report: Structured Content & Tables
**Course:** COSC 50303 Data Mining (Fall 2026)  
**Project:** Mining Silent Data Defects in LLM-Synthesized Agent Environments  
**Author:** Rahul Jha (Department of Computer Science, Texas Christian University)  
**Date:** October 5, 2026  
**Repository:** https://github.com/jharahul968/awm-silent-defects  
**W&B Workspace:** https://wandb.ai/rahul-jha-tcu/awm-silent-defects  

---

## 1. Updated Problem Statement & Proposed Architecture

* **Context:** Agent World Model (AWM) synthesizes 1,000 executable, SQL-backed environments for agentic reinforcement learning.
* **Core Discovery:** Synthesized environments exhibit a **Two-Tier Defect hierarchy**:
  * **Tier 1 (Explicit Crashes):** 1,374 statements throw runtime SQLite exceptions (76.7% foreign key violations, affecting 21.7% of all environments).
  * **Tier 2 (Silent Semantic Corruptions):** Because LLMs omit SQL `CHECK` constraints, SQLite commits logically corrupt data with return code 0 (>92% of injected anomalies accepted).
* **Mining Question:** *Can tabular data-mining algorithms reliably detect and prioritize SQL-valid, semantically corrupted database records across unseen synthetic environments?*
* **Change from Proposal:** In direct response to faculty feedback requiring graduate-level technical rigor, the modeling scope was expanded from 3 initial methods to an **8-model benchmark** across heuristic, statistical, unsupervised, and supervised paradigms, paired with **three controlled experiments** (severity breakdown, leave-one-type-out holdout, feature ablation) and **SHAP interpretability**.

---

## 2. Dataset Statistics & Leakage Prevention

* **Raw Population:** 129,328 original INSERT statements across 1,000 AWM environments.
* **Working Benchmark ($N = 613,017$ instances):**
  * Clean Baseline Instances: 126,490
  * Corrupted (SQLite-Accepted) Instances: 486,527 (79.37% positive rate)
  * Corruptions Rejected by SQLite Constraints: 38,212 (validating that basic DDL catches fewer than 8% of semantic anomalies)
* **Leakage Prevention Protocol:**
  * Strict **Environment-Grouped Holdout Split** (`GroupShuffleSplit`, 75% train / 25% test, Seed 42).
  * Training set: 749 environments (462,633 instances).
  * Test set: 250 environments (150,384 instances).
  * *Zero statements or schemas from any test world ever appear in training.*

---

## 3. Quantitative Validation Results

### Table I: Main Detection Benchmark (Test Set: $N = 150,384$)
*Split: Environment-grouped 75/25 holdout. All metrics computed on held-out worlds.*

| Model Paradigm | Algorithm | PR-AUC | P@100 | R@100 | F1 Score |
|:---|:---|:---:|:---:|:---:|:---:|
| **Heuristic Baseline** | Keyword Rules | 0.8889 | 1.0000 | 0.0008 | 0.6714 |
| **Statistical Baseline** | Robust Z-Score | 0.9355 | 0.7500 | 0.0006 | 0.8850 |
| **Unsupervised Anomaly** | Isolation Forest (iForest) | 0.8942 | 1.0000 | 0.0008 | 0.0000* |
| **Unsupervised Anomaly** | Local Outlier Factor (LOF) | 0.9703 | 1.0000 | 0.0008 | 0.6966 |
| **Supervised Linear** | Balanced Logistic Regression | 0.9978 | 1.0000 | 0.0008 | 0.9853 |
| **Supervised Ensemble** | Random Forest | 0.9997 | 1.0000 | 0.0008 | 0.9940 |
| **Supervised Boosting** | HistGradientBoosting (HGB) | 0.9996 | 1.0000 | 0.0008 | 0.9940 |
| **Supervised Boosting** | XGBoost | **0.9998** | 1.0000 | 0.0008 | **0.9954** |

*\*Note: Unsupervised Isolation Forest contamination threshold produces low binary decision F1 at default 0.5 cutoff, but achieves strong ranking discrimination (PR-AUC = 0.8942).*

---

### Table II: Controlled Experiment 1 — Corruption Severity Breakdown
*Sensitivity of Supervised (HGB) vs Unsupervised (iForest) across corruption magnitudes.*

| Severity Level | Injected Mutation Characteristics | Test Corrupted ($N$) | HGB PR-AUC | iForest PR-AUC |
|:---|:---|:---:|:---:|:---:|
| **Obvious** | $10\times$ negative scale, year 1823, rating 99.0 | 39,787 | 0.9988 | **0.7687** |
| **Medium** | Exact negative scale, year 1950, rating 10.0 | 39,795 | 0.9989 | **0.7504** |
| **Subtle** | $0.01\times$ negative drift, $-30$ yr offset, rating 5.5 | 39,787 | 0.9987 | **0.7257** |

*Finding:* Supervised boosters maintain steady discrimination across severities, whereas unsupervised density models experience progressive sensitivity loss ($0.7687 \rightarrow 0.7257$) as anomalies blend closer to the in-distribution manifold.

---

### Table III: Controlled Experiment 2 — Leave-One-Type-Out Generalization
*Models trained with one corruption category completely excluded from training, tested strictly on the held-out category.*

| Held-Out Defect Category | Test Instances ($N$) | HGB Zero-Shot PR-AUC | Generalization Status |
|:---|:---:|:---:|:---|
| **Negative Inventory** | 8,130 | 0.8829 | Strong Transfer (transfers from negative money) |
| **Negative Money** | 18,558 | 0.8725 | Strong Transfer (transfers from negative inventory) |
| **Anomalous Date** | 91,365 | 0.7487 | Moderate Transfer |
| **Out-of-Bounds Rating** | 1,241 | **0.0980** | **Catastrophic Failure** |
| **Impossible Age** | 75 | **0.0081** | **Catastrophic Failure** |

*Finding:* Detectors generalize well across semantically adjacent numeric corruptions (e.g., negative money $\leftrightarrow$ negative count), but completely fail to detect rare, bounded-scale anomalies (ratings, age) unless explicit training examples are provided.

---

### Table IV: Controlled Experiment 3 — Feature-Group Ablation
*Cumulative contribution of feature subsets using HistGradientBoosting.*

| Feature Set | Dimensionality ($D$) | Features Included | HGB PR-AUC |
|:---|:---:|:---|:---:|
| **Value Only** | 11 | Literals, min/max numerics, string lengths, negative flags, parsed years | 0.9994 |
| **Value + Column Semantic** | 19 | + Column keyword counts (money, date, inventory, rating, age flags) | **0.9998** |
| **Full (Value + Column + Structural)** | 26 | + Foreign keys, unique constraints, table widths, statement indices | 0.9996 |

*Finding:* Column-semantic interaction features provide the critical boost that eliminates remaining false positives. Structural schema features add minimal extra gain for row-level value defects.

---

## 4. Model Interpretation & Error Analysis (Bonus Item #3)

### SHAP Feature Attribution Ranking (TreeExplainer)
1. `min_year` ($\text{mean } |\text{SHAP}| = 0.3446$): Earliest year identified in date strings.
2. `log_min_numeric` ($0.0559$): Magnitude of the most negative literal.
3. `has_negative_value` ($0.0432$): Binary flag for negative numbers.
4. `has_absurd_year` ($0.0423$): Flag for years $<1900$ or $>2100$.
5. `num_negative_values` ($0.0401$): Total negative literal occurrences.
6. `num_money_cols` ($0.0220$): Number of identified currency/cost columns.

### Representative Case Studies (from `analysis/results/error_cases.json`)

#### 1. False Positive Case Analysis (Legitimate Negative Balances & Adjustments)
* **Statement:** `INSERT INTO accounts (id, user_id, name, type, current_balance) VALUES (3, 1, 'Travel Credit Card', 'credit', -850.40);` (`personal_finance_9`)
* **Predicted Probability:** $1.00$ (Model predicted Corrupt, Ground Truth = Clean)
* **Diagnosis:** The model over-relies on `log_min_numeric` and `num_money_cols`. However, in real double-entry accounting and credit-card models, a negative balance is legitimate business logic (representing an outstanding debt or overdraft).
* **Actionable Fix for Final Project:** Incorporate compound contextual features (e.g., `account_type == 'credit'` suppressing negative balance alarms).

#### 2. False Negative Case Analysis ("Feature Masking" by Unrelated IDs)
* **Statement:** `INSERT INTO dishes (id, restaurant_id, name, average_rating) VALUES (12, 24, 'Omakase Sushi Set', 10.0);` (`wishlist_management_1`)
* **Predicted Probability:** $0.013$ (Model predicted Clean, Ground Truth = Corrupt Rating)
* **Diagnosis:** Feature masking. In this row, `restaurant_id = 24` is an integer larger than the corrupt rating `10.0`. Because row-level aggregation collapses numeric values into `log_max_numeric`, the model observes a normal integer magnitude and misses the out-of-bounds rating.
* **Actionable Fix for Final Project:** Extract column-aligned value z-scores rather than row-wide global extrema.

---

## 5. Figures Ready for Inclusion

1. `figures/fig1_model_comparison.png`: High-DPI bar chart showing PR-AUC for all 8 models against random prevalence.
2. `figures/fig2_severity_decay.png`: Sensitivity curve comparing HGB vs Isolation Forest across Obvious, Medium, and Subtle defects.
3. `figures/fig3_type_generalization.png`: Bar chart demonstrating zero-shot transfer drop on unseen corruption categories.
4. `figures/fig4_feature_importance.png`: Top-10 Gini feature importances.
5. `figures/fig5_shap_beeswarm.png`: Global SHAP beeswarm summary distribution.
6. `figures/fig5_shap_bar.png`: Mean absolute SHAP impact ranking.
