"""Model interpretation and comprehensive error analysis for AWM silent defects.

Fulfills Bonus Item #3 (Data or model interpretation via SHAP values) and
Section 2.5 (Interpretation of representative successes and errors).

Generates:
  1. figures/fig5_shap_beeswarm.png   - SHAP summary beeswarm plot
  2. figures/fig5_shap_bar.png        - Mean absolute SHAP feature impact
  3. analysis/results/error_cases.json- Catalog of analyzed false positives,
                                        false negatives, and representative detections
  4. analysis/results/shap_summary.json - Quantitative SHAP values per feature
"""

from __future__ import annotations

import os

# Cap threads to prevent OpenBLAS exhaustion
for _var in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_var] = "8"

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from analysis.detect_silent_defects import ALL_FEATURES, RANDOM_SEED, SPLIT_SEED, TEST_SIZE

DATA_PATH = Path("analysis/results/tier2_dataset.parquet")
SAMPLE_PATH = Path("data/gen_sample.jsonl")
RESULTS_DIR = Path("analysis/results")
FIGURES_DIR = Path("figures")


def load_raw_statements() -> dict[tuple[str, str, int], str]:
    """Build a lookup mapping (scenario, table, stmt_index) -> raw SQL string."""
    lookup = {}
    print("Indexing raw SQL statements from gen_sample.jsonl ...")
    with SAMPLE_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            scenario = rec["scenario"]
            tables = rec.get("sample_data", {}).get("tables", [])
            for t in tables:
                t_name = t.get("table_name", "")
                stmts = t.get("insert_statements", [])
                for idx, stmt in enumerate(stmts):
                    lookup[(scenario, t_name, idx)] = stmt
    return lookup


def main() -> None:
    print(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)

    X = df[ALL_FEATURES].values.astype(float)
    y = df["label"].values.astype(int)
    groups = df["scenario"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    tr_idx, te_idx = next(splitter.split(X, y, groups))

    print(f"Train samples: {len(tr_idx):,} | Test samples: {len(te_idx):,}")

    # Train Random Forest classifier
    print("Fitting Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=8,
    )
    rf.fit(X[tr_idx], y[tr_idx])

    probs = rf.predict_proba(X[te_idx])[:, 1]
    preds = (probs >= 0.5).astype(int)
    y_test = y[te_idx]
    df_test = df.iloc[te_idx].copy().reset_index(drop=True)

    df_test["prob"] = probs
    df_test["pred"] = preds

    # ──────────────────────────────────────────────────────────────────────────
    # 1. SHAP TreeExplainer Analysis
    # ──────────────────────────────────────────────────────────────────────────
    print("\nComputing SHAP values using TreeExplainer...")
    # Sample balanced set of test points for fast and representative SHAP computation
    rng = np.random.default_rng(RANDOM_SEED)
    clean_test_indices = np.where(y_test == 0)[0]
    corr_test_indices = np.where(y_test == 1)[0]

    sample_size_per_class = min(1000, len(clean_test_indices), len(corr_test_indices))
    sample_sub = np.concatenate([
        rng.choice(clean_test_indices, size=sample_size_per_class, replace=False),
        rng.choice(corr_test_indices, size=sample_size_per_class, replace=False),
    ])
    rng.shuffle(sample_sub)

    X_sample = X[te_idx][sample_sub]

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)

    # In modern shap / binary classification, check shape
    if isinstance(shap_values, list):
        sv = shap_values[1]  # positive class (corrupt)
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # Mean absolute SHAP impact
    mean_abs_shap = np.abs(sv).mean(axis=0)
    shap_ranking = sorted(zip(ALL_FEATURES, mean_abs_shap), key=lambda x: -x[1])

    print("\nTop-10 Features by Mean Absolute SHAP Value:")
    for feat, val in shap_ranking[:10]:
        print(f"  {feat:<28} {val:.5f}")

    shap_summary = {
        "ranking": [{"feature": f, "mean_abs_shap": float(v)} for f, v in shap_ranking],
        "sample_size": len(sample_sub),
    }
    with (RESULTS_DIR / "shap_summary.json").open("w") as f:
        json.dump(shap_summary, f, indent=2)

    # Plot 1: SHAP Beeswarm Plot
    plt.figure(figsize=(7.5, 5.0))
    shap.summary_plot(
        sv,
        X_sample,
        feature_names=ALL_FEATURES,
        max_display=12,
        show=False,
    )
    plt.title("SHAP Feature Attribution (Impact on Defect Detection)", fontsize=11, pad=12)
    beeswarm_path = FIGURES_DIR / "fig5_shap_beeswarm.png"
    plt.savefig(beeswarm_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {beeswarm_path}")

    # Plot 2: SHAP Bar Plot
    top10_feats = [f for f, _ in shap_ranking[:10]][::-1]
    top10_vals = [v for _, v in shap_ranking[:10]][::-1]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh(top10_feats, top10_vals, color="#e69f00", edgecolor="black", linewidth=0.6, height=0.6)
    for i, v in enumerate(top10_vals):
        ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8.5)
    ax.set_xlabel("Mean |SHAP Value| (Average Impact on Prediction)")
    ax.set_title("Global Feature Importance via SHAP Values")
    ax.set_xlim(0, max(top10_vals) * 1.18)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    bar_path = FIGURES_DIR / "fig5_shap_bar.png"
    plt.savefig(bar_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {bar_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Detailed Error Analysis (False Positives & False Negatives)
    # ──────────────────────────────────────────────────────────────────────────
    raw_sql_map = load_raw_statements()

    # False Positives (Clean statement flagged as corrupt)
    fp_mask = (df_test["label"] == 0) & (df_test["pred"] == 1)
    df_fp = df_test[fp_mask].sort_values(by="prob", ascending=False)
    print(f"\nTotal False Positives on Test Set: {len(df_fp):,} / {len(clean_test_indices):,}")

    # False Negatives (Corrupt statement missed by model)
    fn_mask = (df_test["label"] == 1) & (df_test["pred"] == 0)
    df_fn = df_test[fn_mask].sort_values(by="prob", ascending=True)
    print(f"Total False Negatives on Test Set: {len(df_fn):,} / {len(corr_test_indices):,}")

    # Extract concrete case studies
    def extract_case(row: pd.Series, case_type: str) -> dict:
        scen = row["scenario"]
        tbl = row["table"]
        stmt_idx = int(row["statement_index"])
        raw_sql = raw_sql_map.get((scen, tbl, stmt_idx), "[Original SQL unavailable in cache]")

        feature_values = {f: float(row[f]) for f in ALL_FEATURES if abs(float(row[f])) > 0.001}
        return {
            "case_type": case_type,
            "scenario": scen,
            "table": tbl,
            "statement_index": stmt_idx,
            "actual_label": int(row["label"]),
            "predicted_prob": float(row["prob"]),
            "corruption_type": str(row["corruption_type"]),
            "severity": str(row["severity"]),
            "sql_statement": raw_sql,
            "key_features": feature_values,
        }

    case_studies = {
        "summary": {
            "total_test_samples": len(df_test),
            "false_positives_count": int(fp_mask.sum()),
            "false_negatives_count": int(fn_mask.sum()),
            "true_positives_count": int(((df_test["label"] == 1) & (df_test["pred"] == 1)).sum()),
            "true_negatives_count": int(((df_test["label"] == 0) & (df_test["pred"] == 0)).sum()),
        },
        "false_positive_cases": [extract_case(row, "false_positive") for _, row in df_fp.head(5).iterrows()],
        "false_negative_cases": [extract_case(row, "false_negative") for _, row in df_fn.head(5).iterrows()],
        "representative_true_positives": [],
    }

    # Add 1 representative true positive per corruption type
    for c_type in ["negative_money", "negative_inventory", "impossible_age", "out_of_bounds_rating", "anomalous_date"]:
        tp_sub = df_test[(df_test["label"] == 1) & (df_test["pred"] == 1) & (df_test["corruption_type"] == c_type)]
        if not tp_sub.empty:
            case_studies["representative_true_positives"].append(extract_case(tp_sub.iloc[0], f"true_positive_{c_type}"))

    error_cases_path = RESULTS_DIR / "error_cases.json"
    with error_cases_path.open("w") as f:
        json.dump(case_studies, f, indent=2)
    print(f"Saved: {error_cases_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Log to Weights & Biases
    # ──────────────────────────────────────────────────────────────────────────
    print("\nLogging interpretation figures and error catalog to Weights & Biases ...")
    run = wandb.init(
        project="awm-silent-defects",
        job_type="interpretation",
        name="interpretability-shap-errors",
        tags=["interpretation", "shap", "error-analysis"],
    )

    run.log({
        "interpretation/shap_beeswarm": wandb.Image(str(beeswarm_path)),
        "interpretation/shap_bar": wandb.Image(str(bar_path)),
        "interpretation/shap_table": wandb.Table(
            columns=["feature", "mean_abs_shap"],
            data=[[d["feature"], d["mean_abs_shap"]] for d in shap_summary["ranking"]],
        ),
    })

    # Log error cases table
    all_cases = case_studies["false_positive_cases"] + case_studies["false_negative_cases"] + case_studies["representative_true_positives"]
    cases_table = wandb.Table(
        columns=["type", "scenario", "table", "prob", "corr_type", "severity", "sql_sample"],
        data=[[c["case_type"], c["scenario"], c["table"], c["predicted_prob"], c["corruption_type"], c["severity"], c["sql_statement"][:120]] for c in all_cases],
    )
    run.log({"interpretation/error_cases_table": cases_table})
    run.finish()

    print("\n✓ Interpretation and error analysis complete!")


if __name__ == "__main__":
    main()
