"""Full-scale Tier-2 silent defect detection pipeline for AWM-1K.

Builds a controlled benchmark of SQL-valid but semantically corrupted
INSERT statements, evaluates eight detection methods under an environment-
grouped train/test split, and logs all results to Weights & Biases.

Three controlled experiments:
  1. Severity breakdown  – same models, performance split by obvious/medium/subtle
  2. Type holdout        – train without one corruption type, test generalization
  3. Feature ablation    – cumulative feature groups (value → +column → +structural)

Usage:
    uv run python analysis/detect_silent_defects.py              # build + full run
    uv run python analysis/detect_silent_defects.py --skip-build # use cached dataset
    uv run python analysis/detect_silent_defects.py --build-only # build only, then exit
    uv run python analysis/detect_silent_defects.py --n-envs 50  # quick smoke test
"""

from __future__ import annotations

import os

# Cap BLAS/OMP threads before loading numpy/scipy to prevent OpenBLAS thread exhaustion on high-core systems
for _var in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_var] = "8"

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

from analysis.build_insert_audit import INSERT_RE, split_sql_list

# Paths & global constants

DATA_DIR     = Path("data")
RESULTS_DIR  = Path("analysis/results")
DATASET_PATH = RESULTS_DIR / "tier2_dataset.parquet"

RANDOM_SEED = 42
SPLIT_SEED  = 42
TEST_SIZE   = 0.25   # fraction of environments for test
K_RANK      = 100    # k for Precision@K / Recall@K
N_JOBS      = 8      # cap parallelism — OpenBLAS has a compiled thread limit (128)

SEVERITIES       = ("obvious", "medium", "subtle")
CORRUPTION_TYPES = (
    "negative_money",
    "negative_inventory",
    "impossible_age",
    "out_of_bounds_rating",
    "anomalous_date",
)

# Column keyword lists

MONEY_WORDS  = ["price", "amount", "cost", "fee", "salary", "fare", "balance",
                "total", "budget", "revenue", "wage", "rate", "charge", "payment"]
COUNT_WORDS  = ["quantity", "count", "stock", "inventory", "available_seats",
                "capacity", "units", "seats", "slots"]
DATE_WORDS   = ["date", "time", "created", "scheduled", "birth", "due",
                "started", "ended", "timestamp"]
RATING_WORDS = ["rating", "score", "stars", "grade"]
AGE_WORDS    = frozenset(["age", "user_age", "patient_age", "customer_age"])

# Corruption functions  (type × severity → new value string)

def _corrupt_money(val: float, severity: str) -> str:
    """Negate a positive monetary value at three magnitudes."""
    if severity == "obvious":
        return f"{-abs(val) * 10:.4f}"
    if severity == "medium":
        return f"{-abs(val):.4f}"
    # subtle: tiny negative — plausible rounding artefact
    return f"{-max(0.01, abs(val) * 0.01):.4f}"


def _corrupt_count(val: int, severity: str) -> str:
    if severity == "obvious":
        return str(-abs(val) * 10)
    if severity == "medium":
        return str(-abs(val))
    return "-1"   # subtle: just one below zero


def _corrupt_age(severity: str) -> str:
    return {"obvious": "285", "medium": "150", "subtle": "120"}[severity]


def _corrupt_rating(severity: str) -> str:
    return {"obvious": "99.0", "medium": "10.0", "subtle": "5.5"}[severity]


def _corrupt_date(val_str: str, year: int, severity: str) -> str:
    new_year = {
        "obvious": 1823,
        "medium":  1950,
        "subtle":  max(1900, year - 30),
    }[severity]
    return val_str.replace(f"'{year}-", f"'{new_year}-")


def candidate_corruptions(col: str, val: str) -> list[tuple[str, str]]:
    """Return [(corruption_type, original_value)] for every applicable type."""
    col_l = col.lower()
    v = val.strip()
    out: list[tuple[str, str]] = []

    if any(w in col_l for w in MONEY_WORDS):
        try:
            if float(v) > 0:
                out.append(("negative_money", v))
        except ValueError:
            pass

    if any(w in col_l for w in COUNT_WORDS):
        try:
            if int(float(v)) > 0:
                out.append(("negative_inventory", v))
        except ValueError:
            pass

    if col_l in AGE_WORDS:
        try:
            age = int(float(v))
            if 0 <= age <= 100:
                out.append(("impossible_age", v))
        except ValueError:
            pass

    if any(w in col_l for w in RATING_WORDS):
        try:
            r = float(v)
            if 1.0 <= r <= 5.0:
                out.append(("out_of_bounds_rating", v))
        except ValueError:
            pass

    m = re.search(r"'(\d{4})-(\d{2})-(\d{2})", v)
    if m and any(w in col_l for w in DATE_WORDS):
        year = int(m.group(1))
        if 1950 <= year <= 2030:
            out.append(("anomalous_date", v))

    return out


def apply_corruption(col: str, val: str, corr_type: str, severity: str) -> str:
    """Apply a specific (type, severity) corruption to a value string."""
    v = val.strip()
    if corr_type == "negative_money":
        return _corrupt_money(float(v), severity)
    if corr_type == "negative_inventory":
        return _corrupt_count(int(float(v)), severity)
    if corr_type == "impossible_age":
        return _corrupt_age(severity)
    if corr_type == "out_of_bounds_rating":
        return _corrupt_rating(severity)
    if corr_type == "anomalous_date":
        m = re.search(r"'(\d{4})-", v)
        year = int(m.group(1)) if m else 2024
        return _corrupt_date(v, year, severity)
    raise ValueError(f"Unknown corruption type: {corr_type}")


def reconstruct_insert(table: str, cols: list[str], vals: list[str]) -> str:
    cols_part = f" ({', '.join(cols)})" if cols else ""
    return f"INSERT INTO {table}{cols_part} VALUES ({', '.join(vals)});"


def parse_insert(stmt: str) -> tuple[str, list[str], list[str]] | None:
    m = INSERT_RE.match(stmt)
    if not m:
        return None
    table = m.group(1) or m.group(2) or m.group(3) or m.group(4)
    cols  = [c.strip(' "`[]') for c in split_sql_list(m.group(5))] if m.group(5) else []
    vals  = split_sql_list(m.group(6)) if m.group(6) else []
    return table, cols, vals

# Feature extraction  (27 features in 3 groups)

FEATURE_GROUPS: dict[str, list[str]] = {
    # Derived purely from the literal values in the INSERT
    "value": [
        "has_negative_value", "num_negative_values",
        "log_max_numeric", "log_min_numeric", "num_numeric_values",
        "has_absurd_year", "min_year",
        "num_null_values", "num_string_values",
        "max_string_length", "avg_string_length",
    ],
    # Derived from column names and their semantic categories
    "column": [
        "num_columns", "num_values", "col_val_delta",
        "num_money_cols", "num_count_cols", "num_date_cols",
        "has_rating_col", "has_age_col",
    ],
    # Derived from schema metadata and statement position
    "structural": [
        "table_fk_count", "table_unique_count", "table_col_count",
        "num_unknown_cols", "statement_index", "sql_length", "col_fill_ratio",
    ],
}
ALL_FEATURES: list[str] = (
    FEATURE_GROUPS["value"]
    + FEATURE_GROUPS["column"]
    + FEATURE_GROUPS["structural"]
)


def _signed_log(x: float) -> float:
    return float(np.sign(x) * np.log1p(abs(x)))


def extract_features(
    cols: list[str],
    vals: list[str],
    schema_meta: dict[str, Any],
) -> dict[str, float]:
    col_l = [c.lower() for c in cols]
    numerics: list[float] = []
    years:    list[int]   = []
    str_lens: list[int]   = []
    null_count = 0

    for v in vals:
        v = v.strip()
        if v.upper() == "NULL":
            null_count += 1
            continue
        try:
            numerics.append(float(v))
        except ValueError:
            pass
        m = re.search(r"'(\d{4})-", v)
        if m:
            years.append(int(m.group(1)))
        if v.startswith("'") and len(v) >= 2:
            str_lens.append(len(v) - 2)

    known_cols = set(schema_meta.get("columns", {}).keys())
    normalized = {c.strip(' "`[]') for c in cols}

    return {
        # Group A: value
        "has_negative_value":  float(any(n < 0 for n in numerics)),
        "num_negative_values": float(sum(1 for n in numerics if n < 0)),
        "log_max_numeric":     _signed_log(max(numerics)) if numerics else 0.0,
        "log_min_numeric":     _signed_log(min(numerics)) if numerics else 0.0,
        "num_numeric_values":  float(len(numerics)),
        "has_absurd_year":     float(any(y < 1900 or y > 2100 for y in years)),
        "min_year":            float(min(years)) if years else 2000.0,
        "num_null_values":     float(null_count),
        "num_string_values":   float(len(str_lens)),
        "max_string_length":   float(max(str_lens)) if str_lens else 0.0,
        "avg_string_length":   float(np.mean(str_lens)) if str_lens else 0.0,
        # Group B: column-semantic
        "num_columns":    float(len(cols)),
        "num_values":     float(len(vals)),
        "col_val_delta":  float(len(cols) - len(vals)),
        "num_money_cols": float(sum(1 for c in col_l if any(w in c for w in MONEY_WORDS))),
        "num_count_cols": float(sum(1 for c in col_l if any(w in c for w in COUNT_WORDS))),
        "num_date_cols":  float(sum(1 for c in col_l if any(w in c for w in DATE_WORDS))),
        "has_rating_col": float(any(any(w in c for w in RATING_WORDS) for c in col_l)),
        "has_age_col":    float(any(c in AGE_WORDS for c in col_l)),
        # Group C: schema-structural
        "table_fk_count":     float(schema_meta.get("fk_count", 0)),
        "table_unique_count": float(schema_meta.get("unique_count", 0)),
        "table_col_count":    float(len(schema_meta.get("columns", {}))),
        "num_unknown_cols":   float(len(normalized - known_cols)) if cols else 0.0,
        "statement_index":    float(schema_meta.get("stmt_idx", 0)),
        "sql_length":         float(schema_meta.get("sql_length", 0)),
        "col_fill_ratio":     (float(len(vals))
                               / max(1.0, float(len(schema_meta.get("columns", {}))))),
    }

# Dataset builder

def _load_jsonl(path: Path) -> dict[str, dict]:
    data: dict[str, dict] = {}
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            data[rec["scenario"]] = rec
    return data


def _schema_meta(cur: sqlite3.Cursor, t_name: str) -> dict[str, Any]:
    try:
        col_info = cur.execute(f'PRAGMA table_info("{t_name}")').fetchall()
        fk_info  = cur.execute(f'PRAGMA foreign_key_list("{t_name}")').fetchall()
        uq_info  = cur.execute(f'PRAGMA index_list("{t_name}")').fetchall()
        return {
            "columns":      {row[1]: row[2] for row in col_info},
            "fk_count":     len(fk_info),
            "unique_count": sum(1 for r in uq_info if r[2]),
        }
    except Exception:
        return {"columns": {}, "fk_count": 0, "unique_count": 0}


def build_dataset(n_envs: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Build the Tier-2 benchmark dataset.

    For every successfully executing clean INSERT:
      - One clean row  (label=0)
      - One corrupted row per applicable (corruption_type × severity)
        if and only if SQLite accepts the corrupted version (label=1)
    """
    print("Loading JSONL files …")
    db_map     = _load_jsonl(DATA_DIR / "gen_db.jsonl")
    sample_map = _load_jsonl(DATA_DIR / "gen_sample.jsonl")

    scenarios = sorted(set(db_map) & set(sample_map))
    if n_envs:
        scenarios = scenarios[:n_envs]

    rows:  list[dict] = []
    stats: dict[str, int] = defaultdict(int)

    for scenario in tqdm(scenarios, desc="Environments"):
        db_entry    = db_map[scenario]
        sample_data = sample_map[scenario].get("sample_data", {})
        tables_data = sample_data.get("tables", [])
        if not isinstance(tables_data, list) or not tables_data:
            stats["skipped_malformed"] += 1
            continue

        conn = sqlite3.connect(":memory:")
        cur  = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")

        # Create schema
        for t in db_entry.get("db_schema", {}).get("tables", []):
            try:
                cur.execute(t["ddl"])
                for idx in t.get("indexes", []):
                    try:
                        cur.execute(idx)
                    except Exception:
                        pass
            except Exception:
                pass

        # Collect per-table schema metadata
        meta_cache: dict[str, dict] = {
            t["name"]: _schema_meta(cur, t["name"])
            for t in db_entry.get("db_schema", {}).get("tables", [])
        }

        for table_item in tables_data:
            t_name = table_item.get("table_name", "")
            stmts  = table_item.get("insert_statements", [])
            schema = meta_cache.get(t_name, {"columns": {}, "fk_count": 0, "unique_count": 0})

            for stmt_idx, stmt in enumerate(stmts):
                parsed = parse_insert(stmt)
                if not parsed:
                    continue
                _, cols, vals = parsed
                if not cols or len(cols) != len(vals):
                    continue

                stmt_meta = {**schema, "stmt_idx": stmt_idx, "sql_length": len(stmt)}

                # Test if clean statement executes successfully
                cur.execute("SAVEPOINT sp;")
                clean_ok = True
                try:
                    cur.execute(stmt)
                except Exception:
                    clean_ok = False
                cur.execute("ROLLBACK TO SAVEPOINT sp;")

                if not clean_ok:
                    stats["skipped_failed_clean"] += 1
                    # Still commit so FK chain stays intact for later rows
                    try:
                        cur.execute(stmt)
                    except Exception:
                        pass
                    continue

                # Add clean row
                feats = extract_features(cols, vals, stmt_meta)
                rows.append({
                    "scenario":        scenario,
                    "table":           t_name,
                    "label":           0,
                    "corruption_type": "none",
                    "severity":        "none",
                    **feats,
                })
                stats["clean_rows"] += 1

                # Try all (col, type, severity) combinations
                for col_idx, (col, val) in enumerate(zip(cols, vals)):
                    for corr_type, _ in candidate_corruptions(col, val):
                        for severity in SEVERITIES:
                            new_val  = apply_corruption(col, val, corr_type, severity)
                            new_vals = list(vals)
                            new_vals[col_idx] = new_val
                            corrupted_stmt = reconstruct_insert(t_name, cols, new_vals)

                            cur.execute("SAVEPOINT sp_c;")
                            accepted = False
                            try:
                                cur.execute(corrupted_stmt)
                                accepted = True
                            except Exception:
                                pass
                            cur.execute("ROLLBACK TO SAVEPOINT sp_c;")

                            if accepted:
                                c_feats = extract_features(cols, new_vals, stmt_meta)
                                rows.append({
                                    "scenario":        scenario,
                                    "table":           t_name,
                                    "label":           1,
                                    "corruption_type": corr_type,
                                    "severity":        severity,
                                    **c_feats,
                                })
                                stats["corrupted_rows"] += 1
                                stats[f"sev_{severity}"] += 1
                                stats[f"type_{corr_type}"] += 1
                            else:
                                stats["corruption_rejected_by_sqlite"] += 1

                # Commit clean statement so FK chain continues
                try:
                    cur.execute(stmt)
                except Exception:
                    pass

        conn.close()

    print(f"\nDataset build complete: {dict(stats)}")
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PATH, index=False)
    print(f"Saved → {DATASET_PATH}  ({len(df):,} rows)")
    return df, dict(stats)

# Evaluation helpers

def _pr_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float]:
    top_k = np.argsort(scores)[::-1][:k]
    tp    = int(y_true[top_k].sum())
    return tp / k, tp / max(1, int(y_true.sum()))


def _eval(
    name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    k: int = K_RANK,
    threshold: float = 0.5,
) -> dict:
    preds = (scores >= threshold).astype(int)
    p_k, r_k = _pr_at_k(y_true, scores, k)
    return {
        "model":        name,
        "pr_auc":       float(average_precision_score(y_true, scores)),
        f"p@{k}":       p_k,
        f"r@{k}":       r_k,
        "precision":    float(precision_score(y_true, preds, zero_division=0)),
        "recall":       float(recall_score(y_true, preds, zero_division=0)),
        "f1":           float(f1_score(y_true, preds, zero_division=0)),
    }


def _rule_scores(df_slice: pd.DataFrame) -> np.ndarray:
    """Deterministic rule-based flags — no training required."""
    flag = (
        # Negative value in a money or count column
        ((df_slice["has_negative_value"] > 0)
         & (df_slice["num_money_cols"] + df_slice["num_count_cols"] > 0))
        # Rating-scale column with a value well above 5
        | ((df_slice["has_rating_col"] > 0)
           & (df_slice["log_max_numeric"] > np.log1p(5.0)))
        # Absurd year in a date column
        | (df_slice["has_absurd_year"] > 0)
        # Age column with a value above 110
        | ((df_slice["has_age_col"] > 0)
           & (df_slice["log_max_numeric"] > np.log1p(110.0)))
    )
    return flag.astype(float).values


def _zscore_scores(X_train_clean: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Max absolute z-score across features, fit on clean training data only."""
    mu  = X_train_clean.mean(axis=0)
    std = X_train_clean.std(axis=0) + 1e-8
    return np.abs((X_test - mu) / std).max(axis=1)

# Model comparison

def _make_split(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(int)
    g = df["scenario"].values
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    tr, te = next(splitter.split(X, y, g))
    return X, y, g, tr, te, X[tr][y[tr] == 0]   # last element = clean-only train X


def run_comparison(
    df: pd.DataFrame,
    feature_cols: list[str],
    tag: str = "main",
    wandb_run: Any = None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Train and evaluate all 8 models. Returns (metrics, tr_idx, te_idx)."""
    X, y, _, tr_idx, te_idx, X_tr_clean = _make_split(df, feature_cols)
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    n_neg    = int((y_tr == 0).sum())
    n_pos    = int(y_tr.sum())
    scale_pw = n_neg / max(1, n_pos)

    results: list[dict] = []

    # 1. Rule-based baseline (no training)
    scores = _rule_scores(df.iloc[te_idx])
    results.append(_eval("rule_baseline", y_te, scores))

    # 2. Z-score (unsupervised, clean train only)
    scores = _zscore_scores(X_tr_clean, X_te)
    results.append(_eval("z_score", y_te, scores))

    # 3. Isolation Forest (unsupervised, clean train only)
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=RANDOM_SEED, n_jobs=N_JOBS)
    iso.fit(X_tr_clean)
    results.append(_eval("isolation_forest", y_te, -iso.decision_function(X_te)))

    # 4. Local Outlier Factor (novelty mode, clean train only)
    if len(X_tr_clean) > 25000:
        rng = np.random.default_rng(RANDOM_SEED)
        lof_tr_idx = rng.choice(len(X_tr_clean), size=25000, replace=False)
        X_lof_tr = X_tr_clean[lof_tr_idx]
    else:
        X_lof_tr = X_tr_clean
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1, n_jobs=N_JOBS)
    lof.fit(X_lof_tr)
    results.append(_eval("local_outlier_factor", y_te, -lof.decision_function(X_te)))

    # 5. Logistic Regression (supervised)
    pipe_lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED),
    )
    pipe_lr.fit(X_tr, y_tr)
    results.append(_eval("logistic_regression", y_te, pipe_lr.predict_proba(X_te)[:, 1]))

    # 6. Random Forest (supervised)
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=N_JOBS,
    )
    rf.fit(X_tr, y_tr)
    results.append(_eval("random_forest", y_te, rf.predict_proba(X_te)[:, 1]))

    # 7. HistGradientBoosting (supervised)
    hgb = HistGradientBoostingClassifier(
        max_iter=300, class_weight="balanced", random_state=RANDOM_SEED,
    )
    hgb.fit(X_tr, y_tr)
    results.append(_eval("hist_gradient_boosting", y_te, hgb.predict_proba(X_te)[:, 1]))

    # 8. XGBoost (supervised)
    # use_label_encoder and eval_metric removed in XGBoost ≥ 2.0
    xgb = XGBClassifier(
        n_estimators=300,
        scale_pos_weight=scale_pw,
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        verbosity=0,
    )
    xgb.fit(X_tr, y_tr)
    results.append(_eval("xgboost", y_te, xgb.predict_proba(X_te)[:, 1]))

    # Print summary
    print(f"\n[{tag}] Model comparison (n_test={len(te_idx):,}, pos_rate={y_te.mean():.3%}):")
    print(f"{'Model':<28} {'PR-AUC':>8} {'P@100':>8} {'R@100':>8} {'F1':>8}")
    for r in results:
        print(f"{r['model']:<28} {r['pr_auc']:>8.4f} {r[f'p@{K_RANK}']:>8.4f}"
              f" {r[f'r@{K_RANK}']:>8.4f} {r['f1']:>8.4f}")

    # W&B
    if wandb_run:
        tbl = wandb.Table(columns=list(results[0].keys()),
                          data=[list(r.values()) for r in results])
        wandb_run.log({f"{tag}/comparison_table": tbl})
        for r in results:
            wandb_run.log({f"{tag}/{r['model']}_pr_auc": r["pr_auc"]})

    return results, tr_idx, te_idx

# Controlled experiment 1: severity breakdown

def run_severity_breakdown(
    df: pd.DataFrame,
    feature_cols: list[str],
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    wandb_run: Any = None,
) -> None:
    """Break down HGB and IF performance on test set by corruption severity."""
    X, y, _, _, _, X_tr_clean = _make_split(df, feature_cols)
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    sev_te = df.iloc[te_idx]["severity"].values

    hgb = HistGradientBoostingClassifier(
        max_iter=300, class_weight="balanced", random_state=RANDOM_SEED,
    )
    hgb.fit(X_tr, y_tr)
    hgb_scores = hgb.predict_proba(X_te)[:, 1]

    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=RANDOM_SEED, n_jobs=N_JOBS)
    iso.fit(X_tr_clean)
    iso_scores = -iso.decision_function(X_te)

    clean_mask = sev_te == "none"
    rows: list[dict] = []
    print("\n=== Severity breakdown ===")
    for sev in ("obvious", "medium", "subtle"):
        corr_mask = sev_te == sev
        if corr_mask.sum() == 0:
            continue
        combined = clean_mask | corr_mask
        yt = y_te[combined]
        hgb_pr = float(average_precision_score(yt, hgb_scores[combined]))
        iso_pr = float(average_precision_score(yt, iso_scores[combined]))
        rows.append({"severity": sev,
                     "n_corrupted": int(corr_mask.sum()),
                     "hgb_pr_auc": hgb_pr,
                     "iso_pr_auc": iso_pr})
        print(f"  {sev:>8}: HGB={hgb_pr:.4f}  IF={iso_pr:.4f}"
              f"  (n_corrupted={int(corr_mask.sum())})")

    if wandb_run and rows:
        tbl = wandb.Table(columns=list(rows[0].keys()),
                          data=[list(r.values()) for r in rows])
        wandb_run.log({"severity/breakdown_table": tbl})

    return rows

# Controlled experiment 2: corruption-type holdout

def run_type_holdout(
    df: pd.DataFrame,
    feature_cols: list[str],
    wandb_run: Any = None,
) -> None:
    """For each corruption type: train HGB without it, test on it exclusively."""
    X = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(int)
    g = df["scenario"].values
    types = df["corruption_type"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    tr_idx, te_idx = next(splitter.split(X, y, g))

    rows: list[dict] = []
    print("\n=== Type holdout (HGB trained without held-out type) ===")
    for held_out in CORRUPTION_TYPES:
        # Train: clean rows + corrupted rows that are NOT the held-out type
        tr_use = tr_idx[(y[tr_idx] == 0) | (types[tr_idx] != held_out)]
        if y[tr_use].sum() == 0:
            continue

        hgb = HistGradientBoostingClassifier(
            max_iter=300, class_weight="balanced", random_state=RANDOM_SEED,
        )
        hgb.fit(X[tr_use], y[tr_use])

        # Test: clean rows + only the held-out type
        te_use = te_idx[(y[te_idx] == 0) | (types[te_idx] == held_out)]
        yt = y[te_use]
        if yt.sum() == 0:
            continue

        scores = hgb.predict_proba(X[te_use])[:, 1]
        pr = float(average_precision_score(yt, scores))
        n_held = int((types[te_use] == held_out).sum())
        rows.append({"held_out_type": held_out,
                     "n_test_corrupted": n_held,
                     "hgb_pr_auc": pr})
        print(f"  {held_out:>25}: PR-AUC={pr:.4f}  (n={n_held})")

    if wandb_run and rows:
        tbl = wandb.Table(columns=list(rows[0].keys()),
                          data=[list(r.values()) for r in rows])
        wandb_run.log({"type_holdout/table": tbl})

    return rows

# Controlled experiment 3: feature-group ablation

def run_feature_ablation(
    df: pd.DataFrame,
    wandb_run: Any = None,
) -> None:
    """Cumulative feature-group ablation with HGB."""
    y = df["label"].values.astype(int)
    g = df["scenario"].values

    ablation_sets = {
        "value only":                  FEATURE_GROUPS["value"],
        "value + column":              FEATURE_GROUPS["value"] + FEATURE_GROUPS["column"],
        "value + column + structural": ALL_FEATURES,
    }

    rows: list[dict] = []
    print("\n=== Feature ablation (HGB) ===")
    for label, feats in ablation_sets.items():
        X = df[feats].values.astype(float)
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SPLIT_SEED)
        tr_idx, te_idx = next(splitter.split(X, y, g))
        hgb = HistGradientBoostingClassifier(
            max_iter=300, class_weight="balanced", random_state=RANDOM_SEED,
        )
        hgb.fit(X[tr_idx], y[tr_idx])
        scores = hgb.predict_proba(X[te_idx])[:, 1]
        pr = float(average_precision_score(y[te_idx], scores))
        rows.append({"feature_set": label, "n_features": len(feats), "hgb_pr_auc": pr})
        print(f"  {label:>35}: n={len(feats):>2}  PR-AUC={pr:.4f}")

    if wandb_run and rows:
        tbl = wandb.Table(columns=list(rows[0].keys()),
                          data=[list(r.values()) for r in rows])
        wandb_run.log({"ablation/table": tbl})

    return rows

# Feature importance

def log_feature_importance(
    df: pd.DataFrame,
    tr_idx: np.ndarray,
    wandb_run: Any = None,
) -> list[dict]:
    X = df[ALL_FEATURES].values.astype(float)
    y = df["label"].values.astype(int)
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=N_JOBS,
    )
    rf.fit(X[tr_idx], y[tr_idx])
    pairs = sorted(zip(ALL_FEATURES, rf.feature_importances_), key=lambda x: -x[1])

    print("\n=== Top-10 RF feature importances ===")
    for feat, imp in pairs[:10]:
        print(f"  {feat:<30} {imp:.4f}")

    if wandb_run:
        tbl = wandb.Table(columns=["feature", "importance"],
                          data=[[f, float(i)] for f, i in pairs])
        wandb_run.log({"feature_importance/rf_table": tbl})

    return [{"feature": f, "importance": float(i)} for f, i in pairs]

# Entry point

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-envs",        type=int,   default=None,
                        help="Limit to N environments (default: all 1,000)")
    parser.add_argument("--skip-build",    action="store_true",
                        help="Load cached dataset from disk instead of rebuilding")
    parser.add_argument("--build-only",    action="store_true",
                        help="Build and save dataset, then exit without running models")
    parser.add_argument("--wandb-project", default="awm-silent-defects",
                        help="W&B project name  (default: awm-silent-defects)")
    parser.add_argument("--wandb-entity",  default=None,
                        help="W&B entity / username  (default: auto-detected)")
    parser.add_argument("--no-wandb",      action="store_true",
                        help="Disable W&B logging entirely")
    args = parser.parse_args()

    # Build or load dataset
    dataset_stats: dict = {}
    if args.skip_build:
        print(f"Loading cached dataset from {DATASET_PATH} …")
        df = pd.read_parquet(DATASET_PATH)
    else:
        df, dataset_stats = build_dataset(n_envs=args.n_envs)

    if args.build_only:
        print("Build complete. Exiting (--build-only).")
        return

    n_envs_actual = df["scenario"].nunique()
    n_corr        = int(df["label"].sum())
    corr_rate     = float(df["label"].mean())
    print(f"\nDataset: {len(df):,} rows | {n_corr:,} corrupted ({corr_rate:.2%}) "
          f"| {n_envs_actual} environments")

    # W&B initialisation
    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "n_environments":   n_envs_actual,
                "n_rows":           len(df),
                "n_corrupted":      n_corr,
                "corruption_rate":  corr_rate,
                "test_size":        TEST_SIZE,
                "random_seed":      RANDOM_SEED,
                "split_seed":       SPLIT_SEED,
                "k_rank":           K_RANK,
                "n_features":       len(ALL_FEATURES),
                "severities":       list(SEVERITIES),
                "corruption_types": list(CORRUPTION_TYPES),
                **dataset_stats,
            },
            tags=["tier2", "silent-defects", "awm"],
        )
        for groupby_col, tag in [("corruption_type", "type"), ("severity", "severity")]:
            counts = df.groupby(groupby_col).size().reset_index(name="count")
            wandb_run.log({
                f"dataset/{tag}_counts": wandb.Table(
                    columns=list(counts.columns),
                    data=counts.values.tolist(),
                )
            })

    # Main model comparison
    comparison_results, tr_idx, te_idx = run_comparison(
        df, ALL_FEATURES, tag="main", wandb_run=wandb_run,
    )

    # Controlled experiment 1: severity breakdown
    severity_results = run_severity_breakdown(df, ALL_FEATURES, tr_idx, te_idx, wandb_run=wandb_run)

    # Controlled experiment 2: type holdout
    type_holdout_results = run_type_holdout(df, ALL_FEATURES, wandb_run=wandb_run)

    # Controlled experiment 3: feature ablation
    ablation_results = run_feature_ablation(df, wandb_run=wandb_run)

    # Feature importance
    feature_importance_results = log_feature_importance(df, tr_idx, wandb_run=wandb_run)

    # Save structured JSON summary
    final_output = {
        "dataset": {
            "n_environments": n_envs_actual,
            "n_rows": len(df),
            "n_corrupted": n_corr,
            "corruption_rate": corr_rate,
            "features": ALL_FEATURES,
        },
        "main_comparison": comparison_results,
        "severity_breakdown": severity_results,
        "type_holdout": type_holdout_results,
        "feature_ablation": ablation_results,
        "feature_importance": feature_importance_results,
    }
    json_path = RESULTS_DIR / "tier2_results.json"
    with json_path.open("w") as fh:
        json.dump(final_output, fh, indent=2)
    print(f"\nStructured results saved → {json_path}")

    if wandb_run:
        wandb_run.finish()

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
