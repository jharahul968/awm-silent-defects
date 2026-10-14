"""Full-scale Tier-2 silent defect detection pipeline for AWM-1K.

Builds a controlled benchmark of SQL-valid but semantically corrupted
INSERT statements, evaluates eight detection methods under an environment-
grouped train/validation/test split, and logs all results to Weights & Biases.

Three controlled experiments:
  1. Severity breakdown  – same models, performance split by obvious/medium/subtle
  2. Type holdout        – train without one corruption type, test generalization
  3. Feature ablation    – cumulative feature groups (value → +column → +structural)

Usage:
    python -m analysis.detect_silent_defects --no-wandb
    python -m analysis.detect_silent_defects --skip-build --no-wandb
    python -m analysis.detect_silent_defects --build-only
"""

from __future__ import annotations

import os

# Cap BLAS/OMP threads before loading numpy/scipy to prevent OpenBLAS thread exhaustion on high-core systems
for _var in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_var] = "8"

import argparse
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
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
    precision_recall_curve,
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
DATASET_STATS_PATH = RESULTS_DIR / "dataset_stats.json"
MODEL_PATH = RESULTS_DIR / "detector.joblib"

RANDOM_SEED = 42
SPLIT_SEED  = 42
TEST_SIZE   = 0.25   # fraction of environments for test
VALIDATION_SIZE = 0.20  # fraction of the remaining environments
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
PROVENANCE_COLUMNS = (
    "source_id", "original_sql", "sql_statement", "corrupted_column",
    "original_value", "corrupted_value", "column_index",
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
        corrupted = -abs(val) * 10
    elif severity == "medium":
        corrupted = -abs(val)
    else:
        corrupted = -max(0.01, abs(val) * 0.01)
    formatted = f"{corrupted:.4f}"
    # Retain the sign for positive source values smaller than the output precision.
    return repr(corrupted) if float(formatted) == 0 else formatted


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

# Feature extraction (26 features in three groups)

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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint() -> dict[str, str]:
    return {name: _file_sha256(DATA_DIR / name)
            for name in ("gen_db.jsonl", "gen_sample.jsonl")}


def dataset_fingerprint(path: str | Path | None = None) -> str:
    return _file_sha256(DATASET_PATH if path is None else Path(path))


def _try_statement(cur: sqlite3.Cursor, statement: str) -> tuple[bool, str | None]:
    """Test a statement without changing the database state."""
    cur.execute("SAVEPOINT candidate")
    try:
        cur.execute(statement)
        return True, None
    except sqlite3.Error as error:
        return False, type(error).__name__
    finally:
        cur.execute("ROLLBACK TO candidate")
        cur.execute("RELEASE candidate")


def build_dataset(n_envs: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Retain executable source rows and accepted, traceable injected variants.

    Label zero means unmodified source data, not manually verified semantics.
    Label one means a rule-based mutation that SQLite accepts.
    """
    db_map = _load_jsonl(DATA_DIR / "gen_db.jsonl")
    sample_map = _load_jsonl(DATA_DIR / "gen_sample.jsonl")
    scenarios = sorted(set(db_map) & set(sample_map))
    if n_envs is not None:
        scenarios = scenarios[:n_envs]
    rows: list[dict] = []
    stats: dict[str, int] = defaultdict(int)

    for scenario in tqdm(scenarios, desc="Environments"):
        db_entry = db_map[scenario]
        tables_data = sample_map[scenario].get("sample_data", {}).get("tables", [])
        if not isinstance(tables_data, list) or not tables_data:
            stats["skipped_malformed_environments"] += 1
            continue
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON")
        for table in db_entry.get("db_schema", {}).get("tables", []):
            try:
                cur.execute(table["ddl"])
            except sqlite3.Error as error:
                stats["schema_errors"] += 1
                stats[f"schema_error_{type(error).__name__}"] += 1
                continue
            for index in table.get("indexes", []):
                try:
                    cur.execute(index)
                except sqlite3.Error as error:
                    stats["index_errors"] += 1
                    stats[f"index_error_{type(error).__name__}"] += 1
        meta_cache = {table["name"]: _schema_meta(cur, table["name"])
                      for table in db_entry.get("db_schema", {}).get("tables", [])}

        for table_item in tables_data:
            t_name = table_item.get("table_name", "")
            schema = meta_cache.get(t_name, {"columns": {}, "fk_count": 0, "unique_count": 0})
            for stmt_idx, stmt in enumerate(table_item.get("insert_statements", [])):
                stats["source_statements_seen"] += 1
                parsed = parse_insert(stmt)
                if not parsed:
                    stats["skipped_unparseable"] += 1
                    continue
                parsed_table, cols, vals = parsed
                if not cols or len(cols) != len(vals):
                    stats["skipped_unsupported_insert"] += 1
                    continue
                accepted, error_kind = _try_statement(cur, stmt)
                if not accepted:
                    stats["skipped_failed_clean"] += 1
                    stats[f"source_error_{error_kind}"] += 1
                    continue
                source_id = json.dumps([scenario, t_name, stmt_idx], separators=(",", ":"))
                shared = {"scenario": scenario, "table": t_name,
                          "source_id": source_id, "original_sql": stmt}
                stmt_meta = {**schema, "stmt_idx": stmt_idx, "sql_length": len(stmt)}
                rows.append({**shared, "label": 0, "corruption_type": "none",
                             "severity": "none", "sql_statement": stmt,
                             "corrupted_column": None, "original_value": None,
                             "corrupted_value": None, "column_index": -1,
                             **extract_features(cols, vals, stmt_meta)})
                stats["clean_rows"] += 1

                for col_idx, (col, val) in enumerate(zip(cols, vals)):
                    for corr_type, _ in candidate_corruptions(col, val):
                        for severity in SEVERITIES:
                            new_val = apply_corruption(col, val, corr_type, severity)
                            if new_val == val.strip():
                                stats["skipped_unchanged_mutations"] += 1
                                continue
                            new_vals = list(vals)
                            new_vals[col_idx] = new_val
                            corrupted_stmt = reconstruct_insert(parsed_table, cols, new_vals)
                            accepted, error_kind = _try_statement(cur, corrupted_stmt)
                            if not accepted:
                                stats["corruption_rejected_by_sqlite"] += 1
                                stats[f"corruption_error_{error_kind}"] += 1
                                continue
                            variant_meta = {**stmt_meta, "sql_length": len(corrupted_stmt)}
                            rows.append({**shared, "label": 1, "corruption_type": corr_type,
                                         "severity": severity, "sql_statement": corrupted_stmt,
                                         "corrupted_column": col, "original_value": val,
                                         "corrupted_value": new_val, "column_index": col_idx,
                                         **extract_features(cols, new_vals, variant_meta)})
                            stats["corrupted_rows"] += 1
                            stats[f"sev_{severity}"] += 1
                            stats[f"type_{corr_type}"] += 1
                # Retain the source state for subsequent foreign-key dependencies.
                cur.execute(stmt)
        conn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No executable source INSERT statements were retained")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PATH, index=False)
    metadata = {"dataset_version": 2, "source_fingerprint": source_fingerprint(),
                "dataset_fingerprint": dataset_fingerprint(), "n_envs_limit": n_envs,
                "builder_stats": dict(stats)}
    DATASET_STATS_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Built {len(df):,} records: {dict(stats)}")
    return df, metadata


# Evaluation and grouped experiments


def split_dataset(df: pd.DataFrame, split_seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the original outer test holdout; reserve validation within training."""
    groups = df["scenario"].to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=split_seed)
    remaining, test = next(outer.split(df, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_SIZE, random_state=split_seed + 1)
    train_rel, validation_rel = next(inner.split(remaining, groups=groups[remaining]))
    train, validation = remaining[train_rel], remaining[validation_rel]
    partitions = [set(groups[index]) for index in (train, validation, test)]
    if any(partitions[i] & partitions[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("Environment partitions overlap")
    return train, validation, test


def _hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, class_weight="balanced", random_state=RANDOM_SEED)


def _rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=N_JOBS)


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED))


def _iforest() -> IsolationForest:
    return IsolationForest(n_estimators=200, contamination="auto", random_state=RANDOM_SEED, n_jobs=N_JOBS)


def train_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    model = _rf()
    model.fit(X, y)
    return model


def model_config() -> dict:
    """Parameters used by the benchmark, including estimator defaults."""
    params = {
        "random_forest": _rf().get_params(),
        "hist_gradient_boosting": _hgb().get_params(),
        "logistic_regression": _lr().steps[-1][1].get_params(),
        "isolation_forest": _iforest().get_params(),
        "local_outlier_factor": LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1, n_jobs=N_JOBS).get_params(),
        "standard_scaler": StandardScaler().get_params(),
        "xgboost": XGBClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=N_JOBS, verbosity=0).get_params(),
        "z_score": {"location": "clean_training_mean", "scale": "clean_training_std", "epsilon": 1e-8},
        "lof_training_cap": 25000,
        "threshold_selection": "highest validation score threshold maximizing F1",
    }
    # XGBoost uses NaN to represent its default missing-value marker.
    for config in params.values():
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, float) and not np.isfinite(value):
                    config[key] = "NaN"
    params["xgboost"]["scale_pos_weight"] = "n_clean_train / n_corrupted_train"
    return params


def load_model_bundle(path: str | Path = MODEL_PATH) -> dict:
    bundle = joblib.load(path)
    required = {"model", "features", "threshold", "split_seed", "model_config", "dataset_fingerprint"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("Detector bundle is missing required metadata; rerun the benchmark")
    if bundle["features"] != ALL_FEATURES:
        raise ValueError("Detector feature list differs from this implementation")
    return bundle


def select_threshold(y_validation: np.ndarray, scores: np.ndarray) -> float:
    """Choose a threshold on validation records, never on the test records."""
    if len(np.unique(y_validation)) != 2:
        raise ValueError("Validation must contain both source and injected records")
    if not np.isfinite(scores).all():
        raise ValueError("Model returned non-finite validation scores")
    precision, recall, thresholds = precision_recall_curve(y_validation, scores)
    f1 = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                   out=np.zeros(len(thresholds)), where=(precision[:-1] + recall[:-1]) > 0)
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best])


def _pr_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float]:
    actual_k = min(k, len(y_true))
    if actual_k == 0:
        return 0.0, 0.0
    top = np.argsort(-scores, kind="stable")[:actual_k]
    tp = int(y_true[top].sum())
    return tp / actual_k, tp / max(1, int(y_true.sum()))


def _eval(name: str, y_true: np.ndarray, scores: np.ndarray,
          k: int = K_RANK, threshold: float = 0.5) -> dict:
    predictions = scores >= threshold
    p_k, r_k = _pr_at_k(y_true, scores, k)
    ap = float(average_precision_score(y_true, scores))
    return {"model": name, "average_precision": ap, "pr_auc": ap,
            f"p@{k}": p_k, f"r@{k}": r_k,
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "threshold": float(threshold), "n_test": len(y_true),
            "n_positive": int(y_true.sum()), "prevalence": float(y_true.mean())}


def _rule_scores(df_slice: pd.DataFrame) -> np.ndarray:
    flag = (
        ((df_slice["has_negative_value"] > 0)
         & (df_slice["num_money_cols"] + df_slice["num_count_cols"] > 0))
        | ((df_slice["has_rating_col"] > 0)
           & (df_slice["log_max_numeric"] > np.log1p(5.0)))
        | (df_slice["has_absurd_year"] > 0)
        | ((df_slice["has_age_col"] > 0)
           & (df_slice["log_max_numeric"] > np.log1p(110.0)))
    )
    return flag.astype(float).to_numpy()


def _zscore_scores(X_train_clean: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    mu = X_train_clean.mean(axis=0)
    std = X_train_clean.std(axis=0) + 1e-8
    return np.abs((X_test - mu) / std).max(axis=1)


def _log_table(wandb_run: Any, name: str, rows: list[dict]) -> None:
    if wandb_run is not None and rows:
        wandb_run.log({name: wandb.Table(columns=list(rows[0]), data=[list(row.values()) for row in rows])})


def _partition_metadata(df: pd.DataFrame, indices: tuple[np.ndarray, np.ndarray, np.ndarray], seed: int) -> dict:
    output = {"split_seed": seed, "outer_test_fraction": TEST_SIZE,
              "inner_validation_fraction": VALIDATION_SIZE,
              "threshold_selection": "validation maximum F1; highest threshold breaks ties",
              "metric": "average_precision_score (AP), legacy pr_auc is an alias",
              "overlapping_environments": 0, "overlapping_sources": 0}
    for name, index in zip(("train", "validation", "test"), indices):
        subset = df.iloc[index]
        output[name] = {"n_environments": int(subset.scenario.nunique()), "n_rows": len(subset),
                        "n_positive": int(subset.label.sum()), "prevalence": float(subset.label.mean())}
    if "source_id" in df:
        source_sets = [set(df.iloc[index].source_id) for index in indices]
        output["overlapping_sources"] = sum(len(source_sets[i] & source_sets[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
        if output["overlapping_sources"]:
            raise AssertionError("Source statement variants cross partitions")
    return output


def run_comparison(df: pd.DataFrame, feature_cols: list[str], tag: str = "main",
                   wandb_run: Any = None) -> tuple[list[dict], dict]:
    X = df[feature_cols].to_numpy(dtype=float)
    y = df.label.to_numpy(dtype=int)
    train, validation, test = split_dataset(df)
    clean = X[train][y[train] == 0]
    if len(clean) == 0:
        raise ValueError("Training partition contains no unmodified records")
    models: dict[str, Any] = {}
    scores: dict[str, np.ndarray] = {}
    validation_scores: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    scores["rule_baseline"] = _rule_scores(df.iloc[test])
    thresholds["rule_baseline"] = 0.5
    validation_scores["z_score"] = _zscore_scores(clean, X[validation])
    scores["z_score"] = _zscore_scores(clean, X[test])

    iso = _iforest().fit(clean)
    models["isolation_forest"] = iso
    validation_scores["isolation_forest"] = -iso.decision_function(X[validation])
    scores["isolation_forest"] = -iso.decision_function(X[test])

    scaler = StandardScaler().fit(clean)
    clean_scaled = scaler.transform(clean)
    if len(clean_scaled) > 25000:
        sample = np.random.default_rng(RANDOM_SEED).choice(len(clean_scaled), size=25000, replace=False)
        clean_scaled = clean_scaled[sample]
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1, n_jobs=N_JOBS).fit(clean_scaled)
    models["local_outlier_factor"] = {"scaler": scaler, "model": lof}
    validation_scores["local_outlier_factor"] = -lof.decision_function(scaler.transform(X[validation]))
    scores["local_outlier_factor"] = -lof.decision_function(scaler.transform(X[test]))

    n_positive = int(y[train].sum())
    scale_weight = int((y[train] == 0).sum()) / max(1, n_positive)
    supervised = {
        "logistic_regression": _lr(), "random_forest": _rf(),
        "hist_gradient_boosting": _hgb(),
        "xgboost": XGBClassifier(n_estimators=300, scale_pos_weight=scale_weight,
                                 random_state=RANDOM_SEED, n_jobs=N_JOBS, verbosity=0),
    }
    for name, model in supervised.items():
        print(f"Fitting {name}...", flush=True)
        model.fit(X[train], y[train])
        models[name] = model
        validation_scores[name] = model.predict_proba(X[validation])[:, 1]
        scores[name] = model.predict_proba(X[test])[:, 1]
    for name, val_scores in validation_scores.items():
        thresholds[name] = select_threshold(y[validation], val_scores)

    results = [_eval(name, y[test], model_scores, threshold=thresholds[name])
               for name, model_scores in scores.items()]
    baseline = _eval("prevalence_baseline", y[test], np.full(len(test), y[train].mean()), threshold=0.0)
    results.append(baseline)
    for result in results:
        print(f"{result['model']}: AP={result['average_precision']:.4f}, F1={result['f1']:.4f}, threshold={result['threshold']:.6g}")
    _log_table(wandb_run, f"{tag}/comparison_table", results)
    context = {"train": train, "validation": validation, "test": test,
               "models": models, "scores": scores, "thresholds": thresholds,
               "protocol": _partition_metadata(df, (train, validation, test), SPLIT_SEED),
               "xgboost_scale_pos_weight": scale_weight}
    return results, context


def _subset_summary(y: np.ndarray) -> dict:
    return {"n_test": len(y), "n_positive": int(y.sum()), "n_clean": int((y == 0).sum()),
            "prevalence": float(y.mean()), "random_average_precision": float(y.mean())}


def run_severity_breakdown(df: pd.DataFrame, context: dict, wandb_run: Any = None) -> list[dict]:
    test = df.iloc[context["test"]]
    y = test.label.to_numpy(dtype=int)
    rows = []
    for severity in SEVERITIES:
        mask = (test.severity == "none") | (test.severity == severity)
        use = mask.to_numpy()
        summary = {"severity": severity, "n_corrupted": int((test.severity == severity).sum()),
                   **_subset_summary(y[use])}
        for model, prefix in (("hist_gradient_boosting", "hgb"), ("isolation_forest", "iso")):
            ap = float(average_precision_score(y[use], context["scores"][model][use]))
            summary[f"{prefix}_average_precision"] = ap
            summary[f"{prefix}_pr_auc"] = ap
        rows.append(summary)
    _log_table(wandb_run, "severity/breakdown_table", rows)
    return rows


def run_type_holdout(df: pd.DataFrame, feature_cols: list[str], wandb_run: Any = None) -> list[dict]:
    X = df[feature_cols].to_numpy(dtype=float)
    y = df.label.to_numpy(dtype=int)
    types = df.corruption_type.to_numpy()
    train, validation, test = split_dataset(df)
    rows = []
    for held_out in CORRUPTION_TYPES:
        tr = train[types[train] != held_out]
        va = validation[(y[validation] == 0) | (types[validation] != held_out)]
        te = test[(y[test] == 0) | (types[test] == held_out)]
        if len(np.unique(y[tr])) != 2 or y[te].sum() == 0:
            continue
        model = _hgb().fit(X[tr], y[tr])
        threshold = select_threshold(y[va], model.predict_proba(X[va])[:, 1])
        metrics = _eval(held_out, y[te], model.predict_proba(X[te])[:, 1], threshold=threshold)
        rows.append({"held_out_type": held_out, "n_test_corrupted": int(y[te].sum()),
                     **_subset_summary(y[te]), "hgb_average_precision": metrics["average_precision"],
                     "hgb_pr_auc": metrics["average_precision"], "precision": metrics["precision"],
                     "recall": metrics["recall"], "f1": metrics["f1"], "threshold": threshold})
    _log_table(wandb_run, "type_holdout/table", rows)
    return rows


def run_feature_ablation(df: pd.DataFrame, context: dict, wandb_run: Any = None) -> list[dict]:
    y = df.label.to_numpy(dtype=int)
    train, validation, test = context["train"], context["validation"], context["test"]
    rows = []
    sets = {"value only": FEATURE_GROUPS["value"],
            "value + column": FEATURE_GROUPS["value"] + FEATURE_GROUPS["column"],
            "value + column + structural": ALL_FEATURES}
    for name, features in sets.items():
        if features == ALL_FEATURES:
            scores = context["scores"]["hist_gradient_boosting"]
            threshold = context["thresholds"]["hist_gradient_boosting"]
        else:
            X = df[features].to_numpy(dtype=float)
            model = _hgb().fit(X[train], y[train])
            threshold = select_threshold(y[validation], model.predict_proba(X[validation])[:, 1])
            scores = model.predict_proba(X[test])[:, 1]
        metrics = _eval(name, y[test], scores, threshold=threshold)
        rows.append({"feature_set": name, "n_features": len(features),
                     "hgb_average_precision": metrics["average_precision"],
                     "hgb_pr_auc": metrics["average_precision"], "f1": metrics["f1"]})
    _log_table(wandb_run, "ablation/table", rows)
    return rows


def run_repeated_splits(df: pd.DataFrame, seeds: list[int], context: dict, wandb_run: Any = None) -> dict:
    """Measure sensitivity to environment partitioning with fixed model parameters."""
    X = df[ALL_FEATURES].to_numpy(dtype=float)
    y = df.label.to_numpy(dtype=int)
    rows = []
    names = ("logistic_regression", "hist_gradient_boosting", "isolation_forest")
    for seed in seeds:
        train, validation, test = split_dataset(df, seed)
        for name in names:
            if seed == SPLIT_SEED:
                scores, threshold = context["scores"][name], context["thresholds"][name]
            else:
                model = {"logistic_regression": _lr, "hist_gradient_boosting": _hgb,
                         "isolation_forest": _iforest}[name]()
                if name == "isolation_forest":
                    model.fit(X[train][y[train] == 0])
                    val_scores = -model.decision_function(X[validation])
                    scores = -model.decision_function(X[test])
                else:
                    model.fit(X[train], y[train])
                    val_scores = model.predict_proba(X[validation])[:, 1]
                    scores = model.predict_proba(X[test])[:, 1]
                threshold = select_threshold(y[validation], val_scores)
            rows.append({"split_seed": seed, **_eval(name, y[test], scores, threshold=threshold)})
    summaries = []
    for name in names:
        subset = [row for row in rows if row["model"] == name]
        summary = {"model": name, "n_splits": len(subset)}
        for metric in ("average_precision", "f1", "precision", "recall"):
            values = [row[metric] for row in subset]
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summaries.append(summary)
    _log_table(wandb_run, "repeated/splits", rows)
    _log_table(wandb_run, "repeated/summary", summaries)
    return {"seeds": seeds, "runs": rows, "summary": summaries,
            "interpretation": "Sample standard deviation across grouped partitions; overlapping holdouts are not independent replicates."}


def low_prevalence_indices(y: np.ndarray, prevalence: float, seed: int = RANDOM_SEED) -> np.ndarray:
    """Keep all source test rows and downsample injections to a stated prevalence."""
    if not 0 < prevalence < 1:
        raise ValueError("Prevalence must be between zero and one")
    clean = np.flatnonzero(y == 0)
    injected = np.flatnonzero(y == 1)
    n_positive = min(len(injected), int(len(clean) * prevalence / (1 - prevalence)))
    if len(clean) == 0 or n_positive == 0:
        raise ValueError("Not enough clean or injected test records for this prevalence")
    chosen = np.random.default_rng(seed).choice(injected, size=n_positive, replace=False)
    index = np.concatenate([clean, chosen])
    return np.random.default_rng(seed + 1).permutation(index)


def run_low_prevalence(df: pd.DataFrame, context: dict, wandb_run: Any = None) -> list[dict]:
    test = df.iloc[context["test"]]
    y = test.label.to_numpy(dtype=int)
    rows = []
    for target in (0.01, 0.05):
        index = low_prevalence_indices(y, target)
        for name, scores in context["scores"].items():
            rows.append({"target_prevalence": target,
                         **_eval(name, y[index], scores[index], threshold=context["thresholds"][name])})
        rows.append({"target_prevalence": target,
                     **_eval("prevalence_baseline", y[index], np.full(len(index), target), threshold=0.0)})
    _log_table(wandb_run, "prevalence/sensitivity", rows)
    return rows


def log_feature_importance(model: RandomForestClassifier, wandb_run: Any = None) -> list[dict]:
    pairs = sorted(zip(ALL_FEATURES, model.feature_importances_), key=lambda item: -item[1])
    rows = [{"feature": feature, "importance": float(importance)} for feature, importance in pairs]
    _log_table(wandb_run, "feature_importance/rf_table", rows)
    return rows


def _load_cached_dataset() -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(DATASET_PATH)
    missing = set(PROVENANCE_COLUMNS) - set(df)
    if missing:
        raise ValueError(f"Cached dataset predates mutation provenance ({sorted(missing)} missing). Rebuild without --skip-build.")
    if not DATASET_STATS_PATH.exists():
        raise ValueError("Cached dataset has no build metadata. Rebuild without --skip-build.")
    metadata = json.loads(DATASET_STATS_PATH.read_text())
    if metadata.get("dataset_version") != 2:
        raise ValueError("Unsupported cached dataset version. Rebuild without --skip-build.")
    if metadata.get("source_fingerprint") != source_fingerprint():
        raise ValueError("Raw source data changed since the cached build. Rebuild without --skip-build.")
    if metadata.get("dataset_fingerprint") != dataset_fingerprint():
        raise ValueError("Cached dataset fingerprint differs from its metadata. Rebuild without --skip-build.")
    return df, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-envs", type=int, default=None, help="Build only the first N sorted environments")
    parser.add_argument("--skip-build", action="store_true", help="Use a provenance-bearing, source-verified cached dataset")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--wandb-project", default="awm-silent-defects")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--repeat-seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    if args.n_envs is not None and args.n_envs < 1:
        parser.error("--n-envs must be positive")
    if args.skip_build and args.n_envs is not None:
        parser.error("--n-envs controls rebuilding and cannot be combined with --skip-build")
    seeds = list(dict.fromkeys(args.repeat_seeds))
    df, dataset_metadata = _load_cached_dataset() if args.skip_build else build_dataset(args.n_envs)
    if args.build_only:
        return

    config = {"model_config": model_config(), "features": ALL_FEATURES,
              "protocol": _partition_metadata(df, split_dataset(df), SPLIT_SEED),
              "dataset_metadata": dataset_metadata,
              "repeat_seeds": seeds, "severities": list(SEVERITIES),
              "corruption_types": list(CORRUPTION_TYPES), "k_rank": K_RANK}
    config["implementation_fingerprint"] = {
        name: _file_sha256(Path(name)) for name in (
            "analysis/detect_silent_defects.py", "analysis/build_insert_audit.py", "uv.lock"
        )
    }
    run = None if args.no_wandb else wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                                                config=config, tags=["midterm", "controlled-injections"])
    try:
        comparison, context = run_comparison(df, ALL_FEATURES, wandb_run=run)
        rf = context["models"]["random_forest"]
        bundle = {"model": rf, "features": ALL_FEATURES,
                  "threshold": context["thresholds"]["random_forest"], "split_seed": SPLIT_SEED,
                  "model_config": config["model_config"],
                  "implementation_fingerprint": config["implementation_fingerprint"],
                  "tracking": ({"run_id": run.id, "url": run.url, "project": run.project,
                                "entity": run.entity, "mode": run.settings.mode}
                               if run is not None else {"mode": "disabled"}),
                  "dataset_fingerprint": dataset_metadata["dataset_fingerprint"]}
        joblib.dump(bundle, MODEL_PATH)
        output = {"dataset": {"n_environments": int(df.scenario.nunique()), "n_rows": len(df),
                              "n_corrupted": int(df.label.sum()), "corruption_rate": float(df.label.mean()),
                              "features": ALL_FEATURES, "dataset_fingerprint": dataset_metadata["dataset_fingerprint"]},
                  "dataset_metadata": dataset_metadata, "protocol": context["protocol"],
                  "model_config": config["model_config"],
                  "implementation_fingerprint": config["implementation_fingerprint"],
                  "tracking": bundle["tracking"],
                  "xgboost_scale_pos_weight": context["xgboost_scale_pos_weight"],
                  "main_comparison": comparison,
                  "prevalence_baseline_note": "Constant scores give AP equal to test prevalence; threshold zero predicts every record positive.",
                  "severity_breakdown": run_severity_breakdown(df, context, run),
                  "type_holdout": run_type_holdout(df, ALL_FEATURES, run),
                  "feature_ablation": run_feature_ablation(df, context, run),
                  "feature_importance": log_feature_importance(rf, run),
                  "repeated_grouped_splits": run_repeated_splits(df, seeds, context, run),
                  "low_prevalence_evaluation": run_low_prevalence(df, context, run),
                  "low_prevalence_note": "Simulated 1% and 5% injection prevalence; fixed validation thresholds, no retraining. Does not estimate natural defect prevalence.",
                  "label_definition": "0=unmodified executable source; 1=SQLite-accepted rule-based injection. Semantic validity has not been exhaustively manually annotated."}
        (RESULTS_DIR / "tier2_results.json").write_text(json.dumps(output, indent=2) + "\n")
        if run is not None:
            run.summary["dataset_fingerprint"] = dataset_metadata["dataset_fingerprint"]
            run.summary["model_bundle"] = str(MODEL_PATH)
            run.summary["test_average_precision"] = {row["model"]: row["average_precision"] for row in comparison}
        print(f"Saved benchmark results and detector bundle in {RESULTS_DIR}")
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
