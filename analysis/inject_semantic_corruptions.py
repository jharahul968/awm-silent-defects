"""Pilot script for injecting and detecting semantic world-state corruptions in AWM."""

import json
import os
import re
import sqlite3
import copy
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from analysis.build_insert_audit import INSERT_RE, split_sql_list

DATA_DIR = Path("data")


def load_jsonl(path: Path, max_records: int = 30) -> list[dict]:
    records = []
    with path.open() as stream:
        for i, line in enumerate(stream):
            if i >= max_records:
                break
            records.append(json.loads(line))
    return records


def parse_statement(statement: str) -> tuple[str, list[str], list[str]] | None:
    match = INSERT_RE.match(statement)
    if not match:
        return None
    table = match.group(1) or match.group(2) or match.group(3) or match.group(4)
    cols_text = match.group(5)
    vals_text = match.group(6)
    cols = [c.strip(' "`[]') for c in split_sql_list(cols_text)] if cols_text else []
    vals = split_sql_list(vals_text) if vals_text else []
    return table, cols, vals


def reconstruct_statement(table: str, cols: list[str], vals: list[str]) -> str:
    cols_part = f" ({', '.join(cols)})" if cols else ""
    vals_part = ", ".join(vals)
    return f"INSERT INTO {table}{cols_part} VALUES ({vals_part});"


def try_corrupt_value(col_name: str, val_str: str) -> tuple[str, str] | None:
    """Attempt a semantic corruption on a value based on column semantics.
    Returns (corrupted_val_str, corruption_type) or None.
    """
    col_lower = col_name.lower()
    val_clean = val_str.strip()

    # Rule 1: Numeric monetary / non-negative quantities -> negative values
    money_words = ["price", "amount", "cost", "fee", "salary", "fare", "balance", "total", "budget"]
    if any(w in col_lower for w in money_words):
        try:
            num = float(val_clean)
            if num > 0:
                # Corrupt to negative price/amount
                return f"-{num:.2f}", "negative_money"
        except ValueError:
            pass

    # Rule 2: Quantities, counts, inventory, stock -> negative or absurd count
    count_words = ["quantity", "count", "stock", "inventory", "available_seats", "capacity"]
    if any(w in col_lower for w in count_words):
        try:
            num = int(val_clean)
            if num > 0:
                return f"-{num * 10}", "negative_inventory"
        except ValueError:
            pass

    # Rule 3: Human / entity attributes: age -> impossible age
    if col_lower in ["age", "user_age", "patient_age"]:
        try:
            num = int(val_clean)
            if 0 <= num <= 100:
                return "285", "impossible_age"
        except ValueError:
            pass

    # Rule 4: Ratings / reviews (scale usually 1-5 or 0-10) -> out of bounds
    if any(w in col_lower for w in ["rating", "score", "stars"]):
        try:
            num = float(val_clean)
            if 1.0 <= num <= 5.0:
                return "99.0", "out_of_bounds_rating"
        except ValueError:
            pass

    # Rule 5: Year / Dates in the extreme past or future
    date_match = re.search(r"'(\d{4})-(\d{2})-(\d{2})", val_clean)
    if date_match and any(w in col_lower for w in ["date", "time", "created", "scheduled", "birth", "due"]):
        year = int(date_match.group(1))
        if 1950 <= year <= 2030:
            # Corrupt to year 1823 (absurd historical) or 2999 (far future)
            new_val = val_clean.replace(f"'{year}-", "'1823-")
            return new_val, "anomalous_date"

    return None


def extract_row_features(table: str, cols: list[str], vals: list[str], col_types: dict[str, str]) -> list[float]:
    """Extract general numeric and structural features for a row."""
    has_negative_num = 0
    max_numeric = 0.0
    min_numeric = 0.0
    num_literals = 0
    has_absurd_year = 0
    has_money_col = 0
    has_count_col = 0
    has_date_col = 0

    money_words = ["price", "amount", "cost", "fee", "salary", "fare", "balance", "total", "budget"]
    count_words = ["quantity", "count", "stock", "inventory", "available_seats", "capacity"]
    date_words = ["date", "time", "created", "scheduled", "birth", "due"]

    for col, val in zip(cols, vals):
        col_lower = col.lower()
        val_clean = val.strip()

        if any(w in col_lower for w in money_words):
            has_money_col += 1
        if any(w in col_lower for w in count_words):
            has_count_col += 1
        if any(w in col_lower for w in date_words):
            has_date_col += 1

        # Check numeric
        try:
            num = float(val_clean)
            num_literals += 1
            if num < 0:
                has_negative_num += 1
            max_numeric = max(max_numeric, num)
            min_numeric = min(min_numeric, num)
        except ValueError:
            pass

        # Check year in date
        date_match = re.search(r"'(\d{4})-", val_clean)
        if date_match:
            year = int(date_match.group(1))
            if year < 1950 or year > 2050:
                has_absurd_year = 1

    return [
        float(len(cols)),
        float(len(vals)),
        float(has_negative_num),
        float(has_absurd_year),
        float(np.log1p(abs(max_numeric))),
        float(np.log1p(abs(min_numeric))),
        float(num_literals),
        float(has_money_col),
        float(has_count_col),
        float(has_date_col),
    ]


def run_pilot(num_scenarios: int = 25):
    print(f"=== Starting Semantic Corruption Pilot on {num_scenarios} Environments ===")
    
    db_items = load_jsonl(DATA_DIR / "gen_db.jsonl", max_records=num_scenarios)
    sample_items = load_jsonl(DATA_DIR / "gen_sample.jsonl", max_records=num_scenarios)

    db_map = {item["scenario"]: item for item in db_items}
    sample_map = {item["scenario"]: item for item in sample_items}

    dataset_rows = []
    scenarios_processed = 0
    total_valid_corruptions = 0
    sqlite_rejected_corruptions = 0

    for scenario_name, db_entry in db_map.items():
        if scenario_name not in sample_map:
            continue
        sample_entry = sample_map[scenario_name]
        sample_data = sample_entry.get("sample_data", {})
        tables_data = sample_data.get("tables", [])
        if not tables_data:
            continue

        # Create test DB in memory
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")

        # Run schema DDL
        for table in db_entry.get("db_schema", {}).get("tables", []):
            try:
                cursor.execute(table["ddl"])
                for idx in table.get("indexes", []):
                    cursor.execute(idx)
            except Exception:
                pass

        table_columns_meta = {}
        for t in db_entry.get("db_schema", {}).get("tables", []):
            t_name = t["name"]
            try:
                info = cursor.execute(f'PRAGMA table_info("{t_name}")').fetchall()
                table_columns_meta[t_name] = {row[1]: row[2] for row in info}
            except Exception:
                pass

        # Execute statements and test corruptions
        for table_item in tables_data:
            t_name = table_item.get("table_name", "")
            for stmt in table_item.get("insert_statements", []):
                parsed = parse_statement(stmt)
                if not parsed:
                    continue
                t, cols, vals = parsed
                if len(cols) != len(vals) or len(cols) == 0:
                    continue

                # First, test if the original clean statement executes cleanly
                cursor.execute("SAVEPOINT sp_test;")
                clean_success = False
                try:
                    cursor.execute(stmt)
                    clean_success = True
                except Exception:
                    clean_success = False
                cursor.execute("ROLLBACK TO SAVEPOINT sp_test;")

                if not clean_success:
                    # Statement was already failing in AWM
                    continue

                # Add the clean statement to dataset
                clean_feats = extract_row_features(t, cols, vals, table_columns_meta.get(t, {}))
                dataset_rows.append({
                    "scenario": scenario_name,
                    "table": t,
                    "statement": stmt,
                    "label": 0, # clean
                    "corruption_type": "none",
                    "features": clean_feats,
                })

                # Try to generate an eligible semantic corruption
                corrupted_applied = False
                for col_idx, (col, val) in enumerate(zip(cols, vals)):
                    corr = try_corrupt_value(col, val)
                    if corr and not corrupted_applied:
                        new_val, corr_type = corr
                        corrupted_vals = list(vals)
                        corrupted_vals[col_idx] = new_val
                        corrupted_stmt = reconstruct_statement(t, cols, corrupted_vals)

                        # TEST IF SQLITE ACCEPTS THE CORRUPTED ROW (IN PLACE OF CLEAN ROW)
                        cursor.execute("SAVEPOINT sp_corr;")
                        try:
                            cursor.execute(corrupted_stmt)
                            # SUCCESS: SQLite accepted it! (No CHECK constraint caught it)
                            cursor.execute("ROLLBACK TO SAVEPOINT sp_corr;")
                            total_valid_corruptions += 1
                            corrupted_applied = True

                            corr_feats = extract_row_features(t, cols, corrupted_vals, table_columns_meta.get(t, {}))
                            dataset_rows.append({
                                "scenario": scenario_name,
                                "table": t,
                                "statement": corrupted_stmt,
                                "label": 1, # corrupted
                                "corruption_type": corr_type,
                                "features": corr_feats,
                            })
                        except sqlite3.IntegrityError as e:
                            # CHECK constraint or FK caught it!
                            sqlite_rejected_corruptions += 1
                            if sqlite_rejected_corruptions <= 3:
                                print(f"[IntegrityError Sample]: {e} | Stmt: {corrupted_stmt[:100]}")
                            cursor.execute("ROLLBACK TO SAVEPOINT sp_corr;")
                        except Exception as e:
                            sqlite_rejected_corruptions += 1
                            cursor.execute("ROLLBACK TO SAVEPOINT sp_corr;")

                # Now commit the clean statement for real so foreign keys for subsequent tables work
                cursor.execute(stmt)

        conn.close()
        scenarios_processed += 1

    print(f"\nScenarios processed: {scenarios_processed}")
    print(f"Total dataset rows constructed: {len(dataset_rows)}")
    clean_count = sum(1 for r in dataset_rows if r["label"] == 0)
    corr_count = sum(1 for r in dataset_rows if r["label"] == 1)
    print(f"  Clean rows: {clean_count}")
    print(f"  Corrupted rows (accepted by SQLite): {corr_count}")
    print(f"  Corrupted candidates rejected by SQLite CHECK constraints: {sqlite_rejected_corruptions}")

    corr_breakdown = defaultdict(int)
    for r in dataset_rows:
        if r["label"] == 1:
            corr_breakdown[r["corruption_type"]] += 1
    print("\nCorruption type distribution:")
    for k, v in corr_breakdown.items():
        print(f"  - {k}: {v}")

    if corr_count < 10:
        print("\n[CRITICAL WARNING]: Too few valid corruptions were found. Heterogeneity or strict schemas prevented enough instances.")
        return

    # Now run ML evaluation on Grouped Split
    print("\n=== Evaluating Detection Models (Environment-Grouped Holdout) ===")
    x = np.asarray([r["features"] for r in dataset_rows], dtype=float)
    y = np.asarray([r["label"] for r in dataset_rows], dtype=int)
    groups = np.asarray([r["scenario"] for r in dataset_rows])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, test_idx = next(gss.split(x, y, groups))

    train_envs = len(set(groups[train_idx]))
    test_envs = len(set(groups[test_idx]))
    print(f"Train set: {len(train_idx)} rows across {train_envs} environments")
    print(f"Test set:  {len(test_idx)} rows across {test_envs} environments")
    print(f"Test set positive prevalence: {y[test_idx].mean():.4f}")

    # Baseline: Majority
    prev = y[test_idx].mean()
    maj_pr_auc = float(average_precision_score(y[test_idx], np.full(len(test_idx), prev)))

    # Model 1: Logistic Regression
    pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=42))
    pipe.fit(x[train_idx], y[train_idx])
    lr_probs = pipe.predict_proba(x[test_idx])[:, 1]
    lr_preds = pipe.predict(x[test_idx])
    lr_pr_auc = float(average_precision_score(y[test_idx], lr_probs))
    lr_prec = float(precision_score(y[test_idx], lr_preds, zero_division=0))
    lr_rec = float(recall_score(y[test_idx], lr_preds, zero_division=0))
    lr_f1 = float(f1_score(y[test_idx], lr_preds, zero_division=0))

    # Model 2: Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    rf.fit(x[train_idx], y[train_idx])
    rf_probs = rf.predict_proba(x[test_idx])[:, 1]
    rf_preds = rf.predict(x[test_idx])
    rf_pr_auc = float(average_precision_score(y[test_idx], rf_probs))
    rf_prec = float(precision_score(y[test_idx], rf_preds, zero_division=0))
    rf_rec = float(recall_score(y[test_idx], rf_preds, zero_division=0))
    rf_f1 = float(f1_score(y[test_idx], rf_preds, zero_division=0))

    # Model 3: Unsupervised Isolation Forest (Trained only on clean training rows!)
    clean_train_x = x[train_idx][y[train_idx] == 0]
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(clean_train_x)
    # IsolationForest outputs negative anomaly scores (lower = more anomalous)
    iso_scores = -iso.decision_function(x[test_idx])
    iso_pr_auc = float(average_precision_score(y[test_idx], iso_scores))

    print("\n--- Test Set Results ---")
    print(f"Majority Baseline PR-AUC:      {maj_pr_auc:.4f}")
    print(f"Isolation Forest PR-AUC:       {iso_pr_auc:.4f}")
    print(f"Logistic Regression PR-AUC:    {lr_pr_auc:.4f} | Prec: {lr_prec:.4f} | Rec: {lr_rec:.4f} | F1: {lr_f1:.4f}")
    print(f"Random Forest PR-AUC:          {rf_pr_auc:.4f} | Prec: {rf_prec:.4f} | Rec: {rf_rec:.4f} | F1: {rf_f1:.4f}")


if __name__ == "__main__":
    run_pilot(num_scenarios=25)
