"""Run the final static-checker and grouped-stability checks for the pilot."""

import json
import re
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from awm.core.db import create_sqlite_database
from analysis.build_insert_audit import INSERT_RE, load_records, split_sql_list


DATA_DIR = Path("data")
RESULT_DIR = Path("analysis/results")
FEATURE_NAMES = [
    "sql_length",
    "column_count",
    "value_count",
    "column_value_count_delta",
    "unknown_column_count",
    "null_count",
    "string_literal_count",
    "numeric_literal_count",
    "datetime_expression_count",
    "apostrophe_count",
    "backslash_count",
    "statement_index",
    "table_column_count",
    "table_foreign_key_count",
    "table_unique_constraint_count",
]


def literal_key(value: str) -> str | None:
    value = value.strip()
    if value.upper() == "NULL" or "(" in value:
        return None
    return value


def schema_metadata(connection: sqlite3.Connection) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    for (table_name,) in tables:
        columns = connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        unique_groups: list[list[str]] = []
        for index in connection.execute(f'PRAGMA index_list("{table_name}")').fetchall():
            if index[2]:
                unique_groups.append([
                    row[2] for row in connection.execute(f'PRAGMA index_info("{index[1]}")').fetchall()
                ])
        primary_key = [row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5]]
        if primary_key:
            unique_groups.append(primary_key)
        foreign_keys: dict[int, list[tuple[str, str, str]]] = defaultdict(list)
        for row in connection.execute(f'PRAGMA foreign_key_list("{table_name}")').fetchall():
            foreign_keys[row[0]].append((row[3], row[2], row[4]))
        metadata[table_name] = {
            "columns": {row[1] for row in columns},
            "unique_groups": [group for group in unique_groups if group],
            "foreign_keys": list(foreign_keys.values()),
        }
    return metadata


def static_flags(statement: str, table: str, metadata: dict[str, dict], seen: dict[str, dict[str, set[tuple[str, ...]]]]) -> list[str]:
    match = INSERT_RE.match(statement)
    if not match:
        return ["unparseable_insert"]
    columns = [column.strip(' "`[]') for column in split_sql_list(match.group(5) or "")]
    values = split_sql_list(match.group(6) or "")
    table_meta = metadata.get(table)
    if table_meta is None:
        return ["unknown_table"]
    flags: list[str] = []
    if len(columns) != len(values):
        flags.append("column_value_mismatch")
    unknown = set(columns) - table_meta["columns"]
    if unknown:
        flags.append("unknown_column")
    if statement.count("'") % 2:
        flags.append("unbalanced_quote")
    if "\\'" in statement:
        flags.append("backslash_quote")
    values_by_column = {
        column: literal_key(value)
        for column, value in zip(columns, values)
    }
    for unique_group in table_meta["unique_groups"]:
        key = tuple(values_by_column.get(column) for column in unique_group)
        if all(value is not None for value in key) and key in seen[table]["unique"]:
            flags.append("duplicate_prior_unique_key")
        elif all(value is not None for value in key):
            seen[table]["unique"].add(key)
    for foreign_key in table_meta["foreign_keys"]:
        source_columns = [item[0] for item in foreign_key]
        target_table = foreign_key[0][1]
        target_columns = [item[2] for item in foreign_key]
        key = tuple(values_by_column.get(column) for column in source_columns)
        if all(value is not None for value in key):
            target_key = tuple(target_columns)
            if key not in seen[target_table]["keys"].get(target_key, set()):
                flags.append("missing_prior_foreign_key")
    blocking_flags = {
        "unparseable_insert",
        "unknown_table",
        "column_value_mismatch",
        "unknown_column",
        "unbalanced_quote",
        "backslash_quote",
        "duplicate_prior_unique_key",
        "missing_prior_foreign_key",
    }
    if set(flags) & blocking_flags:
        return sorted(set(flags))
    for unique_group in table_meta["unique_groups"]:
        key = tuple(values_by_column.get(column) for column in unique_group)
        if all(value is not None for value in key):
            seen[table]["keys"].setdefault(tuple(unique_group), set()).add(key)
    return sorted(set(flags))


def categorize(error_message: str | None) -> str:
    message = (error_message or "").lower()
    if "foreign key" in message:
        return "foreign_key"
    if "unique constraint" in message or "primary key" in message:
        return "uniqueness_or_primary_key"
    if "no column named" in message:
        return "unknown_column"
    if "values for" in message or "has no column" in message:
        return "column_value_or_schema_mismatch"
    if "syntax error" in message or "unrecognized token" in message:
        return "syntax_or_quoting"
    return "other_sql_error"


def run_static_check() -> list[dict]:
    db_records = load_records(DATA_DIR / "gen_db.jsonl")
    sample_records = load_records(DATA_DIR / "gen_sample.jsonl")
    audit_rows = {
        (row["scenario"], row["table"], row["statement_index"]): row
        for row in (json.loads(line) for line in (RESULT_DIR / "insert_audit.jsonl").open())
    }
    output: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="awm-static-") as directory:
        for scenario, db_record in db_records.items():
            db_path, _, _, _ = create_sqlite_database(scenario, db_record["db_schema"], directory)
            connection = sqlite3.connect(db_path)
            metadata = schema_metadata(connection)
            connection.close()
            seen = defaultdict(lambda: {"unique": set(), "keys": {}})
            sample_tables = sample_records[scenario].get("sample_data", {}).get("tables", [])
            if not isinstance(sample_tables, list):
                continue
            for table_data in sample_tables:
                table = table_data.get("table_name", "unknown")
                for index, statement in enumerate(table_data.get("insert_statements", [])):
                    key = (scenario, table, index)
                    flags = static_flags(statement, table, metadata, seen)
                    actual = audit_rows[key]
                    output.append({
                        "scenario": scenario,
                        "table": table,
                        "statement_index": index,
                        "static_flags": flags,
                        "static_predicted_failure": bool(flags),
                        "actual_failure": actual["label"] == "failure",
                        "actual_error_category": categorize(actual["error_message"]),
                    })
    return output


def model_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    predictions = scores >= 0.5
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def run_repeated_splits() -> list[dict]:
    rows = [json.loads(line) for line in (RESULT_DIR / "insert_audit.jsonl").open()]
    x = np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float)
    y = np.asarray([row["label"] == "failure" for row in rows], dtype=int)
    groups = np.asarray([row["scenario"] for row in rows])
    outputs = []
    splitter = GroupShuffleSplit(n_splits=5, test_size=0.20, random_state=42)
    for split, (train, test) in enumerate(splitter.split(x, y, groups), start=1):
        test_y = y[test]
        result = {"split": split, "train_environments": len(set(groups[train])), "test_environments": len(set(groups[test])), "models": {}}
        result["models"]["majority"] = model_metrics(test_y, np.full(len(test), y[train].mean()))
        logistic = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))
        logistic.fit(x[train], y[train])
        result["models"]["logistic_regression"] = model_metrics(test_y, logistic.predict_proba(x[test])[:, 1])
        forest = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=42)
        forest.fit(x[train], y[train])
        result["models"]["random_forest"] = model_metrics(test_y, forest.predict_proba(x[test])[:, 1])
        outputs.append(result)
    return outputs


def main() -> None:
    static_rows = run_static_check()
    static_tp = sum(row["static_predicted_failure"] and row["actual_failure"] for row in static_rows)
    static_fp = sum(row["static_predicted_failure"] and not row["actual_failure"] for row in static_rows)
    static_fn = sum(not row["static_predicted_failure"] and row["actual_failure"] for row in static_rows)
    categories = Counter(row["actual_error_category"] for row in static_rows if row["actual_failure"])
    flags = Counter(flag for row in static_rows for flag in row["static_flags"])
    environment_failures = Counter(row["scenario"] for row in static_rows if row["actual_failure"])
    top_20_share = sum(count for _, count in environment_failures.most_common(20)) / sum(environment_failures.values())
    result = {
        "static_checker": {
            "rows_checked": len(static_rows),
            "predicted_failure_count": static_tp + static_fp,
            "true_positive_count": static_tp,
            "false_positive_count": static_fp,
            "false_negative_count": static_fn,
            "precision": static_tp / (static_tp + static_fp) if static_tp + static_fp else 0,
            "recall": static_tp / (static_tp + static_fn) if static_tp + static_fn else 0,
            "flag_counts": dict(flags),
        },
        "error_categories": dict(categories),
        "failure_concentration": {
            "failed_statements": sum(environment_failures.values()),
            "failed_environments": len(environment_failures),
            "top_20_failure_share": top_20_share,
            "top_20_environments": environment_failures.most_common(20),
        },
        "repeated_grouped_splits": run_repeated_splits(),
    }
    (RESULT_DIR / "final_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    (RESULT_DIR / "static_checker_rows.jsonl").write_text("\n".join(json.dumps(row) for row in static_rows) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()