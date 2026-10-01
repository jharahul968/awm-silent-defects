"""Evaluate pre-execution prediction of AWM seed SQL failures."""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


DATA_PATH = Path("analysis/results/insert_audit.jsonl")
OUTPUT_PATH = Path("analysis/results/predictability_results.json")


def load_rows() -> list[dict]:
    with DATA_PATH.open() as stream:
        return [json.loads(line) for line in stream]


def metrics(name: str, y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    predictions = scores >= threshold
    return {
        "model": name,
        "pr_auc": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "predicted_positive_count": int(predictions.sum()),
    }


def main() -> None:
    rows = load_rows()
    feature_names = [
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
    x = np.asarray([[row[name] for name in feature_names] for row in rows], dtype=float)
    y = np.asarray([row["label"] == "failure" for row in rows], dtype=int)
    groups = np.asarray([row["scenario"] for row in rows])

    first_split = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_indices, heldout_indices = next(first_split.split(x, y, groups))
    heldout_groups = groups[heldout_indices]
    second_split = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=43)
    validation_relative, test_relative = next(
        second_split.split(heldout_indices, y[heldout_indices], heldout_groups)
    )
    validation_indices = heldout_indices[validation_relative]
    test_indices = heldout_indices[test_relative]

    results = {
        "row_count": len(rows),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()),
        "group_counts": {
            "train": int(len(set(groups[train_indices]))),
            "validation": int(len(set(groups[validation_indices]))),
            "test": int(len(set(groups[test_indices]))),
        },
        "feature_names": feature_names,
        "models": [],
    }

    train_y = y[train_indices]
    test_y = y[test_indices]
    majority_score = np.full(len(test_indices), train_y.mean(), dtype=float)
    results["models"].append(metrics("majority", test_y, majority_score))

    static_rule = (
        (x[test_indices, 3] != 0)
        | (x[test_indices, 4] != 0)
        | (x[test_indices, 10] > 0)
        | ((x[test_indices, 9] % 2) == 1)
    )
    results["models"].append(metrics("static_rules", test_y, static_rule.astype(float), threshold=0.5))

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
    )
    logistic.fit(x[train_indices], train_y)
    results["models"].append(metrics("logistic_regression", test_y, logistic.predict_proba(x[test_indices])[:, 1]))

    forest = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    forest.fit(x[train_indices], train_y)
    results["models"].append(metrics("random_forest", test_y, forest.predict_proba(x[test_indices])[:, 1]))

    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()