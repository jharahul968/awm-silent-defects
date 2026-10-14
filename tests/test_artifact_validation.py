"""Checks that the artifact auditor catches mismatched edits and leaves reviews blank."""

import json

import numpy as np
import pandas as pd
import pytest

from analysis.validate_artifacts import (
    _verify_sql_pair,
    confusion_counts,
    review_sample,
    validate_mutation,
)
from analysis.detect_silent_defects import extract_features, parse_insert


def mutation():
    return {
        "original_sql": "INSERT INTO products (id, price, stock) VALUES (1, 10, 3);",
        "sql_statement": "INSERT INTO products (id, price, stock) VALUES (1, -10, 3);",
        "label": 1, "column_index": 1, "corrupted_column": "price",
        "original_value": "10", "corrupted_value": "-10",
    }


def test_mutation_audit_rejects_an_extra_changed_column():
    row = mutation()
    validate_mutation(row)
    row["sql_statement"] = "INSERT INTO products (id, price, stock) VALUES (1, -10, 99);"
    with pytest.raises(ValueError, match="changed columns"):
        validate_mutation(row)


def test_confusion_counts_uses_injected_as_positive():
    counts = confusion_counts(np.array([0, 0, 1, 1, 1]), np.array([0, 1, 1, 1, 0]))
    assert counts["true_positives_count"] == 2
    assert counts["false_positives_count"] == 1
    assert counts["false_negatives_count"] == 1
    assert counts["true_negatives_count"] == 1
    assert counts["injected_test_samples"] == 3


def test_review_sampling_is_reproducible_and_does_not_invent_judgements():
    rows = [
        {"source_id": f"clean_{i}", "label": 0, "corruption_type": "none", "severity": "none"}
        for i in range(20)
    ]
    for kind in ("negative_money", "anomalous_date"):
        for severity in ("obvious", "medium", "subtle"):
            for i in range(3):
                rows.append({"source_id": f"{kind}_{severity}_{i}", "label": 1,
                             "corruption_type": kind, "severity": severity})
    frame = pd.DataFrame(rows)
    first, second = review_sample(frame), review_sample(frame)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 16
    assert (first.review_status == "unreviewed").all()
    assert (first.manual_judgement == "").all()
    assert (first.reviewer == "").all()


def test_pair_check_replays_source_state_and_recomputes_features():
    row = mutation()
    row.update({"source_id": json.dumps(["shop", "products", 0]), "scenario": "shop",
                "table": "products", "statement_index": 0,
                "corruption_type": "negative_money", "severity": "medium"})
    schemas = {"shop": {"db_schema": {"tables": [
        {"name": "products", "ddl": "CREATE TABLE products (id INTEGER PRIMARY KEY, price REAL, stock INTEGER)"},
    ]}}}
    samples = {"shop": {"sample_data": {"tables": [
        {"table_name": "products", "insert_statements": [row["original_sql"]]},
    ]}}}
    _, columns, values = parse_insert(row["sql_statement"])
    # SQLite INTEGER PRIMARY KEY creates no separate unique index.
    row["key_features"] = extract_features(columns, values, {
        "columns": {"id": "INTEGER", "price": "REAL", "stock": "INTEGER"},
        "fk_count": 0, "unique_count": 0, "stmt_idx": 0,
        "sql_length": len(row["sql_statement"]),
    })
    result = _verify_sql_pair(row, schemas, samples, list(row["key_features"]))
    assert result["sql_accepted"]
    assert result["features_recomputed"] == 26
    row["key_features"]["sql_length"] += 1
    with pytest.raises(ValueError, match="sql_length"):
        _verify_sql_pair(row, schemas, samples, list(row["key_features"]))
