"""Regression checks for meaningful demo output and traceable case examples."""

import pandas as pd
import pytest

from demo_app import check_sqlite_input, literal_range_checks, replay_before_source
from analysis.interpret_and_analyze_errors import extract_case


@pytest.fixture
def source_context():
    schema = {"db_schema": {"tables": [
        {"name": "stores", "ddl": "CREATE TABLE stores (id INTEGER PRIMARY KEY)"},
        {"name": "products", "ddl": (
            "CREATE TABLE products (id INTEGER PRIMARY KEY, store_id INTEGER "
            "REFERENCES stores(id), price REAL CHECK (price >= 0))"
        )},
    ]}}
    sample = {"sample_data": {"tables": [
        {"table_name": "stores", "insert_statements": ["INSERT INTO stores (id) VALUES (1)"]},
        {"table_name": "products", "insert_statements": [
            "INSERT INTO products (id, store_id, price) VALUES (9, 1, 25)",
        ]},
    ]}}
    source = {"table": "products", "statement_index": 0}
    return schema, sample, source


def test_rating_and_age_checks_ignore_unrelated_identifiers():
    assert literal_range_checks(["id", "rating", "user_age"], ["99999", "4.5", "28"]) == []
    checks = literal_range_checks(["id", "rating", "user_age"], ["99999", "5.5", "111"])
    assert [check["column"] for check in checks] == ["rating", "user_age"]
    assert [check["literal"] for check in checks] == ["5.5", "111"]


def test_sqlite_demo_preserves_actual_constraints(source_context):
    schema, sample, source = source_context
    accepted, _, meta, failures = check_sqlite_input(
        "INSERT INTO products (id, store_id, price) VALUES (9, 1, 25)", schema, sample, source
    )
    assert accepted
    assert meta["fk_count"] == 1
    assert failures == 0
    accepted, error, _, _ = check_sqlite_input(
        "INSERT INTO products (id, store_id, price) VALUES (9, 1, -25)", schema, sample, source
    )
    assert not accepted
    assert "CHECK" in error
    accepted, error, _, _ = check_sqlite_input(
        "INSERT INTO products (id, store_id, price) VALUES (9, 999, 25)", schema, sample, source
    )
    assert not accepted
    assert "FOREIGN KEY" in error


def test_replay_stops_before_source_without_duplicate_key(source_context):
    schema, sample, _ = source_context
    conn, _ = replay_before_source(schema, sample, "products", 0)
    try:
        assert conn.execute("SELECT id FROM stores").fetchall() == [(1,)]
        assert conn.execute("SELECT count(*) FROM products").fetchone()[0] == 0
    finally:
        conn.close()


def test_missing_source_cannot_be_reported_as_valid_context(source_context):
    schema, sample, _ = source_context
    with pytest.raises(ValueError, match="not found"):
        replay_before_source(schema, sample, "products", 99)


def test_error_case_keeps_evaluated_mutation():
    before = "INSERT INTO reviews (id, rating) VALUES (2, 4.5);"
    after = "INSERT INTO reviews (id, rating) VALUES (2, 5.5);"
    row = pd.Series({
        "source_id": "test/reviews/0", "original_sql": before, "sql_statement": after,
        "corrupted_column": "rating", "column_index": 1, "original_value": "4.5",
        "corrupted_value": "5.5", "scenario": "test", "table": "reviews",
        "statement_index": 0, "label": 1, "prob": 0.1, "pred": 0,
        "corruption_type": "out_of_bounds_rating", "severity": "subtle", "has_rating_col": 1,
    })
    case = extract_case(row, "false_negative", ["has_rating_col"])
    assert case["original_sql"] == before
    assert case["sql_statement"] == after
    assert case["corrupted_column"] == "rating"
    row["sql_statement"] = None
    with pytest.raises(ValueError, match="missing"):
        extract_case(row, "false_negative", ["has_rating_col"])


def test_parallel_shap_matches_serial_and_preserves_sample_order():
    import numpy as np
    import shap
    from sklearn.ensemble import RandomForestClassifier
    from analysis.interpret_and_analyze_errors import compute_shap_values

    x = np.array([[0, 3], [1, 2], [2, 1], [3, 0], [4, 5], [5, 4]], dtype=float)
    y = np.array([0, 0, 1, 1, 0, 1])
    model = RandomForestClassifier(n_estimators=8, max_depth=3, random_state=17).fit(x, y)
    raw = shap.TreeExplainer(model).shap_values(x)
    expected = raw[1] if isinstance(raw, list) else raw[:, :, 1] if raw.ndim == 3 else raw
    serial = compute_shap_values(model, x, n_jobs=1)
    parallel = compute_shap_values(model, x, n_jobs=2)
    assert serial.shape == x.shape
    np.testing.assert_allclose(serial, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(parallel, expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="between 1 and 8"):
        compute_shap_values(model, x, n_jobs=0)
