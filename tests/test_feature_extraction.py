"""Unit tests for the 26 pre-execution feature extraction pipeline."""

from __future__ import annotations

import pytest

from analysis.detect_silent_defects import ALL_FEATURES, extract_features


def test_extract_features_count_and_keys() -> None:
    cols = ["id", "price", "stock", "created_at"]
    vals = ["1", "49.99", "10", "'2025-05-01 12:00:00'"]
    schema_meta = {
        "columns": {"id": "INT", "price": "REAL", "stock": "INT", "created_at": "TEXT"},
        "fk_count": 0,
        "unique_count": 1,
        "stmt_idx": 0,
        "sql_length": 80,
    }

    feats = extract_features(cols, vals, schema_meta)

    assert len(feats) == len(ALL_FEATURES)
    for f in ALL_FEATURES:
        assert f in feats
        assert isinstance(feats[f], (int, float))


def test_extract_features_negative_value_detection() -> None:
    cols = ["id", "price"]
    vals = ["1", "-25.50"]
    schema_meta = {"columns": {"id": "INT", "price": "REAL"}, "fk_count": 0, "unique_count": 1}

    feats = extract_features(cols, vals, schema_meta)
    assert feats["has_negative_value"] == 1.0
    assert feats["num_negative_values"] == 1.0
    assert feats["num_money_cols"] >= 1.0


def test_extract_features_absurd_year_detection() -> None:
    cols = ["id", "event_date"]
    vals = ["1", "'1823-01-15'"]
    schema_meta = {"columns": {"id": "INT", "event_date": "TEXT"}, "fk_count": 0, "unique_count": 0}

    feats = extract_features(cols, vals, schema_meta)
    assert feats["has_absurd_year"] == 1.0
    assert feats["min_year"] == 1823.0

