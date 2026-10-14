"""Unit tests for semantic corruption injection functions."""

from __future__ import annotations

import pytest

from analysis.detect_silent_defects import (
    CORRUPTION_TYPES,
    SEVERITIES,
    apply_corruption,
    candidate_corruptions,
)


def test_candidate_corruptions_money() -> None:
    cands = candidate_corruptions("unit_price", "29.99")
    assert any(c[0] == "negative_money" for c in cands)


def test_candidate_corruptions_inventory() -> None:
    cands = candidate_corruptions("stock_quantity", "100")
    assert any(c[0] == "negative_inventory" for c in cands)


def test_candidate_corruptions_date() -> None:
    cands = candidate_corruptions("created_at", "'2025-04-12 10:00:00'")
    assert any(c[0] == "anomalous_date" for c in cands)


def test_candidate_corruptions_rating() -> None:
    cands = candidate_corruptions("average_rating", "4.5")
    assert any(c[0] == "out_of_bounds_rating" for c in cands)


def test_candidate_corruptions_age() -> None:
    cands = candidate_corruptions("user_age", "35")
    assert any(c[0] == "impossible_age" for c in cands)


@pytest.mark.parametrize("sev", ["obvious", "medium", "subtle"])
def test_apply_corruption_money(sev: str) -> None:
    corrupted = apply_corruption("price", "50.0", "negative_money", sev)
    val = float(corrupted)
    assert val < 0


@pytest.mark.parametrize("severity", ["obvious", "medium", "subtle"])
def test_small_price_stays_negative_after_sql_roundtrip(severity: str) -> None:
    import sqlite3

    value = apply_corruption("unit_price", "0.00002", "negative_money", severity)
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE prices (price REAL)")
        conn.execute(f"INSERT INTO prices VALUES ({value})")
        assert conn.execute("SELECT price < 0 FROM prices").fetchone()[0] == 1


@pytest.mark.parametrize("sev", ["obvious", "medium", "subtle"])
def test_apply_corruption_inventory(sev: str) -> None:
    corrupted = apply_corruption("stock", "20", "negative_inventory", sev)
    val = int(corrupted)
    assert val < 0


@pytest.mark.parametrize("sev", ["obvious", "medium", "subtle"])
def test_apply_corruption_rating(sev: str) -> None:
    corrupted = apply_corruption("rating", "4.0", "out_of_bounds_rating", sev)
    val = float(corrupted)
    assert val > 5.0

