"""Unit tests for SQL statement parsing and reconstruction."""

from __future__ import annotations

import pytest

from analysis.build_insert_audit import INSERT_RE, split_sql_list
from analysis.detect_silent_defects import parse_insert, reconstruct_insert


def test_split_sql_list_basic() -> None:
    items = split_sql_list("1, 'hello', 3.14, NULL")
    assert items == ["1", "'hello'", "3.14", "NULL"]


def test_split_sql_list_nested_quotes() -> None:
    items = split_sql_list("1, 'hello, world', 42")
    assert items == ["1", "'hello, world'", "42"]


def test_sqlite_doubled_quotes_and_nested_functions() -> None:
    import sqlite3

    statement = "INSERT INTO people (id, name, created) VALUES (1, 'O''Brien, Jr.', datetime('2025-01-01', '+1 day'));"
    table, columns, values = parse_insert(statement)
    assert values == ["1", "'O''Brien, Jr.'", "datetime('2025-01-01', '+1 day')"]
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE people (id INTEGER, name TEXT, created TEXT)")
        connection.execute(reconstruct_insert(table, columns, values))
        assert connection.execute("SELECT name, created FROM people").fetchone() == (
            "O'Brien, Jr.", "2025-01-02 00:00:00"
        )


def test_parse_insert_standard() -> None:
    stmt = "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30);"
    parsed = parse_insert(stmt)
    assert parsed is not None
    table, cols, vals = parsed
    assert table == "users"
    assert cols == ["id", "name", "age"]
    assert vals == ["1", "'Alice'", "30"]


def test_parse_insert_bracketed_columns() -> None:
    stmt = 'INSERT INTO "orders" ("order_id", "total_price") VALUES (101, 49.99);'
    parsed = parse_insert(stmt)
    assert parsed is not None
    table, cols, vals = parsed
    assert table == "orders"
    assert cols == ["order_id", "total_price"]
    assert vals == ["101", "49.99"]


def test_reconstruct_insert() -> None:
    reconstructed = reconstruct_insert("products", ["id", "price"], ["5", "19.99"])
    assert reconstructed == "INSERT INTO products (id, price) VALUES (5, 19.99);"

