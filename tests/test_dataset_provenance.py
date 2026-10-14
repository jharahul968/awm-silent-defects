"""Small SQLite benchmark checks for mutation provenance and cache identity."""

import json
import sqlite3

import pytest

from analysis import detect_silent_defects as detector


@pytest.fixture
def benchmark_paths(tmp_path, monkeypatch):
    data = tmp_path / "data"
    results = tmp_path / "results"
    data.mkdir()
    results.mkdir()
    monkeypatch.setattr(detector, "DATA_DIR", data)
    monkeypatch.setattr(detector, "RESULTS_DIR", results)
    monkeypatch.setattr(detector, "DATASET_PATH", results / "tier2_dataset.parquet")
    monkeypatch.setattr(detector, "DATASET_STATS_PATH", results / "dataset_stats.json")
    schema = {"scenario": "shop_1", "db_schema": {"tables": [
        {"name": "products", "ddl": "CREATE TABLE products (id INTEGER PRIMARY KEY, price REAL CHECK(price >= -10), stock INTEGER CHECK(stock >= -10), created_at TEXT)", "indexes": []},
        {"name": "orders", "ddl": "CREATE TABLE orders (id INTEGER PRIMARY KEY, product_id INTEGER REFERENCES products(id))", "indexes": []},
    ]}}
    original = "INSERT INTO products (id, price, stock, created_at) VALUES (1, 10, 3, '1950-01-01');"
    samples = {"scenario": "shop_1", "sample_data": {"tables": [
        {"table_name": "products", "insert_statements": [original, original]},
        {"table_name": "orders", "insert_statements": ["INSERT INTO orders (id, product_id) VALUES (1, 1);"]},
    ]}}
    for filename, record in (("gen_db.jsonl", schema), ("gen_sample.jsonl", samples)):
        (data / filename).write_text(json.dumps(record) + "\n")
    return data, results


def test_savepoint_checks_restore_state_and_release_savepoint():
    connection = sqlite3.connect(":memory:")
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE prices (price INTEGER CHECK(price > 0))")
    assert detector._try_statement(cursor, "INSERT INTO prices VALUES (1)") == (True, None)
    assert detector._try_statement(cursor, "INSERT INTO prices VALUES (-1)") == (False, "IntegrityError")
    assert cursor.execute("SELECT COUNT(*) FROM prices").fetchone()[0] == 0
    assert not connection.in_transaction
    with pytest.raises(sqlite3.OperationalError, match="no such savepoint"):
        cursor.execute("RELEASE candidate")
    connection.close()


def test_dataset_preserves_source_and_actual_sql_and_counts_rejections(benchmark_paths):
    frame, metadata = detector.build_dataset()
    assert len(detector.ALL_FEATURES) == 26
    assert set(detector.PROVENANCE_COLUMNS).issubset(frame)
    stats = metadata["builder_stats"]
    assert stats["clean_rows"] == 2
    assert stats["skipped_failed_clean"] == 1
    assert stats["corruption_rejected_by_sqlite"] == 2
    assert stats["skipped_unchanged_mutations"] == 1
    assert len(frame[frame.table == "orders"]) == 1  # Source FK state was retained.
    assert (frame.sql_length == frame.sql_statement.str.len()).all()
    for _, row in frame.iterrows():
        if row.label == 0:
            assert row.sql_statement == row.original_sql
            assert row.column_index == -1
        else:
            _, columns, values = detector.parse_insert(row.sql_statement)
            assert columns[int(row.column_index)] == row.corrupted_column
            assert values[int(row.column_index)] == row.corrupted_value
            assert row.original_value != row.corrupted_value
    clean_sources = set(frame[frame.label == 0].source_id)
    assert set(frame[frame.label == 1].source_id).issubset(clean_sources)


def test_cached_dataset_rejects_source_drift(benchmark_paths):
    data, _ = benchmark_paths
    detector.build_dataset()
    cached, metadata = detector._load_cached_dataset()
    assert len(cached) > 0
    assert metadata["dataset_version"] == 2
    with (data / "gen_sample.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="source data changed"):
        detector._load_cached_dataset()
