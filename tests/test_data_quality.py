import json

from analysis.profile_data_quality import profile_raw_data


def test_missing_literals_are_distinct_from_omitted_columns(tmp_path):
    schema = {"scenario": "fixture", "db_schema": {"tables": [{
        "name": "products",
        "ddl": "CREATE TABLE products (id INTEGER, price REAL CHECK(price>=0), description TEXT DEFAULT 'unknown')",
    }]}}
    sample = {"scenario": "fixture", "sample_data": {"tables": [{
        "table_name": "products", "insert_statements": [
            "INSERT INTO products (id, price) VALUES (1, NULL);",
            "INSERT INTO products (id, price, description) VALUES (2, 10.0, 'O''Brien, Jr.');",
        ],
    }]}}
    for name, record in (("gen_db.jsonl", schema), ("gen_sample.jsonl", sample)):
        (tmp_path / name).write_text(json.dumps(record) + "\n")
    report = profile_raw_data(tmp_path)
    assert report["insert_statements"] == 2
    assert report["value_literals"] == 5
    assert report["null_literals"] == 1
    assert report["omitted_column_slots"] == 1
    assert report["explicit_null_fraction"] == 0.2
    assert report["schema_counts"]["tables_with_check"] == 1
