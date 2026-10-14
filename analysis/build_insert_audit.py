"""Build a statement-level audit of AWM seed SQL data."""

import argparse
import json
import re
import shutil
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

from awm.core.db import create_sqlite_database


INSERT_RE = re.compile(
    r"^\s*INSERT\s+INTO\s+(?:\"([^\"]+)\"|`([^`]+)`|\[([^\]]+)\]|([\w.]+))"
    r"\s*(?:\((.*?)\))?\s*VALUES\s*\((.*)\)\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def load_records(path: Path) -> dict[str, dict]:
    with path.open() as stream:
        return {record["scenario"]: record for record in map(json.loads, stream)}


def split_sql_list(value: str) -> list[str]:
    """Split SQL values without splitting quoted strings or nested expressions.

    SQLite escapes a quote by doubling it. Consume the pair together so that
    commas later in the same literal do not become column separators.
    """
    items: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    index = 0
    while index < len(value):
        char = value[index]
        if quote:
            if char == quote:
                if index + 1 < len(value) and value[index + 1] == quote:
                    index += 2
                    continue
                quote = None
        elif char in "'\"`":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            items.append(value[start:index].strip())
            start = index + 1
        index += 1
    items.append(value[start:].strip())
    return items


def statement_features(
    statement: str,
    table_name: str,
    statement_index: int,
    table_info: dict[str, dict],
    foreign_keys: dict[str, int],
    unique_constraints: dict[str, int],
) -> dict:
    match = INSERT_RE.match(statement)
    columns_text = match.group(5) if match else None
    values_text = match.group(6) if match else None
    columns = split_sql_list(columns_text) if columns_text else []
    values = split_sql_list(values_text) if values_text else []
    literals = values if values_text else []
    known_columns = set(table_info.get(table_name, {}))
    normalized_columns = {column.strip(' "`[]') for column in columns}
    return {
        "sql_length": len(statement),
        "column_count": len(columns),
        "value_count": len(values),
        "column_value_count_delta": len(columns) - len(values),
        "unknown_column_count": len(normalized_columns - known_columns) if columns else 0,
        "null_count": sum(value.upper() == "NULL" for value in literals),
        "string_literal_count": sum(value.startswith("'") for value in literals),
        "numeric_literal_count": sum(bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value)) for value in literals),
        "datetime_expression_count": sum("datetime(" in value.lower() for value in literals),
        "apostrophe_count": statement.count("'"),
        "backslash_count": statement.count("\\"),
        "statement_index": statement_index,
        "table_column_count": len(table_info.get(table_name, {})),
        "table_foreign_key_count": foreign_keys.get(table_name, 0),
        "table_unique_constraint_count": unique_constraints.get(table_name, 0),
    }


def table_metadata(connection: sqlite3.Connection) -> tuple[dict, dict, dict]:
    table_info: dict[str, dict] = {}
    foreign_keys: dict[str, int] = {}
    unique_constraints: dict[str, int] = {}
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    for (table_name,) in tables:
        columns = connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        table_info[table_name] = {column[1]: column[2] for column in columns}
        foreign_keys[table_name] = len(connection.execute(f'PRAGMA foreign_key_list("{table_name}")').fetchall())
        unique_constraints[table_name] = sum(
            row[2] for row in connection.execute(f'PRAGMA index_list("{table_name}")').fetchall()
        )
    return table_info, foreign_keys, unique_constraints


def audit_scenario(scenario: str, db_record: dict, sample_record: dict, database_dir: Path) -> tuple[list[dict], dict]:
    db_path, _, _, _ = create_sqlite_database(scenario, db_record["db_schema"], str(database_dir))
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    table_info, foreign_keys, unique_constraints = table_metadata(connection)
    rows: list[dict] = []
    error_counts: Counter[str] = Counter()
    sample_data = sample_record.get("sample_data", {})
    tables = sample_data.get("tables") if isinstance(sample_data, dict) else None
    if not isinstance(tables, list):
        connection.close()
        return rows, {
            "scenario": scenario,
            "status": "malformed_sample_payload",
            "statement_count": 0,
            "failed_count": 0,
            "successful_count": 0,
            "error_types": {},
        }
    for table_data in tables:
        table_name = table_data.get("table_name", "unknown")
        statements = table_data.get("insert_statements", [])
        for statement_index, statement in enumerate(statements):
            try:
                connection.execute(statement)
                success = True
                error_type = None
                error_message = None
            except sqlite3.IntegrityError as error:
                success = False
                error_type = "IntegrityError"
                error_message = str(error)
            except sqlite3.Error as error:
                success = False
                error_type = "SQLError"
                error_message = str(error)
            if error_type:
                error_counts[error_type] += 1
            rows.append(
                {
                    "scenario": scenario,
                    "table": table_name,
                    "sql": statement,
                    "label": "success" if success else "failure",
                    "error_type": error_type,
                    "error_message": error_message,
                    **statement_features(statement, table_name, statement_index, table_info, foreign_keys, unique_constraints),
                }
            )
    connection.commit()
    connection.close()
    failures = sum(row["label"] == "failure" for row in rows)
    return rows, {
        "scenario": scenario,
        "status": "failed" if failures else "clean",
        "statement_count": len(rows),
        "failed_count": failures,
        "successful_count": len(rows) - failures,
        "error_types": dict(error_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    db_records = load_records(args.data_dir / "gen_db.jsonl")
    sample_records = load_records(args.data_dir / "gen_sample.jsonl")
    with tempfile.TemporaryDirectory(prefix="awm-audit-") as temporary_dir:
        database_dir = Path(temporary_dir)
        all_rows: list[dict] = []
        environment_rows: list[dict] = []
        for scenario, db_record in db_records.items():
            rows, environment = audit_scenario(scenario, db_record, sample_records[scenario], database_dir)
            all_rows.extend(rows)
            environment_rows.append(environment)
    with (args.output_dir / "insert_audit.jsonl").open("w") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")
    with (args.output_dir / "environment_audit.json").open("w") as stream:
        json.dump(environment_rows, stream, indent=2)
    summary = {
        "environment_count": len(environment_rows),
        "statement_count": len(all_rows),
        "failure_count": sum(row["label"] == "failure" for row in all_rows),
        "failure_environment_count": sum(row["status"] == "failed" for row in environment_rows),
        "malformed_payload_count": sum(row["status"] == "malformed_sample_payload" for row in environment_rows),
        "error_types": dict(Counter(row["error_type"] for row in all_rows if row["error_type"])),
    }
    with (args.output_dir / "audit_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    shutil.rmtree(database_dir, ignore_errors=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
