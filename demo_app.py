"""Inspect a SQL INSERT with the saved detector and an AWM database replay.

Run ``uv run streamlit run demo_app.py`` after generating the detector and
error catalog. Each preset identifies the original database state used for
its prediction; edited statements are checked in that same source context.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from analysis.detect_silent_defects import (
    AGE_WORDS,
    COUNT_WORDS,
    DATE_WORDS,
    MONEY_WORDS,
    RATING_WORDS,
    extract_features,
    load_model_bundle,
    parse_insert,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "analysis/results/detector.joblib"
CASES_PATH = ROOT / "analysis/results/error_cases.json"
DATA_DIR = ROOT / "data"


def literal_range_checks(cols: list[str], vals: list[str]) -> list[dict[str, str]]:
    """Apply column-aligned illustrative checks, independent of model prediction."""
    checks = []
    for col, value in zip(cols, vals):
        name = col.lower()
        try:
            number = float(value)
        except ValueError:
            number = None
        if number is not None and math.isfinite(number):
            message = None
            if any(word in name for word in MONEY_WORDS + COUNT_WORDS) and number < 0:
                message = "Negative money/count literal; debts and adjustments can be legitimate."
            if any(word in name for word in RATING_WORDS) and not 0 <= number <= 5:
                message = "Outside the illustrative 0–5 rating scale; verify this column's scale."
            if name in AGE_WORDS and not 0 <= number <= 110:
                message = "Outside the illustrative 0–110 age range."
            if message:
                checks.append({"column": col, "literal": value, "check": message})
        date = re.match(r"'(\d{4})-\d{2}-\d{2}", value.strip())
        if date and any(word in name for word in DATE_WORDS):
            year = int(date.group(1))
            if year < 1900 or year > 2100:
                checks.append({
                    "column": col,
                    "literal": value,
                    "check": "Outside the illustrative 1900–2100 year range; historical dates may be valid.",
                })
    return checks


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def replay_before_source(
    schema: dict[str, Any], sample: dict[str, Any], table: str, statement_index: int
) -> tuple[sqlite3.Connection, int]:
    """Replay the source order, stopping before the selected original statement."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    failed = 0
    try:
        for definition in schema["db_schema"]["tables"]:
            conn.execute(definition["ddl"])
            for index_sql in definition.get("indexes", []):
                conn.execute(index_sql)
        for source_table in sample["sample_data"]["tables"]:
            for idx, sql in enumerate(source_table.get("insert_statements", [])):
                if source_table["table_name"] == table and idx == statement_index:
                    return conn, failed
                try:
                    conn.execute(sql)
                except sqlite3.Error:
                    failed += 1
        raise ValueError("Selected source statement was not found in the source dataset.")
    except Exception:
        conn.close()
        raise


def check_sqlite_input(
    sql: str, schema: dict[str, Any], sample: dict[str, Any], source: dict[str, Any]
) -> tuple[bool, str, dict[str, Any], int]:
    """Check input against the source schema and state without writing a database."""
    parsed = parse_insert(sql)
    if parsed is None:
        raise ValueError("Expected an INSERT INTO ... VALUES (...) statement.")
    input_table, _, _ = parsed
    conn, prior_failures = replay_before_source(
        schema, sample, source["table"], int(source["statement_index"])
    )
    try:
        name = quote_identifier(input_table)
        columns = conn.execute(f"PRAGMA table_info({name})").fetchall()
        foreign_keys = conn.execute(f"PRAGMA foreign_key_list({name})").fetchall()
        indexes = conn.execute(f"PRAGMA index_list({name})").fetchall()
        if not columns:
            raise ValueError(f"Table {input_table!r} is absent from the selected environment.")
        meta = {
            "columns": {row[1]: row[2] for row in columns},
            "fk_count": len(foreign_keys),
            "unique_count": sum(bool(row[2]) for row in indexes),
            "stmt_idx": int(source["statement_index"]),
            "sql_length": len(sql),
        }
        try:
            conn.execute(sql)
            conn.commit()
            return True, "", meta, prior_failures
        except sqlite3.Error as exc:
            return False, str(exc), meta, prior_failures
    finally:
        conn.close()


@st.cache_resource
def load_detector(modified_ns: int) -> dict[str, Any]:
    return load_model_bundle(MODEL_PATH)


@st.cache_data
def load_cases(modified_ns: int) -> dict[str, Any]:
    return json.loads(CASES_PATH.read_text())


@st.cache_data
def source_records(scenario: str) -> tuple[dict[str, Any], dict[str, Any]]:
    records = []
    for filename in ("gen_db.jsonl", "gen_sample.jsonl"):
        with (DATA_DIR / filename).open() as handle:
            record = next(
                (item for line in handle if (item := json.loads(line))["scenario"] == scenario),
                None,
            )
        if record is None:
            raise ValueError(f"Environment {scenario!r} was not found in {filename}.")
        records.append(record)
    return records[0], records[1]


def example_presets(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    examples = {}
    for group in (
        "representative_true_positives", "representative_true_negatives",
        "false_positive_cases", "false_negative_cases",
    ):
        for idx, case in enumerate(catalog.get(group, []), start=1):
            label = f"{case['case_type']} {idx}: {case['scenario']} / {case['table']}"
            examples[label] = case
    return examples


def main() -> None:
    st.set_page_config(page_title="AWM defect detector", layout="wide")
    st.title("Inspect SQL records in an AWM environment")
    st.write(
        "Edit a source example to compare the saved Random Forest's score with "
        "SQLite execution under that environment's original schema and constraints."
    )
    if not MODEL_PATH.exists() or not CASES_PATH.exists():
        st.error("Generate the saved detector and its error catalog before starting this demo.")
        st.code(
            "uv run python -m analysis.detect_silent_defects --no-wandb\n"
            "uv run python -m analysis.interpret_and_analyze_errors --no-wandb"
        )
        st.stop()
    try:
        bundle = load_detector(MODEL_PATH.stat().st_mtime_ns)
        catalog = load_cases(CASES_PATH.stat().st_mtime_ns)
        metadata = catalog["metadata"]
        if (metadata["dataset_fingerprint"] != bundle["dataset_fingerprint"]
                or metadata["model_config"] != bundle["model_config"]
                or metadata["threshold"] != bundle["threshold"]
                or metadata["split_seed"] != bundle["split_seed"]):
            raise ValueError("Detector and error catalog differ; regenerate the interpretation artifacts.")
        presets = example_presets(catalog)
        if not presets:
            raise ValueError("The error catalog has no source examples.")
    except (OSError, ValueError, KeyError) as exc:
        st.error(str(exc))
        st.stop()

    choice = st.sidebar.selectbox("Source example", list(presets))
    source = presets[choice]
    st.sidebar.caption(f"Environment: {source['scenario']}")
    st.sidebar.caption(f"Source statement: {source['source_id']}")
    st.sidebar.caption(f"Saved decision threshold: {bundle['threshold']:.4f}")
    st.caption(
        f"Source label: {'injected mutation' if source['actual_label'] else 'unmodified source'}. "
        "Unmodified records have not been independently verified as semantically correct."
    )
    sql = st.text_area("SQL INSERT to inspect", value=source["sql_statement"], height=150, key=choice)
    if not sql.strip():
        return
    parsed = parse_insert(sql)
    if parsed is None or not parsed[1] or len(parsed[1]) != len(parsed[2]):
        st.warning("Enter a single INSERT INTO ... (columns) VALUES (...) with matching column/value counts.")
        return
    _, cols, vals = parsed
    try:
        schema, sample = source_records(source["scenario"])
        accepted, sqlite_error, meta, prior_failures = check_sqlite_input(sql, schema, sample, source)
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        st.error(f"Could not reconstruct the source database: {exc}")
        return

    features = extract_features(cols, vals, meta)
    x = np.array([[features[name] for name in bundle["features"]]], dtype=float)
    probability = float(bundle["model"].predict_proba(x)[0, 1])
    flagged = probability >= bundle["threshold"]
    model_column, database_column = st.columns(2)
    with model_column:
        st.subheader("Saved detector")
        st.metric("Injected-defect score", f"{probability:.2%}")
        if flagged:
            st.warning("Flagged for review at the validation-selected threshold.")
        else:
            st.info("Not flagged at this threshold. This is not proof of semantic correctness.")
        st.caption(
            "The score measures resemblance to the controlled benchmark's mutations; "
            "it is not a calibrated probability of a real-world defect."
        )
        st.subheader("Independent literal-range checks")
        st.caption("These are column-aligned heuristics, not SHAP attributions or explanations of this prediction.")
        checks = literal_range_checks(cols, vals)
        if checks:
            st.dataframe(pd.DataFrame(checks), hide_index=True, width="stretch")
        else:
            st.write("No illustrated range check fired; these checks cover only a few kinds of literals.")
    with database_column:
        st.subheader("SQLite execution in the selected environment")
        if accepted:
            st.success("INSERT accepted with the original schema and foreign-key checks enabled.")
        else:
            st.warning(f"INSERT rejected: {sqlite_error}")
        st.caption(
            "An in-memory database replays original records in source order up to this "
            "example. Acceptance confirms execution only; SQLite does not establish business semantics."
        )
        if prior_failures:
            st.caption(f"Earlier source INSERT failures skipped during replay: {prior_failures}.")
        st.dataframe(pd.DataFrame({"Column": cols, "Literal": vals}), hide_index=True, width="stretch")
    if source.get("corrupted_column"):
        st.caption(
            f"Original injected edit: {source['corrupted_column']}: "
            f"{source['original_value']} → {source['corrupted_value']}."
        )
    with st.expander("Original source SQL"):
        st.code(source["original_sql"], language="sql")


if __name__ == "__main__":
    main()
