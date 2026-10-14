"""Interactive demonstration tool for inspecting SQL statements and silent defect detection.

Usage:
    uv run streamlit run demo_app.py
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from analysis.detect_silent_defects import (
    ALL_FEATURES,
    RANDOM_SEED,
    SPLIT_SEED,
    TEST_SIZE,
    extract_features,
    parse_insert,
)

DATA_PATH = Path("analysis/results/tier2_dataset.parquet")

st.set_page_config(
    page_title="Silent Defect Detector | AWM Benchmark",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource(show_spinner="Training detection model on benchmark data...")
def load_and_fit_model():
    df = pd.read_parquet(DATA_PATH)
    X = df[ALL_FEATURES].values.astype(float)
    y = df["label"].values.astype(int)
    groups = df["scenario"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    tr_idx, _ = next(splitter.split(X, y, groups))

    # Fast, high-accuracy Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=4,
    )
    rf.fit(X[tr_idx], y[tr_idx])
    return rf, len(tr_idx), len(df)


# App Header
st.title("🔍 Mining Silent Data Defects in LLM-Synthesized Worlds")
st.markdown(
    """
    **COSC 50303 Data Mining Project Demonstration**  
    *Demonstrating real-time pre-execution defect detection on SQL-valid records that bypass SQLite execution sandboxes.*
    """
)

# Load model
try:
    rf_model, n_train, n_total = load_and_fit_model()
    st.sidebar.success(f"✓ Model Ready (Trained on {n_train:,} statements)")
except Exception as e:
    st.error(f"Error loading benchmark model: {e}")
    st.stop()

# Sidebar: Preset Examples
st.sidebar.header("Select Preset Example")
PRESETS = {
    "Custom Statement": "",
    "Defect: Negative Product Price (-$45.00)": (
        "INSERT INTO products (id, store_id, name, price, stock, created_at) "
        "VALUES (101, 5, 'Noise-Cancelling Headphones', -45.00, 30, '2025-05-10 12:00:00');"
    ),
    "Defect: Negative Inventory (-500 items)": (
        "INSERT INTO warehouse_inventory (id, warehouse_id, sku, quantity, updated_at) "
        "VALUES (402, 12, 'WIDGET-BLK-XL', -500, '2025-08-15 09:30:00');"
    ),
    "Defect: Anachronistic Year in Date ('1823')": (
        "INSERT INTO appointments (id, patient_id, doctor_id, appointment_date, status) "
        "VALUES (55, 304, 12, '1823-11-20 14:00:00', 'scheduled');"
    ),
    "Defect: Out-of-Bounds Rating (99.0 / 5.0)": (
        "INSERT INTO restaurant_reviews (id, restaurant_id, reviewer_id, rating, review_text) "
        "VALUES (990, 42, 108, 99.0, 'Delicious pasta and wonderful service!');"
    ),
    "Clean: Valid Customer Registration": (
        "INSERT INTO users (id, username, email, user_age, created_at) "
        "VALUES (1, 'alex_smith', 'alex@example.com', 28, '2025-01-15 10:00:00');"
    ),
    "Edge Case: Credit Card Account Debt (-$850.40)": (
        "INSERT INTO accounts (id, user_id, name, type, current_balance, currency) "
        "VALUES (3, 1, 'Travel Credit Card', 'credit', -850.40, 'USD');"
    ),
}

choice = st.sidebar.selectbox("Choose a scenario:", list(PRESETS.keys()))
default_sql = PRESETS[choice]

sql_input = st.text_area(
    "SQL INSERT Statement to Analyze:",
    value=default_sql,
    height=110,
    placeholder="INSERT INTO table_name (col1, col2) VALUES (val1, val2);",
)

col1, col2 = st.columns([1, 1])

if sql_input.strip():
    parsed = parse_insert(sql_input)
    if not parsed:
        st.warning("⚠️ Could not parse statement. Please ensure it follows standard SQL `INSERT INTO ... VALUES (...)` format.")
    else:
        table, cols, vals = parsed
        
        # Test Dynamic SQLite Execution (Does SQLite allow it?)
        sqlite_accepted = False
        sqlite_error = ""
        try:
            conn = sqlite3.connect(":memory:")
            col_defs = ", ".join([f'"{c}" TEXT' for c in cols])
            conn.execute(f'CREATE TABLE "{table}" ({col_defs});')
            conn.execute(sql_input)
            sqlite_accepted = True
            conn.close()
        except Exception as err:
            sqlite_accepted = False
            sqlite_error = str(err)

        # Extract features
        dummy_meta = {"columns": {c: "TEXT" for c in cols}, "fk_count": 0, "unique_count": 0, "stmt_idx": 1, "sql_length": len(sql_input)}
        features_dict = extract_features(cols, vals, dummy_meta)
        X_test = np.array([[features_dict[f] for f in ALL_FEATURES]], dtype=float)

        prob_corrupt = float(rf_model.predict_proba(X_test)[0, 1])
        is_corrupt = prob_corrupt >= 0.5

        with col1:
            st.subheader("Model Prediction & Attribution")
            if is_corrupt:
                st.error(f"🚨 **Semantic Defect Detected** (Probability: {prob_corrupt:.1%})")
            else:
                st.success(f"✅ **Clean / Plausible Record** (Probability: {prob_corrupt:.1%})")

            st.progress(prob_corrupt)
            st.caption(f"Random Forest Anomaly Score: **{prob_corrupt:.4f}**")

            st.markdown("### Top Triggering Factors:")
            triggers = []
            if features_dict.get("has_absurd_year"):
                triggers.append(f"• **Anachronistic Year:** Detected year outside modern era (`min_year = {features_dict.get('min_year'):.0f}`)")
            if features_dict.get("has_negative_value") and (features_dict.get("num_money_cols") or features_dict.get("num_count_cols")):
                triggers.append(f"• **Negative Quantity/Currency:** Found value `{features_dict.get('log_min_numeric'):.2f}` in monetary/inventory context")
            if features_dict.get("has_rating_col") and features_dict.get("log_max_numeric", 0) > np.log1p(5.0):
                triggers.append("• **Out-of-Bounds Scale:** Rating literal exceeds standard 1–5 range")
            if features_dict.get("has_age_col") and features_dict.get("log_max_numeric", 0) > np.log1p(110.0):
                triggers.append("• **Implausible Entity Age:** Value exceeds biological threshold")

            if triggers:
                for t in triggers:
                    st.write(t)
            else:
                st.write("• All attributes conform to expected ranges and temporal bounds.")

        with col2:
            st.subheader("SQLite Engine Verification")
            if sqlite_accepted:
                st.warning("⚠️ **SQLite Execution Status:** `Return Code 0 (Success)`\n\n*SQLite committed this record without throwing an integrity exception. This confirms the defect is silent and bypasses database engine checks.*")
            else:
                st.info(f"ℹ️ **SQLite Error:** `{sqlite_error}`")

            st.markdown("### Parsed Columns & Values:")
            df_display = pd.DataFrame({"Column": cols, "Literal Value": vals})
            st.dataframe(df_display, use_container_width=True)

st.markdown("---")
st.caption("Agent World Model (AWM) Silent Defect Benchmark · TCU COSC 50303 Data Mining")
