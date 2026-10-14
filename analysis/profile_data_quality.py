"""Data quality, missingness, and distribution profiling for synthetic databases.

Analyzes attribute sparsity, class balance, and environment-level instance depth
across the benchmark dataset.

Outputs:
  - analysis/results/data_quality_report.json
  - figures/fig0_data_quality.png
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.detect_silent_defects import ALL_FEATURES, parse_insert

DATA_PATH = Path("analysis/results/tier2_dataset.parquet")
RESULTS_DIR = Path("analysis/results")
FIGURES_DIR = Path("figures")


def profile_raw_data(data_dir: Path = Path("data")) -> dict:
    """Count source SQL literals and declared schema types before augmentation.

    SQL NULL counts measure explicit missing literals. Omitted columns may have
    defaults, so they are counted separately rather than classified as missing.
    """
    db_records = [json.loads(line) for line in (data_dir / "gen_db.jsonl").open()]
    column_counts = {}
    declared_types = Counter()
    schema_counts = Counter()
    for record in db_records:
        with sqlite3.connect(":memory:") as conn:
            for table in record.get("db_schema", {}).get("tables", []):
                schema_counts["tables"] += 1
                ddl = table["ddl"]
                schema_counts["tables_with_check"] += bool(re.search(r"\bCHECK\s*\(", ddl, re.I))
                try:
                    conn.execute(ddl)
                    escaped_name = table["name"].replace('"', '""')
                    info = conn.execute(f'PRAGMA table_info("{escaped_name}")').fetchall()
                except sqlite3.Error:
                    schema_counts["ddl_errors"] += 1
                    continue
                column_counts[(record["scenario"], table["name"])] = len(info)
                declared_types.update((row[2] or "UNDECLARED").upper() for row in info)

    counts = Counter()
    environment_count = 0
    with (data_dir / "gen_sample.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            environment_count += 1
            tables = record.get("sample_data", {}).get("tables", [])
            if not isinstance(tables, list) or not tables:
                counts["malformed_payloads"] += 1
                continue
            for table in tables:
                for statement in table.get("insert_statements", []):
                    counts["insert_statements"] += 1
                    parsed = parse_insert(statement)
                    if parsed is None:
                        counts["unparsed_statements"] += 1
                        continue
                    table_name, cols, values = parsed
                    counts["parsed_statements"] += 1
                    if not cols:
                        counts["implicit_column_lists"] += 1
                    elif len(cols) != len(values):
                        counts["column_value_mismatches"] += 1
                    else:
                        width = column_counts.get((record["scenario"], table_name))
                        if width is not None:
                            counts["omitted_column_slots"] += max(0, width - len(cols))
                    counts["value_literals"] += len(values)
                    missing = sum(value.strip().upper() == "NULL" for value in values)
                    counts["null_literals"] += missing
                    counts["statements_with_null"] += missing > 0
                    counts["statements_with_dynamic_datetime"] += bool(
                        re.search(r"\b(?:date|datetime|strftime)\s*\(", statement, re.I)
                    )
    return {
        "environment_records": environment_count,
        **dict(counts),
        "explicit_null_fraction": counts["null_literals"] / max(1, counts["value_literals"]),
        "declared_column_types": dict(declared_types.most_common()),
        "schema_counts": dict(schema_counts),
        "missingness_definition": "Explicit SQL NULL literals in parsed source INSERTs; omitted columns are separate and may have defaults.",
        "quality_limits": "Unmodified SQL-valid rows are reference negatives, not manually verified semantic ground truth. Dynamic date expressions are not expanded by the literal-year features.",
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading benchmark dataset from {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)

    # 1. Missingness & Zero-Density Analysis
    print("Profiling missingness and sparsity ...")
    null_analysis = {}
    for col in ALL_FEATURES:
        null_count = int(df[col].isnull().sum())
        zero_count = int((df[col] == 0).sum())
        null_analysis[col] = {
                "null_count": null_count,
                "null_pct": float(null_count / len(df)),
                "zero_count": zero_count,
                "zero_pct": float(zero_count / len(df)),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
        }

    # 2. Class Imbalance & Defect Distribution
    label_counts = df["label"].value_counts().to_dict()
    corr_breakdown = df[df["label"] == 1]["corruption_type"].value_counts().to_dict()
    sev_breakdown = df[df["label"] == 1]["severity"].value_counts().to_dict()

    # 3. Environment-Level Variance
    env_sizes = df.groupby("scenario").size()

    report = {
        "raw_data": profile_raw_data(),
        "dataset_summary": {
            "total_instances": len(df),
            "total_environments": int(df["scenario"].nunique()),
            "clean_instances": int(label_counts.get(0, 0)),
            "corrupted_instances": int(label_counts.get(1, 0)),
            "corruption_rate": float(df["label"].mean()),
            "overall_imbalance_ratio": f"{label_counts.get(1, 0)}:{label_counts.get(0, 0)}",
            "statements_per_env_mean": float(env_sizes.mean()),
            "statements_per_env_median": float(env_sizes.median()),
            "statements_per_env_std": float(env_sizes.std()),
            "statements_per_env_min": int(env_sizes.min()),
            "statements_per_env_max": int(env_sizes.max()),
        },
        "corruption_distribution": {
            "by_type": {k: int(v) for k, v in corr_breakdown.items()},
            "by_severity": {k: int(v) for k, v in sev_breakdown.items()},
        },
        "feature_profiling": null_analysis,
    }

    report_path = RESULTS_DIR / "data_quality_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {report_path}")

    # 4. Generate Publication Quality Figure for Report (Fig 0)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))

    # Subplot A: Corruption Type Breakdown
    ax1 = axes[0]
    type_keys = ["anomalous_date", "negative_money", "negative_inventory", "out_of_bounds_rating", "impossible_age"]
    counts = [corr_breakdown.get(k, 0) for k in type_keys]
    bars = ax1.bar(range(len(counts)), [c / 1000 for c in counts], color="#0072b2", edgecolor="black", lw=0.6)
    ax1.set_xticks(range(len(counts)))
    ax1.set_xticklabels(["Date", "Money", "Inventory", "Rating", "Age"], fontsize=8.5)
    ax1.set_ylabel("Instances (Thousands)")
    ax1.set_title("(a) Defect Category Volumes", fontsize=9.5)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 8, f"{h:.1f}k", ha="center", va="bottom", fontsize=7.5)

    # Subplot B: Severity Balance
    ax2 = axes[1]
    sev_keys = ["obvious", "medium", "subtle"]
    sev_counts = [sev_breakdown.get(k, 0) for k in sev_keys]
    bars2 = ax2.bar(["Obvious", "Medium", "Subtle"], [s / 1000 for s in sev_counts], color="#e69f00", edgecolor="black", lw=0.6)
    ax2.set_ylabel("Instances (Thousands)")
    ax2.set_title("(b) Severity Partition Balance", fontsize=9.5)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.set_ylim(0, max(sev_counts) / 1000 * 1.25)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 3, f"{h:.1f}k", ha="center", va="bottom", fontsize=8.0)

    # Subplot C: Environment Size Distribution
    ax3 = axes[2]
    sns.histplot(env_sizes, bins=25, ax=ax3, color="#009e73", kde=True)
    ax3.set_xlabel("Statements per Environment")
    ax3.set_ylabel("Environment Count")
    ax3.set_title("(c) Environment Instance Depth", fontsize=9.5)
    ax3.grid(axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig0_path = FIGURES_DIR / "fig0_data_quality.png"
    plt.savefig(fig0_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig0_path}")


if __name__ == "__main__":
    main()
