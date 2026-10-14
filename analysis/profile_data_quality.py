"""Data quality, missingness, and distribution profiling for synthetic databases.

Analyzes attribute sparsity, class balance, and environment-level instance depth
across the benchmark dataset.

Outputs:
  - analysis/results/data_quality_report.json
  - figures/fig0_data_quality.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_PATH = Path("analysis/results/tier2_dataset.parquet")
RESULTS_DIR = Path("analysis/results")
FIGURES_DIR = Path("figures")


def main() -> None:
    print(f"Loading benchmark dataset from {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)

    # 1. Missingness & Zero-Density Analysis
    print("Profiling missingness and sparsity ...")
    null_analysis = {}
    for col in df.columns:
        if col in ["scenario", "table", "corruption_type", "severity"]:
            null_count = int(df[col].isnull().sum())
            null_pct = float(null_count / len(df))
            null_analysis[col] = {"null_count": null_count, "null_percentage": null_pct}
        else:
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
        "dataset_summary": {
            "total_instances": len(df),
            "total_environments": int(df["scenario"].nunique()),
            "clean_instances": int(label_counts.get(0, 0)),
            "corrupted_instances": int(label_counts.get(1, 0)),
            "overall_imbalance_ratio": f"{label_counts.get(1, 0)}:{label_counts.get(0, 0)} (79.4% : 20.6%)",
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
