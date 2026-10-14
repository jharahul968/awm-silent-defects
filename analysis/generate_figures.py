"""Generate publication-ready figures for the AWM Silent Defects project.

Reads from analysis/results/tier2_results.json and creates clean, high-DPI
figures in figures/ suitable for IEEE 2-column conference papers.

Figures generated:
  1. figures/fig1_model_comparison.png   - Average precision across all 8 models
  2. figures/fig2_severity_decay.png     - Performance decay across corruption severities
  3. figures/fig3_type_generalization.png- Leave-one-type-out generalization gap
  4. figures/fig4_feature_importance.png - Top-12 feature importance ranking
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RESULTS_DIR = Path("analysis/results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Clean academic style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_results() -> dict:
    with (RESULTS_DIR / "tier2_results.json").open() as f:
        return json.load(f)


def plot_model_comparison(results: dict) -> None:
    """Compare ranking metrics against the held-out positive prevalence."""
    models_data = [row for row in results["main_comparison"] if row["model"] != "prevalence_baseline"]
    
    # Pretty display names and grouping colors
    model_name_map = {
        "rule_baseline": "Rule Baseline",
        "z_score": "Z-Score",
        "isolation_forest": "Isolation Forest",
        "local_outlier_factor": "Local Outlier Factor",
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "hist_gradient_boosting": "HistGradientBoosting",
        "xgboost": "XGBoost",
    }
    categories = {
        "rule_baseline": "Heuristic",
        "z_score": "Statistical",
        "isolation_forest": "Unsupervised",
        "local_outlier_factor": "Unsupervised",
        "logistic_regression": "Supervised",
        "random_forest": "Supervised",
        "hist_gradient_boosting": "Supervised",
        "xgboost": "Supervised",
    }
    palette = {
        "Heuristic": "#8c8c8c",
        "Statistical": "#e69f00",
        "Unsupervised": "#56b4e9",
        "Supervised": "#009e73",
    }

    names = [model_name_map[m["model"]] for m in models_data]
    pr_aucs = [m["pr_auc"] for m in models_data]
    cats = [categories[m["model"]] for m in models_data]
    colors = [palette[c] for c in cats]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.barh(names, pr_aucs, color=colors, edgecolor="black", linewidth=0.6, height=0.65)
    
    # Value annotations on bars
    for bar in bars:
        w = bar.get_width()
        ax.text(w - 0.04, bar.get_y() + bar.get_height() / 2, f"{w:.4f}",
                ha="right", va="center", color="white" if w > 0.9 else "black",
                fontweight="bold", fontsize=8.5)

    # Prevalence baseline reference line
    prevalence = models_data[0]["prevalence"]
    ax.axvline(prevalence, color="#d55e00", linestyle="--", linewidth=1.2,
               label=f"Random Baseline ({prevalence:.2%})")

    ax.set_xlim(0, 1.01)
    ax.set_xlabel("Average precision")
    ax.set_title("Detection Performance Across 8 Models on Held-Out Environments")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    
    # Legend for categories
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[k], ec="black", lw=0.6) for k in palette]
    labels = list(palette.keys())
    handles.append(plt.Line2D([0], [0], color="#d55e00", linestyle="--", lw=1.2))
    labels.append(f"Prevalence ({prevalence:.1%})")
    ax.legend(handles, labels, loc="lower left", framealpha=0.9, edgecolor="none")

    out_path = FIGURES_DIR / "fig1_model_comparison.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_severity_decay(results: dict) -> None:
    """Figure 2: Performance breakdown across corruption severity levels."""
    sev_data = results["severity_breakdown"]
    
    severities = ["Obvious", "Medium", "Subtle"]
    hgb_scores = [d["hgb_pr_auc"] for d in sev_data]
    iso_scores = [d["iso_pr_auc"] for d in sev_data]

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    x = np.arange(len(severities))

    ax.plot(x, hgb_scores, marker="o", linewidth=2.0, color="#009e73", label="HistGradientBoosting (Supervised)")
    ax.plot(x, iso_scores, marker="s", linewidth=2.0, color="#d55e00", linestyle="--", label="Isolation Forest (Unsupervised)")
    ax.plot(x, [row["prevalence"] for row in sev_data], marker=".", color="gray",
            linestyle=":", label="Subset prevalence")

    for i, (h, s) in enumerate(zip(hgb_scores, iso_scores)):
        ax.annotate(f"{h:.4f}", (x[i], h), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, color="#009e73")
        ax.annotate(f"{s:.4f}", (x[i], s), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8.5, color="#d55e00")

    ax.set_xticks(x)
    ax.set_xticklabels(severities)
    ax.set_xlabel("Corruption Severity")
    ax.set_ylabel("Average precision")
    ax.set_ylim(0, 1.02)
    ax.set_title("Sensitivity vs. Corruption Subtlety")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="center left", framealpha=0.9)

    out_path = FIGURES_DIR / "fig2_severity_decay.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_type_generalization(results: dict) -> None:
    """Figure 3: Generalization performance when corruption types are unseen during training."""
    type_data = results["type_holdout"]
    
    type_name_map = {
        "negative_money": "Money",
        "negative_inventory": "Inventory",
        "anomalous_date": "Date",
        "out_of_bounds_rating": "Rating",
        "impossible_age": "Age",
    }

    # Sort from highest to lowest
    sorted_data = sorted(type_data, key=lambda x: x["hgb_pr_auc"], reverse=True)
    labels = [f"{type_name_map[d['held_out_type']]}\n(n+={d['n_test_corrupted']:,})" for d in sorted_data]
    scores = [d["hgb_pr_auc"] for d in sorted_data]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    colors = ["#0072b2"] * len(scores)
    bars = ax.bar(labels, scores, color=colors, edgecolor="black", linewidth=0.6, width=0.55)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.4f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.scatter(np.arange(len(sorted_data)), [row["prevalence"] for row in sorted_data],
               color="#d55e00", marker="_", s=250, zorder=3, label="Subset prevalence")
    ax.set_ylabel("Average precision on held-out type")
    ax.set_ylim(0, 1.08)
    ax.set_title("Transfer to corruption types excluded from training")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    
    ax.legend(loc="upper right", framealpha=0.9)

    out_path = FIGURES_DIR / "fig3_type_generalization.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_feature_importance(results: dict) -> None:
    """Figure 4: Top Random Forest Feature Importances."""
    fi_data = results["feature_importance"][:10]
    
    feat_name_map = {
        "min_year": "min_year (Earliest parsed year)",
        "log_min_numeric": "log_min_numeric (Signed min value)",
        "has_absurd_year": "has_absurd_year (Year <1900 or >2100)",
        "has_negative_value": "has_negative_value (Any negative numeric)",
        "num_negative_values": "num_negative_values (Count of negatives)",
        "max_string_length": "max_string_length",
        "num_string_values": "num_string_values",
        "num_money_cols": "num_money_cols (Schema keyword match)",
        "avg_string_length": "avg_string_length",
        "log_max_numeric": "log_max_numeric (Signed max value)",
    }

    labels = [feat_name_map.get(d["feature"], d["feature"]) for d in fi_data][::-1]
    importances = [d["importance"] for d in fi_data][::-1]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bars = ax.barh(labels, importances, color="#0072b2", edgecolor="black", linewidth=0.6, height=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.3f}",
                ha="left", va="center", fontsize=8.5)

    ax.set_xlim(0, max(importances) * 1.25)
    ax.set_xlabel("Gini Feature Importance (Random Forest)")
    ax.set_title("Top-10 Predictive Features for Silent Defect Detection")
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    out_path = FIGURES_DIR / "fig4_feature_importance.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    results = load_results()
    plot_model_comparison(results)
    plot_severity_decay(results)
    plot_type_generalization(results)
    plot_feature_importance(results)
    print("All figures successfully generated in figures/")


if __name__ == "__main__":
    main()
