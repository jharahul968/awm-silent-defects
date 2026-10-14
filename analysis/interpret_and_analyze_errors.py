"""Explain the saved detector and preserve the exact SQL behind held-out cases.

Run after the main experiment:
    uv run python -m analysis.interpret_and_analyze_errors --no-wandb

Writes SHAP figures, a SHAP summary and a traceable error/success catalog.
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import wandb
from joblib import Parallel, delayed, parallel_config

from analysis.detect_silent_defects import (
    CORRUPTION_TYPES,
    RANDOM_SEED,
    dataset_fingerprint,
    load_model_bundle,
    split_dataset,
)

DATA_PATH = Path("analysis/results/tier2_dataset.parquet")
MODEL_PATH = Path("analysis/results/detector.joblib")
RESULTS_DIR = Path("analysis/results")
FIGURES_DIR = Path("figures")
PROVENANCE_COLUMNS = (
    "source_id", "original_sql", "sql_statement", "corrupted_column",
    "column_index", "original_value", "corrupted_value",
)


def _json_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def extract_case(row: pd.Series, case_type: str, features: list[str]) -> dict[str, Any]:
    """Keep original and mutated SQL separate; never substitute source SQL for a mutation."""
    if not isinstance(row["sql_statement"], str) or not row["sql_statement"].strip():
        raise ValueError("Case is missing its evaluated SQL statement; rebuild the benchmark.")
    case = {name: _json_value(row[name]) for name in PROVENANCE_COLUMNS}
    case.update({
        "case_type": case_type,
        "scenario": row["scenario"],
        "table": row["table"],
        "statement_index": int(row["statement_index"]),
        "actual_label": int(row["label"]),
        "predicted_prob": float(row["prob"]),
        "predicted_label": int(row["pred"]),
        "corruption_type": row["corruption_type"],
        "severity": row["severity"],
        "key_features": {name: float(row[name]) for name in features},
    })
    return case


def build_case_catalog(df_test: pd.DataFrame, features: list[str], metadata: dict) -> dict:
    label, predicted = df_test["label"], df_test["pred"]
    fp = (label == 0) & (predicted == 1)
    fn = (label == 1) & (predicted == 0)
    tp = (label == 1) & (predicted == 1)
    tn = (label == 0) & (predicted == 0)
    n_unmodified, n_injected = int((label == 0).sum()), int((label == 1).sum())
    catalog = {
        "metadata": metadata,
        "summary": {
            "total_test_samples": len(df_test),
            "unmodified_test_samples": n_unmodified,
            "injected_test_samples": n_injected,
            "false_positives_count": int(fp.sum()),
            "false_negatives_count": int(fn.sum()),
            "true_positives_count": int(tp.sum()),
            "true_negatives_count": int(tn.sum()),
            "false_positive_rate": float(fp.sum() / max(1, n_unmodified)),
            "false_negative_rate": float(fn.sum() / max(1, n_injected)),
        },
        "false_positive_cases": [],
        "false_negative_cases": [],
        "representative_true_positives": [],
        "representative_true_negatives": [],
        "detection_by_type": [],
    }
    for key, mask, ascending in (
        ("false_positive_cases", fp, False), ("false_negative_cases", fn, True),
        ("representative_true_negatives", tn, True),
    ):
        rows = df_test[mask].sort_values("prob", ascending=ascending).drop_duplicates("source_id").head(5)
        case_type = {"false_positive_cases": "false_positive", "false_negative_cases": "false_negative",
                     "representative_true_negatives": "true_negative"}[key]
        catalog[key] = [extract_case(row, case_type, features) for _, row in rows.iterrows()]
    for corruption_type in CORRUPTION_TYPES:
        rows = df_test[tp & (df_test["corruption_type"] == corruption_type)]
        obvious = rows[rows["severity"] == "obvious"]
        if not obvious.empty:
            rows = obvious
        if not rows.empty:
            catalog["representative_true_positives"].append(
                extract_case(rows.iloc[0], f"true_positive_{corruption_type}", features)
            )
        subset = df_test[(label == 1) & (df_test["corruption_type"] == corruption_type)]
        if not subset.empty:
            detected = int(subset["pred"].sum())
            catalog["detection_by_type"].append({
                "corruption_type": corruption_type,
                "test_count": len(subset),
                "detected_count": detected,
                "missed_count": len(subset) - detected,
                "recall": detected / len(subset),
            })
    return catalog


def _shap_chunk(model: Any, x: np.ndarray) -> np.ndarray:
    values = shap.TreeExplainer(model).shap_values(x)
    if isinstance(values, list):
        return np.asarray(values[1])
    if values.ndim == 3:
        return values[:, :, 1]
    return np.asarray(values)


def compute_shap_values(model: Any, x: np.ndarray, n_jobs: int = 4) -> np.ndarray:
    """Compute exact positive-class SHAP values in ordered, independent chunks."""
    if not 1 <= n_jobs <= 8:
        raise ValueError("SHAP worker count must be between 1 and 8.")
    if x.ndim != 2:
        raise ValueError("SHAP input must be a two-dimensional feature matrix.")
    if len(x) == 0:
        return np.empty_like(x, dtype=float)
    jobs = min(n_jobs, len(x))
    if jobs == 1:
        return _shap_chunk(model, x)
    chunks = np.array_split(x, jobs)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        values = Parallel(n_jobs=jobs)(delayed(_shap_chunk)(model, chunk) for chunk in chunks)
    return np.concatenate(values, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--sample-per-class", type=int, default=1000)
    parser.add_argument("--shap-jobs", type=int, default=4, help="Exact SHAP workers, 1–8 (default: 4)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="awm-silent-defects")
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()
    if args.sample_per_class < 1:
        parser.error("--sample-per-class must be positive")
    if not 1 <= args.shap_jobs <= 8:
        parser.error("--shap-jobs must be between 1 and 8")

    bundle = load_model_bundle(args.model_path)
    fingerprint = dataset_fingerprint(args.data_path)
    if fingerprint != bundle["dataset_fingerprint"]:
        raise ValueError("Dataset differs from the saved model's dataset; rerun the main experiment.")
    df = pd.read_parquet(args.data_path)
    missing = set(PROVENANCE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing SQL provenance {sorted(missing)}; rebuild the benchmark.")
    features = bundle["features"]
    train_idx, val_idx, test_idx = split_dataset(df, split_seed=bundle["split_seed"])
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)
    x_test = df_test[features].to_numpy(dtype=float)
    y_test = df_test["label"].to_numpy(dtype=int)
    model = bundle["model"]
    df_test["prob"] = model.predict_proba(x_test)[:, 1]
    df_test["pred"] = (df_test["prob"] >= bundle["threshold"]).astype(int)
    print(f"Saved RF: train={len(train_idx):,}, validation={len(val_idx):,}, test={len(test_idx):,}")

    rng = np.random.default_rng(args.seed)
    unmodified = np.flatnonzero(y_test == 0)
    injected = np.flatnonzero(y_test == 1)
    n_per_class = min(args.sample_per_class, len(unmodified), len(injected))
    if n_per_class == 0:
        raise ValueError("Both labels are required in the held-out SHAP sample.")
    sample_indices = np.concatenate([
        rng.choice(unmodified, n_per_class, replace=False),
        rng.choice(injected, n_per_class, replace=False),
    ])
    rng.shuffle(sample_indices)
    x_sample = x_test[sample_indices]
    values = compute_shap_values(model, x_sample, n_jobs=args.shap_jobs)
    mean_abs = np.abs(values).mean(axis=0)
    ranking = sorted(zip(features, mean_abs), key=lambda item: -item[1])
    metadata = {
        "model": "random_forest",
        "model_path": str(args.model_path),
        "model_config": bundle["model_config"],
        "features": features,
        "threshold": float(bundle["threshold"]),
        "split_seed": int(bundle["split_seed"]),
        "dataset_fingerprint": fingerprint,
        "shap_seed": args.seed,
        "shap_jobs": min(args.shap_jobs, len(x_sample)),
        "shap_method": "exact TreeExplainer, independent ordered chunks",
        "shap_sampling": "balanced held-out sample; not the natural benchmark class distribution",
        "shap_sample_per_class": n_per_class,
        "n_train": len(train_idx), "n_validation": len(val_idx), "n_test": len(test_idx),
    }
    summary = {
        "metadata": metadata,
        "sample_size": len(sample_indices),
        "ranking": [{"feature": name, "mean_abs_shap": float(value)} for name, value in ranking],
    }
    catalog = build_case_catalog(df_test, features, metadata)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "shap_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (RESULTS_DIR / "error_cases.json").write_text(json.dumps(catalog, indent=2) + "\n")

    shap.summary_plot(values, x_sample, feature_names=features, max_display=12, show=False)
    plt.title("Saved Random Forest: held-out SHAP values", fontsize=11)
    beeswarm_path = FIGURES_DIR / "fig5_shap_beeswarm.png"
    plt.savefig(beeswarm_path, bbox_inches="tight", dpi=300)
    plt.close()
    top = ranking[:10][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh([name for name, _ in top], [value for _, value in top], color="#e69f00")
    ax.set_xlabel("Mean absolute SHAP value (balanced held-out sample)")
    ax.set_title("Saved Random Forest feature attribution")
    bar_path = FIGURES_DIR / "fig5_shap_bar.png"
    fig.savefig(bar_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    if not args.no_wandb:
        with wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type="interpretation",
            name="saved-rf-shap-errors",
            config=metadata,
            tags=["interpretation", "shap", "error-analysis"],
        ) as run:
            run.log({
                "interpretation/shap_beeswarm": wandb.Image(str(beeswarm_path)),
                "interpretation/shap_bar": wandb.Image(str(bar_path)),
                "interpretation/shap_table": wandb.Table(
                    columns=["feature", "mean_abs_shap"],
                    data=[[item["feature"], item["mean_abs_shap"]] for item in summary["ranking"]],
                ),
                **{f"interpretation/{name}": value for name, value in catalog["summary"].items()},
            })
            cases = sum((catalog[group] for group in (
                "false_positive_cases", "false_negative_cases", "representative_true_positives",
                "representative_true_negatives",
            )), [])
            columns = ["case_type", "source_id", "scenario", "table", "predicted_prob",
                       "corruption_type", "severity", "corrupted_column", "original_value",
                       "corrupted_value", "original_sql", "sql_statement"]
            run.log({"interpretation/error_cases_table": wandb.Table(
                columns=columns, data=[[case[column] for column in columns] for case in cases],
            )})
            artifact = wandb.Artifact("saved-rf-interpretation", type="interpretation", metadata=metadata)
            for path in (RESULTS_DIR / "shap_summary.json", RESULTS_DIR / "error_cases.json", beeswarm_path, bar_path):
                artifact.add_file(str(path))
            run.log_artifact(artifact)
    print(f"Interpretation complete: {json.dumps(catalog['summary'])}")


if __name__ == "__main__":
    main()
