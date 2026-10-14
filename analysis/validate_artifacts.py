"""Check saved experiment artifacts and export unreviewed SQL pairs.

Run after the benchmark and interpretation scripts:
    python -m analysis.validate_artifacts

This verifies execution and provenance. It does not establish whether the
injected values violate a domain's business rules; the CSV leaves that
judgement for a reviewer.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.detect_silent_defects import (
    ALL_FEATURES,
    CORRUPTION_TYPES,
    DATA_DIR,
    RANDOM_SEED,
    SEVERITIES,
    _eval,
    _file_sha256,
    extract_features,
    load_model_bundle,
    parse_insert,
    source_fingerprint,
    split_dataset,
)
from demo_app import check_sqlite_input


RESULTS_DIR = Path("analysis/results")
CASE_GROUPS = (
    "false_positive_cases", "false_negative_cases",
    "representative_true_positives", "representative_true_negatives",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: float, expected: float, message: str) -> None:
    require(bool(np.isclose(actual, expected, rtol=1e-9, atol=1e-12)), message)


def confusion_counts(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    require(len(labels) == len(predictions), "Prediction and label lengths differ")
    return {
        "total_test_samples": len(labels),
        "unmodified_test_samples": int((labels == 0).sum()),
        "injected_test_samples": int((labels == 1).sum()),
        "false_positives_count": int(((labels == 0) & (predictions == 1)).sum()),
        "false_negatives_count": int(((labels == 1) & (predictions == 0)).sum()),
        "true_positives_count": int(((labels == 1) & (predictions == 1)).sum()),
        "true_negatives_count": int(((labels == 0) & (predictions == 0)).sum()),
    }


def validate_mutation(row: dict) -> None:
    """Check that an injected statement changes exactly its documented literal."""
    before, after = parse_insert(row["original_sql"]), parse_insert(row["sql_statement"])
    require(before is not None and after is not None, "A review pair cannot be parsed")
    source_table, source_columns, source_values = before
    table, columns, values = after
    require(table == source_table and columns == source_columns,
            "Mutation changed the INSERT table or columns")
    require(len(values) == len(source_values), "Mutation changed the number of values")
    if int(row.get("actual_label", row.get("label", 0))) == 0:
        require(row["sql_statement"] == row["original_sql"], "Unmodified row differs from its source")
        return
    index = int(row["column_index"])
    require(0 <= index < len(columns), "Corrupted column index is outside the statement")
    require(columns[index] == row["corrupted_column"], "Corrupted column name/index differ")
    require(source_values[index] == row["original_value"], "Original literal differs from provenance")
    require(values[index] == row["corrupted_value"], "Injected literal differs from provenance")
    changed = [i for i, pair in enumerate(zip(source_values, values)) if pair[0] != pair[1]]
    require(changed == [index], f"Mutation changed columns {changed}, expected only {index}")


def review_sample(frame: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """One test injection per type/severity plus ten unmodified test records."""
    rng = np.random.default_rng(seed)
    indices = []
    for corruption_type in CORRUPTION_TYPES:
        for severity in SEVERITIES:
            candidates = frame.index[(frame.corruption_type == corruption_type)
                                     & (frame.severity == severity)].to_numpy()
            if len(candidates):
                indices.append(int(rng.choice(candidates)))
    unmodified = frame[frame.label == 0].drop_duplicates("source_id").index.to_numpy()
    indices.extend(rng.choice(unmodified, size=min(10, len(unmodified)), replace=False).tolist())
    selected = frame.loc[indices].copy()
    selected["review_status"] = "unreviewed"
    selected["manual_judgement"] = ""
    selected["reviewer"] = ""
    selected["rationale"] = ""
    return selected


def _records_for_scenarios(scenarios: set[str]) -> tuple[dict, dict]:
    outputs = []
    for filename in ("gen_db.jsonl", "gen_sample.jsonl"):
        output = {}
        with (DATA_DIR / filename).open() as stream:
            for line in stream:
                record = json.loads(line)
                if record["scenario"] in scenarios:
                    output[record["scenario"]] = record
        require(set(output) == scenarios, f"Missing review environments in {filename}")
        outputs.append(output)
    return outputs[0], outputs[1]


def _verify_sql_pair(row: dict, schemas: dict, samples: dict, features: list[str]) -> dict:
    scenario = row["scenario"]
    schema, sample = schemas[scenario], samples[scenario]
    table, statement_index = row["table"], int(row["statement_index"])
    tables = [item for item in sample["sample_data"]["tables"] if item["table_name"] == table]
    require(len(tables) == 1, "Original source table is missing or duplicated")
    require(tables[0]["insert_statements"][statement_index] == row["original_sql"],
            "Original SQL differs from the raw source record")
    require(json.loads(row["source_id"]) == [scenario, table, statement_index],
            "Source ID differs from its environment/table/index")
    validate_mutation(row)
    accepted, error, metadata, prior_failures = check_sqlite_input(
        row["sql_statement"], schema, sample, row
    )
    require(accepted, f"Retained SQL rejected during source replay: {error}")
    original_accepted, error, _, _ = check_sqlite_input(row["original_sql"], schema, sample, row)
    require(original_accepted, f"Original SQL rejected during source replay: {error}")
    _, columns, values = parse_insert(row["sql_statement"])
    actual_features = extract_features(columns, values, metadata)
    expected_features = row.get("key_features", row)
    for feature in features:
        close(actual_features[feature], float(expected_features[feature]),
              f"Feature {feature} differs from the actual SQL for {row['source_id']}")
    return {"source_id": row["source_id"], "corruption_type": row["corruption_type"],
            "severity": row["severity"], "sql_accepted": True,
            "features_recomputed": len(features), "earlier_source_failures": prior_failures}


def verify(results_dir: Path = RESULTS_DIR, seed: int = RANDOM_SEED) -> dict:
    paths = {name: results_dir / filename for name, filename in (
        ("dataset", "tier2_dataset.parquet"), ("dataset_stats", "dataset_stats.json"),
        ("model", "detector.joblib"), ("results", "tier2_results.json"),
        ("cases", "error_cases.json"), ("shap", "shap_summary.json"),
    )}
    for name, path in paths.items():
        require(path.exists(), f"Required {name} artifact is missing: {path}")
    hashes = {name: _file_sha256(path) for name, path in paths.items()}
    metadata = json.loads(paths["dataset_stats"].read_text())
    results = json.loads(paths["results"].read_text())
    catalog = json.loads(paths["cases"].read_text())
    shap_summary = json.loads(paths["shap"].read_text())
    bundle = load_model_bundle(paths["model"])
    fingerprint = hashes["dataset"]
    checks = []

    require(source_fingerprint() == metadata["source_fingerprint"], "Raw dataset source fingerprints differ")
    for name, item in (("build metadata", metadata), ("model bundle", bundle),
                       ("results", results["dataset"]), ("SHAP", shap_summary["metadata"]),
                       ("case catalog", catalog["metadata"])):
        require(item["dataset_fingerprint"] == fingerprint, f"Dataset fingerprint differs in {name}")
    require(results["dataset_metadata"] == metadata, "Results/build metadata differ")
    checks.append("Dataset, source, model, SHAP and case catalog fingerprints agree")

    features = bundle["features"]
    require(features == ALL_FEATURES, "Saved feature ordering differs")
    for name, item in (("SHAP", shap_summary["metadata"]), ("case catalog", catalog["metadata"])):
        require(item["model_config"] == bundle["model_config"], f"Model configuration differs in {name}")
        require(item["features"] == features, f"Feature ordering differs in {name}")
        require(item["split_seed"] == bundle["split_seed"], f"Split seed differs in {name}")
        close(item["threshold"], bundle["threshold"], f"Decision threshold differs in {name}")
    require(results["model_config"] == bundle["model_config"], "Benchmark/bundle model configuration differs")
    require(bundle["model"].get_params() == bundle["model_config"]["random_forest"],
            "Saved RF parameters differ from the recorded configuration")
    require(bundle["model"].classes_.tolist() == [0, 1], "Saved RF class ordering differs")
    require(len(bundle["model"].estimators_) == 200, "Saved RF does not contain 200 fitted trees")
    require(shap_summary["sample_size"] == 2 * shap_summary["metadata"]["shap_sample_per_class"],
            "SHAP sample count differs from the recorded balanced sample")
    require({item["feature"] for item in shap_summary["ranking"]} == set(features),
            "SHAP ranking does not cover the saved features")
    require(all(np.isfinite(item["mean_abs_shap"]) and item["mean_abs_shap"] >= 0
                for item in shap_summary["ranking"]), "Invalid mean absolute SHAP values")
    checks.append("Interpretation uses the benchmark RF, features, threshold and split seed")

    frame = pd.read_parquet(paths["dataset"])
    train, validation, test = split_dataset(frame, bundle["split_seed"])
    indices = (train, validation, test)
    environment_sets = [set(frame.iloc[index].scenario) for index in indices]
    source_sets = [set(frame.iloc[index].source_id) for index in indices]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        require(not (environment_sets[i] & environment_sets[j]), "Environment partitions overlap")
        require(not (source_sets[i] & source_sets[j]), "Source variants cross partitions")
    counts = {"n_rows": len(frame), "n_environments": int(frame.scenario.nunique()),
              "n_corrupted": int(frame.label.sum())}
    for key, value in counts.items():
        require(results["dataset"][key] == value, f"Dataset count {key} differs")
    require(metadata["builder_stats"]["clean_rows"] == int((frame.label == 0).sum()), "Build/source row count differs")
    require(metadata["builder_stats"]["corrupted_rows"] == int(frame.label.sum()), "Build/injection row count differs")
    for name, index in zip(("train", "validation", "test"), indices):
        partition = frame.iloc[index]
        recorded = results["protocol"][name]
        require(recorded["n_rows"] == len(partition), f"{name} row count differs")
        require(recorded["n_environments"] == partition.scenario.nunique(), f"{name} environment count differs")
        require(recorded["n_positive"] == partition.label.sum(), f"{name} injection count differs")
        for interpretation in (catalog["metadata"], shap_summary["metadata"]):
            require(interpretation[f"n_{name}"] == len(partition), f"Interpretation {name} count differs")
    require((frame.sql_length == frame.sql_statement.str.len()).all(), "Stored SQL lengths differ from actual statements")
    checks.append("Dataset/partition counts agree; no environments or source variants cross partitions")

    heldout = frame.iloc[test].copy().reset_index(drop=True)
    labels = heldout.label.to_numpy(dtype=int)
    scores = bundle["model"].predict_proba(heldout[features].to_numpy(dtype=float))[:, 1]
    predictions = (scores >= bundle["threshold"]).astype(int)
    recomputed = _eval("random_forest", labels, scores, threshold=bundle["threshold"])
    rf_rows = [row for row in results["main_comparison"] if row["model"] == "random_forest"]
    require(len(rf_rows) == 1, "Main benchmark must contain one RF result")
    for name, value in recomputed.items():
        if isinstance(value, (int, float)):
            close(float(rf_rows[0][name]), value, f"Recomputed RF metric {name} differs")
    confusion = confusion_counts(labels, predictions)
    for name, value in confusion.items():
        require(catalog["summary"][name] == value, f"Error catalog count {name} differs")
    for name, numerator, denominator in (
        ("false_positive_rate", "false_positives_count", "unmodified_test_samples"),
        ("false_negative_rate", "false_negatives_count", "injected_test_samples"),
    ):
        close(catalog["summary"][name], confusion[numerator] / max(1, confusion[denominator]),
              f"Error catalog {name} differs")
    checks.append("Saved RF test predictions reproduce benchmark metrics and error confusion counts")

    heldout["predicted_prob"] = scores
    heldout["predicted_label"] = predictions
    selected = review_sample(heldout, seed)
    cases = [case for group in CASE_GROUPS for case in catalog[group]]
    require(cases, "Case catalog is empty")
    required_scenarios = {row["scenario"] for row in cases} | set(selected.scenario)
    schemas, samples = _records_for_scenarios(required_scenarios)
    case_checks = []
    for case in cases:
        matching = heldout[(heldout.source_id == case["source_id"])
                           & (heldout.sql_statement == case["sql_statement"])
                           & (heldout.corruption_type == case["corruption_type"])
                           & (heldout.severity == case["severity"])]
        require(len(matching) == 1, f"Case does not uniquely identify a held-out record: {case['source_id']}")
        actual = matching.iloc[0]
        require(int(case["actual_label"]) == int(actual.label), "Case label differs from benchmark")
        require(int(case["predicted_label"]) == int(actual.predicted_label), "Case prediction differs from saved RF")
        close(case["predicted_prob"], actual.predicted_prob, "Case score differs from saved RF")
        case_checks.append(_verify_sql_pair(case, schemas, samples, features))
    review_checks = [_verify_sql_pair(row.to_dict(), schemas, samples, features)
                     for _, row in selected.iterrows()]
    checks.append("Catalog cases and stratified review pairs execute under original SQLite state and reproduce their features")

    columns = ["source_id", "scenario", "table", "statement_index", "label", "corruption_type",
               "severity", "corrupted_column", "column_index", "original_value", "corrupted_value",
               "original_sql", "sql_statement", "predicted_prob", "predicted_label",
               "review_status", "manual_judgement", "reviewer", "rationale"]
    selected["statement_index"] = selected.statement_index.astype(int)
    selected["column_index"] = selected.column_index.astype(int)
    review_path = results_dir / "manual_review_pairs.csv"
    # Keep a completed review separate so a later rerun cannot erase judgements.
    if review_path.exists():
        previous = pd.read_csv(review_path, keep_default_na=False)
        if (previous.manual_judgement.str.strip() != "").any():
            review_path = results_dir / "manual_review_pairs_unreviewed.csv"
    selected[columns].to_csv(review_path, index=False)
    money = frame[frame.corruption_type == "negative_money"]
    rounded_zero = money[pd.to_numeric(money.corrupted_value) >= 0]
    recipe_findings = [{
        "finding": "A negative-money recipe rounded to a nonnegative literal",
        "source_id": row.source_id, "severity": row.severity,
        "corrupted_column": row.corrupted_column, "original_value": row.original_value,
        "corrupted_value": row.corrupted_value,
    } for _, row in rounded_zero.iterrows()]
    return {
        "status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": hashes, "source_sha256": metadata["source_fingerprint"],
        "dataset_counts": counts, "split_seed": bundle["split_seed"],
        "threshold": bundle["threshold"], "model_config": bundle["model_config"],
        "checks": checks, "recomputed_rf_metrics": recomputed, "confusion_counts": confusion,
        "catalog_case_checks": case_checks, "review_pair_checks": review_checks,
        "recipe_quality_findings": recipe_findings,
        "manual_review": {"path": str(review_path), "sha256": _file_sha256(review_path),
                          "n_pairs": len(selected), "sampling_seed": seed,
                          "sampling": "One held-out injection per available type/severity and ten unmodified source records",
                          "review_status": "unreviewed", "human_validation_completed": False},
        "scope": "Artifact identity, prediction consistency, SQL execution and mutation traceability; semantic business rules require human review.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    path = args.results_dir / "verification.json"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    try:
        report = verify(args.results_dir, args.seed)
    except (ValueError, KeyError, OSError, IndexError, sqlite3.Error) as error:
        report = {"status": "failed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                  "error": str(error), "human_validation_completed": False}
        path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Passed {len(report['checks'])} artifact checks; {report['manual_review']['n_pairs']} unreviewed pairs exported.")


if __name__ == "__main__":
    main()
