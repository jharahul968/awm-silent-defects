"""Checks for validation selection and environment-level experiment isolation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

from analysis.detect_silent_defects import (
    _eval,
    _partition_metadata,
    _pr_at_k,
    low_prevalence_indices,
    model_config,
    select_threshold,
    split_dataset,
)


def test_split_keeps_original_test_holdout_and_source_families_together():
    rows = [
        {"scenario": f"env_{env:04}", "source_id": f"env_{env:04}:row_1", "label": label}
        for env in range(999) for label in (0, 1, 1)
    ]
    frame = pd.DataFrame(rows)
    train, validation, test = split_dataset(frame)
    _, original_test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
                            .split(frame, groups=frame.scenario))
    np.testing.assert_array_equal(test, original_test)
    partitions = [set(frame.iloc[index].scenario) for index in (train, validation, test)]
    assert [len(group) for group in partitions] == [599, 150, 250]
    assert not (partitions[0] & partitions[1] or partitions[0] & partitions[2]
                or partitions[1] & partitions[2])
    metadata = _partition_metadata(frame, (train, validation, test), 42)
    assert metadata["overlapping_sources"] == 0
    assert sum(metadata[name]["n_rows"] for name in ("train", "validation", "test")) == len(frame)


def test_validation_threshold_works_for_anomaly_scores_below_half():
    y = np.array([0, 0, 1, 1])
    scores = np.array([-0.1, -0.04, 0.03, 0.1])
    threshold = select_threshold(y, scores)
    assert threshold == pytest.approx(0.03)
    metrics = _eval("anomaly", y, scores, threshold=threshold)
    assert metrics["f1"] == 1.0
    assert metrics["average_precision"] == metrics["pr_auc"] == 1.0


def test_threshold_requires_both_classes():
    with pytest.raises(ValueError, match="both"):
        select_threshold(np.zeros(4), np.arange(4, dtype=float))


def test_precision_at_k_uses_available_record_count():
    precision, recall = _pr_at_k(np.array([1, 0, 1]), np.array([0.9, 0.1, 0.8]), 100)
    assert precision == pytest.approx(2 / 3)
    assert recall == 1


def test_prevalence_sensitivity_retains_clean_rows_and_is_reproducible():
    y = np.concatenate([np.zeros(1000, dtype=int), np.ones(4000, dtype=int)])
    first = low_prevalence_indices(y, 0.01)
    second = low_prevalence_indices(y, 0.01)
    np.testing.assert_array_equal(first, second)
    assert len(set(first)) == len(first)
    assert set(np.flatnonzero(y == 0)).issubset(first)
    assert y[first].sum() == 10
    assert y[first].mean() == pytest.approx(0.01, abs=0.0001)


def test_constant_score_reference_reports_prevalence_and_always_positive_f1():
    y = np.array([0, 0, 0, 1])
    metrics = _eval("prevalence_baseline", y, np.full(4, 0.3), threshold=0.0)
    assert metrics["average_precision"] == 0.25
    assert metrics["precision"] == 0.25
    assert metrics["recall"] == 1
    assert metrics["f1"] == pytest.approx(0.4)


def test_model_config_is_json_serializable():
    import json
    config = model_config()
    json.dumps(config, allow_nan=False)
    assert config["random_forest"]["n_estimators"] == 200
    assert config["local_outlier_factor"]["novelty"] is True
