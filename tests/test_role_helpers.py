"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import numpy as np
import pytest

from toto_forecasting_pipeline import (
    INPUT_SCHEMA,
    MAX_CONTEXT,
    MAX_HORIZON,
    MIN_CONTEXT,
    MODEL_ID,
    MODEL_REVISION,
    QUANTILES,
    evaluation_report,
    validate_inputs,
)


def _result(n_variates: int = 1, horizon: int = 4, context_length: int = 64) -> dict:
    quantiles = np.stack([np.full((n_variates, horizon), level) for level in QUANTILES], axis=1)
    return {
        "quantiles": quantiles,
        "quantile_levels": QUANTILES,
        "median": quantiles[:, 4, :],
        "horizon": horizon,
        "context_length": context_length,
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(np.ones((2, 64)), horizon=8, names=["a", "b"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["context_length"] == [MIN_CONTEXT, MAX_CONTEXT]
    assert manifest["schema"]["horizon"] == [1, MAX_HORIZON]
    assert [entry["id"] for entry in manifest["inputs"]] == ["a", "b"]
    assert manifest["inputs"][0]["context_length"] == 64
    assert (manifest["horizon"], manifest["decode_block_size"]) == (8, 768)
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_ids_for_1d_target() -> None:
    manifest = validate_inputs(np.arange(40), horizon=2)
    assert [entry["id"] for entry in manifest["inputs"]] == ["variate-0"]


def test_validate_inputs_rejects_like_forecast() -> None:
    with pytest.raises(ValueError, match="context length"):
        validate_inputs(np.ones(MIN_CONTEXT - 1), horizon=1)
    with pytest.raises(ValueError, match="horizon must be an integer"):
        validate_inputs(np.ones(64), horizon=MAX_HORIZON + 1)
    with pytest.raises(ValueError, match="finite"):
        validate_inputs(np.array([np.nan] + [1.0] * 63), horizon=1)
    with pytest.raises(ValueError, match="decode_block_size must be None or a positive integer"):
        validate_inputs(np.ones(64), horizon=8, decode_block_size=0)
    with pytest.raises(ValueError, match="names must have one entry per variate"):
        validate_inputs(np.ones((2, 64)), horizon=1, names=["only-one"])


def test_evaluation_report_not_measurable_without_truth() -> None:
    report = evaluation_report(_result())
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert "chronological holdout" in report["needs"]
    assert report["point_forecast"] == "median (q=0.5)"
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_sample_sanity_with_truth_and_context() -> None:
    result = _result(horizon=4)
    truth = np.full((1, 4), 0.5)  # equals the median -> zero error, inside the q10-q90 band
    context = np.linspace(0.0, 1.0, 64)[None, :]
    report = evaluation_report(result, truth, context=context, sample_kind="synthetic")
    assert report["verdict"] == "sample-sanity"
    metrics = {m["id"]: m["value"] for m in report["metrics"]}
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["interval_coverage"] == 1.0
    assert report["baselines"][0]["id"] == "last_value_baseline"
    baseline = {m["id"]: m["value"] for m in report["baselines"][0]["metrics"]}
    assert baseline["mae"] == pytest.approx(0.5)
    assert all(m["estimation"] for m in report["metrics"])


def test_evaluation_report_rejects_mismatched_truth() -> None:
    with pytest.raises(ValueError, match="truth shape"):
        evaluation_report(_result(horizon=4), np.zeros((1, 3)))
