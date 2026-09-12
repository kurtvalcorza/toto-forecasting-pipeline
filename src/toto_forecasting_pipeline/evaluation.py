from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(np.abs(actual - predicted)))


def rmse(y_true, y_pred) -> float:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def last_value_baseline(context, horizon: int) -> np.ndarray:
    values = np.asarray(context, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(
            "context must have shape (variates, time) with at least one observation"
        )
    return np.repeat(values[:, -1:], horizon, axis=1)


def interval_coverage(y_true, lower, upper) -> float:
    actual = np.asarray(y_true, dtype=float)
    lower_bound = np.asarray(lower, dtype=float)
    upper_bound = np.asarray(upper, dtype=float)
    if not (actual.shape == lower_bound.shape == upper_bound.shape):
        raise ValueError("coverage arrays must share shape")
    return float(np.mean((actual >= lower_bound) & (actual <= upper_bound)))
