from __future__ import annotations

import numpy as np

MIN_CONTEXT = 32  # observations; shorter targets are rejected before any model work
MAX_CONTEXT = 16_384  # observations per variate
MAX_HORIZON = 4096  # forecast steps per call


def validate_target(target, *, min_context: int = MIN_CONTEXT, max_context: int = MAX_CONTEXT) -> np.ndarray:
    values = np.asarray(target, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("target must be 1D or 2D with shape (variates, time)")
    if not min_context <= values.shape[1] <= max_context:
        raise ValueError(
            f"context length must be between {min_context} and {max_context}"
        )
    if not np.isfinite(values).all():
        raise ValueError("target must contain only finite values")
    return values


def validate_horizon(horizon: int, *, max_horizon: int = MAX_HORIZON) -> int:
    if not isinstance(horizon, int) or not 1 <= horizon <= max_horizon:
        raise ValueError(f"horizon must be an integer from 1 to {max_horizon}")
    return horizon
