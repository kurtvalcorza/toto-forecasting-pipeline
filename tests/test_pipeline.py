import contextlib
import sys
import types

import numpy as np
import pytest

from toto_forecasting_pipeline import (
    TotoForecastPipeline,
    interval_coverage,
    last_value_baseline,
)


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    @property
    def shape(self):
        return self.array.shape

    def unsqueeze(self, axis):
        return FakeTensor(np.expand_dims(self.array, axis))

    def detach(self):
        return self

    def cpu(self):
        return self

    def __array__(self, dtype=None):
        return self.array.astype(dtype) if dtype else self.array


class FakeModel:
    def forecast(self, batch, horizon, decode_block_size, has_missing_values):
        assert decode_block_size == 768
        assert has_missing_values is False
        n_variates = batch["target"].shape[1]
        return FakeTensor(np.zeros((9, 1, n_variates, horizon)))


def _stub_torch(monkeypatch):
    fake_torch = types.SimpleNamespace(
        float32=None,
        bool=None,
        long=None,
        inference_mode=contextlib.nullcontext,
        as_tensor=lambda value, **kwargs: FakeTensor(np.asarray(value)),
        ones_like=lambda value, **kwargs: FakeTensor(np.ones(value.shape)),
        arange=lambda count, **kwargs: FakeTensor(np.arange(count)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_metrics_and_shape(monkeypatch):
    _stub_torch(monkeypatch)
    pipeline = TotoForecastPipeline(FakeModel(), "cpu")
    result = pipeline.forecast(np.arange(64), horizon=4)
    assert result["median"].shape == (1, 4)
    assert result["quantiles"].shape == (1, 9, 4)
    assert last_value_baseline([1, 2], 2).tolist() == [[2, 2]]
    assert interval_coverage([1, 2], [0, 1], [2, 3]) == 1.0


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "768"])
def test_rejects_invalid_decode_block_size(value):
    pipeline = TotoForecastPipeline(FakeModel(), "cpu")
    with pytest.raises(ValueError, match="decode_block_size"):
        pipeline.forecast(np.arange(64), horizon=4, decode_block_size=value)
