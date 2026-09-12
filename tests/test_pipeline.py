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

PATCH_SIZE = 32


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
    """Records the forecast call and enforces upstream's patch-multiple context contract."""

    config = types.SimpleNamespace(patch_size=PATCH_SIZE)

    def __init__(self):
        self.calls = []

    def forecast(self, batch, horizon, decode_block_size, has_missing_values):
        context = batch["target"].shape[-1]
        assert context % PATCH_SIZE == 0, f"upstream requires context % {PATCH_SIZE} == 0, got {context}"
        self.calls.append(
            {
                "context": context,
                "mask": np.asarray(batch["target_mask"].array, dtype=bool),
                "decode_block_size": decode_block_size,
                "has_missing_values": has_missing_values,
            }
        )
        n_variates = batch["target"].shape[1]
        return FakeTensor(np.zeros((9, 1, n_variates, horizon)))


def _pad(tensor, pad, value):
    left, right = pad
    padded = np.pad(tensor.array, [(0, 0)] * (tensor.array.ndim - 1) + [(left, right)], constant_values=value)
    return FakeTensor(padded)


def _stub_torch(monkeypatch):
    fake_torch = types.SimpleNamespace(
        float32=None,
        bool=None,
        long=None,
        inference_mode=contextlib.nullcontext,
        as_tensor=lambda value, **kwargs: FakeTensor(np.asarray(value)),
        ones_like=lambda value, **kwargs: FakeTensor(np.ones(value.shape, dtype=bool)),
        arange=lambda count, **kwargs: FakeTensor(np.arange(count)),
        nn=types.SimpleNamespace(functional=types.SimpleNamespace(pad=_pad)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_metrics_and_shape(monkeypatch):
    _stub_torch(monkeypatch)
    model = FakeModel()
    pipeline = TotoForecastPipeline(model, "cpu")
    result = pipeline.forecast(np.arange(64), horizon=4)
    assert result["median"].shape == (1, 4)
    assert result["quantiles"].shape == (1, 9, 4)
    assert result["context_padding"] == 0
    assert result["patch_size"] == PATCH_SIZE
    assert model.calls[0]["decode_block_size"] == 768
    assert model.calls[0]["has_missing_values"] is False
    assert last_value_baseline([1, 2], 2).tolist() == [[2, 2]]
    assert interval_coverage([1, 2], [0, 1], [2, 3]) == 1.0


def test_context_is_left_padded_to_a_patch_multiple_and_masked(monkeypatch):
    _stub_torch(monkeypatch)
    model = FakeModel()
    pipeline = TotoForecastPipeline(model, "cpu")
    result = pipeline.forecast(np.ones((2, 272)), horizon=48)
    call = model.calls[0]
    assert call["context"] == 288
    assert call["has_missing_values"] is True
    assert result["context_padding"] == 16
    assert result["context_length"] == 272
    assert call["mask"].shape == (1, 2, 288)
    assert not call["mask"][..., :16].any()
    assert call["mask"][..., 16:].all()


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "768"])
def test_rejects_invalid_decode_block_size(value):
    pipeline = TotoForecastPipeline(FakeModel(), "cpu")
    with pytest.raises(ValueError, match="decode_block_size"):
        pipeline.forecast(np.arange(64), horizon=4, decode_block_size=value)


def test_rejects_decode_block_size_that_is_not_a_patch_multiple():
    pipeline = TotoForecastPipeline(FakeModel(), "cpu")
    with pytest.raises(ValueError, match="multiple of the model patch size"):
        pipeline.forecast(np.arange(64), horizon=4, decode_block_size=100)


def test_rejects_context_shorter_than_one_patch(monkeypatch):
    _stub_torch(monkeypatch)
    model = FakeModel()
    model.config = types.SimpleNamespace(patch_size=64)
    pipeline = TotoForecastPipeline(model, "cpu")
    with pytest.raises(ValueError, match="at least the model patch size"):
        pipeline.forecast(np.arange(40), horizon=4)
