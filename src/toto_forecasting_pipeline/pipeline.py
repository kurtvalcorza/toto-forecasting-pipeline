from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .validation import validate_horizon, validate_target

MODEL_ID = "Datadog/Toto-2.0-2.5B"
MODEL_REVISION = "51a2812bbe449437c01b79c0e425ed578f335f5b"
MODEL_LICENSE = "Apache-2.0"
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _validate_decode_block_size(value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("decode_block_size must be None or a positive integer")
    return value


@dataclass
class TotoForecastPipeline:
    _model: Any
    device: str

    @classmethod
    def from_pretrained(cls, device: str | None = None) -> TotoForecastPipeline:
        import torch
        from toto2 import Toto2Model

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = Toto2Model.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            map_location="cpu",
        )
        model = model.to(resolved_device).eval()
        return cls(model, resolved_device)

    def forecast(
        self,
        target,
        *,
        horizon: int,
        decode_block_size: int | None = 768,
    ) -> dict[str, Any]:
        values = validate_target(target)
        validate_horizon(horizon)
        decode_block_size = _validate_decode_block_size(decode_block_size)
        patch_size = int(self._model.config.patch_size)
        if decode_block_size is not None and decode_block_size % patch_size:
            raise ValueError(
                f"decode_block_size must be a multiple of the model patch size ({patch_size})"
            )
        # Upstream patches the context in blocks of `patch_size` and requires the context length
        # to be a multiple of it. Like the upstream GluonTS adapter, pad on the left and mark the
        # padded positions unobserved so they carry no signal into the scaler or attention.
        context_padding = (patch_size - values.shape[1] % patch_size) % patch_size

        import torch

        target_tensor = torch.as_tensor(
            values,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        target_mask = torch.ones_like(target_tensor, dtype=torch.bool)
        if context_padding:
            pad = (context_padding, 0)
            target_tensor = torch.nn.functional.pad(target_tensor, pad, value=0.0)
            target_mask = torch.nn.functional.pad(target_mask, pad, value=False)
        series_ids = torch.arange(
            values.shape[0],
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0)
        with torch.inference_mode():
            forecast = self._model.forecast(
                {
                    "target": target_tensor,
                    "target_mask": target_mask,
                    "series_ids": series_ids,
                },
                horizon=horizon,
                decode_block_size=decode_block_size,
                has_missing_values=context_padding > 0,
            )
        quantiles = np.asarray(forecast.detach().cpu(), dtype=float)
        expected_shape = (len(QUANTILES), 1, values.shape[0], horizon)
        if quantiles.shape != expected_shape:
            raise RuntimeError(
                f"unexpected Toto forecast shape: {quantiles.shape}; "
                f"expected {expected_shape}"
            )
        quantiles = np.transpose(quantiles[:, 0, :, :], (1, 0, 2))
        return {
            "quantiles": quantiles,
            "quantile_levels": QUANTILES,
            "median": quantiles[:, 4, :],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "horizon": horizon,
            "context_length": values.shape[1],
            "context_padding": context_padding,
            "patch_size": patch_size,
            "n_variates": values.shape[0],
            "decode_block_size": decode_block_size,
        }
