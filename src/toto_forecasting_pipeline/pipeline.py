from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .validation import validate_horizon, validate_target

MODEL_ID = "Datadog/Toto-2.0-2.5B"
MODEL_REVISION = "51a2812bbe449437c01b79c0e425ed578f335f5b"
MODEL_LICENSE = "Apache-2.0"
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass
class TotoForecastPipeline:
    _model: Any
    device: str

    @classmethod
    def from_pretrained(cls, device: str | None = None) -> "TotoForecastPipeline":
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

        import torch

        target_tensor = torch.as_tensor(
            values,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        target_mask = torch.ones_like(target_tensor, dtype=torch.bool)
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
                has_missing_values=False,
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
            "n_variates": values.shape[0],
            "decode_block_size": decode_block_size,
        }
