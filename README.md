# Toto 2.0 Forecasting Pipeline

DIMER-oriented zero-shot probabilistic forecasting wrapper for **Datadog Toto 2.0**, initially pinned to the `Toto-2.0-2.5B` checkpoint. The wrapper normalizes multivariate target input, nine quantile outputs, q=0.5 median forecasts, chronological evaluation, and provenance.

## Upstream alignment

- Model: `Datadog/Toto-2.0-2.5B`
- Revision: `51a2812bbe449437c01b79c0e425ed578f335f5b`
- Runtime package: `toto-2==2.0.0`
- Weight license: Apache-2.0
- Weight format: SafeTensors
- Current Toto 2.0 capability: zero-shot multivariate forecasting with nine quantiles
- **Not claimed:** Toto 2.0 fine-tuning or exogenous-variable support; upstream says these are planned rather than currently available

## Public API

```python
from toto_forecasting_pipeline import TotoForecastPipeline

pipe = TotoForecastPipeline.from_pretrained(device="cuda")
result = pipe.forecast(
    [1, 2, 3, 4] * 32,
    horizon=24,
    decode_block_size=768,
)
print(result["median"])
```

`decode_block_size` must be `None` or a positive multiple of the model patch size (32 for this checkpoint). `None` requests a single forward-pass decode; the tutorial uses `768` explicitly. Upstream patches the context in blocks of `patch_size`, so the wrapper left-pads any context whose length is not a multiple of it with masked (unobserved) positions — the upstream scaler and patch embedding are mask-aware, so the pads carry no signal (upstream's own GluonTS adapter truncates to a patch multiple instead); a context shorter than one patch is rejected; the applied `context_padding` and `patch_size` are returned with every result. The 2.5B checkpoint is treated as a GPU release-reference path even though the API allows callers to choose another device explicitly.

## Tutorial

`tutorials/toto_forecasting_colab.ipynb` is `TASK-INFERENCE`. It self-bootstraps in a fresh runtime, validates raw BYOD CSV headers before pandas ingestion, demonstrates chronological backtesting, last-value baseline, MAE/RMSE, q=0.1–0.9 outputs, empirical q10–q90 coverage, and portable output/provenance files.

## Release status

**Candidate.** The 2.5B checkpoint is heavyweight; a clean supported GPU execution record for the exact PR/release revision is required before release-grade promotion.
