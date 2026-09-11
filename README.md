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
pipe = TotoForecastPipeline.from_pretrained()
result = pipe.forecast([1,2,3,4] * 32, horizon=24)
print(result["median"])
```

## Tutorial

`tutorials/toto_forecasting_colab.ipynb` is `TASK-INFERENCE`. It demonstrates chronological backtesting, last-value baseline, MAE/RMSE, q=0.1–0.9 outputs, optional BYOD, and portable output/provenance files.

## Release status

**Candidate.** The 2.5B checkpoint is heavyweight; a clean supported GPU execution record for the exact PR/release revision is required before release-grade promotion.
