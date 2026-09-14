---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: time-series-forecasting
base_model: Datadog/Toto-2.0-2.5B
date_published: "2026-04-17"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/Datadog/Toto-2.0-2.5B)"
---

# Toto 2.0 2.5B — Time-Series Foundation Model (Zero-Shot Multivariate Forecasting)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datadog%2FToto--2.0--2.5B-ffcc4d?style=flat)](https://huggingface.co/Datadog/Toto-2.0-2.5B)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-DataDog%2Ftoto-181717?style=flat&logo=github&logoColor=white)](https://github.com/DataDog/toto)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2605.20119-b31b1b.svg)](https://arxiv.org/abs/2605.20119)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://huggingface.co/Datadog/Toto-2.0-2.5B)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — bootstrap a fresh runtime, resolve and verify the pinned upstream revision, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/toto-forecasting-pipeline/blob/main/tutorials/toto_forecasting_colab.ipynb) [`toto_forecasting_colab.ipynb`](https://github.com/kurtvalcorza/toto-forecasting-pipeline/blob/main/tutorials/toto_forecasting_colab.ipynb)  
  *Zero-shot multivariate probabilistic forecasting with the pinned Toto 2.0 2.5B checkpoint on bundled or your own series, with chronological evaluation; no training occurs.*

> [!NOTE]
> Use a CUDA GPU runtime for this 2.5B-parameter model; the notebook does not target CPU.

---

#### Description

Toto 2.0 is Datadog's time-series foundation-model family for multivariate probabilistic forecasting. This DIMER package initially targets `Datadog/Toto-2.0-2.5B` at immutable revision `51a2812bbe449437c01b79c0e425ed578f335f5b`. Upstream describes a decoder-only u-μP-scaled transformer with alternating temporal/variate attention and a quantile head. This repository performs no adaptation; it adds pinned acquisition, finite target validation, explicit decode-strategy validation, normalized q=0.1–0.9 outputs, q=0.5 median semantics, chronological evaluation, baselines, provenance, and tutorial packaging.

#### Intended Use and Limitations

###### Primary Intended Uses

The supported task is zero-shot multivariate numerical forecasting from historical target series. Intended domains include observability and infrastructure telemetry, operations, demand, capacity planning, and other regularly ordered numerical series where predictive performance can be backtested on held-out future periods. The pipeline is designed as an inference service or strong pretrained baseline. Toto 2.0 fine-tuning and exogenous-variable support are not exposed because upstream identifies them as planned rather than available in the current release.

###### Primary Intended Users

Primary users are ML engineers, site-reliability and observability practitioners, forecasting data scientists, quantitative analysts, researchers, and application developers who understand temporal leakage, multivariate alignment, missing-value semantics, prediction horizons, quantile interpretation, and hardware requirements for multi-billion-parameter models. Users are expected to compare against simple baselines, evaluate on representative historical windows, and monitor for operational regime changes rather than assuming upstream benchmark rank transfers to their own data.

###### Out-of-scope use cases

1. **Capability boundary:** this repository does not implement Toto 2.0 fine-tuning, exogenous-variable conditioning, classification, anomaly detection, or Toto 1.0's Student-T-mixture interface.
2. **Input boundary:** the DIMER wrapper requires finite 1D/2D target histories between 32 and 16,384 steps and horizons of 1–4,096; missing-value handling is not enabled in the initial public contract. `decode_block_size` must be `None` or a positive multiple of the 32-step model patch size and invalid values are rejected before model execution; contexts that are not a multiple of the patch size are left-padded with masked positions rather than rejected.
3. **Runtime boundary:** the 2.5B release-reference tutorial requires a CUDA GPU; callers choosing other devices own the resulting resource and latency constraints.
4. **Decision boundary:** forecasts must not autonomously trigger high-consequence actions without validated operational thresholds and human/domain oversight.

#### Factors

###### Groups

The core model consumes numerical time series and is not inherently demographic, but deployment series can represent people, regions, services, customers, or institutions whose outcomes differ. This repository has not audited the upstream pretraining corpus for demographic representation or group-level errors. When forecasts affect people or resource allocation, operators must define relevant groups in their own context and compare forecast error and interval coverage across them before relying on the model.

###### Instrumentation

Toto may consume series produced by monitoring agents, metrics backends, sensors, transaction systems, databases, APIs, aggregators, or ETL pipelines. Sampling interval, aggregation windows, missingness, clock drift, unit changes, metric renames, counter resets, instrumentation upgrades, and silent collection failures can all alter model inputs. The initial wrapper rejects non-finite values and validates dimensions, but it cannot identify every semantic instrumentation change or determine whether an observed jump is a real event or data defect.

###### Environment

The reference package targets Python 3.12 with `toto-2==2.0.0`, PyTorch 2.7, torchvision 0.22 (pinned to the build matching PyTorch because hosted runtimes ship a torchvision compiled against their own PyTorch, and an orphaned torchvision breaks the upstream gluonts/torchmetrics import chain), NumPy 1.26.4, and pandas 2.2.3. The 2.5B checkpoint is heavyweight, so a CUDA-capable GPU is the practical release-reference environment; upstream recommends Ampere or newer for optimal execution. The tutorial makes `decode_block_size=768` explicit, while the public API also permits `None` for a single forward-pass decode. The data environment assumes ordered numerical histories representative enough that chronological backtesting is meaningful; abrupt regime changes or novel metric behavior can degrade forecasts.

#### Metrics

###### Performance Measures

The repository reports `mae` and `rmse` for q=0.5 median point forecasts on a chronological holdout and computes the same measures for a last-value baseline. The public `evaluation_report` stage writes these measures, the empirical q10–q90 `interval_coverage` and the baseline comparison to a machine-readable report whose verdict is `sample-sanity` on the withheld tutorial holdout and `not-measurable` when no truth is supplied. MAE is directly interpretable in target units, while RMSE emphasizes larger misses. The wrapper also provides `interval_coverage` so users can measure empirical q=0.1–q=0.9 coverage when ground truth exists. Tutorial values are local sample evidence; upstream benchmark rankings are not claimed as reproduced results.

###### Decision thresholds

No alert, action, or anomaly threshold is shipped. The pipeline treats q=0.5 as the median point forecast and exposes q=0.1 through q=0.9 as model quantiles. Quantiles are not converted into guaranteed confidence intervals or operational decisions. A deployment that triggers actions from forecasts must calibrate its own error budget, interval policy, or decision threshold on representative backtests, accounting for asymmetric costs of over- versus under-forecasting.

###### Approaches to uncertainty and variability

Toto 2.0 directly emits nine predictive quantiles, providing distributional information beyond a point forecast, but this repository does not claim those quantiles achieve nominal coverage on arbitrary deployment domains. The tutorial uses one chronological holdout and reports no cross-run dispersion. GPU kernels, context choice, decode strategy, package versions, and distribution shift can affect outputs. Deployments should estimate empirical interval coverage and point-error distributions across multiple rolling or blocked historical windows.

#### Ethical considerations and biases

###### Data

Upstream Toto documentation describes large-scale time-series pretraining with a strong observability focus and additional general-purpose data; this repository does not independently enumerate every source record or certify that all upstream data are free of sensitivity concerns. The DIMER repository distributes wrapper code, tests, documentation, and tutorial logic but not the upstream checkpoint or user data. Operators must review operational metrics for personal, proprietary, security-sensitive, customer, or regulated information before processing or sharing outputs.

###### Human Life

This package is not intended, certified, or externally validated for autonomous decisions in health, safety, criminal justice, employment, credit, housing, or other high-impact domains. No independent board has cleared this wrapper for such use. If a forecast influences sensitive resource allocation or safety-critical operations, deployment requires qualified human oversight, domain-specific validation across relevant failure scenarios, monitoring and fallback procedures, and any regulatory or institutional approval required by the application.

###### Mitigations

Implemented mitigations include an immutable upstream model revision; a committed `dimer-base-manifest.json` whose per-file SHA-256 digests `verify_snapshot` re-checks before every load; the public `validate_inputs` stage, which applies the same model-independent target, horizon and decode-block checks as `forecast` and writes an input manifest with any rejection recorded as a finding; SafeTensors weights; exact runtime package pins; finite numeric target checks; explicit context/horizon ceilings; a no-missing-values initial serving contract for user data (the only unobserved positions are the wrapper's own left padding, added when the context length is not a multiple of the 32-step model patch size and masked so the mask-aware upstream scaler and patch embedding ignore it; upstream's own GluonTS adapter truncates to a patch multiple instead); validation that `decode_block_size` is `None` or a positive multiple of the patch size; normalized and ordered q=0.1–0.9 outputs; explicit q=0.5 median semantics; chronological tutorial evaluation; raw duplicate CSV-header rejection before pandas ingestion; last-value comparison; machine-readable repository/model provenance; unit tests for shape, metric, and decode contracts; and CI validation that distinguishes source checks from the separate clean-GPU notebook execution evidence required for release.

###### Risks and harms

Forecast error can cause under- or over-provisioning, missed incidents, unnecessary intervention, inventory or staffing errors, and poor resource allocation. Multivariate correlations learned from historical data may fail after architecture, product, or instrumentation changes. Quantiles can be misinterpreted as calibrated guarantees, and upstream benchmark strength can encourage automation bias. Large checkpoints also create resource-exhaustion and availability risks. Sensitive telemetry may leak operational details if outputs, inputs, or logs are handled without access controls.

###### Use cases

The pipeline must not be used for unlawful surveillance, social scoring, discriminatory allocation, deceptive manipulation, or high-impact automated decisions based solely on unvalidated forecasts. Operators must not present future-leaked backtests or upstream benchmark results as evidence of local predictive validity. Use must comply with the Apache-2.0 model license, source-data terms, privacy and security requirements, and DIMER policy. Forecast outputs must not be represented as guaranteed future outcomes or calibrated certainty.

## Immutable provenance

- Model: `Datadog/Toto-2.0-2.5B`
- Revision: `51a2812bbe449437c01b79c0e425ed578f335f5b`
- Runtime package: `toto-2==2.0.0`
- Weight format: SafeTensors
- Snapshot manifest: `weights/toto-2.0-2.5b/dimer-base-manifest.json` — `model.safetensors` SHA-256 `dc08942b20751ac906167194d4ca4aa06b4367e80aabe5f1d5153b30b874bdb9` (9817176960 bytes, from the Hub LFS metadata at the pinned revision)
- Upstream repository: https://github.com/DataDog/toto
- Toto 2.0 technical report: https://arxiv.org/abs/2605.20119
