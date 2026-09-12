# Tutorials

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `toto_forecasting_colab.ipynb` | `TASK-INFERENCE` | Zero-shot multivariate probabilistic forecasting with empirical q10–q90 coverage | CUDA GPU (`device='cuda'`, ≥16 GB) | CSV with `timestamp` + numeric targets, gated off by default | **Candidate** — static checks pass; clean-runtime execution evidence is recorded in `../docs/release-verification.md` and must be reviewed for the exact notebook revision before promotion |

## Conformance notes

- The notebook exercises `TotoForecastPipeline` from the repository public API; the pipeline passes the immutable revision to the upstream loader and validates `decode_block_size`.
- The 2.5B checkpoint is loaded in its stored precision; a CUDA GPU is asserted in Section 1 because the CPU path is not the supported release-reference path.
- BYOD CSV headers are inspected before pandas ingestion so duplicate columns cannot be silently renamed; timestamps must be unique, increasing and regularly spaced.
- `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
