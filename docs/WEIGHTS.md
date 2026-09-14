# Weight provenance and DIMER hosting

- Upstream: `Datadog/Toto-2.0-2.5B`
- Immutable revision: `51a2812bbe449437c01b79c0e425ed578f335f5b`
- Runtime package: `toto-2==2.0.0`
- Weight format: SafeTensors (`model.safetensors`, 9817176960 bytes, SHA-256 `dc08942b20751ac906167194d4ca4aa06b4367e80aabe5f1d5153b30b874bdb9`) plus `config.json`
- Upstream license: Apache-2.0
- Local snapshot: `weights/toto-2.0-2.5b/` with `dimer-base-manifest.json` (per-file bytes + SHA-256 for `README.md`, `config.json`, `model.safetensors`; `totalBytes` 9817184746). The manifest was written from the Hub API at the pinned revision (LFS size and SHA-256 for the checkpoint; the two small files downloaded and hashed locally); the approximately 9.8 GB checkpoint is intentionally not stored in this Git repository or on the build machine, so `verify_snapshot` over the real file has not been exercised locally.
- Load-time check: `stage_missing_files()` fetches only absent manifest entries at the immutable revision and `verify_snapshot()` in `src/toto_forecasting_pipeline/pipeline.py` re-hashes every entry and refuses on any mismatch before `Toto2Model.from_pretrained(<directory>)` builds the model from that directory's `config.json` and SafeTensors file.
- DIMER hosting: Apache-2.0 permits redistribution and hosted use subject to license/notice obligations; DIMER may mirror the pinned snapshot in its model store.
- Trust boundary: SafeTensors weights (no code execution on load); the model class and its configuration schema come from the pinned `toto-2==2.0.0` PyPI package, not from the model repository (no `trust_remote_code`).
- Scope: Toto 2.0 is inference-only in the current open release. Fine-tuning and exogenous-variable support documented for Toto 1.0 are not claimed here.
