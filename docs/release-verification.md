# Release verification

`tutorials/toto_forecasting_colab.ipynb` (`TASK-INFERENCE`) is a **release candidate** until the exact
notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON
validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 1.0. This file is
the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile and the notebook-spec version; `metadata.dimer` declares that profile and spec `1.0`;
- the fresh-runtime bootstrap (clone by canonical URL, `DIMER_TUTORIAL_REF`, detached checkout of
  the requested revision, restart-on-stale-import guard) and the recorded `REPO_SHA` in exports;
- `MODEL_ID`/`MODEL_REVISION` are imported from the package rather than hard-coded, the revision is
  a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls, exports, learner-facing statements and gated-off BYOD
  default listed in the validator; forbidden patterns (credential-in-URL, direct `transformers`
  loading that bypasses the pipeline, `trust_remote_code=True` outside the pipeline boundary,
  `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

These are source/provenance checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CUDA runtime with ≥16 GB GPU memory | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle `NvidiaTeslaT4` kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab`, sets `DIMER_TUTORIAL_REF`, and chdirs to a scratch directory so the bootstrap clones the candidate |
| Local WSL harness (pre-flight only) | Workstation, `run_nb.py` sequential cell executor with a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CUDA GPU runtime (Colab, or the Kaggle
   executor above) with `DIMER_TUTORIAL_REF` set to the candidate commit and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path);
4. verify that Section 1 reports `repository_revision` equal to the candidate commit and that the
   installed core package versions equal the `pyproject.toml` pins;
5. verify every default-path stage completes:
   - fresh bootstrap from GitHub at the candidate revision;
   - pinned `Datadog/Toto-2.0-2.5B` acquisition at the immutable revision;
   - deterministic two-variate synthetic sample, chronological holdout and last-value baseline;
   - zero-shot forecast through `TotoForecastPipeline.forecast` with the validated `decode_block_size=768`, q=0.1–0.9 outputs and the q=0.5 median;
   - MAE/RMSE for the model and the baseline plus empirical q10–q90 coverage;
   - `outputs/toto_forecast.csv` and `outputs/toto_provenance.json` written with repository SHA, model revision, runtime versions, GPU device and decode strategy;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, toto-2, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/toto_forecasting_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/toto_forecasting_colab.ipynb`). Wall times are the sum of per-cell
times reported by the executor and include installs and the model download; they are
measurements for the stated runtime, not general estimates.

Pre-flight runtime: WSL2 Ubuntu 24.04 (kernel 6.18.33), Python 3.12.3, Intel Core Ultra 9 275HX (24 threads), 15 GiB RAM, NVIDIA GeForce RTX 5070 Ti Laptop GPU (12,227 MiB, sm_120, driver 610.88). The harness executes the working-copy notebook cell by cell with the package installed non-editably from the same tree, under an empty `HF_HOME`. **Not a supported user runtime and not promotion evidence.**

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-11 | blob `cbb5011831391949` executed; cell 11 then gained `context_padding`/`patch_size` provenance keys (blob `4a40718e6a15`, this revision) | Local WSL harness, GPU: torch **2.7.0+cu128** (deviation from the `2.7.0` PyPI wheel, whose cu126 build has no sm_120 kernel image — same version, different CUDA build), toto-2 2.0.0, numpy 1.26.4, pandas 2.2.3 | Default two-variate sample, all 5 code cells, clean cache | 896.1 s (9.8 GB `model.safetensors` fetch inside the 890 s forecast cell) | PASS — `repository_revision` recorded; `Datadog/Toto-2.0-2.5B` acquired at the pinned revision into an empty cache (`models--Datadog--Toto-2.0-2.5B` only); context 272 → left-padded to 288, horizon 48, `decode_block_size=768`; peak CUDA allocation 9.45 GiB; MAE 0.0732 / RMSE 0.0887 vs last-value baseline 1.1080 / 1.3108; empirical q10–q90 coverage 0.823; `outputs/toto_forecast.csv` sha256 `175d435f…7494`, `outputs/toto_provenance.json` sha256 `b2a34d77…ad7ad` (pre-padding-key provenance) |
| 2026-09-11 | pre-fix working tree of `54afe64` | Local WSL harness, GPU (torch 2.7.0+cu128) | Default sample path | — | FAILED at cell 9 after a successful 9.8 GB load — `einops.EinopsError: can't divide axis of length 272 in chunks of 32`: upstream consumes the context in `patch_size` blocks and the 320−48 tutorial context is not a multiple of 32. Fixed in the pipeline: contexts are left-padded to the next patch multiple with masked positions (`has_missing_values=True` only when padding was added), mirroring the upstream GluonTS adapter; `decode_block_size` must now be a patch multiple |

## Current status

Static CI and unit tests are preparatory evidence. The tutorial remains **Candidate** until a clean supported-class CUDA GPU run for the exact notebook revision under review is appended to the table and reviewed.
