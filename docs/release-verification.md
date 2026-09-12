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
| Kaggle CLI kernel, fresh-interpreter harness | Same Kaggle container; the committed notebook is executed verbatim, cell by cell, by `run_nb.py` in a subprocess of the container Python | Used because the Kaggle kernel pre-imports numpy 2.0.2 and this repository pins numpy 1.26.4: the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install, so the verbatim notebook runs in a fresh interpreter instead; the evidence cell proves the executed file equals the committed blob |
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
| 2026-09-11 | `7fac0a6d6fdd` / blob `f39ec6202ac1` (the notebook blob at this revision; the two later commits change `tools/validate_release_assets.py` lint and documentation only) | Kaggle kernel `kurtvalcorza/dimer-toto-forecast-t4-verify-v2` v4 — fresh container, Tesla T4 15,360 MiB (sm_75, driver 580.159.04), Python 3.12.13, Linux 6.12.90; committed notebook executed verbatim, cell by cell, by `run_nb.py` in a fresh interpreter (executed blob == committed blob, measured in-run); the Kaggle kernel itself pre-imports numpy 2.0.2 and Pillow 11.3.0, which the generalized stale-import guard of this revision would reject after the pinned install, hence the fresh interpreter | Default two-variate sample, all 5 code cells, clean HF cache (`models--Datadog--Toto-2.0-2.5B` only) | 311.9 s (211 s install, 101 s checkpoint fetch + forecast) | **PASS** — `repository_revision` equals the candidate commit; recorded versions torch 2.7.0, torchvision 0.22.0, toto-2 2.0.0, numpy 1.26.4, pandas 2.2.3 (pins); context 272 left-padded by 16 to 288, horizon 48, `decode_block_size=768`; peak CUDA allocation 9.451 GiB; MAE 0.073249 / RMSE 0.088706 vs last-value 1.108021 / 1.310835; empirical q10–q90 coverage 0.822917 — identical to every earlier run; `outputs/toto_forecast.csv` sha256 `b8c9b9c0…30f96` (byte-identical to the v3 run), `outputs/toto_provenance.json` sha256 `d9734aa9…69541` |
| 2026-09-11 | `3cfc6203a3b3` / blob `4a40718e6a15` (this revision; executed file verified equal to the committed blob) | Kaggle kernel `kurtvalcorza/dimer-toto-forecast-t4-verify-v2` v3 — fresh container, Tesla T4 15,360 MiB (sm_75, driver 580.159.04), Python 3.12.13, Linux 6.12.90; committed notebook executed verbatim, cell by cell, by `run_nb.py` in a fresh interpreter (the Kaggle kernel itself pre-imports numpy 2.0.2, which the tutorial's stale-import guard correctly rejects after the pinned numpy 1.26.4 install — kernel v1 halted there by design; kernel v2 at `633375a` failed with `operator torchvision::nms does not exist` from the orphaned runtime torchvision, fixed by the `torchvision==0.22.0` pin in this revision) | Default two-variate sample, all 5 code cells, clean HF cache (`models--Datadog--Toto-2.0-2.5B` is the only cache entry afterwards) | 305.8 s (212 s install, 93 s checkpoint fetch + forecast) | **PASS** — `repository_revision` equals the candidate commit; pinned PyPI torch 2.7.0+cu126, toto-2 2.0.0, numpy 1.26.4, pandas 2.2.3 (recorded), torchvision 0.22.0 (pinned; not in that run's recorded versions — inferred from the successful pinned install); no deviation; context 272 left-padded by 16 to 288 (`context_padding`/`patch_size` recorded), horizon 48, `decode_block_size=768`; peak CUDA allocation 9.451 GiB (identical to the local pre-flight); MAE 0.073249 / RMSE 0.088706 vs last-value baseline 1.108021 / 1.310835; empirical q10–q90 coverage 0.822917 — all identical to the local pre-flight; `outputs/toto_forecast.csv` sha256 `b8c9b9c0…30f96`, `outputs/toto_provenance.json` sha256 `0bcc27bf…6b3b6`; only warning: HF unauthenticated-download notice |
| 2026-09-11 | uncommitted working copy (`git hash-object` `cbb5011831391949`, never committed); cell 11 then gained `context_padding`/`patch_size` provenance keys before the commit (committed blob `4a40718e6a15`) | Local WSL harness, GPU: torch **2.7.0+cu128** (deviation from the `2.7.0` PyPI wheel, whose cu126 build has no sm_120 kernel image — same version, different CUDA build), toto-2 2.0.0, numpy 1.26.4, pandas 2.2.3 | Default two-variate sample, all 5 code cells, clean cache | 896.1 s (9.8 GB `model.safetensors` fetch inside the 890 s forecast cell) | PASS — `repository_revision` recorded; `Datadog/Toto-2.0-2.5B` acquired at the pinned revision into an empty cache (`models--Datadog--Toto-2.0-2.5B` only); context 272 → left-padded to 288, horizon 48, `decode_block_size=768`; peak CUDA allocation 9.45 GiB; MAE 0.0732 / RMSE 0.0887 vs last-value baseline 1.1080 / 1.3108; empirical q10–q90 coverage 0.823; `outputs/toto_forecast.csv` sha256 `175d435f…7494`, `outputs/toto_provenance.json` sha256 `b2a34d77…ad7ad` (pre-padding-key provenance) |
| 2026-09-11 | pre-fix working tree of `54afe64` | Local WSL harness, GPU (torch 2.7.0+cu128) | Default sample path | — | FAILED at cell 9 after a successful 9.8 GB load — `einops.EinopsError: can't divide axis of length 272 in chunks of 32`: upstream consumes the context in `patch_size` blocks and the 320−48 tutorial context is not a multiple of 32. Fixed in the pipeline: contexts are left-padded to the next patch multiple with masked positions (`has_missing_values=True` only when padding was added), mirroring the upstream GluonTS adapter; `decode_block_size` must now be a patch multiple |

## Current status

A clean supported-class execution of the notebook blob at this revision is recorded in the first row above (Kaggle container, fresh interpreter, clean cache, pinned wheels, all stages of the default path, verbatim blob measured in-run, every version recorded); it supersedes the earlier rows, which remain as the audit trail of the review round. Static CI is green on the same branch. The registry status remains **Candidate** until a reviewer confirms the recorded run against the notebook blob under review and an integrator promotes it; promotion is not performed by the builder. The commit that adds a recorded-execution row changes documentation only; the executed source is the commit named in the row.
