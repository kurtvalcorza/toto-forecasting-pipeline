# Release verification

`tutorials/toto_forecasting_colab.ipynb` (`TASK-INFERENCE`, **standalone** carrier) is a **release
candidate** until the exact notebook revision has executed top-to-bottom in a clean supported runtime.
Unit tests, JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 1.1. This file is the
durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec `1.1`,
  `standalone: true` and `generated_from` (repository, generating revision, the carried modules, their concatenated
  SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the
  primary path; one cell tagged `embedded_module` per carried module (`src/toto_forecasting_pipeline/evaluation.py`,
  `validation.py`, `pipeline.py`, in dependency order), each equal to the module after the generator's documented
  rewrites (working-directory-relative weights directory; package-relative imports removed); the inline `MANIFEST`
  equal to the committed snapshot manifest and the inline `PINS` equal to the `pyproject.toml` runtime pins; the
  notebook byte-identical to `tools/build_notebook.py` output; the pinned-install cell with its restart-on-stale-import
  guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cells (and repeated in the inline manifest,
  which the notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and the
  same identity string appears in `README.md`, `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `TotoForecastPipeline.from_pretrained(device='cuda', weights_dir=...)`, `validate_inputs`, `forecast`,
  `evaluation_report`, `last_value_baseline`), the ceiling print (`MIN_CONTEXT`, `MAX_CONTEXT`, `MAX_HORIZON`), the
  chronological holdout (`context = series[:, :-HORIZON]`, `truth = series[:, -HORIZON:]`), `DECODE_BLOCK_SIZE = 768`,
  the padding/patch-size provenance, the exports, the learner-facing forecasting statements (median semantics, model
  quantiles are not confidence intervals, last-value baseline, no fine-tuning, GPU requirement) and the gated-off BYOD
  default listed in the validator; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository
  import on the primary path, a mutable `revision='main'`, direct `toto2` / `safetensors` / `huggingface_hub` use
  **outside the carried module cells**, any worker process or subprocess outside the generator-owned install cell,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

CI also runs `ruff`, `tools/build_notebook.py --check`, and the offline unit suite (`tests/test_pipeline.py`,
`tests/test_snapshot.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`, `tests/test_release_assets.py`;
stubbed `toto2`/`torch`, no weights). These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CUDA runtime with ≥16 GB GPU memory | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle GPU kernel (Tesla T4 15 GB has sufficed for the previous notebook: peak 9.45 GiB), Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebook is standalone) |
| Kaggle CLI kernel, fresh-interpreter harness | Same Kaggle container; the committed notebook is executed verbatim, cell by cell, by `run_nb.py` in a subprocess of the container Python | Used when the kernel pre-imports a distribution the pinned install replaces (numpy 2.0.2 vs the pinned 1.26.4): the stale-import guard correctly halts the in-kernel path, so the verbatim notebook runs in a fresh interpreter instead; the evidence cell proves the executed file equals the committed blob |
| Local WSL harness (pre-flight only) | Workstation RTX 5070 Ti (12 GB; needs a cu128 torch build for sm_120 — a version-equal deviation from the pinned PyPI wheel) | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CUDA GPU runtime (Colab, or the Kaggle executor above) with
   **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from`, `cuda: True`, and that the installed core package versions equal the inline
   `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (define `TotoForecastPipeline`, `validate_inputs`, `evaluation_report`,
     the metric helpers and the ceilings) with no import of the repository package;
   - pinned `Datadog/Toto-2.0-2.5B` acquisition at the immutable revision through the package: the inline `MANIFEST` is
     asserted against the module identity and written to `weights/toto-2.0-2.5b/`, `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reports all three manifest entries (`README.md`, `config.json`, `model.safetensors`) on a
     clean runtime, `verify_snapshot` returns the manifest dict **after hashing the real 9.8 GB checkpoint against the
     Hub-derived digest — the first time that digest is checked against bytes anywhere**, and
     `from_pretrained(device='cuda', weights_dir=WEIGHTS_DIR)` reports `source == 'local-snapshot'`;
   - deterministic two-variate synthetic sample (320 steps, seed 11) with its float32 SHA-256 printed, the ceilings
     surfaced, the final 48 steps withheld chronologically and the last-value baseline computed from the context;
   - `validate_inputs` writes `outputs/toto_forecasting_input_manifest.json` (verdict `accepted`, one recorded
     rejection finding from the short-context probe);
   - zero-shot forecast through `forecast(context, horizon=HORIZON, decode_block_size=DECODE_BLOCK_SIZE)` with
     q=0.1–0.9 outputs, `point_forecast == 'median (q=0.5)'`, context 272 left-padded by 16 to 288, horizon 48,
     `decode_block_size=768`;
   - `evaluation_report` writes `outputs/toto_forecasting_evaluation_report.json` with verdict `sample-sanity`
     carrying `mae`, `rmse`, `interval_coverage` and the `last_value_baseline` comparison (the previous notebook's
     runs recorded MAE 0.073249 / RMSE 0.088706 vs baseline 1.108021 / 1.310835 and coverage 0.822917 on the same
     sample and holdout; the standalone path must be measured, not assumed to reproduce them);
   - `outputs/toto_forecasting_result.json` and `outputs/toto_forecasting_forecast.csv` written with
     `NOTEBOOK_SOURCE`, model revision, model licence, runtime versions, GPU device and decode strategy;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, toto-2, GPU), model identifier
   and immutable revision, whether the model cache was clean, outcome, produced outputs, peak CUDA allocation, and
   any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/toto_forecasting_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/toto_forecasting_colab.ipynb`). Wall times are the sum of per-cell
times reported by the executor and include installs and the model download; they are
measurements for the stated runtime, not general estimates.

### Standalone carrier (NOTEBOOK_SPEC 1.1 §3.6) — current notebook

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| | | | Default sample path | | pending — queued to the GPU lane |

### Previous repository-installing notebook (NOTEBOOK_SPEC 1.0) — audit trail, does not cover the standalone carrier

Pre-flight runtime: WSL2 Ubuntu 24.04 (kernel 6.18.33), Python 3.12.3, Intel Core Ultra 9 275HX (24 threads), 15 GiB RAM, NVIDIA GeForce RTX 5070 Ti Laptop GPU (12,227 MiB, sm_120, driver 610.88). The harness executes the working-copy notebook cell by cell with the package installed non-editably from the same tree, under an empty `HF_HOME`. **Not a supported user runtime and not promotion evidence.**

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-11 | `7fac0a6d6fdd` / blob `f39ec6202ac1` (the notebook blob at this revision; the two later commits change `tools/validate_release_assets.py` lint and documentation only) | Kaggle kernel `kurtvalcorza/dimer-toto-forecast-t4-verify-v2` v4 — fresh container, Tesla T4 15,360 MiB (sm_75, driver 580.159.04), Python 3.12.13, Linux 6.12.90; committed notebook executed verbatim, cell by cell, by `run_nb.py` in a fresh interpreter (executed blob == committed blob, measured in-run); the Kaggle kernel itself pre-imports numpy 2.0.2 and Pillow 11.3.0, which the generalized stale-import guard of this revision would reject after the pinned install, hence the fresh interpreter | Default two-variate sample, all 5 code cells, clean HF cache (`models--Datadog--Toto-2.0-2.5B` only) | 311.9 s (211 s install, 101 s checkpoint fetch + forecast) | **PASS** — `repository_revision` equals the candidate commit; recorded versions torch 2.7.0, torchvision 0.22.0, toto-2 2.0.0, numpy 1.26.4, pandas 2.2.3 (pins); context 272 left-padded by 16 to 288, horizon 48, `decode_block_size=768`; peak CUDA allocation 9.451 GiB; MAE 0.073249 / RMSE 0.088706 vs last-value 1.108021 / 1.310835; empirical q10–q90 coverage 0.822917 — identical to every earlier run; `outputs/toto_forecast.csv` sha256 `b8c9b9c0…30f96` (byte-identical to the v3 run), `outputs/toto_provenance.json` sha256 `d9734aa9…69541` |
| 2026-09-11 | `3cfc6203a3b3` / blob `4a40718e6a15` (this revision; executed file verified equal to the committed blob) | Kaggle kernel `kurtvalcorza/dimer-toto-forecast-t4-verify-v2` v3 — fresh container, Tesla T4 15,360 MiB (sm_75, driver 580.159.04), Python 3.12.13, Linux 6.12.90; committed notebook executed verbatim, cell by cell, by `run_nb.py` in a fresh interpreter (the Kaggle kernel itself pre-imports numpy 2.0.2, which the tutorial's stale-import guard correctly rejects after the pinned numpy 1.26.4 install — kernel v1 halted there by design; kernel v2 at `633375a` failed with `operator torchvision::nms does not exist` from the orphaned runtime torchvision, fixed by the `torchvision==0.22.0` pin in this revision) | Default two-variate sample, all 5 code cells, clean HF cache (`models--Datadog--Toto-2.0-2.5B` is the only cache entry afterwards) | 305.8 s (212 s install, 93 s checkpoint fetch + forecast) | **PASS** — `repository_revision` equals the candidate commit; pinned PyPI torch 2.7.0+cu126, toto-2 2.0.0, numpy 1.26.4, pandas 2.2.3 (recorded), torchvision 0.22.0 (pinned; not in that run's recorded versions — inferred from the successful pinned install); no deviation; context 272 left-padded by 16 to 288 (`context_padding`/`patch_size` recorded), horizon 48, `decode_block_size=768`; peak CUDA allocation 9.451 GiB (identical to the local pre-flight); MAE 0.073249 / RMSE 0.088706 vs last-value baseline 1.108021 / 1.310835; empirical q10–q90 coverage 0.822917 — all identical to the local pre-flight; `outputs/toto_forecast.csv` sha256 `b8c9b9c0…30f96`, `outputs/toto_provenance.json` sha256 `0bcc27bf…6b3b6`; only warning: HF unauthenticated-download notice |
| 2026-09-11 | uncommitted working copy (`git hash-object` `cbb5011831391949`, never committed); cell 11 then gained `context_padding`/`patch_size` provenance keys before the commit (committed blob `4a40718e6a15`) | Local WSL harness, GPU: torch **2.7.0+cu128** (deviation from the `2.7.0` PyPI wheel, whose cu126 build has no sm_120 kernel image — same version, different CUDA build), toto-2 2.0.0, numpy 1.26.4, pandas 2.2.3 | Default two-variate sample, all 5 code cells, clean cache | 896.1 s (9.8 GB `model.safetensors` fetch inside the 890 s forecast cell) | PASS — `repository_revision` recorded; `Datadog/Toto-2.0-2.5B` acquired at the pinned revision into an empty cache (`models--Datadog--Toto-2.0-2.5B` only); context 272 → left-padded to 288, horizon 48, `decode_block_size=768`; peak CUDA allocation 9.45 GiB; MAE 0.0732 / RMSE 0.0887 vs last-value baseline 1.1080 / 1.3108; empirical q10–q90 coverage 0.823; `outputs/toto_forecast.csv` sha256 `175d435f…7494`, `outputs/toto_provenance.json` sha256 `b2a34d77…ad7ad` (pre-padding-key provenance) |
| 2026-09-11 | pre-fix working tree of `54afe64` | Local WSL harness, GPU (torch 2.7.0+cu128) | Default sample path | — | FAILED at cell 9 after a successful 9.8 GB load — `einops.EinopsError: can't divide axis of length 272 in chunks of 32`: upstream consumes the context in `patch_size` blocks and the 320−48 tutorial context is not a multiple of 32. Fixed in the pipeline: contexts are left-padded to the next patch multiple with masked positions (`has_missing_values=True` only when padding was added), mirroring the upstream GluonTS adapter; `decode_block_size` must now be a patch multiple |

## Current status

**No clean-runtime execution of the standalone notebook has been recorded yet**; the run is **pending** and
queued to the GPU lane. The rows above under the previous notebook prove that the pipeline's forecast path,
the pinned 9.8 GB checkpoint fetch through the upstream Hub loader and the sample/holdout produced stable metrics in
a clean Kaggle T4 container, but they executed the earlier repository-installing carrier: the standalone path
(carried module cells, inline manifest, `stage_missing_files` through `hf_hub_download`, `verify_snapshot` over the
real checkpoint, and `Toto2Model.from_pretrained` on the verified directory) has been validated statically only
(parity PASS, carrier probe with the repository package blocked) and never run. Two facts a reviewer should weigh:
the manifest's `model.safetensors` SHA-256 was taken from the Hub's LFS metadata at the pinned revision, not
computed from a local copy (the checkpoint is not kept on the build machine), so the clean run is the first time
that digest is checked against downloaded bytes; and `from_pretrained(weights_dir=...)` was exercised only with a
stubbed `Toto2Model` — the local-directory loading path of `toto-2==2.0.0` was inspected in upstream source, not
executed. Static validation (`tools/validate_release_assets.py`), nbformat validation, a `compile()` sweep over every
code cell, and the offline unit suite passed on the tutorial source at the candidate revision, which is necessary but
not sufficient. The registry status remains **Candidate** until a reviewer confirms a recorded run against the
notebook blob under review and an integrator promotes it; promotion is not performed by the builder.
