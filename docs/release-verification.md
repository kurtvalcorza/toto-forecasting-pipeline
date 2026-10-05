# Release verification

`tutorials/toto_forecasting_colab.ipynb` (`TASK-INFERENCE`, `GUIDED`, **standalone** carrier, generator /3) is a
**release candidate** until the exact notebook revision has executed top-to-bottom in a clean supported runtime with one
**Run all** in the notebook kernel. Unit tests, JSON validation, code-cell compilation, and
`tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER Notebook
Specification 2.2. This file is the durable release-gate record for the notebook. The executions recorded below ran the
previous (/2, in-kernel install) notebook; they do not carry over to the regenerated notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or execution
  counts; no unresolved placeholder markers (including template braces in markdown); every code cell is preceded by an
  explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE` profile, the spec version
  (`2.2`) and the standalone carrier; `metadata.dimer` declares the profile, mode `GUIDED`, `standalone: true` and
  `generated_from` (repository, generating revision, package paths and SHA-256, carried-file digests, generator
  `build_notebook.py/3.0`);
- the standalone carrier and isolated environment (ST1–ST6, PAR1–PAR3, RUN1, RUN10, ENV6): one carrier cell whose
  carried files equal the repository files (`src/toto_forecasting_pipeline/{__init__,evaluation,pipeline,validation}.py`,
  the stage runner, `tutorials/requirements-colab.lock.txt`, the 3-file snapshot manifest, `LICENSE`) with matching
  digests; the lock pins every `pyproject.toml` runtime pin with hashes; a pinned `uv` builds a managed-CPython
  environment with `--require-hashes`, reused per lock digest; no in-kernel install and no restart instruction; Section 1
  stops on a CPU-only runtime; the four Infrastructure cells are titled and collapsed; every learner cell runs a stage;
  the notebook byte-identical to `tools/build_notebook.py` output;
- the stage-runner markers (staging and verification, the synthetic generator, the BYOD header, encoding, delimiter,
  numeric, timestamp and trim checks, the movable chronological holdout, `validate_inputs` with the short-context probe,
  the last-value, seasonal-naive and least-squares trend + season references and the noise floor, `forecast` with the
  figure, `evaluation_report` with the references, per-variate errors and coverage granularity, the export with
  timestamps, the provenance record), the form-parameter defaults (calls that appear only in comments do not count), no
  quality `assert`, and the forbidden patterns (credential-in-URL, any clone or repository import on the primary path, a
  mutable revision, model-library use in the notebook's own cells, `trust_remote_code=True`, `pickle.load`, `torch.load(`,
  `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

CI also runs `ruff`, `tools/build_notebook.py --check`, and the offline unit suite (`tests/`; stubbed `toto2`, no weights),
including `tests/test_release_assets.py` (negative controls proving the validator discriminates) and
`tests/test_notebook_review_fixes.py` (the notebook's own cells with stand-ins, and the model-free stages). These are source
and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab T4 runtime (15 GB is enough; the previous notebook peaked at 9.45 GiB); any kernel Python — the stages run on the isolated environment's CPython 3.12.12 | The runtime the tutorial is written for; a clean one-pass **Run all** in the notebook kernel is promotion evidence |
| Kaggle notebook kernel | Kaggle GPU kernel; the committed notebook run verbatim with **Run all** (no repository checkout) | Reproducible clean-room executor of the same class; the kernel's preloaded numpy no longer matters, because nothing is installed into the kernel |
| Local harness (pre-flight only) | Workstation, sequential cell executor, stand-ins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new GPU runtime (Colab T4, or a Kaggle GPU kernel) with **no repository
   checkout** and a clean model cache;
3. choose **Run all** once with the defaults (`USE_BYOD = False`, `HORIZON = 48`, `HOLDOUT_OFFSET = 0`,
   `CONTEXT_LENGTH = 0`, `SEASON_PERIOD = 0.0`, `RUN_ACTIVITY = False`); no restart is expected; then re-run the export
   cell (Section 8) once;
4. verify the carried-file verification and the isolated environment's versions (CPython 3.12.12, `torch` 2.7.0,
   `toto-2` 2.0.0, `numpy` 1.26.4, `pandas` 2.2.3); the 3-file snapshot staged and verified (9.8 GB); the synthetic
   sample's float32 SHA-256 `4129ad3d…`; context 272 (padded by 16) / horizon 48; the input manifest with the short-context
   refusal; last-value MAE 1.1080 / RMSE 1.3108, seasonal-naive 0.6497 / 0.7234, the trend + season reference
   0.0363 / 0.0479 and the noise floor 0.0349 / 0.0467; the forecast on the GPU with `point_forecast == 'median (q=0.5)'`,
   the peak CUDA memory and the figure; the evaluation report (`sample-sanity`, the model's MAE / RMSE and per-variate rows
   — the previous notebook's runs recorded 0.073249 / 0.088706 and coverage 0.822917 on the same sample; the regenerated
   path must be measured, not assumed to reproduce them — coverage granularity, the interpretation line); the CSV (with
   the `timestamp` column) and result JSON with source, model identity, licence, runtime and peak memory;
5. record the notebook Git blob id, commit, executor (**Run all** in the notebook kernel), `restarted: false`, runtime
   (platform, GPU, Python, PyTorch, toto-2), model identifier and immutable revision, whether the model cache was clean,
   outcome, metrics, peak CUDA memory and outputs in the table below;
6. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/toto_forecasting_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/toto_forecasting_colab.ipynb`). Wall times are the sum of per-cell
times reported by the executor and include installs and the model download; they are
measurements for the stated runtime, not general estimates.

### Standalone carrier (NOTEBOOK_SPEC 1.1 §3.6) — current notebook

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `3d42457` / `e95378f1837e` | Kaggle T4 (`kurtvalcorza/dimer-nb2-toto-forecasting` v2) | Default sample path | 378.0 s | **PASSED** — 10/10 ok code cells executed cleanly, 8 files, 9817 MB staged |

### Previous repository-installing notebook (NOTEBOOK_SPEC 1.0) — audit trail, does not cover the standalone carrier

Pre-flight runtime: WSL2 Ubuntu 24.04 (kernel 6.18.33), Python 3.12.3, Intel Core Ultra 9 275HX (24 threads), 15 GiB RAM, NVIDIA GeForce RTX 5070 Ti Laptop GPU (12,227 MiB, sm_120, driver 610.88). The harness executes the working-copy notebook cell by cell with the package installed non-editably from the same tree, under an empty `HF_HOME`. **Not a supported user runtime and not promotion evidence.**

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-11 | `7fac0a6d6fdd` / blob `f39ec6202ac1` (the notebook blob at this revision; the two later commits change `tools/validate_release_assets.py` lint and documentation only) | Kaggle kernel `kurtvalcorza/dimer-toto-forecast-t4-verify-v2` v4 — fresh container, Tesla T4 15,360 MiB (sm_75, driver 580.159.04), Python 3.12.13, Linux 6.12.90; committed notebook executed verbatim, cell by cell, by `run_nb.py` in a fresh interpreter (executed blob == committed blob, measured in-run); the Kaggle kernel itself pre-imports numpy 2.0.2 and Pillow 11.3.0, which the generalized stale-import guard of this revision would reject after the pinned install, hence the fresh interpreter | Default two-variate sample, all 5 code cells, clean HF cache (`models--Datadog--Toto-2.0-2.5B` only) | 311.9 s (211 s install, 101 s checkpoint fetch + forecast) | **PASS** — `repository_revision` equals the candidate commit; recorded versions torch 2.7.0, torchvision 0.22.0, toto-2 2.0.0, numpy 1.26.4, pandas 2.2.3 (pins); context 272 left-padded by 16 to 288, horizon 48, `decode_block_size=768`; peak CUDA allocation 9.451 GiB; MAE 0.073249 / RMSE 0.088706 vs last-value 1.108021 / 1.310835; empirical q10–q90 coverage 0.822917 — identical to every earlier run; `outputs/toto_forecast.csv` sha256 `b8c9b9c0…30f96` (byte-identical to the v3 run), `outputs/toto_provenance.json` sha256 `d9734aa9…69541` |
| 2026-09-11 | `3cfc6203a3b3` / blob `4a40718e6a15` (this revision; executed file verified equal to the committed blob) | Kaggle kernel `kurtvalcorza/dimer-toto-forecast-t4-verify-v2` v3 — fresh container, Tesla T4 15,360 MiB (sm_75, driver 580.159.04), Python 3.12.13, Linux 6.12.90; committed notebook executed verbatim, cell by cell, by `run_nb.py` in a fresh interpreter (the Kaggle kernel itself pre-imports numpy 2.0.2, which the tutorial's stale-import guard correctly rejects after the pinned numpy 1.26.4 install — kernel v1 halted there by design; kernel v2 at `633375a` failed with `operator torchvision::nms does not exist` from the orphaned runtime torchvision, fixed by the `torchvision==0.22.0` pin in this revision) | Default two-variate sample, all 5 code cells, clean HF cache (`models--Datadog--Toto-2.0-2.5B` is the only cache entry afterwards) | 305.8 s (212 s install, 93 s checkpoint fetch + forecast) | **PASS** — `repository_revision` equals the candidate commit; pinned PyPI torch 2.7.0+cu126, toto-2 2.0.0, numpy 1.26.4, pandas 2.2.3 (recorded), torchvision 0.22.0 (pinned; not in that run's recorded versions — inferred from the successful pinned install); no deviation; context 272 left-padded by 16 to 288 (`context_padding`/`patch_size` recorded), horizon 48, `decode_block_size=768`; peak CUDA allocation 9.451 GiB (identical to the local pre-flight); MAE 0.073249 / RMSE 0.088706 vs last-value baseline 1.108021 / 1.310835; empirical q10–q90 coverage 0.822917 — all identical to the local pre-flight; `outputs/toto_forecast.csv` sha256 `b8c9b9c0…30f96`, `outputs/toto_provenance.json` sha256 `0bcc27bf…6b3b6`; only warning: HF unauthenticated-download notice |
| 2026-09-11 | uncommitted working copy (`git hash-object` `cbb5011831391949`, never committed); cell 11 then gained `context_padding`/`patch_size` provenance keys before the commit (committed blob `4a40718e6a15`) | Local WSL harness, GPU: torch **2.7.0+cu128** (deviation from the `2.7.0` PyPI wheel, whose cu126 build has no sm_120 kernel image — same version, different CUDA build), toto-2 2.0.0, numpy 1.26.4, pandas 2.2.3 | Default two-variate sample, all 5 code cells, clean cache | 896.1 s (9.8 GB `model.safetensors` fetch inside the 890 s forecast cell) | PASS — `repository_revision` recorded; `Datadog/Toto-2.0-2.5B` acquired at the pinned revision into an empty cache (`models--Datadog--Toto-2.0-2.5B` only); context 272 → left-padded to 288, horizon 48, `decode_block_size=768`; peak CUDA allocation 9.45 GiB; MAE 0.0732 / RMSE 0.0887 vs last-value baseline 1.1080 / 1.3108; empirical q10–q90 coverage 0.823; `outputs/toto_forecast.csv` sha256 `175d435f…7494`, `outputs/toto_provenance.json` sha256 `b2a34d77…ad7ad` (pre-padding-key provenance) |
| 2026-09-11 | pre-fix working tree of `54afe64` | Local WSL harness, GPU (torch 2.7.0+cu128) | Default sample path | — | FAILED at cell 9 after a successful 9.8 GB load — `einops.EinopsError: can't divide axis of length 272 in chunks of 32`: upstream consumes the context in `patch_size` blocks and the 320−48 tutorial context is not a multiple of 32. Fixed in the pipeline: contexts are left-padded to the next patch multiple with masked positions (`has_missing_values=True` only when padding was added), mirroring the upstream GluonTS adapter; `decode_block_size` must now be a patch multiple |

## Current status

**No clean-runtime execution of the standalone notebook has been recorded yet**; clean GPU execution evidence is now recorded below. The rows above under the previous notebook prove that the pipeline's forecast path,
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