"""Per-repository template for tools/build_notebook.py /3 (NOTEBOOK_SPEC 2.2 §4 standalone, §25.13 isolated environment).

The generator writes the infrastructure cells (runtime check — which stops on a CPU-only runtime, since the 2.5B
checkpoint's reference path is a CUDA GPU —, carrier, isolated install + stage runner, snapshot staging) from repository
files; this template holds the learner-facing prose, the guided layer and the learner cells. Every learner cell calls
``run_stage(...)``: the carried ``tools/tutorial_stages.py`` runs one stage per process in an isolated, hash-locked
environment, so nothing is installed into the notebook kernel and no restart is needed.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "toto-forecasting-pipeline"

UV = {
    "version": "0.12.15",
    "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "bytes": 20081404,
    "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
}

TEMPLATE = {
    "package": "toto_forecasting_pipeline",
    "repo_name": REPO,
    "weights_key": "toto-2.0-2.5b",
    "modules": ["__init__.py", "evaluation.py", "pipeline.py", "validation.py"],
    "entry_module": "pipeline.py",
    "lock": "tutorials/requirements-colab.lock.txt",
    "managed_python": "3.12.12",
    "uv": UV,
    "disk_gib": {"weights": 9.2, "environment": 8.0},
    "runtime_modules": ["torch", "toto-2", "numpy", "pandas"],
    "install_flags": ["--only-binary", ":all:"],
    "require_gpu": True,
    "stem": "toto_forecasting",
    "notebook_name": "toto_forecasting_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "stage_runner": "tools/tutorial_stages.py",
    "run_all": (
        "Selecting **Run all** in a fresh Linux x86_64 runtime with a CUDA GPU (a 15 GB T4 is enough) builds an isolated Python "
        "environment from the carried hash-locked requirements without touching the notebook kernel's own packages, stages and "
        "digest-verifies the pinned Toto 2.0 checkpoint (9.8 GB SafeTensors), generates a deterministic two-variate synthetic "
        "series in code, withholds the last `HORIZON` steps chronologically, validates the context into an input manifest with "
        "a recorded rejection, computes last-value, seasonal-naive and least-squares trend + season references on the same "
        "context, runs the zero-shot forecast on the GPU and draws it against the truth, writes an evaluation report comparing "
        "the model median with every reference and the series' noise floor, and exports the forecast and provenance — without a "
        "repository clone, DIMER worker, credential, upload, configuration edit or runtime restart (NOTEBOOK_SPEC 2.2 §5). On a "
        "CPU-only runtime Section 1 stops at once with the runtime-change instruction."
    ),
    "byod": (
        "Set `USE_BYOD = True` and either set `BYOD_CSV_PATH` to a CSV in this runtime (any runtime) or leave it empty in Colab to "
        "open the upload dialog, then re-run from Section 4. The CSV has a `timestamp` column and one or more numeric target "
        "columns (comma, semicolon or tab separated), with unique, increasing, regularly spaced timestamps and no empty values, "
        "and at least `MIN_CONTEXT + HORIZON` rows; a longer series than `MAX_CONTEXT + HORIZON` keeps its most recent rows "
        "(reported). Every refusal names the file and the rule. The exported forecast keeps your timestamps, and the result "
        "records the last context timestamp and the spacing. Uploaded files stay inside this runtime."
    ),
    "title": "Toto 2.0 — DIMER zero-shot forecasting tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            f"https://github.com/kurtvalcorza/{REPO}",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/toto_forecasting_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datadog%2FToto--2.0--2.5B-ffcc4d?style=flat",
            "https://huggingface.co/Datadog/Toto-2.0-2.5B",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-DataDog%2Ftoto-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/DataDog/toto",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2605.20119-b31b1b.svg", "https://arxiv.org/abs/2605.20119"),
    ],
    "capability": "zero-shot multivariate probabilistic time-series forecasting with chronological evaluation, using the pinned `Datadog/Toto-2.0-2.5B` checkpoint",
    "intro": (
        "At inference Toto 2.0 consumes the context of one or more variates in 32-step patches and decodes nine forecast "
        "quantiles (q=0.1 … q=0.9) per step of the requested horizon; the pipeline reports q=0.5 as the point forecast (a model "
        "median, not a mean) and keeps all nine quantiles. **No adaptation occurs:** no gradient training, fine-tuning, "
        "exogenous-variable conditioning or preprocessing fitting happens in this notebook — the upstream checkpoint supplies "
        "the weights and the model configuration, and the carried package adds snapshot verification, the input contract "
        "(`validate_target`, `validate_horizon`, `decode_block_size` and patch-multiple rules, mask-aware left padding), the "
        "`mae` / `rmse` / `last_value_baseline` / `interval_coverage` helpers and the public `validate_inputs` / "
        "`evaluation_report` stages.\n\n"
        "**Read this before Section 7: the default series is easy, so the last-value baseline flatters any forecaster.** The two "
        "variates are trends plus known seasons plus small noise. Repeating the last value scored MAE 1.108; repeating the last "
        "season 0.650; a least-squares fit that knows the generator's two seasons 0.036 — next to the noise floor of 0.035 that "
        "no forecaster can beat. Toto's recorded median MAE on this sample is 0.073: far ahead of the naive baselines, but about "
        "twice the floor. Section 7 prints all of these, so you can see what the model adds; on your own series (BYOD) the "
        "seasonal and rolling-origin comparisons are the informative ones."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to —\n\n"
        "1. **Make** a leakage-safe chronological holdout, move it with `HOLDOUT_OFFSET`, and **explain** why no future value may reach the model (Section 5).\n"
        "2. **Read** an input manifest and a refusal message from the forecasting contract (Section 5).\n"
        "3. **Interpret** a quantile forecast from a figure: the median as the point forecast, the q10–q90 band as model quantiles rather than a guaranteed interval (Section 6).\n"
        "4. **Compare** the model with last-value, seasonal-naive and trend + season references and the noise floor, per variate, and **say** what the comparison can and cannot show (Section 7).\n"
        "5. **Predict**, run and **explain** a rolling-origin check at an earlier holdout, in an optional activity (Section 9)."
    ),
    "exclusions": (
        "gradient training or fine-tuning, exogenous/covariate conditioning (not part of the Toto 2.0 open inference release), "
        "missing-value imputation (non-finite targets are rejected), anomaly detection, or calibrated prediction intervals. The "
        "quantiles are model quantiles, not guaranteed coverage; a single holdout is not a deployment-variability estimate."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh **Linux x86_64** runtime **with a CUDA GPU** — Google Colab with a T4 GPU, or a Kaggle GPU kernel. A 15 GB T4 is enough: the 2.5B checkpoint is loaded in its stored float32 precision (9.8 GB of weights) and the previous notebook peaked at 9.45 GiB of CUDA memory on a T4. Section 1 stops at once on a CPU-only runtime. The kernel's own Python version does not matter: the notebook installs nothing into it, and runs every stage with CPython 3.12.12 in an isolated environment built from {n_locked} hash-locked packages (`toto-2` 2.0.0, `torch` 2.7.0, `numpy` 1.26.4, `pandas` 2.2.3). About 9.2 GiB of disk is needed for the checkpoint and about 8 GB for the isolated environment.",
        "- **Knowledge:** basic Python and NumPy. The glossary below explains context window, horizon, quantile and the other forecasting terms.",
        "- **Data:** the default sample is a deterministic two-variate trend + seasonal series (320 steps, fixed seed) generated in code, so nothing else is downloaded and no private data is needed. Optional BYOD: one UTF-8 CSV with a unique `timestamp` column and one or more finite numeric target columns in chronological order, regularly spaced, at least 80 rows (32 context + 48 holdout) at the default horizon; missing values are refused rather than silently imputed. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "guided": {
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who can run cells in a hosted notebook and read short Python and NumPy, and who "
                "want to see a large time-series foundation model used honestly: a chronological holdout, a quantile forecast read "
                "from a figure, and a comparison against references that tells you what the model actually adds. No experience with "
                "Toto or transformers is assumed; the glossary below explains every term.\n\n"
                "**Running it.** Choose a GPU runtime (*Runtime → Change runtime type → T4 GPU* in Colab), then *Runtime → Run all*. The "
                "default path needs no edit, no upload, no account, no token and no runtime restart. Section 2 builds an isolated "
                "environment and Section 3 downloads 9.8 GB, which take the longest.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell calls "
                "`run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated environment's "
                "Python, streams what it prints, and stops the notebook with the stage's own error message if it fails. Stages hand "
                "results to each other only through files.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–9) are the workflow. *Infrastructure cells* (Sections 1–3) are "
                "collapsed and titled **Infrastructure**; you may run them without studying their implementation.\n\n"
                "**Form controls.** `USE_BYOD`, `BYOD_CSV_PATH` and `HORIZON` (Section 4); `HOLDOUT_OFFSET`, `CONTEXT_LENGTH` and "
                "`SEASON_PERIOD` (Section 5); `RUN_ACTIVITY` and `ACTIVITY_OFFSET` (Section 9). Leave them at their defaults for the "
                "first run.\n\n"
                "**Section tags.** **[Concept]** — what the model does and why. **[Evaluation practice]** — how the evidence is "
                "produced and how to read it. **[Engineering]** — reproducibility, provenance and packaging.\n\n"
                "**Predict, then check.** Before Sections 5, 6 and 7 a **Predict before running** prompt asks you to commit to an "
                "expectation; **What to notice** follows each stage; a collapsed **Check your reasoning** answer follows each "
                "checkpoint. The sample answers quote the recorded runs of this sample."
            ),
            (
                "## The task: Input → Model → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Split** | a 2 × 320 series | chronological holdout: the last `HORIZON` = 48 steps withheld | 272 context steps, 48 truth steps per variate |\n"
                "| **Validate** | the context | `validate_inputs` (shape, length 32..16,384, finiteness, decode rules) | an input manifest with a recorded rejection |\n"
                "| **References** | the context | last value; seasonal naive; least-squares trend + season | three reference forecasts |\n"
                "| **Forecast** | the context, left-padded to a 32-step patch multiple | Toto 2.0 (2.5B), zero-shot, on the GPU | nine quantiles per step; the median is the point forecast; a figure |\n"
                "| **Evaluate** | forecast, references, truth | `evaluation_report` + references + the noise floor, per variate | MAE / RMSE, q10–q90 coverage, an interpretation |\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1–3 | [Engineering] | GPU check, carried code, isolated environment, verified checkpoint | versions, digests |\n"
                "| 4. Data | [Evaluation practice] | synthetic series or a BYOD CSV | shape, digest, any trim |\n"
                "| 5. Holdout, validation, references | [Evaluation practice] | split, input manifest, three references | the manifest, reference errors |\n"
                "| 6. Forecast | [Concept] | zero-shot quantile forecast and a figure | median, band, truth |\n"
                "| 7. Evaluation | [Evaluation practice] | the report | the comparison and its interpretation |\n"
                "| 8. Export | [Engineering] | CSV (with timestamps) and result JSON with provenance | the files |\n"
                "| 9. Optional activity | [Evaluation practice] | a rolling-origin check at an earlier holdout | model vs references at two origins |\n"
                "| Troubleshooting | [Engineering] | every failure and what to do | when something fails |\n"
                "| Interpretation and conclusion | [Evaluation practice] | what was and was not shown | your conclusion |"
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **Context window** | The past observations the model sees (272 steps by default; `CONTEXT_LENGTH` can shorten it). |\n"
                "| **Horizon** | How many future steps are forecast (`HORIZON`, 48 by default). |\n"
                "| **Chronological holdout / origin** | The withheld `HORIZON` steps; `HOLDOUT_OFFSET` moves them earlier, and the step before them is the forecast origin. |\n"
                "| **Patch / left padding** | Toto reads the context in 32-step patches; a shorter remainder is padded on the left with masked (unobserved) positions. |\n"
                "| **Decode block** | How many steps are decoded per block (768 by default); at a 48-step horizon it is a single pass either way. |\n"
                "| **Quantile forecast** | For each step, nine values below which the model expects the truth with probability 0.1 … 0.9. |\n"
                "| **Median vs mean** | The q=0.5 quantile is the point forecast here; it is not the average of possible futures. |\n"
                "| **q10–q90 band / coverage** | The range between the 0.1 and 0.9 quantiles; coverage is the share of truth points inside it (nominal 0.8, not guaranteed). |\n"
                "| **Last-value / seasonal naive** | Repeat the last value / the last full season across the horizon. |\n"
                "| **Trend + season reference** | A least-squares fit of level, slope and one sine/cosine pair per season on the context, extended forward. |\n"
                "| **Noise floor** | The error of the exact noiseless generator on the holdout (synthetic series only). |\n"
                "| **Hash-locked environment / stage** | The isolated Python environment every stage runs in; one workflow step run as its own process. |\n\n"
                "</details>"
            ),
        ],
        "infrastructure": {
            "weights": (
                "**Trust boundary (Section 3).** `model.safetensors` holds tensors only — no pickled code — and is accepted only at the "
                "pinned size and SHA-256. **What to notice:** `fetched` lists the 9.8 GB `model.safetensors` on a first run (several "
                "minutes), then the verified-file count."
            ),
        },
    },
    "cells": [
        {
            "md": (
                "## 4. Generate the synthetic sample or supply your own series · [Evaluation practice]\n\n"
                "The default sample is **synthetic**: 320 steps of two variates — a linear trend plus a sine season plus small Gaussian "
                "noise, and a second variate coupled to the first plus a cosine season — from a fixed seed, so it needs no download and "
                "its float32 SHA-256 is printed for the record. It has a real future, so the evaluation can score the forecast, but a "
                "synthetic series says nothing about any deployment domain.\n\n"
                "**Bring your own series.** Set `USE_BYOD = True` and `BYOD_CSV_PATH` (or, in Colab, leave it empty for the upload "
                "dialog). The header is inspected before pandas reads the file, so duplicate names cannot be silently renamed; the "
                "delimiter is detected (comma, semicolon or tab); a text value in a target column is refused with the column and the "
                "values; empty values, unparseable or irregular timestamps and too few rows stop with their own message; a series "
                "longer than `MAX_CONTEXT + HORIZON` keeps its most recent rows and reports the trim; and your timestamps are kept for "
                "the export."
            ),
            "code": (
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_CSV_PATH = ''  # @param {{type:\"string\"}}\n"
                "HORIZON = 48  # @param {{type:\"integer\"}}\n\n"
                "def upload_one(what, field):\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError:\n"
                "        raise RuntimeError(f'{{field}} is empty, and the upload dialog exists only in Google Colab: set {{field}} to {{what}} in this runtime.') from None\n"
                "    uploaded = files.upload()\n"
                "    if not uploaded:\n"
                "        raise RuntimeError(f'The upload was cancelled or empty: no file was received. Run this cell again and choose {{what}}, or set {{field}}.')\n"
                "    if len(uploaded) != 1:\n"
                "        raise ValueError(f'Upload exactly one file ({{what}}); got {{sorted(uploaded)}}.')\n"
                "    upload_name, payload = next(iter(uploaded.items()))\n"
                "    path = ROOT / 'inputs' / Path(upload_name).name\n"
                "    path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    path.write_bytes(payload)\n"
                "    return str(path)\n\n"
                "csv_path = ''\n"
                "if USE_BYOD:\n"
                "    csv_path = BYOD_CSV_PATH or upload_one('one CSV with a timestamp column', 'BYOD_CSV_PATH')\n"
                "run_stage('data', use_byod=USE_BYOD, byod_csv_path=csv_path, horizon=HORIZON)"
            ),
        },
        {
            "md": "**What to notice:** `sample_kind: 'synthetic'`, shape `[2, 320]`, `horizon: 48` and the float32 SHA-256 `4129ad3d…`.",
        },
        {
            "md": (
                "## 5. Chronological holdout, validation and references · [Evaluation practice]\n\n"
                "The `HORIZON` steps ending `HOLDOUT_OFFSET` steps before the end are **withheld** as the truth (the last 48 by "
                "default); only the earlier context — all of it, or its last `CONTEXT_LENGTH` steps — is passed to the model, so no "
                "future value leaks into the forecast. `validate_inputs` applies the model-independent checks `forecast` applies and "
                "returns an **input manifest**, written to `outputs/{stem}_input_manifest.json`; the stage also validates a "
                "deliberately too-short context and records the refusal.\n\n"
                "Three references are computed from the same context: the **last-value baseline**; a **seasonal-naive** forecast that "
                "repeats the last season (57 steps for the sample, the first variate's period); and a **least-squares trend + season "
                "reference** with one sine/cosine pair per season — for the sample, the generator's own two seasons, which makes it an "
                "informed reference, not a naive baseline. For your own series set `SEASON_PERIOD` in steps, or leave it 0 to use the "
                "largest periodogram peak of the detrended context. For the sample the stage also prints the **noise floor**.\n\n"
                "**Predict before running:** which of the three references will come closest to the truth on this series?"
            ),
            "code": (
                "HOLDOUT_OFFSET = 0  # @param {{type:\"integer\"}}\n"
                "CONTEXT_LENGTH = 0  # @param {{type:\"integer\"}}\n"
                "SEASON_PERIOD = 0.0  # @param {{type:\"number\"}}\n"
                "run_stage('validate', holdout_offset=HOLDOUT_OFFSET, context_length=CONTEXT_LENGTH, season_period=SEASON_PERIOD)"
            ),
        },
        {
            "md": (
                "**What to notice:** context length 272, the `short-context-probe` refusal, last-value MAE 1.1080 / RMSE 1.3108, "
                "seasonal-naive 0.6497 / 0.7234, the trend + season reference 0.0363 / 0.0479, and the noise floor 0.0349 / 0.0467.\n\n"
                "<details>\n<summary>Check your reasoning (open after running)</summary>\n\n"
                "The trend + season reference, by far: it knows the generator's two seasons and fits them on 272 steps, so it lands "
                "next to the noise floor. Seasonal naive captures the season but not the trend (and only one variate's period); last "
                "value captures neither. These three numbers are the scale on which to read the model's error in Section 7.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 6. Forecast · [Concept]\n\n"
                "`forecast` returns `quantiles` of shape `(variates, 9, horizon)` at levels 0.1 … 0.9, and `median` — the q=0.5 slice, "
                "which is the **point forecast**. The context is left-padded to the next 32-step patch multiple with masked positions "
                "(272 → 288). The median is a model median, not a mean, and the other quantiles are model quantiles rather than "
                "guaranteed confidence intervals; nothing is calibrated here. The stage prints the peak CUDA memory and draws the "
                "context, the withheld truth, the median and the q10–q90 band per variate to `outputs/{stem}_forecast.png`, shown "
                "below.\n\n"
                "**Predict before running:** will the truth stay inside the q10–q90 band for most of the 48 steps? Will the band widen "
                "further into the horizon?"
            ),
            "code": (
                "run_stage('forecast')\n"
                "from IPython.display import Image, display\n"
                "display(Image(filename=str(OUTPUTS / '{stem}_forecast.png')))"
            ),
        },
        {
            "md": (
                "**What to notice:** `point_forecast: 'median (q=0.5)'`, `context_padding: 16`, `device: 'cuda'`, `peak_cuda_gib` "
                "(about 9.45 on a T4 in the previous notebook), and in the figure the truth mostly inside the band."
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report · [Evaluation practice]\n\n"
                "`evaluation_report` carries the repository's own `mae` and `rmse` on the median, the `interval_coverage` of the "
                "q10–q90 band (nominal 0.8), and the last-value baseline, with the verdict `sample-sanity` — one holdout, no dispersion "
                "estimate, not a benchmark (without withheld truth the verdict is `not-measurable`). The stage adds the seasonal-naive "
                "and trend + season references, the noise floor (sample), **per-variate** MAE / RMSE, the coverage granularity (96 "
                "points: steps of 1/96 ≈ 0.010) and an **interpretation** line. The report is written to "
                "`outputs/{stem}_evaluation_report.json`.\n\n"
                "**Predict before running:** where will Toto's MAE fall on the scale from Section 5 — nearer last-value, seasonal naive, "
                "or the trend + season reference?"
            ),
            "code": "run_stage('evaluate')",
        },
        {
            "md": (
                "**What to notice:** the model's MAE (0.0732 recorded, RMSE 0.0887, coverage 0.823) beside the references, the "
                "per-variate rows, and the interpretation line.\n\n"
                "**Checkpoint:** the model is 15 times better than last-value. What does that show, and what does it not?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "It shows a sound zero-shot forecast: with no knowledge of the series' form the model lands between seasonal naive "
                "(0.650) and the informed trend + season fit (0.036), about twice the noise floor (0.035). It does not show that the "
                "model is 15 times better than a reasonable method: a flat forecast is a weak reference on a trending seasonal series, "
                "and a fit that knows the seasons does better here. On real series, where nobody knows the seasons, the seasonal and "
                "rolling-origin comparisons (the activity) are the informative ones.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance · [Engineering]\n\n"
                "The CSV keeps variate, step, **timestamp** (your timestamps for BYOD; the step index for the sample), median, truth, "
                "the three references and all nine quantiles aligned row by row. `outputs/{stem}_result.json` preserves the forecast "
                "summary, the evaluation report, the input manifest, the sample identity and digest with the last context timestamp "
                "and the spacing, the notebook's source, the model identifier, revision and licence, and the isolated environment's "
                "versions, device and peak CUDA memory. No credentials are recorded. The stage only reads earlier results, so "
                "re-running it alone is safe."
            ),
            "code": "run_stage('export')",
        },
        {
            "md": (
                "## 9. Optional activity: does the model stay ahead at another origin? · [Evaluation practice]\n\n"
                "**Predict → Change → Run → Observe → Explain.** **Predict:** with `ACTIVITY_OFFSET = 48` the holdout moves 48 steps "
                "earlier (steps 224–271, with 224 steps of context). Will the model's lead over last-value and seasonal naive hold? "
                "**Change:** tick `RUN_ACTIVITY` and set `ACTIVITY_OFFSET`. **Run** this cell: it forecasts that earlier holdout. "
                "**Observe** the two rows — canonical and earlier origin — for the model and both naive references. **Explain** "
                "whether one holdout was enough to judge the model. The activity writes only to `outputs/activity/`."
            ),
            "code": (
                "RUN_ACTIVITY = False  # @param {{type:\"boolean\"}}\n"
                "ACTIVITY_OFFSET = 48  # @param {{type:\"integer\"}}\n"
                "if RUN_ACTIVITY:\n"
                "    run_stage('activity', holdout_offset=ACTIVITY_OFFSET)\n"
                "else:\n"
                "    print('Optional activity skipped: tick RUN_ACTIVITY to run it. The canonical outputs are complete.')"
            ),
        },
        {
            "md": (
                "**What to notice (if you ran it):** the `model_mae` column at the two origins and how much it moves.\n\n"
                "<details>\n<summary>Check your reasoning (open after running)</summary>\n\n"
                "No hosted run of the activity is recorded yet, so compare your rows. If the model's error and its lead over the naive "
                "references change noticeably between two origins, one holdout was not enough to rank methods — which is why a "
                "rolling-origin evaluation over many windows is the standard for forecasting. If they hold, that is one more piece "
                "of evidence, still from the same synthetic generator.\n\n"
                "</details>"
            ),
        },
    ],
    "closing": (
        "## Troubleshooting · [Engineering]\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| `No CUDA GPU detected, and this notebook's model needs one` | a CPU-only runtime | *Runtime → Change runtime type → T4 GPU* (Colab) or turn on a GPU accelerator (Kaggle), then *Run all*. |\n"
        "| `Not enough free disk` | a used runtime, or a small disk | Start a fresh runtime; about 9.2 GiB for the checkpoint and 8 GB for the environment are needed. |\n"
        "| `CUDA out of memory` in Section 6 | a GPU smaller than a 15 GB T4, or another process on it | Use a T4 or larger; to free a GPU another process holds, choose *Runtime → Disconnect and delete runtime*, then *Run all* in the fresh runtime. |\n"
        "| `This notebook needs a Linux x86_64 runtime` | macOS, Windows or ARM kernel | Use Colab, Kaggle or a Linux x86_64 Jupyter kernel. |\n"
        "| `uv … wheel size/hash mismatch` or a `--require-hashes` error | a corrupted or substituted download | Re-run Section 2; never remove a pin or a hash. |\n"
        "| a size or SHA-256 mismatch in Section 3 | an interrupted 9.8 GB download or a different file | Delete `weights/toto-2.0-2.5b/` and re-run Section 3; never edit the manifest. |\n"
        "| `Stage '…' failed …: … is missing: run the stage that writes it` | a cell run out of order | Run the notebook in order from Section 4 (or *Run all*). |\n"
        "| `HOLDOUT_OFFSET=… leaves no context` / `CONTEXT_LENGTH=… exceeds` | a holdout or context setting too large for the series | Lower the field, or supply a longer series. |\n"
        "| `BYOD_CSV_PATH is empty, and the upload dialog exists only in Google Colab` | BYOD outside Colab with no path | Set `BYOD_CSV_PATH`. |\n"
        "| `The upload was cancelled or empty` / `Upload exactly one file` | a cancelled or multi-file upload | Run the cell again and choose one CSV. |\n"
        "| `<file>: no timestamp column in the header` | a missing or misnamed column, or another delimiter | Name the column `timestamp`; comma, semicolon and tab are read. |\n"
        "| `<file>: target columns must be numeric; non-numeric values in {{…}}` | a text or identifier column | Remove the column or fix the values. |\n"
        "| `<file>: empty values in {{…}}` | gaps in a target | Fill or drop those rows; this pipeline does not impute. |\n"
        "| `<file>: timestamps must be regularly spaced` / `unique and strictly increasing` / `does not parse` | gaps, duplicates, unsorted rows or an unknown date format | Sort, de-duplicate, resample, or write ISO dates. |\n"
        "| `<file>: N rows; this tutorial needs at least …` | too short for context + holdout | Supply more rows or lower `HORIZON`. |\n"
        "| `trimmed: {{…}}` (a note, not an error) | more than `MAX_CONTEXT + HORIZON` rows | The most recent rows are kept; older rows cannot be context. |\n\n"
        "## Interpretation and limits\n\n"
        "The forecast is zero-shot; no gradient training or fine-tuning occurs. q=0.5 is a model median, and the other quantiles "
        "are model quantiles rather than guaranteed confidence intervals — the reported q10–q90 coverage on one holdout of 96 "
        "points moves in steps of about 0.010 and is not a calibration statement. On the default synthetic series the model is far "
        "ahead of last-value and seasonal naive, but an informed least-squares fit of the generator's seasons is closer to the "
        "noise floor, so the comparison shows a sound workflow rather than a margin over reasonable methods. MAE/RMSE from one "
        "chronological holdout must be repeated over representative periods of a real deployment series (rolling origins), "
        "against seasonal references, before any conclusion. Regime changes, missing values (rejected by the contract), irregular "
        "sampling and horizons far beyond the context all degrade results in ways the pipeline does not detect. The pipeline "
        "provides no fine-tuning, covariate conditioning, anomaly detection or imputation capability.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can acquire and "
        "digest-verify the pinned checkpoint, validate the demonstrated input, execute the public pipeline path, and emit the shown "
        "machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** establish "
        "benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on an unseen "
        "domain.\n\n"
        "## Conclusion · [Evaluation practice]\n\n"
        "Write three sentences: what the chronological holdout guarantees; where the model's error falls between the naive "
        "references, the informed trend + season fit and the noise floor; and what you would need before trusting the model on a "
        "real series.\n\n"
        "<details>\n<summary>Sample conclusion (open after writing yours)</summary>\n\n"
        "The holdout withheld the last 48 steps of both variates before the model or any reference saw them, so nothing leaked. The "
        "model's median MAE (0.073 recorded) is far below last-value (1.108) and seasonal naive (0.650) but about twice the noise "
        "floor (0.035), and a fit that knows the generator's seasons (0.036) does better, so on this easy series the model shows a "
        "good zero-shot forecast rather than a margin over informed methods. Before trusting it on a real series I would compare it "
        "with seasonal references over many rolling origins from representative periods, per variate, and check its band coverage "
        "over many points.\n\n"
        "</details>\n\n"
        "**Next experiments:** run the activity at offsets 48, 96 and 144; set `CONTEXT_LENGTH` to 64 and watch the error and the "
        "band; enable `USE_BYOD` with a CSV from your own domain, set `SEASON_PERIOD` to its cycle, and compare the model with "
        "seasonal naive at several origins.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/toto-forecasting-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/toto-forecasting-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/toto-forecasting-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/DataDog/toto\n"
        "- Paper: https://arxiv.org/abs/2605.20119"
    ),
}
