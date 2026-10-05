"""Stage runner for the standalone Toto 2.0 (2.5B) zero-shot forecasting tutorial (NOTEBOOK_SPEC 2.2 §25.13 isolated environment).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/``) and runs every stage with the interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --outputs OUTPUTS --weights WEIGHTS --stage data --options '{...}'

Nothing is installed into the notebook kernel. Each stage is a separate process and starts from files only: the
verified snapshot under ``--weights``, the series and records of earlier stages under ``RUN_DIR/state`` (NumPy ``.npy``
and JSON), and the learner-facing exports under ``--outputs``. The 2.5B checkpoint runs on a CUDA GPU (peak about 9.5 GiB on a T4). On
failure a stage writes ``RUN_DIR/state/<stage>.error.json``, which the
notebook re-raises in the kernel.

Stages: weights → data → validate → forecast → evaluate → export, plus the optional ``activity`` (a rolling-origin
comparison at an earlier holdout). ``data`` and
``validate`` import no model library, so CI exercises them directly. No stage asserts a quality level.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

STEM = "toto_forecasting"
PACKAGE = "toto_forecasting_pipeline"
SYNTHETIC_STEPS = 320
SYNTHETIC_SEED = 11
SYNTHETIC_PERIODS = (2 * math.pi * 9, 2 * math.pi * 13)  # sin(t / 9) and cos(t / 13): the generator's own seasons
SYNTHETIC_NAIVE_PERIOD = 57  # nearest integer to 2·pi·9 = 56.5 steps, the first variate's season
DELIMITERS = (",", ";", "\t")


class Run:
    """Paths of one run: carried sources and state under ``root``; learner-facing files under ``outputs``."""

    def __init__(self, root: Path, outputs: Path, weights: Path, options: dict[str, Any]) -> None:
        self.root = root
        self.out = outputs
        self.weights = weights
        self.options = options
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    def write_state(self, name: str, value: Any) -> Path:
        path = self.state / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook in order from Section 4)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path


def package(root: Path):
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module(PACKAGE)


def pipeline_module(root: Path):
    package(root)
    return importlib.import_module(f"{PACKAGE}.pipeline")


def snapshot_dir(run: Run, P) -> Path:
    return run.weights / P.MODEL_KEY


def rounded(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [rounded(v, digits) for v in value]
    return value


def load_series(run: Run, needed_by: str):
    import numpy as np

    data = run.read_state("data.json", needed_by)
    return data, np.load(run.state / "series.npy")


# --------------------------------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------------------------------


def stage_weights(run: Run) -> None:
    P = package(run.root)
    M = pipeline_module(run.root)
    snapshot = snapshot_dir(run, P)
    snapshot.mkdir(parents=True, exist_ok=True)
    carried = run.root / "weights" / P.MODEL_KEY / M.MANIFEST_NAME
    manifest = json.loads(carried.read_text(encoding="utf-8"))
    if (manifest["modelId"], manifest["revision"]) != (P.MODEL_ID, P.MODEL_REVISION):
        raise RuntimeError("the carried manifest does not name the identity carried by the package; regenerate the notebook")
    shutil.copyfile(carried, snapshot / M.MANIFEST_NAME)
    print({"model_id": P.MODEL_ID, "revision": P.MODEL_REVISION, "license": P.MODEL_LICENSE, "files": len(manifest["files"]), "total_bytes": manifest["totalBytes"]})
    fetched = P.stage_missing_files(snapshot, allow_download=True)
    print({"weights_dir": str(snapshot), "fetched": fetched})
    verified = P.verify_snapshot(snapshot)
    print({"verified": verified, "checkpoint": M.WEIGHTS_FILE, "note": "model.safetensors holds tensors only (no pickled code); it is accepted only at the pinned size and SHA-256"})
    run.write_state("weights.json", {"snapshot": str(snapshot), "fetched": fetched})


# --------------------------------------------------------------------------------------------------
# data (TRX-m2) and validation with baselines (TRX-m1); no model library
# --------------------------------------------------------------------------------------------------


def synthetic_series():
    """The tutorial's two coupled variates and their noiseless signal (the second variate carries half the first's noise)."""
    import numpy as np

    rng = np.random.default_rng(SYNTHETIC_SEED)
    t = np.arange(SYNTHETIC_STEPS)
    first = 0.01 * t + np.sin(t / 9) + rng.normal(0, 0.04, len(t))
    second = 0.5 * first + np.cos(t / 13) + rng.normal(0, 0.04, len(t))
    clean1 = 0.01 * t + np.sin(t / 9)
    clean2 = 0.5 * clean1 + np.cos(t / 13)
    return np.vstack([first, second]), np.vstack([clean1, clean2])


def sniff_delimiter(path: Path) -> str:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        first = handle.readline()
    counts = {d: first.count(d) for d in DELIMITERS}
    best = max(counts, key=counts.get)
    return best if counts[best] else ","


def read_byod_csv(P, path: Path, horizon: int) -> dict[str, Any]:
    """Read a BYOD CSV with every refusal naming the file and the rule; trim a long series to the most recent rows."""
    import numpy as np
    import pandas as pd

    if not path.is_file():
        raise FileNotFoundError(f"BYOD_CSV_PATH {str(path)!r} is not a file in this runtime: upload the CSV or correct the path.")
    try:
        path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}: not UTF-8 text (byte {exc.start}: {exc.reason}); save the CSV as UTF-8") from None
    delimiter = sniff_delimiter(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle, delimiter=delimiter), [])
    header = [h.strip() for h in header]
    if not header or any(not h for h in header) or len(header) != len(set(header)):
        raise ValueError(f"{path.name}: the header must have non-empty, unique column names (duplicates are refused before pandas could rename them); got {header}")
    if "timestamp" not in header:
        raise ValueError(f"{path.name}: no `timestamp` column in the header {header} (delimiter read as {delimiter!r}; comma, semicolon and tab are recognised)")
    names = [c for c in header if c != "timestamp"]
    if not names:
        raise ValueError(f"{path.name}: no target column besides `timestamp`")
    frame = pd.read_csv(path, sep=delimiter, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    frame.columns = [c.strip() for c in frame.columns]
    bad = {}
    missing = {}
    for column in names:
        raw = frame[column].str.strip()
        numbers = pd.to_numeric(raw.where(raw != ""), errors="coerce")
        text = raw[(raw != "") & numbers.isna()]
        if len(text):
            bad[column] = text.unique().tolist()[:3]
        if (raw == "").any():
            missing[column] = [int(i) + 2 for i in raw.index[raw == ""][:3]]  # +2: header line, 1-based
    if bad:
        raise ValueError(f"{path.name}: target columns must be numeric; non-numeric values in {bad}. Remove text columns (or move identifiers out of the file) and fix the values.")
    if missing:
        raise ValueError(f"{path.name}: empty values in {missing} (CSV line numbers); this pipeline does not impute — fill or drop those rows so the series stays regularly spaced.")
    try:
        timestamps = pd.to_datetime(frame["timestamp"].str.strip(), errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path.name}: the `timestamp` column does not parse as dates/times ({exc})") from None
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError(f"{path.name}: timestamps must be unique and strictly increasing")
    steps = timestamps.diff().dropna()
    if len(steps) and steps.nunique() != 1:
        raise ValueError(f"{path.name}: timestamps must be regularly spaced (found {steps.nunique()} different gaps, e.g. {sorted(map(str, steps.unique()))[:3]}); fill or resample the gaps first")
    series = frame[names].astype(float).to_numpy().T
    keep = P.MAX_CONTEXT + horizon
    trimmed = None
    if series.shape[1] > keep:
        trimmed = {"original_rows": int(series.shape[1]), "kept_rows": keep, "dropped_oldest": int(series.shape[1] - keep), "first_kept_timestamp": str(timestamps.iloc[-keep])}
        series = series[:, -keep:]
        timestamps = timestamps.iloc[-keep:]
    if series.shape[1] < P.MIN_CONTEXT + horizon:
        raise ValueError(f"{path.name}: {series.shape[1]} rows; this tutorial needs at least {P.MIN_CONTEXT + horizon} ({P.MIN_CONTEXT} context + {horizon} holdout)")
    values = P.validate_target(series, max_context=keep)
    return {"series": np.asarray(values, dtype=float), "names": names, "delimiter": delimiter, "trimmed": trimmed, "first": str(timestamps.iloc[0]), "last": str(timestamps.iloc[-1]), "step": str(steps.iloc[0]) if len(steps) else None, "timestamps": [str(x) for x in timestamps]}


def stage_data(run: Run) -> None:
    import numpy as np

    P = package(run.root)
    opts = run.options
    use_byod = bool(opts.get("use_byod", False))
    horizon = P.validate_horizon(int(opts.get("horizon", 48)))
    for path in sorted(run.out.glob(f"{STEM}_*")):
        if path.is_file():
            path.unlink()
    for name in ("data.json", "validated.json", "forecast.json", "evaluate.json"):
        (run.state / name).unlink(missing_ok=True)
    if use_byod:
        path = Path(opts.get("byod_csv_path") or "")
        info = read_byod_csv(P, path, horizon)
        series, names = info["series"], info["names"]
        record = {"sample_kind": "BYOD", "name": path.name, "delimiter": info["delimiter"], "trimmed": info["trimmed"], "time_range": [info["first"], info["last"]], "step": info["step"]}
        (run.state / "timestamps.json").write_text(json.dumps(info["timestamps"]), encoding="utf-8")
        if info["trimmed"]:
            print({"trimmed": info["trimmed"], "note": f"kept the most recent MAX_CONTEXT + HORIZON = {info['trimmed']['kept_rows']} rows; older rows cannot be used as context"})
    else:
        series, _signal = synthetic_series()
        names = ["synthetic-trend-season", "synthetic-coupled-season"]
        record = {"sample_kind": "synthetic", "name": "synthetic_two_variate_320.csv", "generator": "x1 = 0.01·t + sin(t/9) + N(0, 0.04); x2 = 0.5·x1 + cos(t/13) + N(0, 0.04); seed 11"}
        (run.state / "timestamps.json").unlink(missing_ok=True)
    if series.shape[1] < P.MIN_CONTEXT + horizon:
        raise ValueError(f"this tutorial needs at least {P.MIN_CONTEXT + horizon} rows ({P.MIN_CONTEXT} context + {horizon} holdout); got {series.shape[1]}")
    digest = hashlib.sha256(np.ascontiguousarray(series, dtype=np.float32).tobytes()).hexdigest()
    np.save(run.state / "series.npy", series)
    run.write_state("data.json", {**record, "names": names, "horizon": horizon, "shape": list(series.shape), "float32_sha256": digest})
    print({**{k: v for k, v in record.items() if k != "trimmed"}, "shape": list(series.shape), "horizon": horizon, "float32_sha256": digest})


def estimate_period(context) -> tuple[float | None, str]:
    """The dominant period of the linearly detrended context: the largest periodogram peak (3 steps up to half the context),
    refined by the least-squares fit between the neighbouring frequency bins (the FFT grid alone drifts in phase over a
    long context)."""
    import numpy as np

    y = np.asarray(context, dtype=float)
    n = len(y)
    t = np.arange(n, dtype=float)
    X = np.stack([np.ones_like(t), t], 1)
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    power = np.abs(np.fft.rfft(resid)) ** 2
    k_min = 2  # at most half the context per period
    k_max = n // 3  # at least 3 steps per period
    if k_max < k_min:
        return None, "context too short to estimate a season"
    k = int(np.argmax(power[k_min : k_max + 1])) + k_min

    def sse(period: float) -> float:
        D = np.stack([np.ones_like(t), t, np.sin(2 * np.pi * t / period), np.cos(2 * np.pi * t / period)], 1)
        r = y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
        return float(r @ r)

    candidates = np.linspace(n / (k + 1), n / max(k - 1, 1), 201)
    best = float(min(candidates, key=sse))
    return best, f"largest periodogram peak of the detrended context (frequency {k}/{n}), refined by least squares"


def least_squares_forecast(context, horizon: int, periods):
    """Fit level + trend (+ one sine/cosine pair per period) to each variate's context by least squares and extend it."""
    import numpy as np

    values = np.atleast_2d(np.asarray(context, dtype=float))
    n = values.shape[1]
    t_fit = np.arange(n, dtype=float)
    t_new = np.arange(n, n + horizon, dtype=float)
    periods = [p for p in (periods or ()) if p]

    def design(t):
        columns = [np.ones_like(t), t]
        for period in periods:
            columns += [np.sin(2 * np.pi * t / period), np.cos(2 * np.pi * t / period)]
        return np.stack(columns, 1)

    out = np.empty((values.shape[0], horizon))
    for i, y in enumerate(values):
        coef = np.linalg.lstsq(design(t_fit), y, rcond=None)[0]
        out[i] = design(t_new) @ coef
    return out


def seasonal_naive(context, horizon: int, period: int):
    """Repeat the last full season of each variate across the horizon."""
    import numpy as np

    return np.stack([np.resize(row[-period:], horizon) for row in np.atleast_2d(context)])


def split(series, horizon: int, offset: int, context_length: int):
    """Chronological split: the holdout is the ``horizon`` steps ending ``offset`` steps before the series' end; the context
    is everything before it (or its last ``context_length`` steps)."""
    end = series.shape[1] - offset
    if offset < 0 or end - horizon < 1:
        raise ValueError(f"HOLDOUT_OFFSET={offset} leaves no context before a {horizon}-step holdout in a {series.shape[1]}-step series")
    context = series[:, : end - horizon]
    if context_length:
        if context_length > context.shape[1]:
            raise ValueError(f"CONTEXT_LENGTH={context_length} exceeds the {context.shape[1]} steps available before the holdout")
        context = context[:, -context_length:]
    return context, series[:, end - horizon : end], end - horizon


def stage_validate(run: Run) -> None:
    import numpy as np

    P = package(run.root)
    data, series = load_series(run, "validate")
    horizon = data["horizon"]
    offset = int(run.options.get("holdout_offset", 0))
    context_length = int(run.options.get("context_length", 0))
    context, truth, start = split(series, horizon, offset, context_length)
    print({"ceilings": {"MIN_CONTEXT": P.MIN_CONTEXT, "MAX_CONTEXT": P.MAX_CONTEXT, "MAX_HORIZON": P.MAX_HORIZON}, "holdout_offset": offset, "context_length": int(context.shape[1]), "holdout_steps": [start, start + horizon]})
    manifest = P.validate_inputs(context, horizon=horizon, names=data["names"])
    try:
        P.validate_inputs(np.ones(P.MIN_CONTEXT - 1), horizon=horizon)
    except ValueError as exc:
        manifest["findings"].append({"input": "short-context-probe", "verdict": "rejected", "message": str(exc)})
    if data.get("trimmed"):
        manifest["findings"].append({"input": data["name"], "verdict": "trimmed", "message": f"kept the most recent {data['trimmed']['kept_rows']} of {data['trimmed']['original_rows']} rows"})
    run.write_output(f"{STEM}_input_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    season = float(run.options.get("season_period") or 0)
    if data["sample_kind"] == "synthetic":
        periods, period_source, naive_period = list(SYNTHETIC_PERIODS), "the generator's own seasons (sin(t/9), cos(t/13)): an informed reference, not a naive baseline", SYNTHETIC_NAIVE_PERIOD
    elif season > 0:
        periods, period_source, naive_period = [season], "SEASON_PERIOD set in Section 5", int(round(season))
    else:
        estimated, how = estimate_period(context[0])
        periods, period_source = [estimated], how + " (first variate; set SEASON_PERIOD to override)"
        naive_period = int(round(estimated)) if estimated else 1
    last = P.last_value_baseline(context, horizon)
    naive = seasonal_naive(context, horizon, min(naive_period, context.shape[1]))
    fit = least_squares_forecast(context, horizon, periods)
    baselines = {
        "last_value_baseline": {"mae": P.mae(truth, last), "rmse": P.rmse(truth, last), "what": "repeat the last observed value"},
        "seasonal_naive": {"mae": P.mae(truth, naive), "rmse": P.rmse(truth, naive), "what": f"repeat the last {naive_period} steps (one season)", "period_steps": naive_period},
        "least_squares_trend_season": {"mae": P.mae(truth, fit), "rmse": P.rmse(truth, fit), "what": "level + linear trend + one sine/cosine pair per season, fitted to the context by least squares", "period_steps": periods, "period_source": period_source},
    }
    reference = {}
    if data["sample_kind"] == "synthetic":
        _series, signal = synthetic_series()
        noise = truth - signal[:, start : start + horizon]
        reference["noise_floor"] = {"mae": float(np.mean(np.abs(noise))), "rmse": float(np.sqrt(np.mean(noise**2))), "what": "the error of the exact noiseless generator on the holdout: no forecaster can expect to beat it"}
    np.save(run.state / "least_squares.npy", fit)
    np.save(run.state / "seasonal_naive.npy", naive)
    run.write_state("validated.json", {"baselines": baselines, "reference": reference, "holdout_offset": offset, "context_length_option": context_length, "context_length": int(context.shape[1]), "start": start})
    for name, row in {**baselines, **reference}.items():
        print({name: rounded({k: v for k, v in row.items() if k in ("mae", "rmse", "period_steps")})})


# --------------------------------------------------------------------------------------------------
# model stages (CPU reference path)
# --------------------------------------------------------------------------------------------------


def gpu_pipeline(run: Run, P):
    return P.TotoForecastPipeline.from_pretrained(weights_dir=snapshot_dir(run, P), allow_download=False)


def holdout(run: Run, series, horizon: int):
    validated = run.read_state("validated.json", "holdout")
    return split(series, horizon, validated["holdout_offset"], validated["context_length_option"])


def stage_forecast(run: Run) -> None:
    import numpy as np

    P = package(run.root)
    import torch

    data, series = load_series(run, "forecast")
    horizon = data["horizon"]
    context, truth, _start = holdout(run, series, horizon)
    pipe = gpu_pipeline(run, P)
    result = pipe.forecast(context, horizon=horizon)
    np.save(run.state / "quantiles.npy", result["quantiles"])
    peak = round(torch.cuda.max_memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else None
    summary = {k: result[k] for k in ("point_forecast", "horizon", "context_length", "context_padding", "patch_size", "decode_block_size", "n_variates", "device", "source")} | {"quantile_levels": list(result["quantile_levels"]), "peak_cuda_gib": peak}
    run.write_state("forecast.json", summary)
    print(summary)
    q = result["quantiles"]
    for v, name in enumerate(data["names"]):
        for step in range(min(3, horizon)):
            print(f"{name:<26} step {step + 1:>2}  median {q[v, 4, step]:.4f}  truth {truth[v, step]:.4f}  q10 {q[v, 0, step]:.4f}  q90 {q[v, 8, step]:.4f}")
    figure(run, data, context, truth, q)


def figure(run: Run, data: dict[str, Any], context, truth, q) -> None:
    """TOT-M2: context, truth, median and the q10–q90 band per variate, saved as a PNG the kernel displays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_vars, horizon = truth.shape
    shown = min(context.shape[1], 3 * horizon)
    fig, axes = plt.subplots(n_vars, 1, figsize=(9, 2.6 * n_vars), squeeze=False)
    x_ctx = np.arange(-shown, 0)
    x_new = np.arange(horizon)
    for v, ax in enumerate(axes[:, 0]):
        ax.plot(x_ctx, context[v, -shown:], color="0.4", lw=1, label="context")
        ax.plot(x_new, truth[v], color="black", lw=1.2, label="withheld truth")
        ax.fill_between(x_new, q[v, 0], q[v, 8], color="tab:blue", alpha=0.25, label="q10–q90 (model quantiles)")
        ax.plot(x_new, q[v, 4], color="tab:blue", lw=1.5, label="median (point forecast)")
        ax.axvline(-0.5, color="0.7", ls="--", lw=0.8)
        ax.set_title(data["names"][v], fontsize=10)
        ax.set_xlabel("steps relative to the forecast origin")
    axes[0, 0].legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(run.out / f"{STEM}_forecast.png", dpi=110)
    plt.close(fig)
    print(f"wrote outputs/{STEM}_forecast.png")


def result_from_state(run: Run, needed_by: str) -> dict[str, Any]:
    import numpy as np

    summary = run.read_state("forecast.json", needed_by)
    quantiles = np.load(run.state / "quantiles.npy")
    return {**summary, "quantiles": quantiles, "median": quantiles[:, summary["quantile_levels"].index(0.5), :]}


def stage_evaluate(run: Run) -> None:
    P = package(run.root)
    data, series = load_series(run, "evaluate")
    validated = run.read_state("validated.json", "evaluate")
    horizon = data["horizon"]
    context, truth, _start = holdout(run, series, horizon)
    result = result_from_state(run, "evaluate")
    report = P.evaluation_report(result, truth, context=context, sample_kind=data["sample_kind"])
    sn = validated["baselines"]["seasonal_naive"]
    report["baselines"].append({"id": "seasonal_naive", "metrics": [{"id": "mae", "value": sn["mae"]}, {"id": "rmse", "value": sn["rmse"]}], "what": sn["what"], "period_steps": sn["period_steps"]})
    ls = validated["baselines"]["least_squares_trend_season"]
    report["baselines"].append({"id": "least_squares_trend_season", "metrics": [{"id": "mae", "value": ls["mae"]}, {"id": "rmse", "value": ls["rmse"]}], "what": ls["what"], "period_steps": ls["period_steps"], "period_source": ls["period_source"]})
    report["per_variate"] = [{"variate": name, "mae": P.mae(truth[v], result["median"][v]), "rmse": P.rmse(truth[v], result["median"][v])} for v, name in enumerate(data["names"])]
    report["holdout"] = {"offset_from_end": validated["holdout_offset"], "context_length": validated["context_length"]}
    if validated["reference"].get("noise_floor"):
        nf = validated["reference"]["noise_floor"]
        report["reference_points"] = [{"id": "noise_floor", "metrics": [{"id": "mae", "value": nf["mae"]}, {"id": "rmse", "value": nf["rmse"]}], "what": nf["what"]}]
    n_points = truth.size
    report["coverage_granularity"] = f"q10–q90 coverage over {n_points} holdout point(s) moves in steps of 1/{n_points} = {1 / n_points:.4f}"
    model_mae = next(m["value"] for m in report["metrics"] if m["id"] == "mae")
    last_mae = validated["baselines"]["last_value_baseline"]["mae"]
    lines = [f"Toto median MAE {model_mae:.4f} vs last-value {last_mae:.4f}, seasonal-naive {sn['mae']:.4f} and least-squares trend + season {ls['mae']:.4f}"]
    if validated["reference"].get("noise_floor"):
        lines.append(f"noise floor {validated['reference']['noise_floor']['mae']:.4f}: the model is well ahead of the naive baselines, but a fit that knows the generator's two seasons sits near the floor, so the margin over last-value mostly shows how weak a flat forecast is on a trending seasonal series")
    elif model_mae <= ls["mae"]:
        lines.append("Toto beats the trend + season fit on this holdout; repeat over several holdouts (the activity) before reading it as skill")
    else:
        lines.append("the trend + season fit does at least as well as Toto on this holdout; a simple model may suffice for this series")
    report["interpretation"] = "; ".join(lines)
    run.write_output(f"{STEM}_evaluation_report.json", report)
    run.write_state("evaluate.json", {"model_mae": model_mae, "interpretation": report["interpretation"]})
    print(json.dumps(rounded(report), indent=2))


def stage_export(run: Run) -> None:
    import importlib.metadata
    import platform

    import numpy as np
    import pandas as pd

    P = package(run.root)
    data, series = load_series(run, "export")
    horizon = data["horizon"]
    context, truth, start = holdout(run, series, horizon)
    result = result_from_state(run, "export")
    baseline = P.last_value_baseline(context, horizon)
    fit = np.load(run.state / "least_squares.npy")
    naive = np.load(run.state / "seasonal_naive.npy")
    stamps_path = run.state / "timestamps.json"
    stamps = json.loads(stamps_path.read_text(encoding="utf-8")) if stamps_path.is_file() else None
    rows = []
    for v in range(result["median"].shape[0]):
        for step in range(horizon):
            row = {"variate": data["names"][v], "step": step + 1, "timestamp": stamps[start + step] if stamps else start + step, "median": float(result["median"][v, step]), "truth": float(truth[v, step]), "last_value_baseline": float(baseline[v, step]), "seasonal_naive": float(naive[v, step]), "least_squares_trend_season": float(fit[v, step])}
            for index, level in enumerate(result["quantile_levels"]):
                row[f"q{int(round(level * 100)):02d}"] = float(result["quantiles"][v, index, step])
            rows.append(row)
    pd.DataFrame(rows).to_csv(run.out / f"{STEM}_forecast.csv", index=False)
    source = json.loads((run.root / "source.json").read_text(encoding="utf-8")) if (run.root / "source.json").is_file() else {}
    payload = {
        "forecast": {k: result[k] for k in ("point_forecast", "quantile_levels", "horizon", "context_length", "n_variates")} | {"median": result["median"].tolist()},
        "evaluation_report": json.loads((run.out / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8")),
        "input_manifest": json.loads((run.out / f"{STEM}_input_manifest.json").read_text(encoding="utf-8")),
        "sample": {k: data[k] for k in ("sample_kind", "name", "shape", "float32_sha256")} | {"trimmed": data.get("trimmed"), "last_context_timestamp": stamps[start - 1] if stamps else start - 1, "frequency": data.get("step") or "1 step (synthetic index)"},
        "notebook_source": source,
        "repository_revision": source.get("revision"),
        "model_id": P.MODEL_ID,
        "model_revision": P.MODEL_REVISION,
        "model_license": P.MODEL_LICENSE,
        "runtime": {"python": platform.python_version(), **{d: importlib.metadata.version(d) for d in ("torch", "toto-2", "numpy", "pandas")}, "device": result["device"], "peak_cuda_gib": result.get("peak_cuda_gib")},
    }
    run.write_output(f"{STEM}_result.json", payload)
    print({"rows": len(rows), "columns": list(rows[0]), "runtime": payload["runtime"]})
    print(sorted(p.name for p in run.out.iterdir()))


def stage_activity(run: Run) -> None:
    """TOT-m3: a rolling-origin check — forecast a holdout that ends ``holdout_offset`` steps earlier and print the model
    beside last-value and seasonal-naive at that origin and at the canonical one; writes only to outputs/activity/."""
    P = package(run.root)
    data, series = load_series(run, "activity")
    horizon = data["horizon"]
    validated = run.read_state("validated.json", "activity")
    canonical = result_from_state(run, "activity")
    offset = int(run.options.get("holdout_offset", horizon))
    context, truth, start = split(series, horizon, offset, 0)
    before = sorted(p.name for p in run.out.glob(f"{STEM}_*"))
    pipe = gpu_pipeline(run, P)
    result = pipe.forecast(context, horizon=horizon)
    period = validated["baselines"]["seasonal_naive"]["period_steps"]
    _c0, truth0, _s0 = holdout(run, series, horizon)
    rows = [
        {"origin": f"canonical (offset {validated['holdout_offset']})", "model_mae": round(P.mae(truth0, canonical["median"]), 4), "last_value_mae": round(validated["baselines"]["last_value_baseline"]["mae"], 4), "seasonal_naive_mae": round(validated["baselines"]["seasonal_naive"]["mae"], 4)},
        {"origin": f"earlier (offset {offset})", "model_mae": round(P.mae(truth, result["median"]), 4), "last_value_mae": round(P.mae(truth, P.last_value_baseline(context, horizon)), 4), "seasonal_naive_mae": round(P.mae(truth, seasonal_naive(context, horizon, min(period, context.shape[1]))), 4)},
    ]
    for row in rows:
        print(row)
    run.write_output(f"activity/{STEM}_activity_offset_{offset}.json", {"rows": rows, "holdout_steps": [start, start + horizon]})
    if sorted(p.name for p in run.out.glob(f"{STEM}_*")) != before:
        raise RuntimeError("the activity changed the canonical outputs; it must write only to outputs/activity/")
    print(f"wrote outputs/activity/{STEM}_activity_offset_{offset}.json (the canonical outputs are unchanged)")


STAGES = {
    "weights": stage_weights,
    "data": stage_data,
    "validate": stage_validate,
    "forecast": stage_forecast,
    "evaluate": stage_evaluate,
    "export": stage_export,
    "activity": stage_activity,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--options", default="{}")
    args = parser.parse_args(argv)
    run = Run(args.root.resolve(), args.outputs.resolve(), args.weights.resolve(), json.loads(args.options))
    error_file = run.state / f"{args.stage}.error.json"
    error_file.unlink(missing_ok=True)
    try:
        STAGES[args.stage](run)
    except BaseException as exc:  # noqa: BLE001 -- every failure is reported to the kernel with its own message
        traceback.print_exc()
        error_file.write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
