"""Zero-shot probabilistic forecasting with the pinned ``Datadog/Toto-2.0-2.5B`` checkpoint.

The class loads the checkpoint only from a digest-verified local snapshot (``weights/<key>/``:
``config.json`` + ``model.safetensors``) or, when explicitly allowed, from the Hugging Face Hub at
the pinned revision. Model construction happens inside the upstream ``toto2`` package from its own
``Toto2ModelConfig``; the weights are SafeTensors, and no model-repository code is executed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import interval_coverage, last_value_baseline, mae, rmse
from .validation import MAX_CONTEXT, MAX_HORIZON, MIN_CONTEXT, validate_horizon, validate_target

MODEL_ID = "Datadog/Toto-2.0-2.5B"
MODEL_REVISION = "51a2812bbe449437c01b79c0e425ed578f335f5b"
MODEL_LICENSE = "Apache-2.0"
MODEL_KEY = "toto-2.0-2.5b"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MEDIAN_INDEX = 4  # QUANTILES[4] == 0.5: the point forecast is the model median, not a mean
POINT_FORECAST = "median (q=0.5)"
DEFAULT_DECODE_BLOCK_SIZE = 768


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the 9.8 GB checkpoint). Returns the relative paths fetched; `verify_snapshot` still runs
    after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _validate_decode_block_size(value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("decode_block_size must be None or a positive integer")
    return value


INPUT_SCHEMA: dict[str, Any] = {
    "input": "1D array (time) or 2D array (variates, time) of finite numbers, cast to float32",
    "context_length": [MIN_CONTEXT, MAX_CONTEXT],
    "horizon": [1, MAX_HORIZON],
    "decode_block_size": "None (single forward-pass decode) or a positive multiple of the model patch size",
    "model_dependent_checks": (
        "forecast() additionally requires context_length >= patch_size and "
        "decode_block_size % patch_size == 0, "
        "read from the loaded model config (32 for the pinned checkpoint)"
    ),
    "preprocessing": (
        "context left-padded to the next patch multiple with masked (unobserved) positions; "
        "upstream toto2 scales the context causally inside the model"
    ),
}


def _check_inputs(target: Any, horizon: int, decode_block_size: int | None) -> tuple[np.ndarray, int | None]:
    """Raise ValueError naming the first violated model-independent ceiling; return the validated inputs."""
    values = validate_target(target)
    validate_horizon(horizon)
    return values, _validate_decode_block_size(decode_block_size)


def validate_inputs(
    target: Any,
    *,
    horizon: int,
    decode_block_size: int | None = DEFAULT_DECODE_BLOCK_SIZE,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-variate observations, verdict).

    Rejection is reported by raising exactly as ``forecast`` would for the model-independent checks
    (target shape and finiteness, context length, horizon, ``decode_block_size`` type); the two
    patch-size checks need the loaded model and stay in ``forecast``. A caller that wants a finding
    recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    values, block = _check_inputs(target, horizon, decode_block_size)
    if names is not None and len(names) != values.shape[0]:
        raise ValueError("names must have one entry per variate")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {
                "id": names[i] if names else f"variate-{i}",
                "context_length": int(values.shape[1]),
                "min": float(values[i].min()),
                "max": float(values[i].max()),
            }
            for i in range(values.shape[0])
        ],
        "horizon": horizon,
        "decode_block_size": block,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any],
    truth: Any = None,
    *,
    context: Any = None,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``truth`` (the withheld future, shape ``(variates, horizon)``) the report carries the
    repository's ``mae`` / ``rmse`` on the median forecast, ``interval_coverage`` of the q10–q90 band,
    and — when ``context`` is supplied — the same ``mae`` / ``rmse`` for ``last_value_baseline``, with
    verdict ``sample-sanity``. Without ``truth`` the verdict is ``not-measurable``.
    """
    median = np.asarray(result["median"], dtype=float)
    base = {
        "task": "zero-shot probabilistic time-series forecasting",
        "point_forecast": POINT_FORECAST,
        "score_semantics": (
            "quantile levels 0.1..0.9 are model quantiles, not calibrated confidence intervals"
        ),
        "sample_kind": sample_kind,
        "horizon": int(result["horizon"]),
        "context_length": int(result["context_length"]),
        "n_variates": int(median.shape[0]),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if truth is None:
        return {
            **base,
            "metrics": [],
            "baselines": [],
            "verdict": "not-measurable",
            "reason": "no withheld future values were supplied for the forecast horizon",
            "needs": (
                "a chronological holdout: withhold the final `horizon` observations of the target, "
                "forecast from the remaining context, and score the median with mae/rmse against them "
                "and against the last_value_baseline, repeated over representative periods"
            ),
        }
    actual = np.asarray(truth, dtype=float)
    if actual.ndim == 1:
        actual = actual[None, :]
    if actual.shape != median.shape:
        raise ValueError(f"truth shape {actual.shape} != median shape {median.shape}")
    estimation = "single chronological holdout, no dispersion estimate"
    metrics = [
        {"id": "mae", "value": mae(actual, median), "estimation": estimation},
        {"id": "rmse", "value": rmse(actual, median), "estimation": estimation},
    ]
    quantiles = np.asarray(result["quantiles"], dtype=float)
    levels = list(result["quantile_levels"])
    if quantiles.shape[:1] == actual.shape[:1] and 0.1 in levels and 0.9 in levels:
        metrics.append(
            {
                "id": "interval_coverage",
                "band": [0.1, 0.9],
                "value": interval_coverage(
                    actual, quantiles[:, levels.index(0.1), :], quantiles[:, levels.index(0.9), :]
                ),
                "nominal": 0.8,
                "estimation": estimation,
            }
        )
    baselines = []
    if context is not None:
        baseline = last_value_baseline(context, int(result["horizon"]))
        baselines.append(
            {
                "id": "last_value_baseline",
                "metrics": [
                    {"id": "mae", "value": mae(actual, baseline)},
                    {"id": "rmse", "value": rmse(actual, baseline)},
                ],
            }
        )
    return {
        **base,
        "metrics": metrics,
        "baselines": baselines,
        "verdict": "sample-sanity",
        "reason": (
            f"one chronological holdout of {actual.shape[1]} step(s) on the tutorial sample; not a benchmark"
        ),
        "needs": (
            "repeated holdouts over representative periods of the deployment series for any "
            "generalisable claim"
        ),
    }


@dataclass
class TotoForecastPipeline:
    _model: Any
    device: str
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> TotoForecastPipeline:
        import torch
        from toto2 import Toto2Model

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' was requested but no CUDA device is available; the 2.5B checkpoint's "
                "release-reference path is a CUDA GPU with at least 16 GB of memory"
            )
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            # A directory argument makes toto2 build Toto2ModelConfig from config.json and load
            # model.safetensors from that directory (no Hub resolution, no cache lookup).
            model = Toto2Model.from_pretrained(str(root), map_location="cpu")
            source = "local-snapshot"
        elif allow_download:
            model = Toto2Model.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                map_location="cpu",
            )
            source = "hf-hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        model = model.to(resolved_device).eval()
        return cls(model, resolved_device, source)

    def forecast(
        self,
        target,
        *,
        horizon: int,
        decode_block_size: int | None = DEFAULT_DECODE_BLOCK_SIZE,
    ) -> dict[str, Any]:
        values, decode_block_size = _check_inputs(target, horizon, decode_block_size)
        patch_size = int(self._model.config.patch_size)
        if decode_block_size is not None and decode_block_size % patch_size:
            raise ValueError(
                f"decode_block_size must be a multiple of the model patch size ({patch_size})"
            )
        # Upstream patches the context in blocks of `patch_size` and requires the context length
        # to be a multiple of it. Pad on the left and mark the padded positions unobserved: the
        # upstream causal scaler and patch embedding are mask-aware, so zero pads carry no signal.
        # (Upstream's own GluonTS adapter truncates to a patch multiple instead; padding keeps every
        # observed value.) At least one full patch of real context is required.
        if values.shape[1] < patch_size:
            raise ValueError(f"context length must be at least the model patch size ({patch_size})")
        context_padding = (patch_size - values.shape[1] % patch_size) % patch_size

        import torch

        target_tensor = torch.as_tensor(
            values,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        target_mask = torch.ones_like(target_tensor, dtype=torch.bool)
        if context_padding:
            pad = (context_padding, 0)
            target_tensor = torch.nn.functional.pad(target_tensor, pad, value=0.0)
            target_mask = torch.nn.functional.pad(target_mask, pad, value=False)
        series_ids = torch.arange(
            values.shape[0],
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0)
        with torch.inference_mode():
            forecast = self._model.forecast(
                {
                    "target": target_tensor,
                    "target_mask": target_mask,
                    "series_ids": series_ids,
                },
                horizon=horizon,
                decode_block_size=decode_block_size,
                has_missing_values=context_padding > 0,
            )
        quantiles = np.asarray(forecast.detach().cpu(), dtype=float)
        expected_shape = (len(QUANTILES), 1, values.shape[0], horizon)
        if quantiles.shape != expected_shape:
            raise RuntimeError(
                f"unexpected Toto forecast shape: {quantiles.shape}; "
                f"expected {expected_shape}"
            )
        quantiles = np.transpose(quantiles[:, 0, :, :], (1, 0, 2))
        return {
            "quantiles": quantiles,
            "quantile_levels": QUANTILES,
            "median": quantiles[:, MEDIAN_INDEX, :],
            "point_forecast": POINT_FORECAST,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "horizon": horizon,
            "context_length": values.shape[1],
            "context_padding": context_padding,
            "patch_size": patch_size,
            "n_variates": values.shape[0],
            "decode_block_size": decode_block_size,
            "device": self.device,
            "source": self.source,
        }
