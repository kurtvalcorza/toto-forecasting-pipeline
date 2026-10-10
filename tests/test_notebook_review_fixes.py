"""Regression tests for the 2026-10-05 notebook review (TOT-M1, TOT-M2, TOT-m1..m4, TOT-S1, TOT-S3).

They need numpy and pandas (CI's dependencies) but no model: they exec the notebook's own kernel cells with stand-ins for
`run_stage` and `google.colab`, run the stage runner's model-free stages (`data`, `validate`) in-process, and check the
generated notebook statically. Each test names its finding.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
NB = ROOT / "tutorials" / "toto_forecasting_colab.ipynb"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGES = _load("tot_tutorial_stages", TOOLS / "tutorial_stages.py")


def _nb() -> dict:
    return json.loads(NB.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _md() -> str:
    return "\n".join(_src(c) for c in _nb()["cells"] if c["cell_type"] == "markdown")


def _cell_with(needle: str) -> str:
    found = [_src(c) for c in _nb()["cells"] if c["cell_type"] == "code" and needle in _src(c)]
    assert len(found) == 1, needle
    return found[0]


def _set(src: str, name: str, value) -> str:
    out = []
    for line in src.split("\n"):
        if line.startswith(f"{name} = "):
            line = f"{name} = {value!r}" + (line[line.index("  #"):] if "  #" in line else "")
        out.append(line)
    return "\n".join(out)


@contextlib.contextmanager
def _colab(queue):
    files = types.SimpleNamespace(calls=0)

    def upload():
        files.calls += 1
        return queue.pop(0)

    files.upload = upload
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules.update({"google": google, "google.colab": colab})
    try:
        yield files
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _kernel(tmp_path: Path) -> tuple[dict, list]:
    calls: list = []
    ns = {"ROOT": tmp_path / "run", "Path": Path, "shutil": shutil, "run_stage": lambda stage, **o: calls.append((stage, o))}
    return ns, calls


def _run(tmp_path: Path, options: dict | None = None):
    root = tmp_path / "run"
    if not (root / "src").exists():
        root.mkdir(parents=True, exist_ok=True)
        (root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    return STAGES.Run(root, tmp_path / "outputs", tmp_path / "weights", options or {})


def _quiet(fn, *args) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fn(*args)
    return out.getvalue()


@pytest.fixture
def fake_gpu(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tool = bin_dir / "nvidia-smi"
    tool.write_text("#!/bin/sh\necho 'Tesla T4, 15360 MiB'\n")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{__import__('os').environ['PATH']}")


def _csv(path: Path, rows: int, freq: str = "D", sep: str = ",", extra: dict | None = None) -> Path:
    t = np.arange(rows)
    frame = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=rows, freq=freq).astype(str), "y": 0.01 * t + np.sin(2 * np.pi * t / 12)})
    for name, values in (extra or {}).items():
        frame[name] = values
    frame.to_csv(path, index=False, sep=sep)
    return path


# ---------------------------------------------------------------- TOT-M1: isolated runtime, GPU checked up front


def test_M1_no_kernel_install_and_no_restart_instruction() -> None:
    text = NB.read_text(encoding="utf-8")
    assert "Restart the runtime" not in text and "restart the runtime" not in text
    own = [_src(c) for c in _nb()["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources")]
    for src in own:
        assert not re.search(r"['\"]-m['\"]\s*,\s*['\"]pip['\"]|^\s*[%!]\s*pip\b|['\"]pip install", src, re.M), src[:80]
        assert "import torch" not in src and "from toto_forecasting_pipeline" not in src
    install = _cell_with("# @title Infrastructure: build (or reuse) the isolated")
    assert "'--require-hashes'" in install and "--managed-python" in install
    assert "MPLBACKEND='Agg'" in install and "'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP'" in install


def _exec_check_cell(ns: dict, **overrides) -> None:
    src = _cell_with("# @title Infrastructure: check the runtime")
    src = re.sub(r"'environment': [0-9.]+}", "'environment': 0.0}", src, count=1)
    src = re.sub(r"\{'weights': max\(0\.0, [0-9.]+", "{'weights': max(0.0, 0.0", src, count=1)
    for name, value in overrides.items():
        src = _set(src, name, value)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, "<check>", "exec"), ns)


def test_M1_cpu_runtime_stops_in_section_1(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    with pytest.raises(RuntimeError, match=r"No CUDA GPU detected, and this notebook's model needs one.*Nothing has been downloaded yet"):
        _exec_check_cell({})


def test_M1_section1_is_idempotent(tmp_path, monkeypatch, fake_gpu) -> None:
    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    _exec_check_cell(ns)
    first = ns["ROOT"]
    (first / "tutorial_stages.py").write_text("# carried")
    _exec_check_cell(ns)
    assert ns["ROOT"] == first and (first / "tutorial_stages.py").is_file()
    _exec_check_cell(ns, NEW_RUN_DIRECTORY=True)
    assert ns["ROOT"] != first


def test_M1_second_run_all_reuses_the_matching_environment(tmp_path, monkeypatch, fake_gpu) -> None:
    import subprocess as real_subprocess

    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    _exec_check_cell(ns)
    ns["ENV_ROOT"] = tmp_path / "uvroot"
    ns["NOTEBOOK_SOURCE"] = {"revision": "test"}
    src = _cell_with("# @title Infrastructure: build (or reuse) the isolated")
    version = re.search(r"UV = ENV_ROOT / 'uv-([0-9.]+)'", src).group(1)
    ns["ENV_ROOT"].mkdir()
    (ns["ENV_ROOT"] / f"uv-{version}").write_bytes(b"uv stand-in")
    (ns["ENV_ROOT"] / f"uv-{version}.sha256").write_text(hashlib.sha256(b"uv stand-in").hexdigest())
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append([str(c) for c in cmd])
        if cmd[1] == "venv":
            python = Path(cmd[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("")
        out = json.dumps({"python": "3.12.12", "torch": "x", "toto-2": "x", "numpy": "x", "pandas": "x", "cuda": True})
        return types.SimpleNamespace(stdout=out + "\n", returncode=0)

    fake = types.SimpleNamespace(run=fake_run, Popen=real_subprocess.Popen, PIPE=real_subprocess.PIPE, STDOUT=real_subprocess.STDOUT)
    for attempt in range(2):
        ns["subprocess"] = fake
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src.replace("import zipfile\n", "import zipfile\nsubprocess = globals()['subprocess']\n", 1), "<install>", "exec"), ns)
        ns["subprocess"] = fake
        assert ns["environment_reused"] is (attempt == 1)
    assert len([c for c in commands if c[1] in ("venv", "pip")]) == 2


# ---------------------------------------------------------------- TOT-M2: guided layer, figure, activity

GUIDED = ("## How to use this notebook", "**Who this notebook is for.**", "## The task: Input → Model → Output", "## Roadmap", "<summary><strong>Glossary</strong>", "Predict before running", "**What to notice:**", "Check your reasoning", "## Troubleshooting", "## Conclusion", "Sample conclusion")


def test_M2_guided_layer_and_collapsed_infrastructure() -> None:
    nb = _nb()
    md = _md()
    for heading in GUIDED:
        assert heading in md, heading
    assert md.count("Predict before running") >= 3 and "{{" not in md
    infra = [c for c in nb["cells"] if c["cell_type"] == "code" and _src(c).startswith("# @title Infrastructure:")]
    assert len(infra) == 4 and all(c["metadata"].get("cellView") == "form" for c in infra)


def test_M2_forecast_figure_is_drawn_and_shown() -> None:
    cell = _cell_with("run_stage('forecast')")
    assert "display(Image(filename=str(OUTPUTS / 'toto_forecasting_forecast.png')))" in cell and "import torch" not in cell
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("def figure("):source.index("def result_from_state")]
    assert "fill_between(x_new, q[v, 0], q[v, 8]" in body and "withheld truth" in body and 'f"{STEM}_forecast.png"' in body


def test_M2_rolling_origin_activity_writes_only_to_activity() -> None:
    cell = _cell_with("RUN_ACTIVITY = False  # @param")
    assert "ACTIVITY_OFFSET = 48  # @param" in cell and "run_stage('activity', holdout_offset=ACTIVITY_OFFSET)" in cell
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("def stage_activity"):source.index("STAGES = {")]
    assert '"seasonal_naive_mae"' in body and 'write_output(f"activity/' in body and "!= before" in body


def test_no_quality_assert_in_the_stage_runner() -> None:
    assert not [n for n in ast.walk(ast.parse((TOOLS / "tutorial_stages.py").read_text(encoding="utf-8"))) if isinstance(n, ast.Assert)]


# ---------------------------------------------------------------- TOT-S1 / TOT-S3: references with a scale


def test_S1_references_reproduce_the_review(tmp_path) -> None:
    """TOT-S1: last-value 1.108021 / 1.310835, seasonal naive (57) 0.649661 / 0.723387, the informed harmonic fit
    0.036266 / 0.047889 and the noise floor 0.034875 / 0.046741, all from the context only."""
    run = _run(tmp_path, {"use_byod": False, "horizon": 48})
    _quiet(STAGES.stage_data, run)
    assert json.loads((run.state / "data.json").read_text())["float32_sha256"].startswith("4129ad3d")
    _quiet(STAGES.stage_validate, run)
    v = json.loads((run.state / "validated.json").read_text())
    b, ref = v["baselines"], v["reference"]
    assert (round(b["last_value_baseline"]["mae"], 6), round(b["last_value_baseline"]["rmse"], 6)) == (1.108021, 1.310835)
    assert (round(b["seasonal_naive"]["mae"], 6), round(b["seasonal_naive"]["rmse"], 6), b["seasonal_naive"]["period_steps"]) == (0.649661, 0.723387, 57)
    assert (round(b["least_squares_trend_season"]["mae"], 6), round(b["least_squares_trend_season"]["rmse"], 6)) == (0.036266, 0.047889)
    assert (round(ref["noise_floor"]["mae"], 6), round(ref["noise_floor"]["rmse"], 6)) == (0.034875, 0.046741)


def test_S3_report_carries_per_variate_errors_and_the_interpretation() -> None:
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("def stage_evaluate"):source.index("def stage_export")]
    assert 'report["per_variate"]' in body and '"id": "seasonal_naive"' in body and "coverage_granularity" in body and 'report["interpretation"]' in body
    md = _md()
    assert "the default series is easy, so the last-value baseline flatters any forecaster" in md and "1/96" in md


# ---------------------------------------------------------------- TOT-m1: BYOD path, guarded upload, named refusals


def test_m1_byod_path_field_and_upload_fallback(tmp_path, monkeypatch) -> None:
    cell = _cell_with("USE_BYOD = False  # @param")
    monkeypatch.setitem(sys.modules, "google.colab", None)
    ns, calls = _kernel(tmp_path)
    exec(compile(_set(_set(cell, "USE_BYOD", True), "BYOD_CSV_PATH", "/d/series.csv"), "<s4>", "exec"), ns)
    assert calls[0] == ("data", {"use_byod": True, "byod_csv_path": "/d/series.csv", "horizon": 48})
    ns, calls = _kernel(tmp_path)
    with pytest.raises(RuntimeError, match="BYOD_CSV_PATH is empty, and the upload dialog exists only in Google Colab"):
        exec(compile(_set(cell, "USE_BYOD", True), "<s4>", "exec"), ns)
    monkeypatch.undo()
    for queue, message in (([{}], "cancelled or empty"), ([{"a.csv": b"x", "b.csv": b"y"}], "Upload exactly one file")):
        with _colab(queue) as files:
            ns, calls = _kernel(tmp_path)
            with pytest.raises((RuntimeError, ValueError), match=message):
                exec(compile(_set(cell, "USE_BYOD", True), "<s4>", "exec"), ns)
            assert files.calls == 1 and calls == []


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"site": ["x"] * 120}, r"bad\.csv: target columns must be numeric; non-numeric values in \{'site': \['x'\]\}"),
        ({"z": [1.0] * 5 + [""] + [1.0] * 114}, r"bad\.csv: empty values in \{'z': \[7\]\}"),
    ],
    ids=["text-value", "blank-value"],
)
def test_m1_bad_values_name_the_column_and_row(tmp_path, extra, message) -> None:
    path = _csv(tmp_path / "bad.csv", 120, extra=extra)
    with pytest.raises(ValueError, match=message):
        _quiet(STAGES.stage_data, _run(tmp_path, {"use_byod": True, "byod_csv_path": str(path), "horizon": 48}))


def test_m1_unparseable_timestamp_and_utf16_are_named(tmp_path) -> None:
    frame = pd.read_csv(_csv(tmp_path / "base.csv", 120))
    frame.loc[3, "timestamp"] = "yesterday"
    frame.to_csv(tmp_path / "dates.csv", index=False)
    with pytest.raises(ValueError, match=r"dates\.csv: the `timestamp` column does not parse"):
        _quiet(STAGES.stage_data, _run(tmp_path, {"use_byod": True, "byod_csv_path": str(tmp_path / "dates.csv"), "horizon": 48}))
    utf16 = tmp_path / "utf16.csv"
    utf16.write_text((tmp_path / "base.csv").read_text(), encoding="utf-16")
    with pytest.raises(ValueError, match=r"utf16\.csv: not UTF-8 text"):
        _quiet(STAGES.stage_data, _run(tmp_path / "b", {"use_byod": True, "byod_csv_path": str(utf16), "horizon": 48}))


# ---------------------------------------------------------------- TOT-m2: timestamps survive the export


def test_m2_byod_export_keeps_timestamps(tmp_path, monkeypatch) -> None:
    path = _csv(tmp_path / "hourly.csv", 120, freq="h")
    run = _run(tmp_path, {"use_byod": True, "byod_csv_path": str(path), "horizon": 48})
    _quiet(STAGES.stage_data, run)
    _quiet(STAGES.stage_validate, run)
    run.write_state("forecast.json", {"point_forecast": "median (q=0.5)", "horizon": 48, "context_length": 72, "n_variates": 1, "device": "cuda", "source": "stand-in", "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]})
    np.save(run.state / "quantiles.npy", np.zeros((1, 9, 48)))
    run.write_output("toto_forecasting_evaluation_report.json", {"verdict": "sample-sanity"})
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "x")  # the model libraries are not installed in CI
    _quiet(STAGES.stage_export, run)
    rows = pd.read_csv(run.out / "toto_forecasting_forecast.csv")
    assert rows["timestamp"].iloc[0] == "2024-01-04 00:00:00" and rows["timestamp"].iloc[-1] == "2024-01-05 23:00:00"
    sample = json.loads((run.out / "toto_forecasting_result.json").read_text())["sample"]
    assert sample["last_context_timestamp"] == "2024-01-03 23:00:00" and sample["frequency"] == "0 days 01:00:00"


# ---------------------------------------------------------------- TOT-m3: experiments change one form field


def test_m3_holdout_offset_and_context_length_are_form_fields(tmp_path) -> None:
    cell = _cell_with("HOLDOUT_OFFSET = 0  # @param")
    assert "CONTEXT_LENGTH = 0  # @param" in cell and "run_stage('validate', holdout_offset=HOLDOUT_OFFSET, context_length=CONTEXT_LENGTH, season_period=SEASON_PERIOD)" in cell
    assert "DECODE_BLOCK_SIZE = None" not in NB.read_text(encoding="utf-8")
    run = _run(tmp_path, {"use_byod": False, "horizon": 48})
    _quiet(STAGES.stage_data, run)
    run.options = {"holdout_offset": 48, "context_length": 96}
    printed = _quiet(STAGES.stage_validate, run)
    assert "'holdout_offset': 48, 'context_length': 96, 'holdout_steps': [224, 272]" in printed
    with pytest.raises(ValueError, match="CONTEXT_LENGTH=500 exceeds"):
        STAGES.split(np.zeros((1, 320)), 48, 0, 500)


# ---------------------------------------------------------------- TOT-m4: GPU prerequisite as measured


def test_m4_gpu_prerequisite_is_stated_as_measured() -> None:
    md = _md()
    assert "A 15 GB T4 is enough" in md and "9.45 GiB" in md and "at least 16 GB" not in md


def test_stage_processes_import_neither_ipython_nor_google_colab() -> None:
    """Stages run in the isolated environment, which has neither IPython nor google.colab: only kernel cells may use
    them (the figure display and the BYOD upload dialog). A carried module that imported either would fail on Colab;
    there is no worker and no google.colab stub to give a ModuleSpec (swin2sr-x4-super-resolution-pipeline 34eac6c
    pattern)."""
    build = _load("tot_build_notebook_carried", TOOLS / "build_notebook.py")
    template = build.load_template(TOOLS / "notebook_template.py")
    carried = [ROOT / source for dest, source in build.carried_sources(ROOT, template).items() if dest.endswith(".py")]
    assert any(path.name == "tutorial_stages.py" for path in carried)
    assert any(path.name == "pipeline.py" for path in carried)
    offenders = [str(path) for path in carried if re.search(r"^\s*(from|import)\s+(IPython|google)\b", path.read_text(encoding="utf-8"), re.M)]
    assert not offenders, offenders
    text = NB.read_text(encoding="utf-8")
    assert "sys.modules['google" not in text and 'sys.modules[\\"google' not in text and "_WORKER_SOURCE" not in text


def test_m4_status_records_agree_that_one_pass_run_all_is_pending() -> None:
    """TOT-m4: the 2026-09-14 Kaggle run of the previous notebook passed only after a restart, so no document may call
    Run all verified; STATUS.md cites the spec the notebook declares, and the run row records the restart."""
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "Specification 2.2" in status and "Specification 1.1" not in status
    assert "verified — clean-runtime" not in registry and "| pending — no hosted one-pass `Run all`" in registry
    assert "Passed after a restart — not one-pass evidence." in record and "restarted_after_install_cell: true" in record
    assert "No clean-runtime execution of the standalone notebook has been recorded yet" not in record
