"""Refused user input is shown as one clean line (NOTEBOOK_SPEC 3.0 DAT19/DAT20).

Each refusal path runs the generated notebook's own kernel code (the ``run_stage`` helper of the install cell and the
learner's Section 4 BYOD cell) against the real stage runner in a child process, as the notebook does, with only
``google.colab``'s upload dialog stood in. For every path the cell must raise ``Refused`` (an exception, so Run all
stops at that cell), render as exactly one ``Refused: <file>: <reason>`` line through IPython's ``_render_traceback_``
hook, name the file the learner supplied, and print no ``Traceback``. The earlier run's outputs are gone, so none can be
read as this input's result. A genuine failure still raises ``RuntimeError`` with the stage's traceback. The content
case is the hosted refusal run's input, ``BYOD_CSV_PATH=/etc/hostname``: a file that exists in every fresh runtime but
has no ``timestamp`` column. CI's dependencies suffice: the data stage imports no model library.
"""
# ruff: noqa: E501

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STEM = "toto_forecasting"
NOTEBOOK = ROOT / "tutorials" / f"{STEM}_colab.ipynb"
REFUSAL_CELL_ID = "toto_forecasting-15"  # the hosted refusal run's halting cell: Section 4's BYOD cell


def _cells() -> list[dict]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [{**c, "source": "".join(c["source"])} for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(needle: str) -> dict:
    found = [c for c in _cells() if needle in c["source"]]
    assert len(found) == 1, needle
    return found[0]


def _byod_cell(path: str) -> str:
    lines = []
    for line in _cell("USE_BYOD = False  # @param")["source"].split("\n"):
        if line.startswith("USE_BYOD = "):
            line = "USE_BYOD = True" + line[line.index("  #"):]
        elif line.startswith("BYOD_CSV_PATH = "):
            line = f"BYOD_CSV_PATH = {path!r}" + line[line.index("  #"):]
        lines.append(line)
    return "\n".join(lines)


def _kernel(tmp_path: Path) -> dict:
    """The notebook's own stage helper, bound to a run directory that carries the package and the stage runner."""
    run = tmp_path / "run"
    (run / "state").mkdir(parents=True)
    shutil.copytree(ROOT / "src", run / "src")
    shutil.copyfile(ROOT / "tools" / "tutorial_stages.py", run / "tutorial_stages.py")
    install = _cell("def run_stage(stage, **options):")["source"]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    ns = {"ROOT": run, "OUTPUTS": outputs, "WEIGHTS": tmp_path / "weights", "PYTHON": Path(sys.executable), "ENV": env, "Path": Path, "json": json, "subprocess": subprocess, "shutil": shutil}
    exec(compile(install[install.index("REFUSED_EXIT = 3"):], "<install>", "exec"), ns)  # noqa: S102 - the notebook's own helpers
    return ns


@contextlib.contextmanager
def _colab(*uploads: dict):
    queue = list(uploads)
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: queue.pop(0))
    google = types.ModuleType("google")
    google.colab = colab
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules.update({"google": google, "google.colab": colab})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _run_all(ns: dict, cells: list[str]) -> tuple[BaseException | None, str, int]:
    """Run cells in order as Run all does: the first cell that raises stops the run. Returns the exception, the cell
    output (stdout plus the exception as IPython renders it) and how many cells ran."""
    out, ran, raised = io.StringIO(), 0, None
    with contextlib.redirect_stdout(out):
        for source in cells:
            ran += 1
            try:
                exec(compile(source, f"<cell {ran}>", "exec"), ns)  # noqa: S102 - the notebook's own cells
            except BaseException as exc:  # noqa: BLE001 -- what IPython would show is asserted below
                raised = exc
                break
    shown = out.getvalue()
    if raised is not None and hasattr(raised, "_render_traceback_"):
        shown += "\n".join(raised._render_traceback_()) + "\n"
    return raised, shown, ran


def _assert_refused(raised, shown: str, ran: int, file: str) -> str:
    assert type(raised).__name__ == "Refused", (raised, shown)
    lines = raised._render_traceback_()
    assert len(lines) == 1 and lines[0].startswith(f"Refused: {file}: "), lines
    assert [line for line in shown.splitlines() if "Refused:" in line] == lines, shown
    assert "Traceback" not in shown and "Error" not in shown.replace("Refused", ""), shown
    assert ran == 1, "Run all must stop at the refusing cell"
    return lines[0]


def test_the_refusal_cell_is_section_4s_byod_cell() -> None:
    assert _cell("USE_BYOD = False  # @param")["id"] == REFUSAL_CELL_ID


def test_a_file_without_a_timestamp_column_is_one_refused_line_and_leaves_no_earlier_results(tmp_path: Path) -> None:
    """The hosted refusal run's input (``USE_BYOD=True;BYOD_CSV_PATH=/etc/hostname``)."""
    ns = _kernel(tmp_path)
    for name in (f"{STEM}_forecast.csv", f"{STEM}_evaluation.json"):
        (ns["OUTPUTS"] / name).write_text("{}", encoding="utf-8")
    for name in ("data.json", "forecast.json"):
        (ns["ROOT"] / "state" / name).write_text("{}", encoding="utf-8")
    hostname = tmp_path / "hostname"
    hostname.write_text("colab-runtime\n", encoding="utf-8")
    raised, shown, ran = _run_all(ns, [_byod_cell(str(hostname)), "print('AFTER THE REFUSAL')"])
    line = _assert_refused(raised, shown, ran, "hostname")
    assert "no `timestamp` column" in line and "run the Section 4 cell again" in line
    assert "AFTER THE REFUSAL" not in shown
    assert list(ns["OUTPUTS"].glob(f"{STEM}_*")) == []
    assert not (ns["ROOT"] / "state" / "data.json").exists() and not (ns["ROOT"] / "state" / "forecast.json").exists()
    error = json.loads((ns["ROOT"] / "state" / "data.error.json").read_text(encoding="utf-8"))
    assert error["refused"] is True and error["file"] == "hostname"


@pytest.mark.parametrize(
    ("name", "text", "rule"),
    [
        ("text.csv", "timestamp,value\n" + "".join(f"2024-01-{d:02d},x{d}\n" for d in range(1, 29)), "target columns must be numeric"),
        ("gap.csv", "timestamp,value\n2024-01-01,1\n2024-01-02,2\n2024-01-05,3\n", "regularly spaced"),
        ("dates.csv", "timestamp,value\nnot-a-date,1\n", "does not parse"),
    ],
)
def test_byod_contract_violations_are_one_refused_line(tmp_path: Path, name: str, text: str, rule: str) -> None:
    ns = _kernel(tmp_path)
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    raised, shown, ran = _run_all(ns, [_byod_cell(str(path))])
    assert rule in _assert_refused(raised, shown, ran, name)


def test_a_missing_path_is_one_refused_line(tmp_path: Path) -> None:
    ns = _kernel(tmp_path)
    missing = str(tmp_path / "absent.csv")
    raised, shown, ran = _run_all(ns, [_byod_cell(missing)])
    assert "is not a file in this runtime" in _assert_refused(raised, shown, ran, missing)


@pytest.mark.parametrize(("uploads", "file"), [({}, "(no file)"), ({"a.csv": b"x", "b.csv": b"y"}, "a.csv, b.csv")])
def test_upload_dialog_refusals_are_one_refused_line(tmp_path: Path, uploads: dict, file: str) -> None:
    ns = _kernel(tmp_path)
    with _colab(uploads):
        raised, shown, ran = _run_all(ns, [_byod_cell("")])
    _assert_refused(raised, shown, ran, file)


def test_an_empty_path_outside_colab_is_one_refused_line(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)  # importing it raises ImportError
    ns = _kernel(tmp_path)
    raised, shown, ran = _run_all(ns, [_byod_cell("")])
    _assert_refused(raised, shown, ran, "BYOD_CSV_PATH")


def test_a_genuine_failure_is_still_an_error_with_its_traceback(tmp_path: Path) -> None:
    """A stage run out of order is a notebook failure, not a refusal: RuntimeError, and the stage's traceback shows."""
    ns = _kernel(tmp_path)
    raised, shown, ran = _run_all(ns, ["run_stage('validate')"])
    assert type(raised) is RuntimeError and "Stage 'validate' failed (exit 1)" in str(raised)
    assert "Traceback" in shown and "Refused:" not in shown
