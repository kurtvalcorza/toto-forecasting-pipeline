# ruff: noqa: E501  -- control payloads and messages are kept on one line
"""Run the static release-asset validator and prove it discriminates.

A validator that passes on the committed tree proves little unless a mutated tree fails,
so each negative control below re-runs the validator against a copy carrying one defect
that has actually shipped in, or been found to evade the checks of, DIMER tutorials:
editable self-install in either spelling, hard-coded or rebound revision, persisted
outputs, drifting identity, conflicting release status, an enabled or non-form BYOD gate,
a required call surviving only in a comment, an in-kernel install, a restart instruction, an edited carried file and a
quality assert in the stage runner.

The notebook-level controls run the structural, isolation and content checks directly (``_notebook_checks``): a mutated
notebook also fails the generator-parity check (PAR3), which would otherwise mask the specific rule under test.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_release_assets.py"
COPIED = (
    "LICENSE",
    "MODEL_CARD.md",
    "README.md",
    "STATUS.md",
    "pyproject.toml",
    "docs",
    "tutorials",
    "src",
    "tools",
    "weights",
)
# weights/ is copied for its committed manifest only; the git-ignored checkpoints never enter the tmp tree.
IGNORED = shutil.ignore_patterns("__pycache__", ".cache", "*.ckpt", "*.safetensors", "*.bin", "*.pt", "*.pth")


def _load_validator(root: Path):
    spec = importlib.util.spec_from_file_location("validate_release_assets", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = root
    return module


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    for name in COPIED:
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, tmp_path / name, ignore=IGNORED)
        else:
            shutil.copy2(source, tmp_path / name)
    return tmp_path


def _notebook_path(module) -> Path:
    return module.ROOT / "tutorials" / module.NOTEBOOK_NAME


def _edit_notebook(module, mutate) -> None:
    path = _notebook_path(module)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    mutate(notebook)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _replace_in_code(module, old: str, new: str) -> None:
    def mutate(notebook: dict) -> None:
        hits = 0
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            text = "".join(cell["source"])
            if old in text:
                hits += 1
                cell["source"] = [text.replace(old, new)]
        assert hits, f"control marker not found in notebook: {old!r}"

    _edit_notebook(module, mutate)


def _notebook_checks(module) -> None:
    """Every notebook check except PAR3 byte parity, in validate_notebooks order."""
    path = _notebook_path(module)
    spec = module.NOTEBOOKS[path.name]
    template = module._load_tool(spec["template"]).TEMPLATE
    build = module._load_tool("build_notebook")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells, markdown = module._validate_notebook_structure(path, notebook, spec, template, build)
    carrier_index, runner = module._validate_carrier(path, notebook, build, template)
    module._validate_isolation(path, notebook, code_cells, carrier_index)
    module._validate_notebook_content(path, code_cells, markdown, carrier_index, runner, spec, template)


def test_committed_tree_passes() -> None:
    module = _load_validator(ROOT)
    expected = ["model-card", "identity-consistency", "weight-facts", "release-status", "notebooks+parity"]
    assert module.validate_all() == expected


@pytest.mark.parametrize("flag", ["'-e', ", "'--editable', "])
def test_control_editable_self_install_is_rejected(tree: Path, flag: str) -> None:
    module = _load_validator(tree)
    _replace_in_code(module, "run_stage('forecast')", f"subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', {flag}'.'], check=True)\nrun_stage('forecast')")
    with pytest.raises(module.ValidationError, match="notebook kernel|editable self-install"):
        _notebook_checks(module)


def test_control_missing_profile_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    _edit_notebook(module, lambda notebook: notebook["metadata"]["dimer"].pop("notebook_profile"))
    with pytest.raises(module.ValidationError, match="notebook_profile"):
        module.validate_notebooks()


def test_control_persisted_output_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)

    def mutate(notebook: dict) -> None:
        cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
        cell["execution_count"] = 1

    _edit_notebook(module, mutate)
    with pytest.raises(module.ValidationError, match="execution_count"):
        module.validate_notebooks()


def test_control_hard_coded_revision_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    _, revision = module._package_identity(module._primary_template())
    _replace_in_code(module, "run_stage('forecast')", f"print('{revision}')\nrun_stage('forecast')")
    with pytest.raises(module.ValidationError, match="revision may appear only in the carried files"):
        _notebook_checks(module)


def test_control_edited_carried_file_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    _replace_in_code(module, "MIN_CONTEXT = 32", "MIN_CONTEXT = 8")
    with pytest.raises(module.ValidationError, match="carried files differ from the repository"):
        module.validate_notebooks()


@pytest.mark.parametrize("spelling", ["{gate} = True", "{gate}=True"])
def test_control_enabled_byod_gate_is_rejected(tree: Path, spelling: str) -> None:
    module = _load_validator(tree)
    gate = "USE_BYOD"
    line = f'{gate} = False  # @param {{type:"boolean"}}'
    _replace_in_code(module, line, f"{line}\n{spelling.format(gate=gate)}")
    with pytest.raises(module.ValidationError, match="assigned exactly once"):
        _notebook_checks(module)


def test_control_byod_gate_without_form_annotation_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    gate = "USE_BYOD"
    _replace_in_code(module, f'{gate} = False  # @param {{type:"boolean"}}', f"{gate} = False")
    with pytest.raises(module.ValidationError, match="Colab form parameter|lack required markers"):
        _notebook_checks(module)


def test_control_required_call_only_in_comment_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    _replace_in_code(module, "run_stage('evaluate')", "pass  # run_stage('evaluate')")
    with pytest.raises(module.ValidationError, match="lack required markers|must run a stage"):
        _notebook_checks(module)


def test_control_restart_instruction_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    _replace_in_code(module, "run_stage('forecast')", "print('Restart the runtime, then rerun from the top.')\nrun_stage('forecast')")
    with pytest.raises(module.ValidationError, match="must not instruct a runtime restart"):
        _notebook_checks(module)


def test_control_quality_assert_in_runner_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    runner = tree / "tools" / "tutorial_stages.py"
    runner.write_text(runner.read_text(encoding="utf-8").replace("    print({**{k: v for k, v in record.items()", "    assert series.shape[1] > 0\n    print({**{k: v for k, v in record.items()", 1), encoding="utf-8")
    with pytest.raises(module.ValidationError, match="carried stage runner|carried files differ"):
        module.validate_notebooks()


def test_control_identity_drift_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    _, revision = module._package_identity(module._primary_template())
    readme = tree / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8").replace(revision, "0" * 40), encoding="utf-8")
    with pytest.raises(module.ValidationError, match="README.md"):
        module.validate_identity_consistency()


def test_control_conflicting_release_status_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    status = tree / "STATUS.md"
    promoted = status.read_text(encoding="utf-8").replace("**Candidate", "**Release-grade")
    status.write_text(promoted, encoding="utf-8")
    with pytest.raises(module.ValidationError, match="status"):
        module.validate_release_status()


def test_control_placeholder_in_model_card_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    card = tree / "MODEL_CARD.md"
    card.write_text(card.read_text(encoding="utf-8") + "\nTODO: fill in.\n", encoding="utf-8")
    with pytest.raises(module.ValidationError, match="placeholder"):
        module.validate_model_card()
