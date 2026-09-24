"""task.yaml: the keys a task may set, and what happens when it sets others.

`load_task` rejects unknown keys rather than ignoring them, because a key that
is silently dropped is a setting the operator believes is in force and is not --
and on this harness the settings in question decide whether a measurement means
anything (`tools:`) or which environment produced it (`image:`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import sandbox, trial  # noqa: E402

TASKS = sorted(p.parent.name for p in ROOT.glob("tasks/*/task.yaml"))


def _write_task(tmp_path: Path, cfg: dict) -> Path:
    root = tmp_path / "atask"
    root.mkdir(parents=True)
    (root / "task.yaml").write_text(yaml.safe_dump(cfg))
    return root


@pytest.mark.parametrize("name", TASKS)
def test_shipped_tasks_load(name: str) -> None:
    task = trial.load_task(name)
    assert task.name == name
    assert task.timeout_s > 0


@pytest.mark.parametrize("name", TASKS)
def test_shipped_tasks_name_an_image_that_resolves(name: str) -> None:
    task = trial.load_task(name)
    resolved = sandbox.resolve_image(task.image)
    assert resolved.endswith(".sqsh")
    # A task may override `image:`; all it must do is resolve to a real image
    # name, not necessarily the default.
    assert isinstance(resolved, str) and resolved, (
        f"{name} resolves to {resolved!r}, not a usable image name")


def test_image_key_is_optional(tmp_path: Path) -> None:
    """Absent means the default. A task should not have to say the obvious."""
    root = _write_task(tmp_path, {"name": "atask"})
    task = trial.load_task(str(root))
    assert task.image == ""
    assert sandbox.resolve_image(task.image) == sandbox.DEFAULT_IMAGE


def test_image_key_is_threaded_through(tmp_path: Path) -> None:
    """A task naming another image gets it, with no code change anywhere."""
    root = _write_task(tmp_path, {"name": "atask", "image": "otherenv"})
    task = trial.load_task(str(root))
    assert task.image == "otherenv"
    assert sandbox.resolve_image(task.image).endswith("otherenv.sqsh")


def test_image_key_accepts_a_path(tmp_path: Path) -> None:
    root = _write_task(tmp_path, {"name": "atask", "image": "/tmp/some.sqsh"})
    assert sandbox.resolve_image(trial.load_task(str(root)).image) == \
        "/tmp/some.sqsh"


def test_unknown_key_is_refused(tmp_path: Path) -> None:
    """A typo must stop the run, not quietly change the experiment."""
    root = _write_task(tmp_path, {"name": "atask", "imagee": "python311"})
    with pytest.raises(SystemExit, match="imagee"):
        trial.load_task(str(root))


# --------------------------------------------------------------------------
# variants: one folder, several builds of the same environment
# --------------------------------------------------------------------------
def _variant_cfg() -> dict:
    return {"name": "atask", "prompts": {"a": "prompts/a.md", "b": "prompts/b.md"},
            "variants": {"default": "two",
                         "one": {"prompt": "a", "images": 1},
                         "two": {"prompt": "b", "images": 10}}}


def test_no_variants_block_means_no_variant(tmp_path: Path) -> None:
    root = _write_task(tmp_path, {"name": "atask"})
    task = trial.load_task(str(root))
    assert task.variant == "" and task.variants == {}
    assert task.environment == root.resolve() / "environment"
    assert task.data_path == root.resolve() / "data.jsonl"
    assert task.prompt_path() == root.resolve() / "instruction"


def test_variant_on_a_task_without_variants_is_refused(tmp_path: Path) -> None:
    root = _write_task(tmp_path, {"name": "atask"})
    with pytest.raises(SystemExit, match="declares no `variants:`"):
        trial.load_task(str(root), "one")


def test_variant_selects_environment_data_and_prompt(tmp_path: Path) -> None:
    root = _write_task(tmp_path, _variant_cfg())
    task = trial.load_task(str(root), "one")
    r = root.resolve()
    assert task.variant == "one"
    assert task.environment == r / "environment" / "one"
    assert task.data_path == r / "environment" / "one" / "data.jsonl"
    assert task.default_prompt == "a"
    assert task.prompt_path() == r / "prompts" / "a.md"
    assert task.prompt_path("prompts/x.md") == r / "prompts" / "x.md"
    # The hook is shared code at the task's environment/, not per variant.
    assert task.setup_hook == r / "environment" / "setup.py"


def test_variant_default_is_the_named_default_then_the_first(tmp_path: Path) -> None:
    root = _write_task(tmp_path, _variant_cfg())
    assert trial.load_task(str(root)).variant == "two"
    cfg = _variant_cfg()
    del cfg["variants"]["default"]
    root = _write_task(tmp_path / "second", cfg)
    assert trial.load_task(str(root)).variant == "one"


def test_unknown_variant_is_refused(tmp_path: Path) -> None:
    root = _write_task(tmp_path, _variant_cfg())
    with pytest.raises(SystemExit, match="not declared"):
        trial.load_task(str(root), "three")


def test_variant_must_name_a_declared_prompt(tmp_path: Path) -> None:
    cfg = _variant_cfg()
    cfg["variants"]["one"]["prompt"] = "zzz"
    root = _write_task(tmp_path, cfg)
    with pytest.raises(SystemExit, match="not under `prompts:`"):
        trial.load_task(str(root))
    cfg["variants"]["one"] = {"images": 1}
    root = _write_task(tmp_path / "second", cfg)
    with pytest.raises(SystemExit, match="needs a `prompt:` key"):
        trial.load_task(str(root))


def test_task_image_recipe_is_resolved_inside_task(tmp_path):
    root = _write_task(tmp_path, {"name": "atask", "image": "office", "image_dockerfile": "Dockerfile"})
    (root / "Dockerfile").write_text("FROM python:3.11-slim\n")
    assert trial.load_task(str(root)).image_dockerfile == root / "Dockerfile"


@pytest.mark.parametrize("recipe", ["missing", "../outside", "/etc/passwd"])
def test_task_image_recipe_rejects_missing_or_external_files(tmp_path, recipe):
    root = _write_task(tmp_path, {"name": "atask", "image_dockerfile": recipe})
    with pytest.raises(SystemExit, match="image_dockerfile"):
        trial.load_task(str(root))


def test_office_task_has_its_own_build_recipe():
    task = trial.load_task("knowledge_work")
    assert task.image_dockerfile.is_file()
    text = task.image_dockerfile.read_text()
    assert "libreoffice" in text and "PyMuPDF" in text
