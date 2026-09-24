"""The Dockerfile is the source of truth, and two runs never build it twice.

No enroot and no daemon needed: these test the reading of the Dockerfile and the
locking around the build, not the build itself. The build itself is exercised
end to end by tests/test_image_anonymity.py, which runs against its output.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import sandbox  # noqa: E402
from core.sandbox import build  # noqa: E402


# ==========================================================================
# reading the Dockerfile
# ==========================================================================
def test_repo_dockerfile_parses() -> None:
    spec = sandbox.parse_dockerfile()
    assert spec["base"] == "python:3.11-slim"
    assert spec["workdir"] == "/workspace"
    assert spec["run"], "no RUN steps"


def test_continuations_stay_valid_shell() -> None:
    """A folded RUN is handed to bash verbatim, so the `\\` must survive.

    Rewriting `apt-get install \\`+newline to a bare newline split the command
    in two and the second half was a list of package names, which bash then
    tried to execute. That failed loudly (`curl: command not found`) but it
    could just as easily have half-built an image.
    """
    spec = sandbox.parse_dockerfile()
    apt = next(r for r in spec["run"] if "apt-get" in r)
    for line in apt.split("\n")[:-1]:
        assert line.rstrip().endswith("\\"), f"continuation lost: {line!r}"


def test_unsupported_instruction_is_refused(tmp_path: Path) -> None:
    """COPY/ADD would put a file of ours in the image. Refuse, never ignore."""
    p = tmp_path / "Dockerfile"
    p.write_text("FROM python:3.11-slim\nCOPY core/ /opt/harness/\n")
    with pytest.raises(ValueError, match="COPY"):
        sandbox.parse_dockerfile(p)


def test_from_is_required(tmp_path: Path) -> None:
    p = tmp_path / "Dockerfile"
    p.write_text("RUN echo hi\n")
    with pytest.raises(ValueError, match="FROM"):
        sandbox.parse_dockerfile(p)


def test_the_image_carries_no_agent() -> None:
    """The Dockerfile must not install the agent, its SDK or its runtime.

    These belong to the adapter (core/agents/), and an image that shipped them
    would weld one vendor to the sandbox -- and, for the SDK, would let an agent
    running `pip list` find the harness's own library in its environment. They
    are legitimate as a pre-baked CACHE, but that is a deliberate act and this
    test is what makes it deliberate rather than accidental.
    """
    text = sandbox.DOCKERFILE.read_text()
    body = "\n".join(ln for ln in text.split("\n")
                     if not ln.lstrip().startswith("#"))
    for banned in ("@anthropic-ai/claude-code", "claude-agent-sdk", "nodejs",
                   "npm"):
        assert banned not in body, (
            f"{banned!r} is installed by the Dockerfile; it belongs in the "
            f"adapter's install_script(). Remove it, or say in the header why "
            f"it is being cached here.")


# ==========================================================================
# the lock
# ==========================================================================
def _slow_build(lock_path: str, marker: str, hold: float) -> None:
    """Stand in for build_image: take the lock, record that we ran, hold it."""
    with build._file_lock(lock_path):
        with open(marker, "a") as f:
            f.write(f"{os.getpid()} {time.time():.4f}\n")
        time.sleep(hold)


def test_file_lock_serialises_processes(tmp_path: Path) -> None:
    """Two processes must not be inside the build at the same time.

    This is the property that matters on a cluster, where two `run.py`
    invocations on two nodes share a filesystem and neither knows about the
    other. `ensure_image` re-checks after acquiring, so in the real path the
    loser does not build at all -- here we let both through so the OVERLAP is
    what is being measured.
    """
    lock_path = str(tmp_path / "image.lock")
    marker = str(tmp_path / "who_ran")
    hold = 0.6
    procs = [multiprocessing.Process(target=_slow_build,
                                     args=(lock_path, marker, hold))
             for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)

    stamps = sorted(float(ln.split()[1])
                    for ln in Path(marker).read_text().splitlines())
    assert len(stamps) == 4
    for a, b in zip(stamps, stamps[1:]):
        assert b - a >= hold * 0.9, (
            f"two processes entered the build {b - a:.3f}s apart, but each "
            f"holds it for {hold}s -- the lock did not serialise them")


def _racing_ensure(image: str, built: str) -> None:
    """ensure_image with a build that just creates the file."""
    def fake_build(img, log=None):
        time.sleep(0.4)
        with open(built, "a") as f:
            f.write(f"{os.getpid()}\n")
        Path(img).write_text("image")
        return img

    build.build_image = fake_build
    build.ensure_image(image, log=lambda m: None)


def test_ensure_image_builds_exactly_once(tmp_path: Path) -> None:
    """N processes, image absent, exactly one build.

    The loser wakes up holding the lock, finds the winner's image on the
    re-check, and returns without building. Demonstrated for real against
    enroot too: two concurrent `python -m core.sandbox --build` with the image
    deleted produced one `enroot import`/`export` pair and one "another process
    built it while we waited".
    """
    image = str(tmp_path / "python311.sqsh")
    built = str(tmp_path / "builds")
    Path(built).touch()
    procs = [multiprocessing.Process(target=_racing_ensure, args=(image, built))
             for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)

    n = len([ln for ln in Path(built).read_text().splitlines() if ln.strip()])
    assert n == 1, f"{n} processes built the image; exactly 1 should have"
    assert os.path.exists(image)


def test_ensure_image_is_a_noop_when_present(tmp_path: Path) -> None:
    """The common case is one os.path.exists and nothing else."""
    image = tmp_path / "python311.sqsh"
    image.write_text("image")
    called = []
    real, build.build_image = build.build_image, lambda *a, **k: called.append(1)
    try:
        assert build.ensure_image(str(image)) is False
    finally:
        build.build_image = real
    assert not called
    assert not (tmp_path / "python311.sqsh.lock").exists(), \
        "took the lock for an image that was already there"


# ==========================================================================
# where things live
# ==========================================================================
def test_image_name_resolves_under_the_images_dir() -> None:
    """A task's `image:` is a NAME; a path is also accepted as itself."""
    assert sandbox.image_path("python311") == \
        os.path.join(sandbox.IMAGES_DIR, "python311.sqsh")
    assert sandbox.image_path() == sandbox.image_path(sandbox.DEFAULT_IMAGE_NAME)
    assert sandbox.image_path("/tmp/other.sqsh") == "/tmp/other.sqsh"


def test_no_operator_home_baked_into_the_defaults() -> None:
    """No committed default may name a person.

    A literal `/data/<someone>/...` in the code is both machine-specific and, if
    it ever reached the container, a way for the agent to name the operator.
    Every path is an environment variable with a portable default instead.

    STRING LITERALS only, via the AST: prose in a docstring may of course say
    the words `/data/<someone>` -- explaining why the rule exists is not
    breaking it -- and a comment is not code either.
    """
    import ast

    for f in ("paths.py", "build.py", "runtime.py"):
        path = Path(sandbox.__file__).parent / f
        tree = ast.parse(path.read_text())
        docstrings = {id(ast.get_docstring(n, clean=False))
                      for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.ClassDef,
                                        ast.FunctionDef, ast.AsyncFunctionDef))}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)):
                continue
            if id(node.value) in docstrings:
                continue
            assert "/data/" not in node.value, (
                f"{f}:{node.lineno}: hardcoded host path in a string literal: "
                f"{node.value!r}")


def test_build_uses_the_rootfs_directory_it_cleans(monkeypatch):
    monkeypatch.setattr(build.paths, 'DEFAULT_DATA_PATH', '/tmp/expected-build-data')
    monkeypatch.setenv('ENROOT_DATA_PATH', '/tmp/different-site-default')
    assert build._build_env()['ENROOT_DATA_PATH'] == '/tmp/expected-build-data'


def test_office_default_selects_its_recipe():
    from core.trial import load_task
    task = load_task('knowledge_work')
    assert task.image == 'python311-office'
    assert task.image_dockerfile == task.root / 'Dockerfile'
    steps = '\n'.join(build.parse_dockerfile(task.image_dockerfile)['run'])
    assert 'pymupdf' in steps.lower()
    assert 'import fitz' in steps
    assert 'soffice --headless --version' in steps


@pytest.mark.parametrize('async_call', [False, True])
def test_task_image_rebuilds_legacy_and_changed_recipe(tmp_path, monkeypatch, async_call):
    import asyncio
    image = tmp_path / 'office.sqsh'
    image.write_bytes(b'legacy image missing office dependencies')
    recipe = tmp_path / 'Dockerfile'
    recipe.write_text('FROM python:3.11-slim\nRUN pip install pymupdf\n')
    built = []
    def fake_build(candidate, **kwargs):
        assert kwargs['dockerfile'] == recipe
        assert image.read_bytes() != b'incomplete image'
        built.append(candidate)
        Path(candidate).write_bytes(recipe.read_bytes())
    monkeypatch.setattr(build, 'build_image', fake_build)
    def ensure():
        if async_call:
            return asyncio.run(build.ensure_image_async(str(image), dockerfile=recipe))
        return build.ensure_image(str(image), dockerfile=recipe)
    assert ensure() is True
    assert image.read_bytes() == recipe.read_bytes()
    assert ensure() is False
    recipe.write_text(recipe.read_text() + 'RUN pip install pillow\n')
    assert ensure() is True
    assert len(built) == 2
    assert ensure() is False
    # Replacing a file under the same name cannot reuse its previous receipt.
    image.write_bytes(b'replaced image')
    assert ensure() is True
    assert len(built) == 3


def test_failed_task_image_build_preserves_previous_image(tmp_path, monkeypatch):
    image = tmp_path / 'office.sqsh'
    image.write_bytes(b'previous image')
    recipe = tmp_path / 'Dockerfile'
    recipe.write_text('FROM python:3.11-slim\n')
    def fail(candidate, **kwargs):
        Path(candidate).write_bytes(b'incomplete image')
        raise RuntimeError('dependency or audit failed')
    monkeypatch.setattr(build, 'build_image', fail)
    with pytest.raises(RuntimeError, match='dependency or audit failed'):
        build.ensure_image(str(image), dockerfile=recipe)
    assert image.read_bytes() == b'previous image'
    assert not Path(str(image) + '.build.json').exists()
    assert not list(tmp_path.glob('.image-*'))


def _racing_task_ensure(image, recipe, counter):
    def fake_build(candidate, **kwargs):
        time.sleep(0.2)
        with open(counter, 'a') as f:
            f.write('build\n')
        Path(candidate).write_bytes(b'office image')
    build.build_image = fake_build
    build.ensure_image(image, dockerfile=Path(recipe))


def test_stale_task_image_rebuilt_once_across_processes(tmp_path):
    image = tmp_path / 'office.sqsh'
    image.write_bytes(b'stale image')
    recipe = tmp_path / 'Dockerfile'
    recipe.write_text('FROM python:3.11-slim\n')
    counter = tmp_path / 'counter'
    procs = [multiprocessing.Process(target=_racing_task_ensure,
                args=(str(image), str(recipe), str(counter))) for _ in range(4)]
    for p in procs: p.start()
    for p in procs:
        p.join(20)
        assert p.exitcode == 0
    assert counter.read_text().splitlines() == ['build']
    assert build.ensure_image(str(image), dockerfile=recipe) is False
