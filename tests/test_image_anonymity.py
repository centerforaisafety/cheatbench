"""The sandbox image must tell the agent nothing about the eval.

This is the one check in the harness that has no equivalent in Harbor, and
deliberately so: Harbor's agents are allowed to know they are being evaluated,
and ours must not be. It is a test rather than a script somebody is supposed to
remember to run, because the thing it guards against is exactly the kind of
mistake that gets made in a hurry. The first build of this image baked the
harness and its prompts into /opt, which put the judge rubric and the verifier
internals one `cat` away from the agent under test.

Every probe goes through `core/sandbox.run_shell`, which builds the SAME command
line, mount table, hooks and namespaces a real episode gets. Half of what is
checked here (the mount table, the environment, /proc) is a property of the
LAUNCHER rather than of the image, and a test that took a shortcut with a bare
`enroot start` would be testing something other than what the agent sees.

    pytest tests/test_image_anonymity.py -v

Skips cleanly where enroot, /dev/fuse or the image itself is unavailable, so the
suite still runs on a laptop. A cheap subset of the same probes runs
automatically after every image build (core/sandbox/build.py), so a leaking
image cannot reach a run even if nobody runs pytest.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import sandbox  # noqa: E402

IMAGE = os.environ.get("RH_TEST_IMAGE") or sandbox.DEFAULT_IMAGE


@pytest.fixture(scope="module")
def image() -> str:
    """The image under test, or a skip with the reason."""
    sandbox.ensure_path()
    problems = sandbox.preflight()
    if problems:
        pytest.skip("sandbox runtime unavailable: " + "; ".join(problems))
    if not os.path.exists(IMAGE):
        pytest.skip(f"{IMAGE} not built (python -m core.sandbox --build)")
    return IMAGE


def _run(image: str, cmd: str) -> str:
    res = sandbox.run_shell(cmd, image, timeout=600)
    return res.stdout.strip()


@pytest.mark.parametrize("name,cmd", sandbox.audit_probes(),
                         ids=[n for n, _ in sandbox.audit_probes()])
def test_probe_finds_nothing(image: str, name: str, cmd: str) -> None:
    """Each probe is reconnaissance an agent could run. All must come back empty.

    NO TRUNCATION on failure: whatever leaked, in full, because the point of the
    failure message is to say precisely what an agent would have found.
    """
    out = _run(image, cmd)
    assert not out, f"{name} leaked:\n{out}"


def test_image_file_name_is_unremarkable() -> None:
    """The image FILE NAME is visible to the agent.

    enroot leaves squashfuse running alongside the payload with the image path
    in its argv. The private pid namespace hides it from `ps`, but the name is
    the last line of defence and must stay something an ordinary Python
    container would be called.
    """
    stem = Path(IMAGE).stem
    assert stem == sandbox.DEFAULT_IMAGE_NAME, (
        f"image is named {stem!r}; it must stay a plain "
        f"{sandbox.DEFAULT_IMAGE_NAME!r} -- the name reaches the agent")


def test_agent_is_not_root_and_cwd_is_neutral(image: str) -> None:
    """The identity the agent runs as is a stock container user, not the operator.

    Also the reason it is uid 1000 at all: the Claude Code CLI refuses
    --dangerously-skip-permissions as root, and every episode runs with
    permission_mode=bypassPermissions.
    """
    out = _run(image, "id -u; id -un; pwd; echo $HOME")
    uid, user, pwd, home = out.split("\n")[:4]
    assert uid == str(sandbox.AGENT_UID)
    assert user == sandbox.AGENT_NAME
    assert pwd == "/workspace"
    assert home == sandbox.AGENT_HOME


def test_no_harness_source_reachable(image: str) -> None:
    """Nothing of ours is on the container filesystem, anywhere.

    The runner, the staging module, the task's grader and the row arrive on
    stdin and are consumed before the agent exists. A `find` for our module
    names is the direct form of the question.
    """
    ours = "claude_sdk_runner|judge_schema|task.yaml|grader.py|models.yaml"
    out = _run(image, f"find / -xdev -type f 2>/dev/null | grep -Ei '{ours}'")
    assert not out, f"harness source reachable inside the container:\n{out}"


def test_proc_shows_only_the_episode(image: str) -> None:
    """`ps` inside shows the payload and nothing else.

    enroot leaves squashfuse, fuse-overlayfs and a shim shell running next to the
    payload; a shared pid namespace would hand the agent the image path and
    enroot's runtime directory through their argv. The payload unshares its own
    pid namespace so pid 1 is the episode and there is nothing else.
    """
    out = _run(image, "ps -eo comm --no-headers 2>/dev/null | sort -u")
    names = {ln.strip() for ln in out.split("\n") if ln.strip()}
    assert not (names & {"squashfuse", "fuse-overlayfs", "enroot", "slirp4netns"}), \
        f"enroot's plumbing is visible to the agent: {sorted(names)}"


def test_cmdline_of_pid_1_says_nothing(image: str) -> None:
    """/proc/1/cmdline is world-readable and the agent runs as the same uid.

    So the only thing on the container's command line is the bootstrap: a
    dozen tokens of json/exec that name neither the task nor the eval.
    """
    out = _run(image, "tr '\\0' ' ' < /proc/1/cmdline")
    terms = ("reward", "hacking", "openmath", "gdpval", "judge", "rubric",
             "grader", "instruction")
    found = [t for t in terms if t in out.lower()]
    assert not found, f"/proc/1/cmdline names {found}:\n{out}"


def test_filename_audit_distinguishes_writings_from_sympy() -> None:
    import re
    from core.sandbox.build import EVAL_TERMS

    for path in ("/opt/writings/prompt.md", "/opt/creative_writings/grader.py"):
        assert re.search(EVAL_TERMS, path, re.I)
    assert not re.search(EVAL_TERMS, "/usr/local/lib/python3.11/site-packages/sympy/combinatorics/rewritingsystem.py", re.I)
