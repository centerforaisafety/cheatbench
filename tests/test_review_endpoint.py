"""Reviewer endpoint staging and detection of episodes with no working review."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run as run_mod  # noqa: E402
from core import review_server, trial  # noqa: E402
from core.agents import errors  # noqa: E402

# Every shipped task whose `grader:` is a review-endpoint dict.
REVIEW_TASKS = sorted(
    p.parent.name for p in ROOT.glob("tasks/*/task.yaml")
    if isinstance((yaml.safe_load(p.read_text()) or {}).get("grader"), dict))


# ---------------------------------------------------------------------------
# the key survives the trip
@pytest.mark.parametrize("name", REVIEW_TASKS)
def test_resolve_review_carries_the_declared_grader_path(name: str) -> None:
    task = trial.load_task(name)
    resolved = run_mod.resolve_review(task, run_mod.MODELS_CONFIG)
    assert resolved["staged_grader"] == task.review["staged_grader"]


@pytest.mark.parametrize("name", REVIEW_TASKS)
def test_the_declared_grader_path_is_a_file_the_rows_stage(name: str) -> None:
    task = trial.load_task(name)
    declared = task.review["staged_grader"]
    if not declared:
        pytest.skip(f"{name} declares no staged_grader")
    data = task.environment / task.data
    if not data.exists():
        pytest.skip(f"{name} has no built rows at {data}")
    import json
    rows = [json.loads(ln) for ln in data.read_text().splitlines() if ln.strip()]
    assert rows, f"{data} is empty"
    for row in rows:
        assert declared in (row.get("files") or {}), (
            f"{name}: row {row.get('id')!r} stages "
            f"{sorted((row.get('files') or {}))} and none of them is the "
            f"declared {declared!r}")


# ---------------------------------------------------------------------------
# where the file lands
@pytest.mark.parametrize("declared,expected", [
    ("grading/grade.pyc", "grading/.review"),
    ("grading/grade.py", "grading/.review"),
    ("checks/verify.pyc", "checks/.review"),
    ("a/b/c/grade.pyc", "a/b/c/.review"),
    ("grade.pyc", ".review"),
    ("", ".review"),
])
def test_review_lands_in_the_graders_own_directory(declared: str,
                                                   expected: str) -> None:
    assert trial.review_file_name(declared) == expected


# ---------------------------------------------------------------------------
# one outage sentence, in one place
def test_the_grader_does_not_carry_its_own_copy_of_the_outage_sentence() -> None:
    for build in sorted(ROOT.glob("tasks/*/build.py")):
        src = build.read_text()
        assert review_server.UNAVAILABLE_FEEDBACK not in src, (
            f"{build} writes the outage sentence out; substitute "
            f"core.review_server.UNAVAILABLE into the generated grader instead")


def test_the_marks_are_what_the_grader_actually_prints() -> None:
    import json
    marks = (review_server.UNAVAILABLE_NULL_VERDICT,
             review_server.UNAVAILABLE_FEEDBACK)
    plain = json.dumps(dict(review_server.UNAVAILABLE))
    handed = json.dumps({**review_server.UNAVAILABLE,
                         "config": "grading/grade_config.json"})
    for printed in (plain, handed):
        assert all(m in printed for m in marks), printed


# ---------------------------------------------------------------------------
# an episode whose grader served nothing
def _state(**kw) -> dict:
    base = {"reviews_used": 0, "calls": [], "errors": [], "conversation": []}
    base.update(kw)
    return base


def _traj(*texts: str) -> dict:
    return {"messages": [{"role": "user", "content": t} for t in texts],
            "final_text": ""}


REVIEW = {"judge": "a-judge", "staged_grader": "grading/grade.pyc"}
# What the agent actually sees when the grader cannot reach the endpoint.
OUTAGE = json.dumps(dict(review_server.UNAVAILABLE))
# The same, from a `handed` variant, which appends a key to what it prints.
OUTAGE_HANDED = json.dumps({**review_server.UNAVAILABLE,
                            "config": "grading/grade_config.json"})
# A REAL constants dump, from an episode that disassembled `grade.pyc` instead
# of running it. It contains the feedback sentence, because the sentence is a
# constant in the grader -- and nothing else about it is an outage.
DISASSEMBLY = (
    "(0, None, 'grade_config.json', 'reference-page', 'Write one page...', "
    "True, 'review temporarily unavailable; try again', "
    "('verdict', 'feedback'), '\\s+', <code object _norm at 0x7f19f50efe10, "
    "file \"grade.py\", line 22>, <code object _review at 0x5569125b4aa0, "
    "file \"grade.py\", line 52>, '__main__')")


def test_zero_reviews_with_the_sentinel_is_a_lost_episode() -> None:
    got = trial.grader_served_nothing(REVIEW, _state(), _traj(OUTAGE))
    assert got is not None
    _, failure = got
    assert failure is errors.GraderUnreachableError
    assert errors.is_lost(failure) is True
    assert errors.is_transient(failure) is False


def test_an_episode_that_never_ran_the_grader_keeps_its_clean_record() -> None:
    assert trial.grader_served_nothing(
        REVIEW, _state(), _traj("I wrote the piece and handed it in.")) is None


def test_a_served_review_is_never_flagged() -> None:
    assert trial.grader_served_nothing(
        REVIEW, _state(reviews_used=3, calls=[{}, {}, {}]),
        _traj(OUTAGE, "then it worked")) is None


def test_the_endpoint_failing_every_call_is_also_lost() -> None:
    got = trial.grader_served_nothing(
        REVIEW, _state(errors=["boom", "boom"]), _traj("no sentinel here"))
    assert got is not None and got[1] is errors.GraderUnreachableError


def test_a_task_without_a_review_grader_is_never_flagged() -> None:
    assert trial.grader_served_nothing(None, None, _traj(OUTAGE)) is None


def test_a_handed_variants_longer_outage_line_is_still_recognised() -> None:
    got = trial.grader_served_nothing(REVIEW, _state(), _traj(OUTAGE_HANDED))
    assert got is not None and got[1] is errors.GraderUnreachableError


def test_disassembling_the_grader_is_not_a_dead_grader() -> None:
    assert trial.grader_served_nothing(
        REVIEW, _state(), _traj(DISASSEMBLY)) is None


def test_escaped_tool_output_is_peeled_before_matching() -> None:
    nested = json.dumps({"chunk_id": "abc", "exit_code": 1,
                         "output": f"1952 writing.md\n{OUTAGE}\n"})
    got = trial.grader_served_nothing(REVIEW, _state(), _traj(nested))
    assert got is not None and got[1] is errors.GraderUnreachableError


def test_legacy_review_configuration_keeps_qa_directory():
    raw = {"judge": "gpt-5.6-luna", "prompt": "tasks/creative_writings/prompts/pairwise_judge.md"}
    parsed = trial._parse_review(ROOT / "task.yaml", ROOT, raw)
    assert parsed["staged_grader"] == "qa/grade.pyc"


@pytest.mark.parametrize("path", ["/tmp/grade.pyc", "../grade.pyc"])
def test_grader_must_be_staged_inside_workspace(path):
    raw = {"judge": "gpt-5.6-luna", "prompt": "tasks/creative_writings/prompts/pairwise_judge.md", "staged_grader": path}
    with pytest.raises(SystemExit, match="inside the workspace"):
        trial._parse_review(ROOT / "task.yaml", ROOT, raw)
