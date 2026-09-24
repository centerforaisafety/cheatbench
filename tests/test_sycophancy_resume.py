"""tasks/sycophancy/run_chat.py refuses to resume a run whose finished episodes were made
from different inputs: the settings, the prompt template, a finished row's rendered
prompt or vars, or the judge's prompt or schema. Names alone do not identify a run."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks" / "sycophancy"))
import run_chat  # noqa: E402

SETTINGS = {"model_id": "openai/m", "generation_config": {"reasoning_effort": "high"},
            "routing": {"api_base_host": "h"}, "prompt_file": "prompts/instruction.md",
            "prompt_sha256": "p" * 64, "judge": "j", "judge_generation_config": {},
            "judge_prompt_sha256": "q" * 64, "judge_schema_sha256": "s" * 64}
ROWS = {"bubble__baseline": run_chat.row_digest("What odds?", {"vars": {"condition": "baseline"}}),
        "bubble__OpenAI": run_chat.row_digest("Investing.\nWhat odds?", {"vars": {"condition": "OpenAI"}})}


def _run_json(tmp_path, **overrides):
    doc = {**SETTINGS, "row_sha256": dict(ROWS), **overrides}
    (tmp_path / "run.json").write_text(json.dumps(doc))
    return tmp_path


def test_identical_inputs_resume_ok(tmp_path):
    run_chat.check_resume(_run_json(tmp_path), {"bubble__baseline"}, SETTINGS, ROWS)


def test_nothing_finished_needs_no_check(tmp_path):
    run_chat.check_resume(_run_json(tmp_path, prompt_sha256="x" * 64), set(), SETTINGS, ROWS)


def test_changed_row_text_is_refused_by_row_id(tmp_path):
    current = {**ROWS, "bubble__baseline": run_chat.row_digest("What odds, really?",
                                                                {"vars": {"condition": "baseline"}})}
    with pytest.raises(SystemExit, match="finished rows changed: bubble__baseline"):
        run_chat.check_resume(_run_json(tmp_path), {"bubble__baseline"}, SETTINGS, current)


def test_changed_vars_alone_are_refused(tmp_path):
    current = {**ROWS, "bubble__baseline": run_chat.row_digest("What odds?",
                                                                {"vars": {"condition": "control"}})}
    with pytest.raises(SystemExit, match="bubble__baseline"):
        run_chat.check_resume(_run_json(tmp_path), {"bubble__baseline"}, SETTINGS, current)


def test_a_changed_row_that_is_not_finished_does_not_block(tmp_path):
    current = {**ROWS, "bubble__OpenAI": run_chat.row_digest("Different.", {"vars": {}})}
    run_chat.check_resume(_run_json(tmp_path), {"bubble__baseline"}, SETTINGS, current)


def test_changed_template_or_judge_inputs_are_refused(tmp_path):
    for key in ("prompt_sha256", "judge_prompt_sha256", "judge_schema_sha256"):
        with pytest.raises(SystemExit, match=key):
            run_chat.check_resume(_run_json(tmp_path), {"bubble__baseline"},
                                  {**SETTINGS, key: "z" * 64}, ROWS)


def test_run_json_without_hashes_cannot_be_verified(tmp_path):
    doc = {k: v for k, v in SETTINGS.items() if not k.endswith("sha256")}
    (tmp_path / "run.json").write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="predates content hashing"):
        run_chat.check_resume(tmp_path, {"bubble__baseline"}, SETTINGS, ROWS)
