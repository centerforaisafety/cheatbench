"""ATIF is the ONE internal trajectory, and one renderer reads it.

These tests defend four properties:

  * the ported models validate what the spec says they validate;
  * a `Trajectory` round-trips through JSON unchanged;
  * the Claude adapter's conversion puts each piece of the episode in the ATIF
    field that means it -- reasoning in `reasoning_content`, never folded into
    the message; every tool call paired with its own result;
  * a HISTORICAL transcript (SDK-shaped, written long before ATIF existed) still
    converts and still renders, because re-judging old runs is the thing that
    tells a rubric change apart from a harness drift.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.agents import ClaudeSDKAgent, trajectory_from_transcript
from core.render import render_trajectory
from core.trajectory import (
    SCHEMA_VERSION,
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)

ROOT = Path(__file__).resolve().parent.parent

# A real run made before any of this existed. Its transcripts are SDK-shaped.
LEGACY_RUN = ROOT / "outputs" / "openmath_claude-opus-5_5243" / "trajectories"


def _trajectory() -> Trajectory:
    return Trajectory(
        schema_version=SCHEMA_VERSION,
        session_id="sess-1",
        agent=Agent(name="claude-sdk", version="2.0.1",
                    model_name="claude-opus-5"),
        steps=[
            Step(step_id=1, source="user", message="do the thing"),
            Step(step_id=2, source="agent", message="on it",
                 reasoning_content="privately reconsidering",
                 tool_calls=[ToolCall(tool_call_id="t1", function_name="Bash",
                                      arguments={"command": "ls"})],
                 observation=Observation(results=[
                     ObservationResult(source_call_id="t1", content="a\nb")]),
                 metrics=Metrics(prompt_tokens=10, completion_tokens=5),
                 llm_call_count=1),
        ],
        final_metrics=FinalMetrics(total_prompt_tokens=10, total_steps=2),
    )


# --------------------------------------------------------------------------
# the ported models
# --------------------------------------------------------------------------
def test_trajectory_round_trips_through_json() -> None:
    traj = _trajectory()
    again = Trajectory.model_validate(json.loads(json.dumps(traj.to_json_dict())))
    assert again == traj
    assert again.schema_version == "ATIF-v1.8"


def test_step_ids_must_be_sequential_from_one() -> None:
    with pytest.raises(ValueError, match="sequential from 1"):
        Trajectory(agent=Agent(name="a", version="1"),
                   steps=[Step(step_id=2, source="user", message="x")])


def test_observation_cannot_reference_an_unknown_call() -> None:
    """The pairing is validated, so a mis-paired result cannot reach the judge."""
    with pytest.raises(ValueError, match="source_call_id"):
        Trajectory(
            agent=Agent(name="a", version="1"),
            steps=[Step(step_id=1, source="agent", message="",
                        observation=Observation(results=[
                            ObservationResult(source_call_id="nope",
                                              content="x")]))])


def test_non_agent_steps_reject_agent_only_fields() -> None:
    with pytest.raises(ValueError, match="only applicable when source is 'agent'"):
        Step(step_id=1, source="user", message="x", reasoning_content="no")


def test_schema_version_is_pinned_in_one_place() -> None:
    """Both adapters emit the same version because they read the same constant."""
    assert SCHEMA_VERSION == "ATIF-v1.8"
    assert Trajectory(agent=Agent(name="a", version="1"),
                      steps=[Step(step_id=1, source="user", message="x")]
                      ).schema_version == SCHEMA_VERSION


# --------------------------------------------------------------------------
# the Claude SDK conversion
# --------------------------------------------------------------------------
SDK_TRANSCRIPT = {
    "agent": "claude-sdk",
    "model": "claude-opus-5",
    "messages": [
        {"_type": "SystemMessage", "subtype": "init",
         "data": {"tools": ["Bash"], "mcp_servers": []}},
        {"_type": "AssistantMessage", "content": [
            {"_type": "ThinkingBlock", "thinking": "maybe I should just look"},
            {"_type": "TextBlock", "text": "Checking the directory."},
            {"_type": "ToolUseBlock", "id": "tu1", "name": "Bash",
             "input": {"command": "ls -la"}}]},
        {"_type": "UserMessage", "content": [
            {"_type": "ToolResultBlock", "tool_use_id": "tu1",
             "is_error": False, "content": "total 0"}]},
        {"_type": "ResultMessage", "subtype": "success", "num_turns": 1,
         "total_cost_usd": 0.5,
         "usage": {"input_tokens": 3, "output_tokens": 7,
                   "cache_read_input_tokens": 100,
                   "cache_creation_input_tokens": 20},
         "result": "Done."},
    ],
}


def test_sdk_transcript_converts_to_valid_atif() -> None:
    traj = ClaudeSDKAgent(model="claude-opus-5").to_trajectory(SDK_TRANSCRIPT)

    assert traj.schema_version == SCHEMA_VERSION
    assert traj.agent.name == "claude-sdk"
    # The init message becomes the agent block, not a step.
    assert traj.agent.extra == {"tools": ["Bash"], "mcp_servers": []}

    step = traj.steps[0]
    assert step.source == "agent"
    # Extended thinking is its OWN field. Folding it into the message would make
    # what the model SAID indistinguishable from what it privately reasoned.
    assert step.reasoning_content == "maybe I should just look"
    assert step.message == "Checking the directory."
    assert "maybe I should just look" not in step.message

    # The tool call and its result live on the same step, paired by id.
    assert step.tool_calls[0].function_name == "Bash"
    assert step.tool_calls[0].arguments == {"command": "ls -la"}
    assert step.observation.results[0].source_call_id == "tu1"

    # Totals land in final_metrics: Harbor's rule is input + cache read +
    # cache creation, which is what Anthropic's own session totals report.
    assert traj.final_metrics.total_prompt_tokens == 3 + 100 + 20
    assert traj.final_metrics.total_completion_tokens == 7
    assert traj.final_metrics.total_cost_usd == 0.5

    assert Trajectory.model_validate(traj.to_json_dict()) == traj


def test_an_episode_that_captured_nothing_is_still_valid_atif() -> None:
    """The worst failures must still be recordable and judgeable."""
    traj = ClaudeSDKAgent(model="m").to_trajectory({"messages": []})
    assert len(traj.steps) == 1
    assert "no messages" in traj.steps[0].message


def test_render_puts_reasoning_and_calls_in_the_judges_shape() -> None:
    text = render_trajectory(
        ClaudeSDKAgent(model="claude-opus-5").to_trajectory(SDK_TRANSCRIPT))
    lines = text.split("\n")
    assert lines[0].startswith("[init] agent=claude-sdk")
    assert "THINKING: maybe I should just look" in lines
    assert "ASSISTANT: Checking the directory." in lines
    assert "[0] TOOL_CALL Bash: command=ls -la" in lines
    assert "    -> RESULT[Bash]: total 0" in lines
    assert lines[-1] == "FINAL MESSAGE: Done."


def test_text_is_never_truncated_and_base64_is_elided() -> None:
    """The no-truncation rule. A clipped command has previously hidden the
    behaviour a verdict turned on, so this is load-bearing, not tidiness."""
    # Prose with spaces, not an unbroken alphabet run: the elision rule keys on
    # the latter, so `a/a/a/...` would (correctly) be treated as a payload.
    long_command = "grep -rn " + " ".join(f"--include=*.{i}" for i in range(4000))
    blob = "QWxhZGRpbjpvcGVuIHNlc2FtZQ" * 40
    transcript = {
        "agent": "claude-sdk", "model": "m",
        "messages": [
            {"_type": "AssistantMessage", "content": [
                {"_type": "ToolUseBlock", "id": "t", "name": "Bash",
                 "input": {"command": long_command}}]},
            {"_type": "UserMessage", "content": [
                {"_type": "ToolResultBlock", "tool_use_id": "t",
                 "content": blob}]},
        ],
    }
    text = render_trajectory(
        ClaudeSDKAgent(model="m").to_trajectory(transcript))
    assert long_command in text          # never cut
    assert blob not in text              # payload, not content
    assert "base64 blob" in text


def test_tool_errors_survive_the_conversion() -> None:
    transcript = {
        "agent": "claude-sdk", "model": "m",
        "messages": [
            {"_type": "AssistantMessage", "content": [
                {"_type": "ToolUseBlock", "id": "t", "name": "Bash",
                 "input": {"command": "false"}}]},
            {"_type": "UserMessage", "content": [
                {"_type": "ToolResultBlock", "tool_use_id": "t",
                 "is_error": True, "content": "boom"}]},
        ],
    }
    traj = ClaudeSDKAgent(model="m").to_trajectory(transcript)
    assert traj.steps[0].observation.results[0].extra == {"is_error": True}
    assert "-> RESULT[Bash] (ERROR): boom" in render_trajectory(traj)


# --------------------------------------------------------------------------
# back-compat: historical runs must stay judgeable
# --------------------------------------------------------------------------
@pytest.mark.skipif(not LEGACY_RUN.exists(),
                    reason="the historical run is not present in this checkout")
def test_every_historical_transcript_converts_and_renders() -> None:
    paths = sorted(LEGACY_RUN.glob("*/transcript.json"))
    assert paths, "expected the recorded openmath run to have transcripts"
    for path in paths:
        transcript = json.loads(path.read_text())
        traj = trajectory_from_transcript(transcript)
        assert isinstance(traj, Trajectory)
        assert traj.steps
        text = render_trajectory(traj)
        assert text.startswith("[init] agent=claude-sdk")
        # The evidence actually survived: these episodes all used tools.
        assert "TOOL_CALL" in text
        # And it is still ATIF.
        Trajectory.model_validate(traj.to_json_dict())


def test_a_transcript_from_before_adapters_were_named_reads_as_the_sdk() -> None:
    """There was only one adapter then, so this is history, not a guess."""
    traj = trajectory_from_transcript(
        {"messages": [{"_type": "ResultMessage", "result": "done"}]})
    assert traj.agent.name == "claude-sdk"
    assert render_trajectory(traj).endswith("FINAL MESSAGE: done")


def test_an_unknown_agent_is_refused_rather_than_guessed() -> None:
    with pytest.raises(SystemExit, match="not a registered adapter"):
        trajectory_from_transcript({"agent": "grok-cli", "messages": []})


# --------------------------------------------------------------------------
# thinking is CAPTURED, empty or not
#
# Under `display: "omitted"` -- the Opus 5 default -- ThinkingBlocks arrive with
# an empty `thinking` field. Filtering them on truthiness made a transcript in
# which the model reasoned and the text was withheld look exactly like one in
# which it never reasoned, and that cost a real diagnosis. The renderer decides
# what is worth showing; the converter decides nothing away.
# --------------------------------------------------------------------------
def _assistant(*blocks: dict) -> dict:
    return {"agent": "claude-sdk", "model": "claude-opus-5",
            "messages": [{"_type": "AssistantMessage", "content": list(blocks)}]}


def test_an_empty_thinking_block_survives_as_evidence_that_thinking_was_on() -> None:
    traj = ClaudeSDKAgent(model="claude-opus-5").to_trajectory(
        _assistant({"_type": "ThinkingBlock", "thinking": "", "signature": "s"},
                   {"_type": "TextBlock", "text": "Answer."}))
    step = traj.steps[0]
    # Nothing readable, so nothing to render...
    assert step.reasoning_content is None
    assert "THINKING:" not in render_trajectory(traj)
    # ...but the block's EXISTENCE is on the record, which is the whole point:
    # thinking-on-but-omitted is not the same fact as no thinking at all.
    assert step.extra == {"thinking_blocks": 1}
    assert Trajectory.model_validate(traj.to_json_dict()) == traj


def test_an_absent_thinking_block_says_nothing_at_all() -> None:
    traj = ClaudeSDKAgent(model="claude-opus-5").to_trajectory(
        _assistant({"_type": "TextBlock", "text": "Answer."}))
    assert traj.steps[0].extra is None


def test_a_real_thinking_block_renders_exactly_as_before() -> None:
    """The judge's view of real reasoning must not move by one byte."""
    traj = ClaudeSDKAgent(model="claude-opus-5").to_trajectory(
        _assistant({"_type": "ThinkingBlock", "thinking": " maybe I should look ",
                    "signature": "s"},
                   {"_type": "TextBlock", "text": "Answer."}))
    step = traj.steps[0]
    assert step.reasoning_content == "maybe I should look"
    assert "THINKING: maybe I should look" in render_trajectory(traj)
    assert step.extra == {"thinking_blocks": 1}


def test_redacted_thinking_from_openrouter_is_read_as_reasoning() -> None:
    """OpenRouter reuses Anthropic's envelope to pass PLAIN reasoning through."""
    import base64

    inner = json.dumps({"text": "proxied thought", "type": "reasoning.text"})
    data = "openrouter.reasoning:" + base64.b64encode(inner.encode()).decode()
    traj = ClaudeSDKAgent(model="claude-opus-5").to_trajectory(
        _assistant({"_type": "RedactedThinkingBlock", "data": data},
                   {"_type": "TextBlock", "text": "Answer."}))
    step = traj.steps[0]
    assert step.reasoning_content == "proxied thought"
    assert step.extra == {"redacted_thinking_blocks": 1}
    assert "THINKING: proxied thought" in render_trajectory(traj)


def test_genuine_redacted_ciphertext_is_counted_but_never_rendered() -> None:
    """It cannot be decrypted; a blob in the judge's log is noise, not evidence."""
    traj = ClaudeSDKAgent(model="claude-opus-5").to_trajectory(
        _assistant({"type": "redacted_thinking", "data": "EroBCkYIBBgCKkA..."},
                   {"_type": "TextBlock", "text": "Answer."}))
    step = traj.steps[0]
    assert step.reasoning_content is None
    assert step.extra == {"redacted_thinking_blocks": 1}
    assert "EroBCkYIBBgC" not in render_trajectory(traj)


def test_a_message_of_nothing_but_an_empty_thought_is_still_a_step() -> None:
    traj = ClaudeSDKAgent(model="claude-opus-5").to_trajectory(
        _assistant({"_type": "ThinkingBlock", "thinking": "", "signature": "s"}))
    assert len(traj.steps) == 1
    assert traj.steps[0].extra == {"thinking_blocks": 1}
