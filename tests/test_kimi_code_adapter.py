"""The Kimi Code adapter: its policy, its invocation, and its journal as ATIF.

Three things this file pins, in the order they would hurt if lost:

  * the closed-book policy lands in `$KIMI_CODE_HOME/config.toml` as
    `[tools] disabled = ["FetchURL", "WebSearch"]` -- the CLI's default is
    FetchURL ON, and only that block turns it off (verified on kimi-code 0.42.0
    by the model enumerating its own tools);
  * the model's reasoning reaches the judge. Neither the CLI's stream-json
    output nor its wire.jsonl journal carries thinking, so the runner taps the
    gateway responses and the adapter joins reasoning to steps by response id.
    A journal step whose `messageId` matches a `forwarder.call` record must
    render a THINKING block;
  * the forwarder is never optional for this adapter, and the CLI never sees
    the credential: the blob carries the forwarder module every time, and the
    only key the CLI is configured with is a placeholder.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import trial  # noqa: E402
from core.agents import AgentFactory, make_agent  # noqa: E402
from core.agents.kimi_code import KimiCodeAgent  # noqa: E402
from core.agents.kimi_code_runner import (  # noqa: E402
    compose_command,
    mcp_json,
    summarise,
    usage_from_calls,
)

CLOSED = {"web_search": False, "web_fetch": False}
MODEL = "openrouter/moonshotai/kimi-k3"
PIN = {"provider": {"order": ["moonshotai"], "allow_fallbacks": False}}


def _agent(policy: dict = CLOSED, **kw) -> KimiCodeAgent:
    kw.setdefault("api_base_url", "https://gateway.invalid")
    agent = make_agent("kimi-code", model=MODEL, **kw)
    agent.apply_tool_policy(policy)
    return agent


def _blob(agent: KimiCodeAgent, grader_src: str = "", prompt: str = "p") -> dict:
    return json.loads(agent.blob(
        row={"id": "row", "answer": "THE-KEY-42"}, prompt=prompt, files=[],
        skip_dirs=(), grader_src=grader_src, stage_src="", timeout_s=60,
        env={}, install=None))


# --------------------------------------------------------------------------
# registry + identity
# --------------------------------------------------------------------------
def test_registered_under_its_own_name() -> None:
    assert AgentFactory.get_agent_class("kimi-code") is KimiCodeAgent
    assert KimiCodeAgent.name() == "kimi-code"
    assert "kimi-code" in AgentFactory.names()


# --------------------------------------------------------------------------
# THE GUARD: the closed-book policy becomes `[tools] disabled`
# --------------------------------------------------------------------------
def test_closed_book_disables_both_web_tools_in_config_toml() -> None:
    agent = _agent(CLOSED)
    toml = agent.config_toml()
    assert "[tools]" in toml
    disabled = json.loads(toml.split("disabled = ", 1)[1].split("\n", 1)[0])
    assert set(disabled) == {"FetchURL", "WebSearch"}
    assert KimiCodeAgent.WEB_FETCH_TOOL not in agent.tools
    assert KimiCodeAgent.WEB_SEARCH_TOOL not in agent.tools
    assert agent.setup() == []


# These tasks use the common sandbox runner; Harbor and direct chat do not.
@pytest.mark.parametrize("name", sorted(p.parent.name for p
                                        in ROOT.glob("tasks/*/task.yaml")
                                        if p.parent.name not in {"software_engineering", "sycophancy"}))
def test_every_shipped_task_composes_a_closed_config(name: str) -> None:
    task = trial.load_task(name)
    agent = _agent(task.tools)
    assert task.tools == {**CLOSED, **({"mcp_only": True} if name in {"chess", "go"} else {})}, (
        f"tasks/{name} changed its tools: block; this test asserts the "
        f"closed-book composition and must be updated deliberately")
    assert '"FetchURL"' in agent.config_toml()
    assert agent.setup() == []
    # ...and the config rides in the blob, verbatim, for the runner to write.
    assert _blob(agent)["task"]["kimi"]["config_toml"] == agent.config_toml()


def test_open_fetch_keeps_fetchurl_and_still_disables_search() -> None:
    agent = _agent({"web_search": False, "web_fetch": True})
    assert KimiCodeAgent.WEB_FETCH_TOOL in agent.tools
    assert agent.disabled_tools() == ["WebSearch"]


def test_web_search_true_is_refused_there_is_no_provider() -> None:
    agent = make_agent("kimi-code", model=MODEL)
    with pytest.raises(ValueError) as excinfo:
        agent.apply_tool_policy({"web_search": True, "web_fetch": True})
    assert "WebSearch" in str(excinfo.value)


def test_both_web_keys_are_enforceable_and_unknown_keys_refused() -> None:
    assert KimiCodeAgent.ENFORCEABLE == {"web_search", "web_fetch", "mcp_only"}
    agent = make_agent("kimi-code", model=MODEL)
    with pytest.raises(ValueError) as excinfo:
        agent.apply_tool_policy({"web_search": False, "some_future_tool": False})
    assert "kimi-code" in str(excinfo.value)
    assert "some_future_tool" in str(excinfo.value)


def test_max_turns_becomes_the_step_cap() -> None:
    agent = _agent(CLOSED, max_turns=150)
    assert "[loop_control]\nmax_steps_per_turn = 150" in agent.config_toml()
    assert "[loop_control]" not in _agent(CLOSED, max_turns=None).config_toml()


# --------------------------------------------------------------------------
# the invocation, against Harbor's shape
# --------------------------------------------------------------------------
def test_command_matches_harbors_shape_with_no_permission_flag() -> None:
    agent = _agent(CLOSED)
    command = compose_command(_blob(agent, prompt="solve it")["task"]["kimi"],
                              "solve it")
    for fragment in ("kimi ", "--prompt 'solve it'", "--output-format stream-json",
                     "</dev/null", "nvm.sh"):
        assert fragment in command, fragment
    # Print mode refuses both; the CLI runs `auto` on its own.
    assert "--yolo" not in command and "--auto" not in command


def test_generation_config_is_delivered_as_thinking_effort_env() -> None:
    agent = _agent(CLOSED, generation_config={"thinking_effort": "high"})
    assert agent.cli_env()["KIMI_MODEL_THINKING_EFFORT"] == "high"
    assert agent.setup() == []
    assert "KIMI_MODEL_THINKING_EFFORT" not in _agent(CLOSED).cli_env()


def test_an_untranslated_generation_key_is_a_preflight_problem() -> None:
    agent = _agent(CLOSED, generation_config={"unknown_reasoning_knob": "high"})
    problems = agent.setup()
    assert problems and "unknown_reasoning_knob" in problems[0]


def test_no_base_url_is_a_preflight_problem(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    agent = make_agent("kimi-code", model=MODEL)
    agent.apply_tool_policy(CLOSED)
    assert any("api_base_url" in p for p in agent.setup())


# --------------------------------------------------------------------------
# the blob: the forwarder always travels, the credential never does
# --------------------------------------------------------------------------
def test_blob_always_carries_the_forwarder_and_the_full_model_id() -> None:
    agent = _agent(CLOSED, extra_body=PIN)
    payload = _blob(agent)
    assert "class Upstream" in payload["modules"]["forwarder"]
    assert payload["task"]["routing"]["extra_body"] == PIN
    assert payload["task"]["routing"]["api_base_url"] == "https://gateway.invalid"
    # No prefix stripping: the gateway routes on the id as written.
    assert payload["task"]["kimi"]["model_name"] == MODEL
    # Without extra_body the forwarder STILL travels (it is the reasoning tap).
    assert "class Upstream" in _blob(_agent(CLOSED))["modules"]["forwarder"]
    assert agent.resolved_routing()["forwarder"] is True
    assert _agent(CLOSED).resolved_routing()["forwarder"] is True


def test_the_cli_only_ever_gets_a_placeholder_key() -> None:
    agent = _agent(CLOSED)
    payload = _blob(agent)
    assert KimiCodeAgent.CLI_KEY_ENV == ()
    assert payload["task"]["kimi"]["placeholder_key"]
    assert "KIMI_MODEL_API_KEY" not in agent.cli_env()
    assert KimiCodeAgent.SUPPORTS_FORWARDER is True


def test_blob_carries_row_and_grader_only_when_the_task_ships_one() -> None:
    with_grader = _blob(_agent(CLOSED), "SECRET GRADER SOURCE")
    assert with_grader["task"]["row"]["answer"] == "THE-KEY-42"
    assert with_grader["modules"]["grader"] == "SECRET GRADER SOURCE"
    without = _blob(_agent(CLOSED), "")
    assert "row" not in without["task"]
    assert "THE-KEY-42" not in json.dumps(without)


def test_mcp_json_is_the_url_shape_never_a_command() -> None:
    text = mcp_json("http://127.0.0.1:1/tok/mcp", "grader")
    doc = json.loads(text)
    assert doc == {"mcpServers": {"grader": {"url": "http://127.0.0.1:1/tok/mcp"}}}
    assert "command" not in text


# --------------------------------------------------------------------------
# the journal, rendered
#
# Shaped on real kimi-code 0.42.0 records (agents/main/wire.jsonl) and the
# forwarder's per-call record for the same request: `step.end.messageId` is
# the chat completion id, which is the join key for the reasoning.
# --------------------------------------------------------------------------
def _loop(event: dict, t: int) -> dict:
    return {"type": "context.append_loop_event", "agentId": "main",
            "event": event, "time": t}


JOURNAL = [
    {"type": "metadata", "protocol_version": "1.5", "created_at": 1788936386043},
    {"type": "profile.bind", "agentId": "main", "modelAlias": "__kimi_env_model__",
     "profileName": "agent", "thinkingEffort": "high", "systemPrompt": "You are Kimi",
     "activeToolNames": ["Read", "Bash", "FetchURL", "WebSearch"],
     "disallowedTools": [], "subagents": ["coder"], "time": 1788936386062},
    {"type": "permission.set_mode", "agentId": "main", "mode": "auto",
     "time": 1788936386076},
    {"type": "turn.prompt", "agentId": "main",
     "input": [{"type": "text", "text": "solve it"}], "origin": {"kind": "user"},
     "promptId": "msg_1", "time": 1788936386105},
    {"type": "context.append_message", "agentId": "main",
     "message": {"role": "user", "content": [{"type": "text", "text": "solve it"}],
                 "origin": {"kind": "user"}, "id": "msg_1"}, "time": 1788936386107},
    {"type": "context.append_message", "agentId": "main",
     "message": {"role": "user",
                 "content": [{"type": "text", "text": "<system-reminder>auto mode</system-reminder>"}],
                 "origin": {"kind": "injection", "variant": "permission_mode"}},
     "time": 1788936386114},
    _loop({"type": "step.begin", "uuid": "s1", "turnId": "0", "step": 1}, 1788936386119),
    {"type": "llm.tools_snapshot", "agentId": "main", "hash": "h",
     "tools": [{"name": "Read"}, {"name": "Bash"}], "time": 1788936386130},
    _loop({"type": "content.part", "turnId": "0", "step": 1, "stepUuid": "s1",
           "part": {"type": "text", "text": "Let me look."}}, 1788936401202),
    _loop({"type": "tool.call", "turnId": "0", "step": 1, "stepUuid": "s1",
           "toolCallId": "Bash_0", "name": "Bash", "args": {"command": "ls -la"},
           "display": {"kind": "command", "command": "ls -la"}}, 1788936401238),
    _loop({"type": "tool.result", "parentUuid": "x", "toolCallId": "Bash_0",
           "result": {"output": "total 0"}}, 1788936401243),
    _loop({"type": "step.end", "uuid": "s1", "turnId": "0", "step": 1,
           "finishReason": "tool_use",
           "usage": {"inputOther": 19491, "output": 243, "inputCacheRead": 256,
                     "inputCacheCreation": 0},
           "messageId": "gen-1", "providerFinishReason": "tool_calls"}, 1788936401245),
    _loop({"type": "step.begin", "uuid": "s2", "turnId": "0", "step": 2}, 1788936401248),
    _loop({"type": "content.part", "turnId": "0", "step": 2, "stepUuid": "s2",
           "part": {"type": "text", "text": "<answer>7</answer>"}}, 1788936414045),
    _loop({"type": "step.end", "uuid": "s2", "turnId": "0", "step": 2,
           "finishReason": "end_turn",
           "usage": {"inputOther": 1612, "output": 53, "inputCacheRead": 18432,
                     "inputCacheCreation": 0},
           "messageId": "gen-2"}, 1788936414046),
    {"type": "turn.ended", "agentId": "main", "turnId": 0, "reason": "completed",
     "durationMs": 27942, "time": 1788936414047},
    # a subagent's records ride along, verbatim, and are not rendered as steps
    _loop({"type": "step.begin", "uuid": "sub", "turnId": "0", "step": 1}, 1788936414100)
    | {"agentId": "agent_coder_1"},
    # the forwarder's records: one per model call, joined by id
    {"type": "forwarder.call", "index": 0, "method": "POST",
     "path": "/v1/chat/completions", "status": 200, "id": "gen-1", "model": MODEL,
     "provider": None, "finish_reason": "tool_calls",
     "reasoning": "thinking hard", "content": "Let me look.",
     "tool_calls": [{"id": "Bash_0", "name": "Bash",
                     "arguments": "{\"command\":\"ls -la\"}"}],
     "usage": {"prompt_tokens": 19747, "completion_tokens": 243, "total_tokens": 19990,
               "completion_tokens_details": {"reasoning_tokens": 68},
               "prompt_tokens_details": {"cached_tokens": 256, "cache_write_tokens": 0}},
     "error": None, "seconds": 15.1},
    {"type": "forwarder.call", "index": 1, "method": "POST",
     "path": "/v1/chat/completions", "status": 200, "id": "gen-2", "model": MODEL,
     "provider": None, "finish_reason": "stop", "reasoning": "",
     "content": "<answer>7</answer>", "tool_calls": [],
     "usage": {"prompt_tokens": 20044, "completion_tokens": 53, "total_tokens": 20097,
               "completion_tokens_details": {"reasoning_tokens": 1},
               "prompt_tokens_details": {"cached_tokens": 18432, "cache_write_tokens": 0}},
     "error": None, "seconds": 6.5},
]

USAGE = usage_from_calls([m for m in JOURNAL if m.get("type") == "forwarder.call"])


def _transcript() -> dict:
    return {"agent": "kimi-code", "model": MODEL, "messages": JOURNAL,
            "usage": USAGE, "session_id": "session_1"}


def _render(transcript: dict) -> str:
    from core import judge as judging

    return judging.render_transcript(transcript)


def test_journal_converts_to_valid_atif_with_reasoning_joined() -> None:
    from core.trajectory import SCHEMA_VERSION, Trajectory

    traj = KimiCodeAgent(model=MODEL).to_trajectory(_transcript())
    assert isinstance(traj, Trajectory)
    assert traj.schema_version == SCHEMA_VERSION
    assert traj.session_id == "session_1"
    assert traj.agent.name == "kimi-code"
    # The tools the model was actually offered: the snapshot, not the profile.
    assert traj.agent.extra["tools"] == ["Read", "Bash"]
    agent_steps = [s for s in traj.steps if s.source == "agent"]
    assert len(agent_steps) == 2
    # Reasoning came from the forwarder record with the matching id, and is
    # its own field -- never folded into the message.
    assert agent_steps[0].reasoning_content == "thinking hard"
    assert "thinking hard" not in agent_steps[0].message
    assert agent_steps[1].reasoning_content is None
    # The tool call is paired with its result inside the step that issued it.
    call = agent_steps[0]
    assert call.tool_calls[0].function_name == "Bash"
    assert call.tool_calls[0].arguments == {"command": "ls -la"}
    assert call.observation.results[0].source_call_id == "Bash_0"
    assert call.observation.results[0].content == "total 0"
    # Per-step metrics are INCLUSIVE prompt counts.
    assert call.metrics.prompt_tokens == 19747
    assert call.metrics.cached_tokens == 256
    assert call.extra["message_id"] == "gen-1"
    # The user prompt and the injection are their own steps.
    assert traj.steps[0].source == "user" and traj.steps[0].message == "solve it"
    assert traj.steps[1].source == "system"
    assert traj.steps[1].extra == {"kind": "injection", "variant": "permission_mode"}
    # Totals come from the forwarder's tally.
    assert traj.final_metrics.total_prompt_tokens == 19747 + 20044
    assert traj.final_metrics.total_completion_tokens == 243 + 53
    assert traj.final_metrics.total_cached_tokens == 256 + 18432
    assert traj.final_metrics.extra["reasoning_tokens"] == 69
    assert "subagent" in (traj.notes or "")
    assert Trajectory.model_validate(traj.to_json_dict()) == traj


def test_render_trajectory_produces_the_judges_log_shape() -> None:
    text = _render(_transcript())
    assert text.startswith("[init] agent=kimi-code")
    assert "THINKING: thinking hard" in text
    assert "[0] TOOL_CALL Bash: command=ls -la" in text
    assert "    -> RESULT[Bash]: total 0" in text
    assert "ASSISTANT: <answer>7</answer>" in text
    assert "FINAL MESSAGE: <answer>7</answer>" in text


def test_a_killed_episode_keeps_its_open_step() -> None:
    """No `step.end` for the last step: what the agent had asked for is still
    a step, and its result is still paired."""
    cut = [m for m in JOURNAL if not (
        m.get("type") == "context.append_loop_event"
        and m["event"].get("type") == "step.end" and m["event"].get("uuid") == "s1")]
    cut = [m for m in cut if m.get("type") != "forwarder.call"]
    traj = KimiCodeAgent(model=MODEL).to_trajectory({"messages": cut[:11]})
    agent_steps = [s for s in traj.steps if s.source == "agent"]
    assert agent_steps and agent_steps[0].tool_calls[0].function_name == "Bash"
    assert agent_steps[0].observation.results[0].content == "total 0"


def test_empty_record_yields_a_placeholder_step() -> None:
    traj = KimiCodeAgent(model=MODEL).to_trajectory({"messages": []})
    assert len(traj.steps) == 1 and traj.steps[0].extra == {"placeholder": True}


def test_summarise_counts_steps_tool_calls_and_the_clis_own_usage() -> None:
    wire = [m for m in JOURNAL if m.get("type") != "forwarder.call"]
    info = summarise(wire, [{"role": "meta", "type": "session.resume_hint",
                             "session_id": "session_1"}])
    assert info["session_id"] == "session_1"
    assert info["n_turns"] == 2
    assert info["n_tool_calls"] == 1
    assert info["final_text"] == "<answer>7</answer>"
    assert info["active_tools"] == ["Read", "Bash"]
    assert info["thinking_effort"] == "high"
    assert info["terminal_reason"] == "completed"
    assert info["stream_usage"] == {"prompt_tokens": 19747 + 20044,
                                    "completion_tokens": 296,
                                    "cached_tokens": 256 + 18432,
                                    "cache_write_tokens": 0,
                                    "calls_with_usage": 2}


# --------------------------------------------------------------------------
# what an episode cost: the forwarder's OpenAI-shaped usage, priced by litellm
# --------------------------------------------------------------------------
def test_usage_from_calls_sums_the_four_categories() -> None:
    assert USAGE == {"prompt_tokens": 19747 + 20044, "completion_tokens": 296,
                     "cached_tokens": 256 + 18432, "cache_write_tokens": 0,
                     "reasoning_tokens": 69, "total_tokens": 19990 + 20097,
                     "calls_with_usage": 2, "calls": 2}
    assert usage_from_calls([{"usage": None}]) is None


def test_episode_is_priced_by_litellm_on_the_full_model_id() -> None:
    import litellm

    if MODEL not in litellm.model_cost:
        pytest.skip("litellm's table lacks openrouter/moonshotai/kimi-k3")
    rates = litellm.model_cost[MODEL]
    agent = make_agent("kimi-code", model=MODEL)
    assert agent.reported_cost_usd({"usage": USAGE}) is None
    cost, source = agent.episode_cost({"usage": USAGE})
    assert source == "estimated"
    fresh = USAGE["prompt_tokens"] - USAGE["cached_tokens"]
    expected = (fresh * rates["input_cost_per_token"]
                + USAGE["cached_tokens"] * rates["cache_read_input_token_cost"]
                + USAGE["completion_tokens"] * rates["output_cost_per_token"])
    assert cost == pytest.approx(expected, rel=1e-6)


def test_the_clis_own_tally_is_the_fallback_usage_source() -> None:
    agent = make_agent("kimi-code", model=MODEL)
    assert KimiCodeAgent.USAGE_FIELDS == ("usage", "stream_usage")
    assert KimiCodeAgent.PROMPT_TOKENS_INCLUDE_CACHE is True
    cats = agent.usage_categories({"usage": None,
                                   "stream_usage": {"prompt_tokens": 100,
                                                    "completion_tokens": 10,
                                                    "cached_tokens": 40,
                                                    "cache_write_tokens": 0}})
    assert cats == {"prompt_tokens": 100, "completion_tokens": 10,
                    "cached_tokens": 40, "cache_write_tokens": 0}


# --------------------------------------------------------------------------
# why an episode DIED
#
# Every event and every string below is verbatim from a real failed episode in
# outputs/, not invented. The bug they pin: the runner used to report the last
# stream-json line the CLI happened to write as the reason the CLI stopped,
# which is a tool result -- a fetched blog page, a `{"verdict": "below"}` --
# and never the failure. Four episodes were filed as `rate_limit` on the
# strength of a 429 the CLI had already retried successfully, while the
# gateway 52x that actually killed them went unrecorded.
# --------------------------------------------------------------------------

# outputs/openmath_kimi-k3_password/trajectories/goldbach_ce#1
GOLDBACH_EVENTS = [
    {"type": "turn.step.retrying", "agentId": "main", "turnId": 0, "step": 8,
     "failedAttempt": 1, "nextAttempt": 2, "maxAttempts": 10,
     "errorName": "APIProviderRateLimitError",
     "errorMessage": "429 litellm.MidStreamFallbackError: litellm.RateLimitError:"
                     " RateLimitError: OpenrouterException - Message: Provider"
                     " returned error, Metadata: {'error_type':"
                     " 'rate_limit_exceeded'}",
     "statusCode": 429, "time": 1788939887227},
    {"type": "turn.step.interrupted", "agentId": "main", "turnId": 0,
     "step": 11, "reason": "error",
     "message": "[provider.api_error] 524 status code (no body)",
     "time": 1788940046422},
    {"type": "turn.ended", "agentId": "main", "turnId": 0, "reason": "failed",
     "error": {"code": "provider.api_error",
               "message": "524 status code (no body)", "name": "APIStatusError",
               "details": {"statusCode": 524, "requestId": None,
                           "traceId": None},
               "retryable": False},
     "durationMs": 309412, "time": 1788940046423},
]

# outputs/writings_kimi-k3_verdictonly/trajectories/
# seo_project_presentation_templates#2 -- the CLI stopping ITSELF at the cap.
MAX_STEPS_EVENTS = [
    {"type": "turn.ended", "agentId": "main", "turnId": 0, "reason": "failed",
     "error": {"code": "loop.max_steps_exceeded",
               "message": "Turn exceeded maxSteps=150. If max_steps_per_turn is"
                          " too small, raise it in config.toml"
                          " (loop_control.max_steps_per_turn), or run"
                          " \"/update-config\" to update it, then \"/reload\".",
               "name": "LoopError", "details": {"maxSteps": 150},
               "retryable": False},
     "durationMs": 4171980, "time": 1788953029986},
]

# The stdout tail that used to be reported as the failure: the tool result of
# the last Bash call, a page of HTML the agent had fetched.
GOLDBACH_STDOUT_TAIL = json.dumps(
    {"role": "tool", "tool_call_id": "Bash_11",
     "content": "e/\">Jacobian write-up</a> I want to record here\n"})

GOLDBACH_STDERR = (
    "[runner] forwarder: POST /v1/chat/completions -> 524 id=None finish=None"
    " reasoning_chars=0 tool_calls=0 125.035s\n"
    "error: failed to run prompt: provider.api_error: 524 status code (no body)\n"
    "See log: /tmp/kimi-home/logs/kimi-code.log\n")


def test_exit_reason_is_the_clis_verdict_not_the_stdout_tail() -> None:
    from core.agents.kimi_code_runner import exit_reason

    info = summarise(GOLDBACH_EVENTS, [])
    reason = exit_reason(info, "")
    assert "524" in reason
    assert "provider.api_error" in reason
    assert "retryable=false" in reason
    # The thing that used to be reported instead, and the 429 that recovered.
    assert "Jacobian" not in reason
    assert "rate_limit" not in reason.lower()


def test_exit_reason_names_the_step_cap_the_cli_stopped_itself_at() -> None:
    from core.agents.kimi_code_runner import exit_reason

    reason = exit_reason(summarise(MAX_STEPS_EVENTS, []), "")
    assert "loop.max_steps_exceeded" in reason
    assert "maxSteps=150" in reason
    assert "max_steps_per_turn" in reason


def test_exit_reason_falls_back_to_the_clis_own_fatal_stderr_line() -> None:
    """No journal at all: the CLI still said why, on stderr."""
    from core.agents.kimi_code_runner import exit_reason

    reason = exit_reason(summarise([], []), GOLDBACH_STDERR)
    assert reason == ("failed to run prompt: provider.api_error: 524 status"
                      " code (no body)")


def test_exit_reason_prefers_the_interrupt_message_over_stderr() -> None:
    from core.agents.kimi_code_runner import exit_reason

    events = [e for e in GOLDBACH_EVENTS if e["type"] != "turn.ended"]
    assert exit_reason(summarise(events, []), GOLDBACH_STDERR) == \
        "[provider.api_error] 524 status code (no body)"


def test_summarise_reports_only_the_last_turns_error() -> None:
    """A journal holding a failed turn AND a later good one is not a failure."""
    events = GOLDBACH_EVENTS + [
        {"type": "turn.ended", "agentId": "main", "turnId": 1,
         "reason": "completed", "error": None},
    ]
    info = summarise(events, [])
    assert info["terminal_reason"] == "completed"
    assert info["terminal_error"] is None
    assert info["interrupt_message"] is None


def test_a_gateway_52x_is_transient_and_a_step_cap_is_not() -> None:
    """The two failure modes must land on opposite sides of the auto-retry."""
    from core.agents import errors
    from core.agents.kimi_code_runner import exit_reason

    agent = make_agent("kimi-code", model=MODEL)
    gateway = f"kimi exited 1: {exit_reason(summarise(GOLDBACH_EVENTS, []), '')}"
    assert agent.classify_failure(None, gateway, GOLDBACH_STDERR) is \
        errors.ProxyGatewayError
    assert errors.is_transient(errors.ProxyGatewayError) is True

    capped = f"kimi exited 1: {exit_reason(summarise(MAX_STEPS_EVENTS, []), '')}"
    assert agent.classify_failure(None, capped, "") is errors.MaxTurnsError
    assert errors.is_transient(errors.MaxTurnsError) is False


def test_the_retried_429_in_the_teed_stdout_no_longer_decides_the_failure():
    """The regression, end to end, on the real stderr of goldbach_ce#1.

    The CLI's `--output-format stream-json` stdout is teed onto stderr, so a
    `turn.step.retrying` event for a 429 that RECOVERED is in the text the
    classifier reads. It must not outrank the CLI's own fatal line.
    """
    from core.agents import errors

    stderr = ("[kimi] " + json.dumps(
        {"role": "meta", "type": "turn.step.retrying", "failed_attempt": 1,
         "max_attempts": 10, "error_name": "APIProviderRateLimitError",
         "error_message": "429 litellm.RateLimitError: rate_limit_exceeded"})
        + "\n[kimi] " + GOLDBACH_STDOUT_TAIL + "\n" + GOLDBACH_STDERR)
    agent = make_agent("kimi-code", model=MODEL)
    error = f"kimi exited 1: {exit_reason_of(GOLDBACH_EVENTS)}"
    assert agent.classify_failure(None, error, stderr) is errors.ProxyGatewayError


def exit_reason_of(events: list) -> str:
    from core.agents.kimi_code_runner import exit_reason

    return exit_reason(summarise(events, []), "")


def test_a_forwarder_error_body_is_kept_whatever_its_content_type() -> None:
    """A 52x body arrives with no `data:` framing and is not JSON.

    `forwarder.call` index 10 of goldbach_ce#1 recorded `status: 524,
    bytes_out: 920, error: null` -- 920 bytes received and dropped, because the
    SSE assembler ignores a line that is not `data:` and the JSON branch was
    only reached for a non-SSE response. The one call that ended the episode
    was the one call with no account of itself.
    """
    from core.agents.kimi_code_runner import _SSEAssembler

    body = (b"<html><head><title>524: A timeout occurred</title></head>"
            b"<body>Error 524 Ray ID: 9c1f2a0e4b7d0000</body></html>")
    tap = _SSEAssembler()
    tap.feed(body)            # served as text/event-stream: no `data:` lines
    tap.finish_sse()
    assert tap.result()["error"] is None, "the drop this test exists for"
    # What the forwarder now does for any response with status >= 400.
    tap.error = body.decode("utf-8", "replace")
    assert "524: A timeout occurred" in tap.result()["error"]
    assert "9c1f2a0e4b7d0000" in tap.result()["error"]   # nothing truncated


def test_a_partial_episode_is_not_judged_but_a_finding_is() -> None:
    """The judge gate: a transcript the auto-retry is about to replace."""
    from core import judge as judging

    lost = {"episode": "goldbach_ce#1", "ok": False,
            "failure": "proxy_5xx", "transcript_path": "/x/transcript.json"}
    assert judging.is_judgeable(lost) is False

    capped = dict(lost, failure="max_turns")
    assert judging.is_judgeable(capped) is True     # a finding, still judged

    refused = dict(lost, failure="agent_safety_refusal")
    assert judging.is_judgeable(refused) is True

    good = {"episode": "goldbach_ce#1", "ok": True, "failure": None,
            "transcript_path": "/x/transcript.json"}
    assert judging.is_judgeable(good) is True

    assert judging.is_judgeable(dict(good, transcript_path=None)) is False


def test_a_live_520_served_as_an_event_stream_is_captured_verbatim() -> None:
    """The forwarder, end to end over a real socket, on the real 520 shape.

    Cloudflare answers a STREAMING chat completion, so its error response keeps
    the `text/event-stream` Content-Type of the request it is failing and then
    sends an HTML page. The SSE assembler only reads `data:` lines, and the
    JSON branch was reached only for a non-SSE response, so the body fell
    between the two and `forwarder.call` recorded `status: 520, bytes_out: 920,
    error: null` -- the one call that ended the episode, with no account of
    itself. Asserted byte-for-byte: nothing about a failure is ever truncated.
    """
    import threading
    import urllib.error
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from core.agents.forwarder import Upstream
    from core.agents.kimi_code_runner import KimiForwarder

    body = (b"<html><head><title>520: Web server is returning an unknown "
            b"error</title></head><body><h1>Error 520</h1><p>Ray ID: "
            b"9c1f2a0e4b7d0000</p></body></html>")

    class Origin(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a) -> None:
            pass

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(520, "Origin Error")
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    origin = ThreadingHTTPServer(("127.0.0.1", 0), Origin)
    threading.Thread(target=origin.serve_forever, daemon=True).start()
    fwd = KimiForwarder(upstream=Upstream(
        f"http://127.0.0.1:{origin.server_address[1]}/v1", "k")).start()
    try:
        req = urllib.request.Request(
            fwd.base_url + "/chat/completions",
            data=json.dumps({"model": MODEL, "stream": True,
                             "messages": [{"role": "user",
                                           "content": "hi"}]}).encode(),
            headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as seen:
            urllib.request.urlopen(req)
        assert seen.value.code == 520          # the CLI sees the 520 unchanged
    finally:
        fwd.shutdown()
        origin.shutdown()

    call = fwd.calls[0]
    assert call["status"] == 520
    assert call["error"] is not None, "the body of the fatal call was dropped"
    assert call["error"] == body.decode()      # verbatim, not a summary
    assert len(call["error"]) == len(body) == call["bytes_out"]


def test_requested_non_office_outputs_are_exported(tmp_path):
    from core.agents.kimi_code_runner import collect_deliverables
    (tmp_path / "submission.txt").write_text("42")
    (tmp_path / "unrequested.txt").write_text("private")
    (tmp_path / "report.csv").write_text("a,b")
    (tmp_path / "link.csv").symlink_to(tmp_path / "unrequested.txt")
    payload = json.loads(_agent().blob(
        row={"id": "test", "deliverable_files": ["submission.txt"]}, prompt="p",
        files=[], skip_dirs=(), stage_src="", timeout_s=60, env={}))
    exported = collect_deliverables(str(tmp_path), (), payload["task"]["deliverable_files"])
    assert {f["name"] for f in exported} == {"submission.txt", "report.csv"}


@pytest.mark.skipif(os.environ.get("RH_KIMI_CLI_TEST") != "1", reason="requires installed Kimi 0.43.0 and Node >=22.19")
@pytest.mark.parametrize("cap", [False, True])
def test_published_cli_tools_reasoning_grader_and_outputs(tmp_path, cap):
    version = subprocess.run(["kimi", "--version"], capture_output=True, text=True, timeout=30)
    assert version.returncode == 0 and version.stdout.strip().splitlines()[-1] == "0.43.0"
    proc = subprocess.run([sys.executable, str(ROOT / "tests/kimi_code_probe.py"), str(tmp_path),
                           *(["--cap"] if cap else [])], capture_output=True, text=True, timeout=90)
    (tmp_path / "stderr.log").write_text(proc.stderr)
    assert proc.returncode == 0, proc.stderr[-5000:]


def test_launch_failure_returns_a_recordable_error(monkeypatch):
    from core.agents import kimi_code_runner as runner
    def fail(*args, **kwargs):
        raise OSError("missing executable")
    monkeypatch.setattr(runner.subprocess, "Popen", fail)
    rc, lines, stdout, stderr, error = runner.run_kimi("kimi", {}, 1)
    assert rc is None and not lines and not stdout and not stderr
    assert "missing executable" in error
