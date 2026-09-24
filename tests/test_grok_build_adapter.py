"""The Grok Build adapter: routing, the closed-book composition, cost, ATIF.

What this file pins, and why each line matters:

  * the model id reaches the CLI VERBATIM (`--model openrouter/x-ai/grok-4.6`
    and `model = "openrouter/x-ai/grok-4.6"` in config.toml) -- the whole
    point of the adapter is that the gateway's OpenRouter route gets the full
    id and no prefix is ever stripped;
  * the config block carries `base_url`, `env_key` and
    `supports_reasoning_effort = true`, and NEVER a temperature, top_p or
    token cap -- the CLI forwards exactly what the model entry says;
  * a closed-book policy always produces `--disable-web-search` plus
    `disable_web_search = true` / `web_fetch = false`, and the declared tool
    list carries neither web tool;
  * the MCP entry is the URL shape and never names a command;
  * the CLI's `input_tokens` is EXCLUSIVE of its cache legs, and the estimate
    is taken on the inclusive count (the mistake this repo's cost tests exist
    to prevent, in the other direction from codex);
  * a real session's chat_history converts to ATIF with reasoning on the
    assistant step, every tool call paired with its result, and renders in
    the judge's shape.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

try:                                    # 3.11+
    import tomllib
except ModuleNotFoundError:             # pragma: no cover - host interpreter
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.agents import AgentFactory, make_agent  # noqa: E402
from core.agents.grok_build import GrokBuildAgent  # noqa: E402
from core.render import render_trajectory  # noqa: E402

MODEL = "openrouter/x-ai/grok-4.6"
GATEWAY = "https://gateway.example"
CLOSED = {"web_search": False, "web_fetch": False}
OPEN = {"web_search": True, "web_fetch": True}


def _agent(policy: dict = CLOSED, **kw) -> GrokBuildAgent:
    kw.setdefault("api_base_url", GATEWAY)
    agent = make_agent("grok-build", model=MODEL, **kw)
    agent.apply_tool_policy(policy)
    return agent


def _runner_ns() -> dict:
    """The runner's helpers, exec'd from source the way the container does."""
    g: dict = {"__name__": "rh_runner"}
    exec(compile(GrokBuildAgent.RUNNER.read_text(), "<runner>", "exec"), g)
    return g


def _cfg(agent: GrokBuildAgent, grader_src: str = "") -> dict:
    payload = json.loads(agent.blob(
        row={"id": "row", "answer": "THE-KEY-42"}, prompt="do the thing",
        files=[], skip_dirs=(), grader_src=grader_src, stage_src="",
        timeout_s=60, env={}, install=None))
    return payload["task"]["grok"]


def _argv(agent: GrokBuildAgent) -> list[str]:
    ns = _runner_ns()
    return ns["compose_argv"](_cfg(agent), "solve it")


def _config(agent: GrokBuildAgent, grader_url: str = "") -> str:
    ns = _runner_ns()
    return ns["config_toml"](_cfg(agent), grader_url)


# --------------------------------------------------------------------------
# registry and credentials
# --------------------------------------------------------------------------
def test_it_is_registered_under_its_own_name() -> None:
    assert "grok-build" in AgentFactory.names()
    assert AgentFactory.get_agent_class("grok-build") is GrokBuildAgent
    assert GrokBuildAgent.name() == "grok-build"


def test_the_credential_is_the_gateway_key_unless_the_entry_says_otherwise() -> None:
    assert GrokBuildAgent.API_KEY_ENV == "OPENAI_API_KEY"
    agent = _agent(api_key_env="OPENROUTER_API_KEY")
    assert agent.API_KEY_ENV == "OPENROUTER_API_KEY"
    # The container's key variable is what the CLI is told to read.
    assert 'env_key = "OPENROUTER_API_KEY"' in _config(agent)


# --------------------------------------------------------------------------
# routing: the full id, verbatim, at the gateway
# --------------------------------------------------------------------------
def test_the_model_id_is_sent_verbatim_with_no_prefix_stripped() -> None:
    agent = _agent()
    argv = _argv(agent)
    assert argv[argv.index("--model") + 1] == MODEL
    assert f'model = "{MODEL}"' in _config(agent)
    assert f'[model."{MODEL}"]' in _config(agent)


def test_the_base_url_carries_v1_and_comes_from_the_entry_first(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example")
    assert _agent().api_base() == GATEWAY + "/v1"
    assert _agent(api_base_url=GATEWAY + "/v1/").api_base() == GATEWAY + "/v1"
    # Entry names nothing: the OPENAI_BASE_URL passthrough, as for codex.
    assert _agent(api_base_url="").api_base() == "https://env.example/v1"


def test_config_block_is_the_documented_byok_shape_and_sets_no_sampling() -> None:
    text = _config(_agent())
    assert f'base_url = "{GATEWAY}/v1"' in text
    assert 'env_key = "OPENAI_API_KEY"' in text
    assert 'api_backend = "chat_completions"' in text
    # Without this the CLI logs "model does not support reasoning effort;
    # ignoring" and `--reasoning-effort` never reaches the wire.
    assert "supports_reasoning_effort = true" in text
    for forbidden in ("temperature", "top_p", "max_completion_tokens",
                      "max_tokens", "seed"):
        assert forbidden not in text, forbidden


def test_auxiliary_models_are_pinned_to_the_same_entry() -> None:
    text = _config(_agent())
    for key in ("default", "session_summary", "image_description", "web_search"):
        assert f'{key} = "{MODEL}"' in text


def test_extra_body_is_refused_rather_than_dropped() -> None:
    agent = _agent(extra_body={"provider": {"order": ["xAI"]}})
    assert any("extra_body" in p for p in agent.setup())


def test_no_base_url_and_a_non_xai_key_is_refused(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    agent = _agent(api_base_url="")
    assert any("no base URL" in p for p in agent.setup())
    native = _agent(api_base_url="", api_key_env="XAI_API_KEY")
    assert native.setup() == []
    # Harbor's native path: one prefix stripped, no [model] block.
    xai = make_agent("grok-build", model="xai/grok-4.6", api_key_env="XAI_API_KEY")
    xai.apply_tool_policy(CLOSED)
    assert xai.model_slug == "grok-4.6"
    assert "[model." not in _config(xai)


# --------------------------------------------------------------------------
# the closed-book composition
# --------------------------------------------------------------------------
def test_closed_book_disables_both_web_tools_everywhere() -> None:
    agent = _agent(CLOSED)
    assert "--disable-web-search" in _argv(agent)
    text = _config(agent)
    assert "disable_web_search = true" in text
    assert "web_fetch = false" in text
    assert "backend_tools = false" in text
    assert "web_search" not in agent.tools and "web_fetch" not in agent.tools
    assert agent.setup() == []


@pytest.mark.parametrize("name", sorted(p.parent.name for p
                                        in ROOT.glob("tasks/*/task.yaml")))
def test_every_shipped_task_composes_a_closed_web_command(name: str) -> None:
    from core import trial

    task = trial.load_task(name)
    agent = _agent(task.tools)
    assert "--disable-web-search" in _argv(agent)
    assert agent.setup() == []


def test_open_web_policy_offers_both_tools_and_omits_the_flag() -> None:
    agent = _agent(OPEN)
    assert "--disable-web-search" not in _argv(agent)
    assert "web_search" in agent.tools and "web_fetch" in agent.tools
    text = _config(agent)
    assert "disable_web_search = false" in text
    assert "web_fetch = true" in text


def test_the_two_keys_can_differ() -> None:
    """Separate knobs, unlike codex: search closed, fetch open is expressible."""
    agent = _agent({"web_search": False, "web_fetch": True})
    text = _config(agent)
    assert "disable_web_search = true" in text and "web_fetch = true" in text
    assert "--disable-web-search" not in _argv(agent)
    assert agent.tools[-1] == "web_fetch" and "web_search" not in agent.tools


def test_an_unenforceable_key_is_refused_by_name() -> None:
    agent = make_agent("grok-build", model=MODEL)
    with pytest.raises(ValueError) as excinfo:
        agent.apply_tool_policy({"web_search": False, "some_future_tool": False})
    assert "grok-build" in str(excinfo.value)
    assert "some_future_tool" in str(excinfo.value)


def test_side_channels_are_pinned_off() -> None:
    text = _config(_agent())
    for line in ("telemetry = false", "trace_upload = false",
                 "disable_codebase_upload = true",
                 "disable_workspace_teleport = true", "remote_fetch = false",
                 "auto_update = false", "[subagents]\nenabled = false",
                 "[memory]\nenabled = false", "ask_user_question = false",
                 "image_gen = false", "video_gen = false"):
        assert line in text, line
    assert "--disallowed-tools" in _argv(_agent())


# --------------------------------------------------------------------------
# the invocation, against Harbor's shape
# --------------------------------------------------------------------------
def test_argv_matches_harbors_headless_shape() -> None:
    agent = _agent(max_turns=42)
    argv = _argv(agent)
    joined = " ".join(argv)
    for fragment in ("--no-auto-update", "-p solve it", "--always-approve",
                     "--output-format streaming-json", "--session-id ",
                     f"--model {MODEL}", "--max-turns 42",
                     "--reasoning-effort high", "--disable-web-search",
                     "--disallowed-tools send_feedback,image_edit",
                     "--cwd /workspace"):
        assert fragment in joined, fragment
    # argv, not a shell string: the instruction is one element, unquoted.
    assert argv[argv.index("-p") + 1] == "solve it"


def test_each_blob_draws_a_fresh_session_id() -> None:
    agent = _agent()
    assert _cfg(agent)["session_id"] != _cfg(agent)["session_id"]


def test_reasoning_effort_default_is_recorded_and_the_enum_is_checked() -> None:
    assert _agent().resolved_generation_config() == {"reasoning_effort": "high"}
    low = _agent(generation_config={"reasoning_effort": "low"})
    assert "--reasoning-effort low" in " ".join(_argv(low))
    bad = _agent(generation_config={"reasoning_effort": "turbo"})
    assert any("reasoning_effort" in p for p in bad.setup())
    unknown = _agent(generation_config={"reasoning_summary": "detailed"})
    assert any("reasoning_summary" in p for p in unknown.setup())


def test_permission_mode_maps_to_the_cli_flag() -> None:
    assert _agent().permission_flags() == ["--always-approve"]
    assert _agent(permission_mode="dontAsk").permission_flags() == [
        "--permission-mode", "dontAsk"]


# --------------------------------------------------------------------------
# the grader: url shape, never a command
# --------------------------------------------------------------------------
def test_config_toml_never_names_a_command() -> None:
    text = _config(_agent(), "http://127.0.0.1:12345/deadbeef/mcp")
    assert '[mcp_servers.grader]\nurl = "http://127.0.0.1:12345/deadbeef/mcp"' in text
    assert "command" not in text and "args" not in text
    assert "tool_timeout_sec = 300" in text


@pytest.mark.skipif(tomllib is None, reason="no TOML parser on this interpreter")
def test_config_toml_parses_and_the_scalar_is_top_level() -> None:
    doc = tomllib.loads(_config(_agent(), "http://127.0.0.1:1/t/mcp"))
    assert doc["disable_web_search"] is True
    assert doc["model"][MODEL]["base_url"] == GATEWAY + "/v1"
    assert doc["model"][MODEL]["env_key"] == "OPENAI_API_KEY"
    assert doc["mcp_servers"]["grader"] == {
        "url": "http://127.0.0.1:1/t/mcp", "tool_timeout_sec": 300}
    assert doc["subagents"]["enabled"] is False


def test_the_row_travels_only_with_a_grader() -> None:
    agent = _agent()
    without = json.loads(agent.blob(
        row={"id": "row", "answer": "THE-KEY-42"}, prompt="p", files=[],
        skip_dirs=(), grader_src="", stage_src="", timeout_s=60, env={}))
    assert "row" not in without["task"]
    assert "THE-KEY-42" not in json.dumps(without["task"])
    with_grader = json.loads(agent.blob(
        row={"id": "row", "answer": "THE-KEY-42"}, prompt="p", files=[],
        skip_dirs=(), grader_src="def make_tool(r, s, w): ...", stage_src="",
        timeout_s=60, env={}))
    assert with_grader["task"]["row"]["answer"] == "THE-KEY-42"


# --------------------------------------------------------------------------
# cost: the CLI's input count is EXCLUSIVE of its cache legs
# --------------------------------------------------------------------------
# A REAL grok 1.0.24 session through the gateway (the host smoke run): the
# `end` event's usage. total_tokens was 38350 = 25152 + 12928 + 0 + 270, which
# is the proof that `input_tokens` is the uncached remainder.
END_USAGE = {"input_tokens": 25152, "output_tokens": 270,
             "cache_read_input_tokens": 12928,
             "cache_creation_input_tokens": 0, "reasoning_tokens": 188}
RATES = {"input_cost_per_token": 2e-06, "output_cost_per_token": 6e-06,
         "cache_read_input_token_cost": 5e-07}


def test_usage_keys_are_the_clis_own_and_exclusive() -> None:
    assert GrokBuildAgent.USAGE_KEYS == (
        "input_tokens", "output_tokens", "cache_read_input_tokens",
        "cache_creation_input_tokens")
    assert GrokBuildAgent.PROMPT_TOKENS_INCLUDE_CACHE is False
    cats = _agent().usage_categories({"usage": END_USAGE})
    assert cats == {"prompt_tokens": 25152 + 12928, "completion_tokens": 270,
                    "cached_tokens": 12928, "cache_write_tokens": 0}


def test_the_estimate_prices_the_inclusive_count_at_the_openrouter_rates() -> None:
    import litellm

    pricing = litellm.model_cost.get(MODEL)
    if not pricing:
        pytest.skip(f"{MODEL} absent from this litellm's table")
    for key, rate in RATES.items():
        if pricing.get(key) != rate:
            pytest.skip(f"{MODEL} {key} changed upstream: {pricing.get(key)}")
    cost, source = _agent().episode_cost({"usage": END_USAGE})
    assert source == "estimated"
    expected = (25152 * 2e-06) + (12928 * 5e-07) + (270 * 6e-06)
    assert cost == pytest.approx(expected, rel=1e-9)


def test_reasoning_tokens_are_inside_output_tokens_not_on_top() -> None:
    cats = _agent().usage_categories({"usage": END_USAGE})
    assert cats["completion_tokens"] == 270


def test_stream_usage_is_the_fallback_for_a_killed_episode() -> None:
    cats = _agent().usage_categories({"usage": None, "stream_usage": END_USAGE})
    assert cats["prompt_tokens"] == 25152 + 12928


def test_a_reported_cost_comes_only_from_the_cli_and_passes_through() -> None:
    agent = _agent()
    assert agent.reported_cost_usd({"cost_usd": 1.23}) is None
    assert agent.reported_cost_usd({"reported_cost_usd": True}) is None
    assert agent.reported_cost_usd({"reported_cost_usd": 0.0127}) == 0.0127
    cost, source = agent.episode_cost({"reported_cost_usd": 0.0127,
                                       "usage": END_USAGE})
    assert (cost, source) == (0.0127, "reported")


def test_the_runner_takes_cost_only_when_the_server_stamped_it_complete() -> None:
    ns = _runner_ns()
    events = [{"type": "usage", "usage": END_USAGE},
              {"type": "end", "stopReason": "end_turn", "sessionId": "s",
               "num_turns": 3, "usage": END_USAGE,
               "total_cost_usd": 0.05, "total_cost_usd_ticks": 500000000}]
    out = ns["summarise_stream"](events)
    assert out["reported_cost_usd"] == 0.05
    assert out["usage"] == END_USAGE and out["stream_usage"] == END_USAGE
    assert out["stop_reason"] == "end_turn" and out["num_turns"] == 3
    partial = dict(events[1], cost_is_partial=True)
    assert ns["summarise_stream"]([partial])["reported_cost_usd"] is None
    assert ns["summarise_stream"]([])["usage"] is None


# --------------------------------------------------------------------------
# chat_history -> ATIF, on a real session
# --------------------------------------------------------------------------
# The host smoke session (grok 1.0.24, openrouter/x-ai/grok-4.6 through the
# gateway), with the CLI's long system prompt shortened. Shapes are verbatim.
CHAT_HISTORY = [
    {"type": "system", "content": "You are Grok released by xAI."},
    {"type": "user", "content": [{"type": "text",
                                  "text": "<user_info>\nOS Version: linux\n</user_info>"}]},
    {"type": "user", "content": [{"type": "text",
                                  "text": "<user_query>\nList the files.\n</user_query>"}],
     "prompt_index": 0},
    {"type": "reasoning", "id": "",
     "summary": [{"type": "summary_text", "text": "The user wants me to list files."}]},
    {"type": "assistant", "content": "I'll list the current directory.",
     "tool_calls": [{"id": "call-1", "name": "run_terminal_command",
                     "arguments": "{\"command\":\"ls\",\"description\":\"List files\"}"}],
     "model_id": MODEL},
    {"type": "tool_result", "tool_call_id": "call-1", "content": "exit: 0\na.txt\nb.py\n"},
    {"type": "reasoning", "id": "",
     "summary": [{"type": "summary_text", "text": "Now read a.txt."}]},
    {"type": "assistant", "content": "",
     "tool_calls": [{"id": "call-2", "name": "read_file",
                     "arguments": "{\"target_file\":\"a.txt\"}"}],
     "model_id": MODEL},
    {"type": "tool_result", "tool_call_id": "call-2", "content": "1→hello\n"},
    {"type": "reasoning", "id": "",
     "summary": [{"type": "summary_text", "text": "Answer."}]},
    {"type": "assistant", "content": "There are **2** files.\n\nDONE", "model_id": MODEL},
]


def _raw() -> dict:
    return {"messages": CHAT_HISTORY, "usage": END_USAGE, "session_id": "sess",
            "terminal_reason": "end_turn", "final_text": "There are **2** files.\n\nDONE"}


def test_a_real_session_converts_with_calls_paired_and_reasoning_attached() -> None:
    traj = _agent().to_trajectory(_raw())
    assert traj.session_id == "sess"
    assert traj.agent.model_name == MODEL
    assert [s.source for s in traj.steps] == [
        "system", "user", "user", "agent", "agent", "agent"]
    first, second, last = traj.steps[3], traj.steps[4], traj.steps[5]
    assert first.reasoning_content == "The user wants me to list files."
    assert first.tool_calls[0].function_name == "run_terminal_command"
    assert first.tool_calls[0].arguments == {"command": "ls", "description": "List files"}
    assert first.observation.results[0].source_call_id == "call-1"
    assert first.observation.results[0].content == "exit: 0\na.txt\nb.py\n"
    assert second.tool_calls[0].arguments == {"target_file": "a.txt"}
    assert second.observation.results[0].content == "1→hello\n"
    assert last.tool_calls is None and last.message.endswith("DONE")
    assert traj.notes is None
    assert traj.final_metrics.total_prompt_tokens == 25152 + 12928
    assert traj.final_metrics.total_cached_tokens == 12928
    assert traj.final_metrics.total_completion_tokens == 270
    assert traj.extra["final_text"].endswith("DONE")


def test_render_puts_the_session_in_the_judges_shape() -> None:
    text = render_trajectory(_agent().to_trajectory(_raw()))
    assert "[init] agent=grok-build" in text
    assert "THINKING: The user wants me to list files." in text
    assert "[0] TOOL_CALL run_terminal_command: command=ls" in text
    assert "-> RESULT[run_terminal_command]: exit: 0" in text
    assert "[1] TOOL_CALL read_file: target_file=a.txt" in text
    assert text.rstrip().endswith("FINAL MESSAGE: There are **2** files.\n\nDONE")


def test_trailing_reasoning_is_kept_as_its_own_step() -> None:
    raw = {"messages": CHAT_HISTORY[:4]}
    traj = _agent().to_trajectory(raw)
    assert traj.steps[-1].source == "agent"
    assert traj.steps[-1].reasoning_content == "The user wants me to list files."


def test_an_orphan_result_and_an_unknown_type_are_counted_not_lost_silently() -> None:
    raw = {"messages": [{"type": "tool_result", "tool_call_id": "nope", "content": "x"},
                        {"type": "mystery"}]}
    traj = _agent().to_trajectory(raw)
    assert traj.steps[0].extra == {"placeholder": True}
    assert "unknown call 'nope'" in traj.notes and "'mystery'" in traj.notes


def test_an_episode_that_captured_nothing_is_still_valid_atif() -> None:
    traj = _agent().to_trajectory({"messages": []})
    assert len(traj.steps) == 1 and traj.steps[0].extra == {"placeholder": True}
    assert _agent().readable({"type": "tool_result", "tool_call_id": "x",
                              "content": "y"}) == ""


def test_the_live_tail_renders_one_assistant_line() -> None:
    block = _agent().readable(CHAT_HISTORY[4])
    assert "ASSISTANT: I'll list the current directory." in block
    assert "TOOL_CALL run_terminal_command" in block


def test_the_record_envelope_carries_the_grok_specific_tail() -> None:
    rec = _agent().record({"messages": CHAT_HISTORY, "usage": END_USAGE,
                           "grok_command": "grok -p x", "session_dir": "/tmp/s",
                           "reported_cost_usd": None, "usage_calls": [],
                           "usage_is_incomplete": False,
                           "install": {"version": "1.0.24"}})
    assert rec["grok_command"] == "grok -p x" and rec["session_dir"] == "/tmp/s"
    assert rec["cost_source"] == "estimated" or rec["cost_usd"] is None
    assert _agent().version() is None  # no pin, no container yet
