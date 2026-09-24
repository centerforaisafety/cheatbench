"""The Codex adapter's command line, and the one thing in it that is a trap.

`-c web_search=` MUST be on every Codex command line this harness composes.

The CLI's default is not `disabled`, it is `cached`, and `cached` still carries
the `open`/`click` URL-fetch operations of `web__run`. Those execute on OpenAI's
servers, outside the container's network namespace, so a closed-book episode
that omitted the flag would be able to fetch the REAL terrytao.wordpress.com
instead of the mirror injected into the episode's namespace -- the task's
planted post would simply not be there, and nothing in the run record would say
so. Harbor omits the flag when nothing asked for it, so this is exactly the line
a faithful port would have got wrong.

The rest of the file pins the shape of the invocation against Harbor's, and pins
the refusal behaviour of the tool policy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import trial  # noqa: E402
from core.agents import AgentFactory, make_agent  # noqa: E402
from core.agents.codex import CodexAgent  # noqa: E402
from core.agents.codex_runner import compose_command  # noqa: E402

CLOSED = {"web_search": False, "web_fetch": False}
OPEN = {"web_search": True, "web_fetch": True}


def _agent(policy: dict, **kw) -> CodexAgent:
    agent = make_agent("codex", model="openai/gpt-5.6-sol", **kw)
    agent.apply_tool_policy(policy)
    return agent


def _command(agent: CodexAgent, instruction: str = "do the thing") -> str:
    payload = json.loads(agent.blob(
        row={"id": "row"}, prompt=instruction, files=[], skip_dirs=(),
        grader_src="", stage_src="", timeout_s=60, env={}, install=None))
    return compose_command(payload["task"]["codex"], instruction,
                           "/tmp/codex-home/codex.txt")


# --------------------------------------------------------------------------
# THE GUARD
# --------------------------------------------------------------------------
def test_closed_book_always_passes_web_search_disabled() -> None:
    """Never rely on the CLI default: `cached` still fetches URLs server-side."""
    assert "-c web_search=disabled" in _command(_agent(CLOSED))


# These tasks use the common sandbox runner; Harbor and direct chat do not.
@pytest.mark.parametrize("name", sorted(p.parent.name for p
                                        in ROOT.glob("tasks/*/task.yaml")
                                        if p.parent.name not in {"software_engineering", "sycophancy"}))
def test_every_shipped_task_composes_a_closed_web_command(name: str) -> None:
    """The tasks we actually run, through the real task.yaml, end to end."""
    task = trial.load_task(name)
    agent = _agent(task.tools)
    expected = {**CLOSED, **({"mcp_only": True} if name in {"chess", "go"} else {})}
    assert task.tools == expected, (
        f"tasks/{name} changed its tools: block; this test asserts the "
        f"closed-book composition and must be updated deliberately")
    assert "-c web_search=disabled" in _command(agent)
    assert CodexAgent.WEB_TOOL not in agent.tools
    assert agent.setup() == []


def test_open_web_policy_says_live_not_cached() -> None:
    assert "-c web_search=live" in _command(_agent(OPEN))
    assert CodexAgent.WEB_TOOL in _agent(OPEN).tools


# --------------------------------------------------------------------------
# the tool policy
# --------------------------------------------------------------------------
def test_both_web_keys_are_enforceable() -> None:
    """One tool (`web__run`) serves both, verified against codex-cli 0.152.0."""
    assert CodexAgent.ENFORCEABLE == {"web_search", "web_fetch", "mcp_only"}


def test_the_two_web_keys_may_not_disagree() -> None:
    """One knob cannot express two different answers; refuse, do not pick."""
    agent = make_agent("codex", model="openai/gpt-5.6-sol")
    with pytest.raises(ValueError) as excinfo:
        agent.apply_tool_policy({"web_search": True, "web_fetch": False})
    assert "web_fetch" in str(excinfo.value)


def test_an_unenforceable_key_is_refused_by_name() -> None:
    agent = make_agent("codex", model="openai/gpt-5.6-sol")
    with pytest.raises(ValueError) as excinfo:
        agent.apply_tool_policy({"web_search": False, "some_future_tool": False})
    message = str(excinfo.value)
    assert "codex" in message and "some_future_tool" in message


# --------------------------------------------------------------------------
# the invocation, against Harbor's shape
# --------------------------------------------------------------------------
def test_command_matches_harbors_shape() -> None:
    command = _command(_agent(CLOSED), "solve it")
    for fragment in (
            "codex exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--model gpt-5.6-sol",      # provider prefix stripped, as Harbor does
            "--json",
            "--enable unified_exec",
            "-c model_reasoning_effort=high",
            "--strict-config",
            "--disable image_generation",
            "-c analytics.enabled=false",
            "-c check_for_update_on_startup=false",
            "</dev/null",               # the prompt is argv, never stdin
            "| tee ",
    ):
        assert fragment in command, fragment
    # The instruction is ARGV, after `--`, shell-quoted.
    assert command.split(" -- ")[1].startswith("'solve it'")


def test_ignore_user_config_yields_to_a_base_url(monkeypatch) -> None:
    """`--ignore-user-config` skips $CODEX_HOME/config.toml, which is the ONLY
    place codex >= 0.118 reads openai_base_url from. It cannot be passed when
    there is a base URL to deliver."""
    # The no-base-URL half of this test asserts on the AMBIENT environment, and
    # `run.py`/`core.llm_agents` load_dotenv() the repo's own .env at import.
    # A .env that points the run at a proxy therefore makes the first assertion
    # fail for a reason that has nothing to do with the adapter. Clear it.
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    agent = _agent(CLOSED)
    assert "--ignore-user-config" in _command(agent)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    assert "--ignore-user-config" not in _command(agent)


def test_credentials_are_per_adapter() -> None:
    assert CodexAgent.API_KEY_ENV == "OPENAI_API_KEY"
    assert (AgentFactory.get_agent_class("claude-sdk").API_KEY_ENV
            == "ANTHROPIC_API_KEY")


def _blob(agent: CodexAgent, grader_src: str) -> dict:
    return json.loads(agent.blob(
        row={"id": "row", "answer": "THE-KEY-42"}, prompt="p", files=[],
        skip_dirs=(), grader_src=grader_src, stage_src="",
        timeout_s=60, env={}, install=None))


def test_blob_carries_the_row_and_the_grader_when_the_task_ships_one() -> None:
    """Both ride on STDIN, which is the only channel that can hold them.

    This test replaces one that asserted the opposite. The old adapter forwarded
    neither, on the reasoning that codex takes an MCP server only as a command
    the agent could re-run -- true of the stdio transport, false of the url one,
    and the cost of the mistake was that gdpval's grader existed on the Claude
    adapter and not on this one. See `CodexAgent.blob`.
    """
    payload = _blob(_agent(CLOSED), "SECRET GRADER SOURCE")
    assert payload["task"]["row"]["answer"] == "THE-KEY-42"
    assert payload["modules"]["grader"] == "SECRET GRADER SOURCE"
    # ...and nowhere else. Not in the command line the container is started
    # with, and not in the environment: the agent can read both out of /proc.
    command = compose_command(payload["task"]["codex"], "p", "/tmp/log")
    assert "THE-KEY-42" not in command
    assert "SECRET GRADER SOURCE" not in command
    assert "THE-KEY-42" not in json.dumps(payload["task"]["env"])


def test_a_task_with_no_grader_carries_neither() -> None:
    """openmath ships no grader, and its episodes must not acquire a row.

    "" is the one convention for "no grader" and it has to survive the whole
    way: no row in the payload, an empty grader module, and `grader_state` left
    an explicit None by the runner rather than an empty dict.
    """
    payload = _blob(_agent(CLOSED), "")
    assert "row" not in payload["task"]
    assert "THE-KEY-42" not in json.dumps(payload)
    assert payload["modules"]["grader"] == ""


# --------------------------------------------------------------------------
# the rollout, rendered
# --------------------------------------------------------------------------
ROLLOUT = [
    {"timestamp": "t0", "type": "session_meta",
     "payload": {"id": "sess-1", "cli_version": "0.152.0", "cwd": "/workspace"}},
    {"timestamp": "t1", "type": "turn_context",
     "payload": {"model": "gpt-5.6-sol", "effort": "high"}},
    {"timestamp": "t2", "type": "response_item",
     "payload": {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "solve it"}]}},
    {"timestamp": "t3", "type": "response_item",
     "payload": {"type": "reasoning",
                 "summary": [{"type": "summary_text", "text": "thinking hard"}]}},
    {"timestamp": "t4", "type": "response_item",
     "payload": {"type": "function_call", "call_id": "c1", "name": "shell",
                 "arguments": "{\"command\": \"ls -la\"}"}},
    {"timestamp": "t5", "type": "response_item",
     "payload": {"type": "function_call_output", "call_id": "c1",
                 "output": "{\"output\": \"total 0\"}"}},
    {"timestamp": "t6", "type": "event_msg",
     "payload": {"type": "token_count",
                 "info": {"total_token_usage": {"input_tokens": 10,
                                                "output_tokens": 3}}}},
    {"timestamp": "t7", "type": "response_item",
     "payload": {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "<answer>7</answer>"}]}},
]


def _render(transcript: dict) -> str:
    """A transcript through the ONE path: adapter -> ATIF -> the one renderer."""
    from core import judge as judging

    return judging.render_transcript(transcript)


def test_rollout_converts_to_valid_atif() -> None:
    from core.trajectory import SCHEMA_VERSION, Trajectory

    traj = CodexAgent(model="gpt-5.6-sol").to_trajectory(
        {"agent": "codex", "messages": ROLLOUT})
    assert isinstance(traj, Trajectory)
    assert traj.schema_version == SCHEMA_VERSION
    assert traj.session_id == "sess-1"
    assert traj.agent.name == "codex"
    assert traj.agent.version == "0.152.0"
    # Reasoning is its OWN field and is never folded into the message.
    assert any(s.reasoning_content == "thinking hard" for s in traj.steps)
    assert not any("thinking hard" in s.message
                   for s in traj.steps if isinstance(s.message, str))
    # Every tool call is paired with its result inside the step that issued it.
    call = next(s for s in traj.steps if s.tool_calls)
    assert call.tool_calls[0].function_name == "shell"
    assert call.observation.results[0].source_call_id == call.tool_calls[0].tool_call_id
    # A trajectory round-trips through JSON unchanged.
    assert Trajectory.model_validate(traj.to_json_dict()) == traj


def test_render_trajectory_produces_the_judges_log_shape() -> None:
    text = _render({"agent": "codex", "messages": ROLLOUT})
    assert text.startswith("[init] agent=codex version=0.152.0")
    assert "THINKING: thinking hard" in text
    assert "[0] TOOL_CALL shell: command=ls -la" in text
    assert "    -> RESULT[shell]: total 0" in text
    assert "ASSISTANT: <answer>7</answer>" in text
    assert "FINAL MESSAGE: <answer>7</answer>" in text


def test_one_renderer_serves_every_adapter() -> None:
    """There is no per-adapter renderer left to disagree with the judge."""
    from core import render
    from core.agents import Agent, ClaudeSDKAgent, CodexAgent as _Codex

    for cls in (ClaudeSDKAgent, _Codex):
        assert not hasattr(cls, "render_trajectory")
    assert not hasattr(Agent, "render_trajectory")
    # The renderer takes a Trajectory and nothing else, so it has no argument
    # it could branch on.
    traj = _Codex(model="m").to_trajectory({"messages": ROLLOUT})
    assert render.render_trajectory(traj) == _render(
        {"agent": "codex", "messages": ROLLOUT})


def test_legacy_sdk_transcript_still_converts() -> None:
    """A transcript from before adapters were named can only be the SDK's."""
    legacy = {"messages": [{"_type": "ResultMessage", "result": "done"}]}
    text = _render(legacy)
    assert text.endswith("FINAL MESSAGE: done")
    assert text.startswith("[init] agent=claude-sdk")


def test_render_never_truncates_text_but_elides_base64() -> None:
    # Prose, not an unbroken alphabet run: the elision rule keys on the latter,
    # so a wall of `x` would (correctly) be treated as a payload.
    long_text = "the conjecture remains open. " * 2000
    blob = "QWxhZGRpbjpvcGVuIHNlc2FtZQ" * 40
    rollout = [{"type": "response_item",
                "payload": {"type": "message", "role": "assistant",
                            "content": [{"text": long_text + "\n" + blob}]}}]
    text = _render({"agent": "codex", "messages": rollout})
    assert long_text in text
    assert blob not in text
    assert "base64 blob" in text


def test_summarise_counts_tool_calls_and_totals_tokens() -> None:
    from core.agents.codex_runner import summarise

    info = summarise(ROLLOUT)
    assert info["session_id"] == "sess-1"
    assert info["n_tool_calls"] == 1
    assert info["n_turns"] == 1
    assert info["usage"] == {"input_tokens": 10, "output_tokens": 3}
    assert info["final_text"] == "<answer>7</answer>"


def test_find_rollout_picks_the_deepest_newest_file(tmp_path: Path) -> None:
    from core.agents.codex_runner import find_rollout

    home = tmp_path / "codex-home"
    day = home / "sessions" / "2026" / "09" / "01"
    day.mkdir(parents=True)
    (home / "sessions" / "2026" / "stray.jsonl").write_text("{}\n")
    (day / "rollout-2026-09-01T01-00-00-aaa.jsonl").write_text("{}\n")
    newest = day / "rollout-2026-09-01T02-00-00-bbb.jsonl"
    newest.write_text("{}\n")
    assert find_rollout(str(home)) == str(newest)


# --------------------------------------------------------------------------
# reasoning must ACCUMULATE, not overwrite
#
# `reasoning` is a buffer the next message/tool_call drains. Two reasoning items
# with nothing between them therefore belong to the same upcoming step, and the
# conversion used to ASSIGN, silently discarding the earlier one. Measured on a
# real rollout: one such adjacency lost 991 of 4025 summary chars before the
# judge could read them, while a rollout without an adjacency lost nothing --
# which is exactly why this needs a test rather than an eyeball.
# --------------------------------------------------------------------------
def _summary(*texts: str) -> dict:
    return {"type": "response_item",
            "payload": {"type": "reasoning",
                        "summary": [{"type": "summary_text", "text": t}
                                    for t in texts]}}


def _rollout(*items: dict) -> dict:
    return {"agent": "codex", "messages": [
        {"type": "session_meta", "payload": {"id": "s", "cli_version": "0.152.0"}},
        *items,
        {"type": "response_item",
         "payload": {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "done"}]}},
    ]}


def test_consecutive_reasoning_items_are_both_kept() -> None:
    traj = CodexAgent(model="gpt-5.6-sol").to_trajectory(
        _rollout(_summary("first thought"), _summary("second thought")))
    reasoning = [s.reasoning_content for s in traj.steps if s.reasoning_content]
    assert reasoning == ["first thought\nsecond thought"]


def test_a_reasoning_item_with_no_summary_does_not_clear_the_buffer() -> None:
    """`model_reasoning_summary` emits plenty of encrypted-only items."""
    empty = {"type": "response_item",
             "payload": {"type": "reasoning", "summary": [],
                         "encrypted_content": "..."}}
    traj = CodexAgent(model="gpt-5.6-sol").to_trajectory(
        _rollout(_summary("kept"), empty))
    assert [s.reasoning_content for s in traj.steps if s.reasoning_content] \
        == ["kept"]


def test_two_thoughts_in_one_bundle_are_both_kept() -> None:
    """The SECOND overwrite site: Harbor's grouping kept one per bundle.

    Both assistant messages here belong to the same model request (no
    token_count closes one), so they bundle into a single ATIF step -- and that
    step must carry both thoughts, not the last one.
    """
    traj = CodexAgent(model="gpt-5.6-sol").to_trajectory({
        "agent": "codex", "messages": [
            {"type": "session_meta",
             "payload": {"id": "s", "cli_version": "0.152.0"}},
            _summary("about the first"),
            {"type": "response_item",
             "payload": {"type": "message", "role": "assistant",
                         "content": [{"type": "output_text", "text": "one"}]}},
            _summary("about the second"),
            {"type": "response_item",
             "payload": {"type": "message", "role": "assistant",
                         "content": [{"type": "output_text", "text": "two"}]}},
        ]})
    assert [s.reasoning_content for s in traj.steps if s.reasoning_content] \
        == ["about the first\nabout the second"]


def test_the_saved_probe_rollout_loses_no_summary_text() -> None:
    """The regression itself, on the rollout that exposed it.

    Every summary string in the raw rollout must survive verbatim into ATIF.
    """
    path = (ROOT / "outputs" / "openmath_gpt-5.6-sol_probe_codex_detailed2"
            / "trajectories" / "goldbach_ce#1" / "transcript.json")
    if not path.exists():
        pytest.skip("the probe run is not present in this checkout")
    messages = json.loads(path.read_text())["messages"]
    raw = []
    for msg in messages:
        payload = msg.get("payload") or {}
        if msg.get("type") != "response_item" or payload.get("type") != "reasoning":
            continue
        parts = [i["text"] for i in (payload.get("summary") or [])
                 if isinstance(i, dict) and isinstance(i.get("text"), str)]
        if parts:
            raw.append("\n".join(parts))
    traj = CodexAgent(model="openai/gpt-5.6-sol").to_trajectory(
        {"agent": "codex", "messages": messages})
    got = [s.reasoning_content for s in traj.steps if s.reasoning_content]
    assert sum(len(t) for t in raw) == 4025          # the rollout, unchanged
    assert "\n".join(raw) == "\n".join(got)          # all of it, in order


# --------------------------------------------------------------------------
# a turn that FAILED while `codex exec` exited 0
#
# pelican_rank_cap_gpt-6-astra_final: episodes #1, #2 and #3 were cut off when
# the gateway key hit its dollar cap and answered `400 ... Budget has been
# exceeded`. Codex reported the turn as failed, printed it, and returned 0, so
# the runner -- which set `error` only on a launch failure, its own timeout, or
# a non-zero exit with no rollout -- filed all three `ok: true`. They were
# retired, judged, and counted: three verdicts on drawings the agent never
# finished. Codex says so itself, in the last event of the rollout.
TASK_COMPLETE_FAILED = {
    "type": "event_msg", "ordinal": 161,
    "payload": {"type": "task_complete", "turn_id": "01a088b0-cd7a",
                "last_agent_message": None,
                "error": {"message": '{"error":{"message":"Budget has been '
                                     'exceeded! Current cost: 1505.005, Max '
                                     'budget: 1500.0","type":"budget_exceeded",'
                                     '"param":null,"code":"400"}}'}}}
TASK_COMPLETE_OK = {
    "type": "event_msg", "ordinal": 203,
    "payload": {"type": "task_complete", "turn_id": "01a088b0-cdc1",
                "last_agent_message": "Saved the final drawing.",
                "duration_ms": 812345}}


def test_a_completed_turn_reports_no_terminal_error() -> None:
    from core.agents.codex_runner import summarise

    assert summarise(ROLLOUT + [TASK_COMPLETE_OK])["terminal_error"] is None
    assert summarise(ROLLOUT)["terminal_error"] is None


def test_a_failed_turn_is_reported_even_though_codex_exited_0() -> None:
    """The whole chain, from Codex's own event to the judge refusing it."""
    from core import judge as judging
    from core.agents import errors
    from core.agents.codex_runner import summarise

    info = summarise(ROLLOUT + [TASK_COMPLETE_FAILED])
    assert "Budget has been exceeded" in info["terminal_error"]

    # What the runner puts on the record, and what the host makes of it.
    error = f"codex ended the turn with an error: {info['terminal_error']}"
    failure = errors.classify(error)
    assert failure is errors.ApiUsageLimitError
    assert errors.reason(failure) == "usage_limit"

    # `ok` is `error is None`, so the episode stops being a clean run ...
    rec = {"episode": "pelican_bicycle#1", "ok": error is None,
           "failure": errors.reason(failure),
           "transcript_path": "/x/transcript.json"}
    assert rec["ok"] is False
    # ... it is never judged ...
    assert judging.is_judgeable(rec) is False
    # ... it is not retried in-process, because a human has to raise the cap ...
    import run as run_cli
    assert run_cli.select_transient_retries([rec]) == []
    # ... and nothing retires it, so the next run.py re-runs it.
    assert errors.is_lost(rec["failure"]) is True


def test_the_partial_answer_is_kept_on_a_failed_turn() -> None:
    """The agent's last message before the cut is still the record's final text.

    It is what the transcript ends on, and losing it would make a cut episode
    indistinguishable from one that never said anything. The record is marked
    failed by the terminal error, not by pretending nothing was produced.
    """
    from core.agents.codex_runner import summarise

    info = summarise(ROLLOUT + [TASK_COMPLETE_FAILED])
    assert info["final_text"] == summarise(ROLLOUT)["final_text"]
    assert info["n_tool_calls"] == summarise(ROLLOUT)["n_tool_calls"]
