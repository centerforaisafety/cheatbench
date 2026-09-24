"""The Muse Code adapter: its command line, its settings file, its trajectory.

Three things are pinned here, in order of how expensive they would be to get
wrong silently:

  * `--disable-web-tools` MUST be on every closed-book `muse exec` line. The
    CLI's default tool spec carries `web_search`, which the CLI runs against
    Meta's search endpoint from inside the container -- outside the sandbox's
    network namespace. There is no CLI default to fall back on.
  * the grader is registered in `settings.json` by URL and NEVER by a command:
    a `command` in that file is a path to the grader's source and the answer
    key, exactly as it would be in codex's config.toml.
  * the session log converts to ATIF the way the judge expects. The fixture
    under tests/fixtures/ is a REAL Muse Code 1.0.3 session log, recorded by
    driving the CLI through a two-turn episode (one `bash` call, one final
    message) against a loopback stand-in for the Meta API; the second fixture
    is the same CLI failing at the API with a rejected key.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import render, trial  # noqa: E402
from core.agents import AgentFactory, make_agent  # noqa: E402
from core.agents import errors  # noqa: E402
from core.agents import muse_code_runner as runner  # noqa: E402
from core.agents.muse_code import MuseCodeAgent  # noqa: E402
from core.trajectory import SCHEMA_VERSION, Trajectory  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
SESSION = runner.read_session(str(FIXTURES / "muse_code_session.jsonl"))
AUTHFAIL = runner.read_session(str(FIXTURES / "muse_code_session_authfail.jsonl"))

CLOSED = {"web_search": False, "web_fetch": False}
OPEN = {"web_search": True, "web_fetch": True}
MODEL = "meta/muse-spark-1.3"


def _agent(policy: dict, **kw) -> MuseCodeAgent:
    agent = make_agent("muse-code", model=MODEL, **kw)
    agent.apply_tool_policy(policy)
    return agent


def _cfg(agent: MuseCodeAgent, grader_src: str = "", env: dict | None = None,
         monkeypatch=None, base_url: str = "") -> dict:
    if monkeypatch is not None:
        if base_url:
            monkeypatch.setenv("MUSE_BASE_URL", base_url)
        else:
            monkeypatch.delenv("MUSE_BASE_URL", raising=False)
    payload = json.loads(agent.blob(
        row={"id": "row"}, prompt="do the thing", files=[], skip_dirs=(),
        grader_src=grader_src, stage_src="", timeout_s=60, env=env or {},
        install=agent.install()))
    return payload


def _command(agent: MuseCodeAgent) -> str:
    cfg = _cfg(agent)["task"]["muse"]
    return runner.compose_command(cfg, "/tmp/muse-prompt/prompt.txt",
                                  "/tmp/muse-prompt/muse.jsonl")


# --------------------------------------------------------------------------
# THE GUARD
# --------------------------------------------------------------------------
def test_closed_book_always_passes_disable_web_tools() -> None:
    assert "--disable-web-tools" in _command(_agent(CLOSED))
    assert MuseCodeAgent.WEB_TOOL not in _agent(CLOSED).tools


@pytest.mark.parametrize("name", sorted(p.parent.name for p
                                        in ROOT.glob("tasks/*/task.yaml")))
def test_every_shipped_task_composes_a_closed_web_command(name: str) -> None:
    task = trial.load_task(name)
    agent = _agent(task.tools, **({"api_base_url": "https://gateway.example/v1"}
                                  if task.tools.get("mcp_only") else {}))
    assert not task.tools.get("web_search") and not task.tools.get("web_fetch")
    assert "--disable-web-tools" in _command(agent)
    assert agent.setup() == []


def test_open_web_policy_omits_the_flag_and_offers_web_search() -> None:
    agent = _agent(OPEN)
    assert "--disable-web-tools" not in _command(agent)
    assert MuseCodeAgent.WEB_TOOL in agent.tools
    assert agent.setup() == []


# --------------------------------------------------------------------------
# the tool policy
# --------------------------------------------------------------------------
def test_both_web_keys_are_enforceable() -> None:
    """One flag removes the one web function; verified on the wire."""
    assert MuseCodeAgent.ENFORCEABLE == {"web_search", "web_fetch", "mcp_only"}


def test_the_two_web_keys_may_not_disagree() -> None:
    agent = make_agent("muse-code", model=MODEL)
    with pytest.raises(ValueError) as excinfo:
        agent.apply_tool_policy({"web_search": True, "web_fetch": False})
    assert "web_fetch" in str(excinfo.value)


def test_an_unenforceable_key_is_refused_by_name() -> None:
    agent = make_agent("muse-code", model=MODEL)
    with pytest.raises(ValueError) as excinfo:
        agent.apply_tool_policy({"web_search": False, "some_future_tool": False})
    assert "muse-code" in str(excinfo.value)
    assert "some_future_tool" in str(excinfo.value)


# --------------------------------------------------------------------------
# the invocation, against Harbor's shape
# --------------------------------------------------------------------------
def test_command_matches_harbors_shape() -> None:
    command = _command(_agent(CLOSED, generation_config={"reasoning_effort": "xhigh"},
                              max_turns=150))
    for fragment in (
            'export PATH="$HOME/.local/bin:$PATH"',
            "muse exec",
            "--json",
            "--yolo",
            "--user-input-auto-resolve",
            "--prompt-file /tmp/muse-prompt/prompt.txt",
            "--model muse-spark-1.3",       # the meta/ prefix is stripped
            "--reasoning-effort xhigh",
            "--disable-web-tools",
            "--max-model-steps 150",
            "</dev/null",                   # the prompt is a file, never stdin
            "| tee ",
    ):
        assert fragment in command, fragment
    # The base URL is pinned in settings.json, never passed as a flag: the
    # flag alone makes the CLI withhold its bearer for a non-Meta origin.
    assert "--base-url" not in command


def test_reasoning_effort_defaults_to_high_and_is_recorded() -> None:
    agent = _agent(CLOSED)
    assert "--reasoning-effort high" in _command(agent)
    assert agent.resolved_generation_config() == {"reasoning_effort": "high"}


def test_an_unknown_reasoning_effort_is_refused_at_setup() -> None:
    agent = _agent(CLOSED, generation_config={"reasoning_effort": "extreme"})
    assert any("reasoning_effort" in p for p in agent.setup())


def test_an_untranslated_generation_key_is_refused_at_setup() -> None:
    agent = _agent(CLOSED, generation_config={"reasoning_summary": "detailed"})
    assert any("reasoning_summary" in p for p in agent.setup())


def test_incomplete_release_pin_is_refused() -> None:
    """An exact Muse pin includes the R-build identifier."""
    agent = _agent(CLOSED, version="1.0.2")
    assert any("pinning" in p for p in agent.setup())


def test_model_slug_strips_exactly_one_meta_prefix() -> None:
    assert MuseCodeAgent(model="meta/muse-spark-1.3").model_slug == "muse-spark-1.3"
    assert MuseCodeAgent(model="muse-spark-1.3").model_slug == "muse-spark-1.3"
    assert MuseCodeAgent(model="openai/x").model_slug == "openai/x"


def test_install_fetches_the_binary_at_install_time() -> None:
    install = _agent(CLOSED).install()
    assert install["name"] == "muse-code"
    assert "https://dev.meta.ai/install.sh" in install["install"]
    assert "MUSE_NO_MODIFY_PATH=1" in install["install"]
    # `muse --version` is what makes the launcher download the real binary.
    assert "muse --version" in install["install"]
    assert "muse --version" in install["version_cmd"]


def test_credentials_are_per_adapter() -> None:
    assert MuseCodeAgent.API_KEY_ENV == "META_API_KEY"
    assert "MUSE_BASE_URL" in MuseCodeAgent.PASSTHROUGH_ENV
    # The grader's judge credentials travel for the runner and are stripped
    # from the CLI's environment again.
    for name in MuseCodeAgent.GRADER_ONLY_ENV:
        assert name in MuseCodeAgent.PASSTHROUGH_ENV


def test_blob_carries_the_row_and_the_grader_when_the_task_ships_one() -> None:
    payload = _cfg(_agent(CLOSED), grader_src="def make_tool(r, s, w): ...")
    assert payload["task"]["row"] == {"id": "row"}
    assert payload["modules"]["grader"].startswith("def make_tool")
    assert payload["task"]["muse"]["api_key_env"] == "META_API_KEY"


def test_a_task_with_no_grader_carries_neither() -> None:
    payload = _cfg(_agent(CLOSED))
    assert "row" not in payload["task"]
    assert payload["modules"]["grader"] == ""


# --------------------------------------------------------------------------
# the two routes: native Meta, or the gateway behind the shim
# --------------------------------------------------------------------------
def test_no_base_url_is_the_native_route_with_a_bare_model_id(monkeypatch) -> None:
    payload = _cfg(_agent(CLOSED), monkeypatch=monkeypatch)
    muse = payload["task"]["muse"]
    assert muse["route"] == "native"
    assert muse["base_url"] == ""
    assert muse["model_slug"] == "muse-spark-1.3"
    assert "forwarder" not in payload["modules"]
    assert "--base-url" not in _command(_agent(CLOSED))


def test_an_api_meta_ai_origin_stays_native_and_goes_on_the_flag(monkeypatch) -> None:
    monkeypatch.delenv("MUSE_BASE_URL", raising=False)
    agent = make_agent("muse-code", model=MODEL,
                       api_base_url="https://api.meta.ai/v1")
    agent.apply_tool_policy(CLOSED)
    assert agent.route == "native"
    assert agent.model_slug == "muse-spark-1.3"
    assert ["--base-url", "https://api.meta.ai/v1"] == agent.cli_flags()[-2:]


def test_any_other_origin_is_the_gateway_route_with_the_full_model_id(monkeypatch) -> None:
    monkeypatch.delenv("MUSE_BASE_URL", raising=False)
    agent = make_agent("muse-code", model=MODEL,
                       api_key_env="OPENAI_API_KEY",
                       api_base_url="https://litellm.safe.ai")
    agent.apply_tool_policy(CLOSED)
    assert agent.route == "gateway"
    assert agent.model_slug == MODEL                            # NOT stripped
    assert agent.api_key_env == "OPENAI_API_KEY"                   # host side
    assert agent.container_key_envs() == ("META_API_KEY",)         # CLI side
    assert "--base-url" not in agent.cli_flags()      # the runner adds the shim
    payload = json.loads(agent.blob(
        row={"id": "row"}, prompt="x", files=[], skip_dirs=(), grader_src="",
        stage_src="", timeout_s=60, env={}, install=None))
    muse = payload["task"]["muse"]
    assert muse["route"] == "gateway"
    assert muse["base_url"] == "https://litellm.safe.ai"
    assert muse["model_slug"] == MODEL
    assert muse["api_key_env"] == "META_API_KEY"
    assert "class Upstream" in payload["modules"]["forwarder"]
    assert agent.setup() == []


def test_the_shipped_entry_routes_directly_to_meta(monkeypatch) -> None:
    """configs/models.yaml's muse-spark-1.3, as run.py would build it."""
    import yaml
    from core import routing

    monkeypatch.setenv("OPENAI_BASE_URL", "https://litellm.safe.ai")
    entry = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["muse-spark-1.3"]
    r = routing.resolve("muse-spark-1.3", entry, where="models.yaml")
    agent = make_agent("muse-code", model=entry["model"],
                       generation_config=entry.get("generation_config"),
                       api_key_env=r["api_key_env"], api_base_url=r["api_base_url"],
                       extra_body=r["extra_body"])
    agent.apply_tool_policy(CLOSED)
    assert agent.route == "native"
    assert agent.model_slug == "muse-spark-1.3"
    assert agent.api_key_env == "META_API_KEY"
    assert agent.resolved_base_url() == "https://api.meta.ai/v1"


def test_muse_base_url_env_is_the_fallback_when_the_entry_routes_nothing(monkeypatch) -> None:
    monkeypatch.setenv("MUSE_BASE_URL", "http://127.0.0.1:4242")
    agent = _agent(CLOSED)
    assert agent.route == "gateway"
    assert _cfg(agent)["task"]["muse"]["base_url"] == "http://127.0.0.1:4242"


# --------------------------------------------------------------------------
# the shim's catalog and endpoint composition
# --------------------------------------------------------------------------
def test_catalog_is_the_one_shape_the_cli_accepts() -> None:
    doc = json.loads(runner.catalog_json(MODEL))
    assert doc["data"][0]["id"] == MODEL


def test_upstream_base_drops_a_trailing_v1() -> None:
    assert runner.upstream_base("https://litellm.safe.ai/v1") == "https://litellm.safe.ai"
    assert runner.upstream_base("https://litellm.safe.ai") == "https://litellm.safe.ai"
    assert runner.upstream_base("https://litellm.safe.ai/") == "https://litellm.safe.ai"


# --------------------------------------------------------------------------
# settings.json: the grader by URL
# --------------------------------------------------------------------------
def test_settings_never_name_a_command() -> None:
    doc = runner.settings_json(grader_url="http://127.0.0.1:5/t/mcp")
    text = json.dumps(doc)
    assert "command" not in text
    assert "stdio" not in text
    assert doc["mcp_servers"]["grader"] == {
        "enabled": True, "transport": "streamable_http",
        "url": "http://127.0.0.1:5/t/mcp"}


def test_settings_are_minimal_when_there_is_nothing_to_say() -> None:
    assert runner.settings_json() == {"schema_version": 1}


def test_prepare_homes_writes_settings_only_when_needed(tmp_path: Path) -> None:
    cfg = {"config_home": str(tmp_path / "c"), "data_home": str(tmp_path / "d"),
           "prompt_dir": str(tmp_path / "p")}
    prompt = runner.prepare_homes(cfg, "solve it")
    assert Path(prompt).read_text() == "solve it"
    assert not (tmp_path / "c" / "muse" / "settings.json").exists()
    runner.prepare_homes(cfg, "solve it", grader_url="http://127.0.0.1:9/x/mcp")
    doc = json.loads((tmp_path / "c" / "muse" / "settings.json").read_text())
    assert doc["mcp_servers"]["grader"]["url"] == "http://127.0.0.1:9/x/mcp"


# --------------------------------------------------------------------------
# the session store
# --------------------------------------------------------------------------
def test_find_session_picks_the_newest_main_log_and_never_a_subagent(tmp_path: Path) -> None:
    import os
    import time

    sessions = tmp_path / "muse" / "sessions" / "2026" / "09" / "09"
    old = sessions / "aaa"
    new = sessions / "bbb"
    sub = new / "subagent" / "ccc"
    for d in (old, new, sub):
        d.mkdir(parents=True)
        (d / "session.jsonl").write_text("{}\n")
    past = time.time() - 100
    os.utime(old / "session.jsonl", (past, past))
    future = time.time() + 100
    os.utime(sub / "session.jsonl", (future, future))
    assert runner.find_session(str(tmp_path)) == str(new / "session.jsonl")
    assert runner.subagent_sessions(str(new / "session.jsonl")) == [
        str(sub / "session.jsonl")]
    assert runner.find_session(str(tmp_path / "nowhere")) is None


def test_summarise_counts_calls_tools_and_totals_usage() -> None:
    info = runner.summarise(SESSION)
    assert info["session_id"] == "01a084e3-9f2a-7123-a48f-fecd864cc825"
    assert info["model"] == "muse-spark-1.3"
    assert info["version"] == "1.0.3"
    assert info["n_turns"] == 2
    assert info["n_tool_calls"] == 1
    assert info["terminal"] == "completed"
    assert info["final_text"].startswith("Done. The directory listing")
    usage = runner.usage_of(SESSION)
    assert usage == {"input_tokens": 2700, "output_tokens": 140,
                     "cache_read_tokens": 2200, "cache_write_tokens": 0,
                     "cached_tokens": 2200, "reasoning_tokens": 50,
                     "model_calls": 2}
    assert runner.add_usage(usage, {"input_tokens": 10})["input_tokens"] == 2710


def test_a_completed_run_with_no_model_output_is_an_error() -> None:
    """The gateway's empty completions: the CLI exits 0 with nothing in it."""
    info = {"terminal": "completed", "final_text": "", "n_tool_calls": 0}
    usage = {"model_calls": 1, "output_tokens": 0}
    reason = runner.empty_completion_error(info, usage)
    assert reason and "empty completions" in reason
    assert MuseCodeAgent(model=MODEL).classify_failure(None, reason) \
        is errors.UnknownApiError
    # Any real output, or a real failure, or no call at all, is not this case.
    assert runner.empty_completion_error(info, {"model_calls": 1, "output_tokens": 3}) is None
    assert runner.empty_completion_error(dict(info, final_text="DONE"), usage) is None
    assert runner.empty_completion_error(dict(info, n_tool_calls=1), usage) is None
    assert runner.empty_completion_error(dict(info, terminal="failed"), usage) is None
    assert runner.empty_completion_error(info, None) is None
    # The real fixture episode has output and is untouched.
    assert runner.empty_completion_error(runner.summarise(SESSION),
                                         runner.usage_of(SESSION)) is None


def test_summarise_reports_the_api_failure() -> None:
    info = runner.summarise(AUTHFAIL)
    assert info["terminal"] == "failed"
    assert "META_API_KEY was rejected" in info["terminal_reason"]
    assert info["n_turns"] == 0
    assert runner.usage_of(AUTHFAIL) is None


# --------------------------------------------------------------------------
# the session log as ATIF
# --------------------------------------------------------------------------
def test_session_converts_to_valid_atif() -> None:
    traj = MuseCodeAgent(model=MODEL).to_trajectory(
        {"agent": "muse-code", "messages": SESSION})
    assert isinstance(traj, Trajectory)
    assert traj.schema_version == SCHEMA_VERSION
    assert traj.session_id == "01a084e3-9f2a-7123-a48f-fecd864cc825"
    assert traj.agent.name == "muse-code"
    assert traj.agent.version == "1.0.3"
    assert traj.agent.model_name == "muse-spark-1.3"
    # user prompt, tool-call response, final response
    assert [s.source for s in traj.steps] == ["user", "agent", "agent"]
    call = traj.steps[1]
    assert call.tool_calls[0].function_name == "bash"
    assert call.tool_calls[0].arguments["command"] == "ls -la && echo FAKE_TOOL_RAN"
    assert call.observation.results[0].source_call_id == call.tool_calls[0].tool_call_id
    assert "FAKE_TOOL_RAN" in call.observation.results[0].content
    # Reasoning is its OWN field and is never folded into the message.
    assert call.reasoning_content.startswith("(fake reasoning, turn 1)")
    assert "(fake reasoning" not in call.message
    assert call.metrics.prompt_tokens == 1200
    assert call.metrics.cached_tokens == 1000
    assert call.metrics.completion_tokens == 80
    assert call.extra["finish_reason"] == "tool_calls"
    final = traj.steps[2]
    assert final.message.startswith("Done.")
    assert final.tool_calls is None
    assert traj.final_metrics.total_prompt_tokens == 2700
    assert traj.final_metrics.total_cached_tokens == 2200
    assert traj.final_metrics.total_completion_tokens == 140
    assert traj.extra["final_text"].startswith("Done.")
    assert traj.notes is None
    assert Trajectory.model_validate(traj.to_json_dict()) == traj


def test_render_trajectory_produces_the_judges_log_shape() -> None:
    traj = MuseCodeAgent(model=MODEL).to_trajectory(
        {"agent": "muse-code", "messages": SESSION})
    text = render.render_trajectory(traj)
    assert text.startswith("[init] agent=muse-code version=1.0.3")
    assert "THINKING: (fake reasoning, turn 1)" in text
    assert "TOOL_CALL bash: command=ls -la && echo FAKE_TOOL_RAN" in text
    assert "-> RESULT[bash]:" in text and "FAKE_TOOL_RAN" in text
    assert "ASSISTANT: Done." in text
    assert "FINAL MESSAGE: Done." in text


def test_muse_1_1_preserves_the_actual_meta_reasoning_summary_once() -> None:
    # Live Meta response -> Muse 1.1.1 session, 2026-09-11. Unrelated runtime
    # records and opaque ciphertext are omitted from this reduced fixture.
    # The HTTP summary and durable summary had exactly this same text.
    records = runner.read_session(str(FIXTURES / "muse_code_session_1_1_reasoning.jsonl"))
    summary = "Solving a Chinese Remainder Theorem system manually under abstract output constraints."
    traj = MuseCodeAgent(model=MODEL).to_trajectory({"messages": records})
    assert traj.agent.version == "1.1.1"
    assert len(traj.steps) == 2
    step = traj.steps[1]
    # Both live deltas and the final committed summary exist in the session.
    # They must not become duplicated reasoning or separate response steps.
    assert step.reasoning_content == summary
    assert not step.extra.get("reasoning_withheld")
    assert "312658" in step.message
    assert step.metrics.extra["reasoning_tokens"] == 1489
    text = render.render_trajectory(traj)
    assert text.count(summary) == 1
    assert "THINKING: " + summary in text


@pytest.mark.parametrize("raw_reasoning", ["Full native reasoning text", ""])
def test_native_reasoning_text_and_encrypted_only_are_distinguished(raw_reasoning) -> None:
    records = json.loads(json.dumps(SESSION))
    for record in records:
        event = record.get("payload", {}).get("event", {})
        if event.get("kind") == "reasoning_summary_committed":
            event["text"] = ""
        elif event.get("kind") == "reasoning_committed":
            event["text"] = raw_reasoning
    traj = MuseCodeAgent(model=MODEL).to_trajectory({"messages": records})
    for step in traj.steps[1:]:
        assert step.reasoning_content == (raw_reasoning or None)
        assert bool(step.extra.get("reasoning_withheld")) == (not raw_reasoning)
    text = render.render_trajectory(traj)
    assert "ZmFrZS1lbmNyeXB0ZWQ=" not in text
    if raw_reasoning:
        assert "THINKING: " + raw_reasoning in text


def test_an_api_failure_is_a_recorded_trajectory_not_a_crash() -> None:
    traj = MuseCodeAgent(model=MODEL).to_trajectory(
        {"agent": "muse-code", "messages": AUTHFAIL})
    assert [s.source for s in traj.steps] == ["user"]
    assert "META_API_KEY was rejected" in traj.notes
    assert traj.final_metrics is None
    assert Trajectory.model_validate(traj.to_json_dict()) == traj


def test_an_empty_record_yields_the_placeholder_step() -> None:
    traj = MuseCodeAgent(model=MODEL).to_trajectory({"messages": []})
    assert len(traj.steps) == 1
    assert traj.steps[0].extra == {"placeholder": True}


def test_the_live_tail_renders_one_record_at_a_time() -> None:
    agent = MuseCodeAgent(model=MODEL)
    blocks = [agent.readable(rec) for rec in SESSION]
    assert any("TOOL_CALL bash" in b for b in blocks)
    assert any("ASSISTANT: Done." in b for b in blocks)
    # A session header carries no step and renders nothing, not a placeholder.
    assert agent.readable(SESSION[1]) == ""
    # The call record arrives before its result: the call renders alone with
    # no empty observation, and the result record renders as a bare RESULT.
    call_block = next(b for b in blocks if "TOOL_CALL bash" in b)
    assert "RESULT" not in call_block
    result_block = next(b for b in blocks if "FAKE_TOOL_RAN" in b
                        and "TOOL_CALL" not in b)
    assert "RESULT[" in result_block


def test_transcript_round_trips_through_the_factory() -> None:
    from core.agents import trajectory_from_transcript

    traj = trajectory_from_transcript(
        {"agent": "muse-code", "model": MODEL, "messages": SESSION})
    assert traj.agent.name == "muse-code"
    assert len(traj.steps) == 3


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------
def test_muse_never_claims_a_reported_cost_and_the_estimate_prices_it() -> None:
    agent = MuseCodeAgent(model=MODEL)
    assert agent.reported_cost_usd({"usage": {"input_tokens": 1}}) is None
    usage = runner.usage_of(SESSION)
    cost, source = agent.episode_cost({"usage": usage})
    assert source == "estimated"
    # 500 fresh @ $1.25/M + 2200 cached @ $0.15/M + 140 out @ $4.25/M
    assert cost == pytest.approx(500 * 1.25e-6 + 2200 * 0.15e-6 + 140 * 4.25e-6)


def test_usage_keys_are_muse_spelling_and_prompt_is_inclusive() -> None:
    assert MuseCodeAgent.USAGE_KEYS == ("input_tokens", "output_tokens",
                                        "cache_read_tokens", "cache_write_tokens")
    assert MuseCodeAgent.PROMPT_TOKENS_INCLUDE_CACHE is True


# --------------------------------------------------------------------------
# failure classification, in the CLI's own words
# --------------------------------------------------------------------------
def test_a_rejected_key_is_an_authentication_error() -> None:
    agent = MuseCodeAgent(model=MODEL)
    assert agent.classify_failure(
        None, "your API key from META_API_KEY was rejected") \
        is errors.AgentAuthenticationError
    assert agent.classify_failure(
        None, "failed to fetch model catalog: Provider returned malformed "
              "response data") is errors.UnknownApiError


def test_registry_key_matches_the_adapters_name() -> None:
    assert AgentFactory.get_agent_class("muse-code") is MuseCodeAgent
    assert MuseCodeAgent.name() == "muse-code"
