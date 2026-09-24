"""Model identity, shared generation settings, and default harness configuration."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run as run_py                                          # noqa: E402
from core import judge as judging                             # noqa: E402
from core.agents import AgentFactory, make_agent              # noqa: E402
from core.agents.claude_sdk import ClaudeSDKAgent             # noqa: E402
from core.agents.codex import CodexAgent                      # noqa: E402

MODELS = ROOT / "configs" / "models.yaml"


def _write(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


# --------------------------------------------------------------------------
# the one key
# --------------------------------------------------------------------------
def test_the_shipped_config_loads_for_a_claude_and_a_gpt_entry() -> None:
    claude = run_py.load_config(MODELS, "claude-opus-5")
    # Provider-prefixed: a litellm proxy rejects the bare id outright with
    # `LLM Provider NOT provided`.
    assert claude["model"] == "anthropic/claude-opus-5"
    assert claude["generation_config"] == {
        "thinking": {"type": "adaptive", "display": "summarized"}}

    gpt = run_py.load_config(MODELS, "gpt-5.6-sol")
    assert gpt["model"] == "openai/gpt-5.6-sol"
    assert gpt["generation_config"] == {"reasoning_effort": "high"}
    assert gpt["harness"]["options"] == {"reasoning_summary": "detailed"}


def test_the_old_options_key_is_refused_rather_than_ignored(tmp_path) -> None:
    path = _write(tmp_path, {"models": {"m": {"model": "x",
                                              "options": {"effort": "max"}}}})
    with pytest.raises(SystemExit, match="generation_config"):
        run_py.load_config(path, "m")


def test_a_judge_entry_with_the_old_options_key_is_refused(tmp_path) -> None:
    path = _write(tmp_path, {
        "models": {"m": {"model": "openai/x", "options": {"a": 1}}}})
    with pytest.raises(SystemExit, match="options"):
        judging.load_judge_config(path, "m")


# --------------------------------------------------------------------------
# every consumer declares what it can deliver
# --------------------------------------------------------------------------
def test_each_adapter_delivers_the_generation_config_its_own_way() -> None:
    codex = make_agent("codex", model="openai/gpt-5.6-sol",
                       generation_config={"reasoning_effort": "low",
                                          "reasoning_summary": "detailed"})
    codex.apply_tool_policy({"web_search": False, "web_fetch": False})
    flags = codex.cli_flags()
    pairs = list(zip(flags, flags[1:]))
    assert ("-c", "model_reasoning_effort=low") in pairs
    assert ("-c", "model_reasoning_summary=detailed") in pairs

    # The Claude adapter delivers the same kind of setting as an SDK option,
    # carried to the in-container runner on stdin.
    claude = make_agent("claude-sdk", model="claude-opus-5",
                        generation_config={"thinking": {"type": "adaptive"}})
    claude.apply_tool_policy({"web_search": False, "web_fetch": False})
    blob = json.loads(claude.blob(
        row={"id": "r"}, prompt="p", files=[], skip_dirs=(), stage_src="",
        timeout_s=1, env={}))
    assert blob["task"]["model_options"] == {"thinking": {"type": "adaptive"}}


def test_an_adapter_refuses_a_key_it_cannot_deliver_by_name() -> None:
    """A Claude knob handed to codex, and vice versa: legible, at preflight."""
    codex = make_agent("codex", model="openai/gpt-5.6-sol",
                       generation_config={"thinking": {"type": "adaptive"}})
    codex.apply_tool_policy({"web_search": False, "web_fetch": False})
    problems = codex.setup()
    assert any("'thinking'" in p and "codex" in p for p in problems), problems

    claude = make_agent("claude-sdk", model="claude-opus-5",
                        generation_config={"reasoning_summary": "detailed"})
    claude.apply_tool_policy({"web_search": False, "web_fetch": False})
    problems = claude.setup()
    assert any("'reasoning_summary'" in p and "claude-sdk" in p
               for p in problems), problems


def test_config_may_not_widen_the_tasks_tool_policy() -> None:
    """The task's `tools:` block is the only source of the tool set.

    The in-container runner refuses these keys too (its own guard, which is the
    last line); this is the first, on the host, before any container starts.
    """
    for key in ("tools", "allowed_tools", "disallowed_tools"):
        claude = make_agent("claude-sdk", model="claude-opus-5",
                            generation_config={key: ["Edit"]})
        claude.apply_tool_policy({"web_search": False, "web_fetch": False})
        assert any(f"'{key}'" in p for p in claude.setup())


def test_codex_records_the_effort_default_it_actually_applies() -> None:
    """run.json must state the applied default, not an empty dict."""
    agent = make_agent("codex", model="openai/gpt-5.6-sol")
    assert agent.resolved_generation_config() == {"reasoning_effort": "high"}
    agent = make_agent("codex", model="openai/gpt-5.6-sol",
                       generation_config={"reasoning_effort": "low"})
    assert agent.resolved_generation_config() == {"reasoning_effort": "low"}


@pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-6-sol"])
def test_judge_preserves_effort_and_records_unsupported_display_setting(model):
    sent, withheld = judging.split_generation_config(
        model, f"openai/{model}",
        {"reasoning_effort": "high", "reasoning_summary": "detailed"})
    assert sent == {"reasoning_effort": "high"}
    assert withheld == {"reasoning_summary": "detailed"}
    cfg = judging.load_judge_config(MODELS, model)
    assert cfg["model"] == f"openai/{model}"
    assert cfg["generation_config"] == sent
    assert cfg["unsupported"] == {}


def test_any_model_can_be_selected_without_a_judges_block(tmp_path):
    path = _write(tmp_path, {"models": {
        "default": "other", "other": {"model": "openai/other"},
        "new-model": {"model": "openai/new", "generation_config": {"temperature": 0.4}}
    }})
    cfg = judging.load_judge_config(path, "new-model")
    assert cfg["model"] == "openai/new"
    assert cfg["generation_config"] == {"temperature": 0.4}
    for name in ("typo", "default"):
        with pytest.raises(SystemExit, match="model"):
            judging.load_judge_config(path, name)


def test_judge_client_uses_plain_model_and_announces_display_omission(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    captured = {}
    def factory(model, generation_config, **kwargs):
        captured.update(model=model, generation_config=generation_config, **kwargs)
        return object()
    monkeypatch.setattr(judging, "get_llm_agent_class", factory)
    assert judging.DEFAULT_JUDGE == "gpt-6-sol"
    judging.make_judge_agent(judging.DEFAULT_JUDGE, MODELS)
    assert captured["model"] == "openai/gpt-6-sol"
    assert captured["generation_config"] == {"reasoning_effort": "high"}
    assert "reasoning_summary" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# --model and --agent are independent axes
# --------------------------------------------------------------------------
def test_the_default_agent_is_a_registered_adapter() -> None:
    """--agent unset is a HARNESS default, never derived from the model."""
    assert run_py.DEFAULT_AGENT in AgentFactory.names()


def test_every_model_entry_names_a_default_harness() -> None:
    """Each model selects a registered adapter with a recorded version policy."""
    doc = yaml.safe_load(MODELS.read_text())
    assert "models" not in doc and "default" not in doc
    for name, entry in doc.items():
        assert set(entry) <= {"model", "generation_config", "max_turns",
                              "permission_mode", "base_url", "api_key_env",
                              "api_base_url", "api_base_url_env", "extra_body", "harness"}, name
        assert entry["harness"]["name"] in AgentFactory.names()



# --------------------------------------------------------------------------
# the shipped file, checked against every consumer
# --------------------------------------------------------------------------
def test_every_shipped_generation_key_is_deliverable_by_someone() -> None:
    """A key no consumer translates would be a setting that does nothing."""
    # The judge client forwards every key as written, so this checks the agent
    # adapters only: every shipped key must be one some REGISTERED adapter
    # translates -- all of them, not a hand-kept two, so a new adapter's own
    # knob (delivered by that adapter) is not flagged as undeliverable.
    known: set = set()
    for name in AgentFactory.names():
        known |= set(AgentFactory.get_agent_class(name).GENERATION_KEYS)
    doc = yaml.safe_load(MODELS.read_text())
    for name, entry in doc.items():
        unknown = set(entry.get("generation_config") or {}) - known
        assert not unknown, f"{name}: {sorted(unknown)}"


def test_every_judge_entry_still_resolves() -> None:
    doc = yaml.safe_load(MODELS.read_text())
    for name, entry in doc.items():
        if name == "default" or "/" not in entry["model"]:
            continue
        cfg = judging.load_judge_config(MODELS, name)
        assert "/" in cfg["model"]


# --------------------------------------------------------------------------
# the in-container runner's transcript, which is where thinking is first kept
# --------------------------------------------------------------------------
RUNNER_PROBE = """
import json, sys
sys.path.insert(0, {root!r})
from core.agents import claude_sdk_runner as r
out = {{
    "empty_thinking": r._slim({{"_type": "AssistantMessage", "content": [
        {{"_type": "ThinkingBlock", "thinking": "", "signature": "sig"}}]}}),
    "real_thinking": r._slim({{"_type": "AssistantMessage", "content": [
        {{"_type": "ThinkingBlock", "thinking": "a thought", "signature": "sig"}}]}}),
    "openrouter": r._slim({{"_type": "AssistantMessage", "content": [
        {{"_type": "RedactedThinkingBlock", "data": "openrouter.reasoning:eyJ0"}}]}}),
    "ciphertext": r._slim({{"_type": "AssistantMessage", "content": [
        {{"_type": "RedactedThinkingBlock", "data": "EroBCkYIBBgC"}}]}}),
}}
open({out!r}, "w").write(json.dumps(out))
"""


def test_the_runner_keeps_every_thinking_block(tmp_path) -> None:
    """The transcript is where an omitted-display thinking block first exists.

    Run out of process: importing the runner replaces this process's stdout,
    which it does deliberately (its contract is one JSON line on stdout).
    """
    result = tmp_path / "slim.json"
    proc = subprocess.run(
        [sys.executable, "-c",
         RUNNER_PROBE.format(root=str(ROOT), out=str(result))],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(result.read_text())

    # An empty thinking block is KEPT -- with it, the message survives at all.
    assert got["empty_thinking"] == {
        "_type": "AssistantMessage",
        "content": [{"_type": "ThinkingBlock", "thinking": ""}]}
    # A real one is unchanged, `signature` dropped as before (a multi-KB blob
    # with nothing to read in it).
    assert got["real_thinking"]["content"] == [
        {"_type": "ThinkingBlock", "thinking": "a thought"}]
    # OpenRouter's envelope carries plain reasoning, so its payload is kept for
    # the host-side converter to decode.
    assert got["openrouter"]["content"] == [
        {"_type": "RedactedThinkingBlock",
         "data": "openrouter.reasoning:eyJ0"}]
    # Genuine ciphertext cannot be read: the block survives, the blob does not.
    assert got["ciphertext"]["content"] == [{"_type": "RedactedThinkingBlock"}]


def test_consolidated_terminus_profile_is_accepted():
    cfg = run_py.load_config(MODELS, "gpt-5.6-sol-chat")
    agent = make_agent("terminus-2", model=cfg["model"],
                       generation_config=cfg["generation_config"])
    agent.apply_tool_policy({"web_search": False, "web_fetch": False})
    assert not agent.setup()
    assert cfg["generation_config"] == {"reasoning_effort": "high"}

def test_model_selection_is_required_before_launch(capsys):
    with pytest.raises(SystemExit) as error:
        run_py.build_parser().parse_args(['chess'])
    assert error.value.code == 2
    assert '--model' in capsys.readouterr().err


def test_flat_model_entries_reach_all_consumers(tmp_path):
    from core.chat_config import load_chat_config
    path = _write(tmp_path, {'example': {
        'model': 'openai/example',
        'api_base_url': 'https://example.invalid/v1',
        'generation_config': {'reasoning_effort': 'high'},
        'harness': {'name': 'codex', 'version': '0.156.1',
                    'options': {'reasoning_summary': 'detailed'}},
    }})
    for loader in (run_py.load_config, judging.load_judge_config, load_chat_config):
        cfg = loader(path, 'example')
        assert cfg['model'] == 'openai/example'
        assert cfg['generation_config'] == {'reasoning_effort': 'high'}
    override = tmp_path / 'override.yaml'
    override.write_text('example:\n  generation_config:\n    reasoning_effort: low\n')
    assert load_chat_config(path, 'example', override)['generation_config'] == {'reasoning_effort': 'low'}
