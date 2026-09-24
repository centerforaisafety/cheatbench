"""Exercise configuration at the CLI/SDK boundaries, without paid model calls."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

import run
from core import judge
from core.agents import AgentFactory, make_agent
from core.agents.config import resolve_version
from core.chat_config import load_chat_config
from core.model_settings import api_generation, harness_settings
from tasks.sycophancy import run_chat

MODELS = Path(__file__).resolve().parents[1] / "configs/models.yaml"


@pytest.fixture(autouse=True)
def endpoints(monkeypatch):
    for name in ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL", "LITELLM_BASE_URL",
                 "GEMINI_BASE_URL", "DEEPSEEK_BASE_URL"):
        monkeypatch.setenv(name, "https://gateway.example/v1")
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LITELLM_API_KEY",
                 "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "META_API_KEY", "XAI_API_KEY"):
        monkeypatch.setenv(name, "offline-test")


def agent_for(name, override=None, **kwargs):
    cfg = run.load_config(MODELS, name)
    harness = override or cfg["harness"]["name"]
    generation, extra = harness_settings(cfg, harness)
    agent = make_agent(harness, model=cfg["model"], generation_config=generation,
                       extra_body=extra, api_base_url=cfg["api_base_url"],
                       api_key_env=cfg["api_key_env"], **kwargs)
    agent.apply_tool_policy({"web_search": False, "web_fetch": False})
    return agent


def test_codex_options_reach_cli_but_not_judge_or_sycophancy():
    agent = agent_for("gpt-6-sol")
    flags = agent.cli_flags()
    assert "model_reasoning_effort=high" in flags
    assert "model_reasoning_summary=detailed" in flags
    for loader in (judge.load_judge_config, load_chat_config, run_chat.load_config):
        cfg = loader(MODELS, "gpt-6-sol")
        assert cfg["generation_config"] == {"reasoning_effort": "high"}
        assert cfg["unsupported"] == {}


def test_t2_override_preserves_model_route_effort_and_uses_its_own_pin():
    cfg = run.load_config(MODELS, "gpt-6-sol")
    original = deepcopy(cfg)
    agent = agent_for("gpt-6-sol", "terminus-2")
    assert agent.model == cfg["model"]
    assert agent.api_base_url == cfg["api_base_url"]
    assert agent.api_key_env == cfg["api_key_env"]
    assert agent.generation_config == {"reasoning_effort": "high"}
    version = resolve_version("terminus-2", model_harness=cfg["harness"], models_path=MODELS)
    assert version.version == agent.FIXED_VERSION
    assert cfg == original


@pytest.mark.parametrize("name,api_model", [
    ("muse-spark-1.3", "openai/muse-spark-1.3"),
    ("deepseek-v4-pro", "openai/deepseek-v4-pro"),
    ("kimi-k3", "openai/openrouter/moonshotai/kimi-k3"),
    ("grok-4.6", "openai/openrouter/x-ai/grok-4.6"),
    ("grok-4.7", "openai/openrouter/x-ai/grok-4.7"),
    ("gemini-3.8-flash", "openai/gemini/gemini-3.8-flash"),
    ("claude-opus-5-5", "anthropic/anthropic/claude-opus-5-5"),
])
def test_t2_override_uses_correct_protocol_and_wire_model(name, api_model):
    agent = agent_for(name, "terminus-2")
    assert agent.setup() == []
    assert agent.api_model() == api_model
    assert agent.resolved_routing()["api_model_id"] == api_model
    if name == "kimi-k3":
        assert agent.extra_body["thinking"] == {"type": "enabled", "effort": "high"}
    if name.startswith("claude"):
        assert agent.generation_config["output_config"] == {"effort": "high"}


def test_gemini_turn_cap_is_a_model_limit_not_an_infrastructure_failure():
    from core.agents import errors
    assert errors.classify("Maximum session turns exceeded") is errors.MaxTurnsError


def test_claude_shared_settings_reach_sdk_and_messages_api():
    agent = agent_for("claude-opus-5-5")
    payload = json.loads(agent.blob(row={"id": "probe"}, prompt="probe", files=[],
                                    skip_dirs=(), stage_src="", timeout_s=1, env={}))
    assert payload["task"]["model_options"] == {
        "effort": "high", "thinking": {"type": "adaptive", "display": "summarized"}}
    assert payload["task"]["permission_mode"] == "bypassPermissions"
    cfg = judge.load_judge_config(MODELS, "claude-opus-5-5")
    assert cfg["generation_config"]["output_config"] == {"effort": "high"}
    assert cfg["generation_config"]["thinking"]["display"] == "summarized"


def test_kimi_shared_effort_reaches_environment_and_api_body():
    agent = agent_for("kimi-k3")
    assert agent.generation_config == {"thinking_effort": "high"}
    assert "KIMI_MODEL_THINKING_EFFORT" in agent.cli_env()
    cfg = load_chat_config(MODELS, "kimi-k3")
    assert cfg["extra_body"]["thinking"] == {"type": "enabled", "effort": "high"}
    assert cfg["extra_body"]["provider"]["allow_fallbacks"] is False


def test_muse_summary_is_scoped_to_harness_and_effort_to_both():
    cfg = run.load_config(MODELS, "muse-spark-1.3")
    gen, extra = harness_settings(cfg, "muse-code")
    assert gen == {"reasoning_effort": "high"}
    assert extra == {"reasoning": {"summary": "detailed"}}
    assert load_chat_config(MODELS, "muse-spark-1.3")["extra_body"] == {}
    _, alternate_extra = harness_settings(cfg, "terminus-2")
    assert alternate_extra == {}


def test_gemini_turn_limit_is_written_to_actual_cli_settings():
    agent = agent_for("gemini-3.8-flash", max_turns=7)
    settings = agent.settings()
    assert settings["model"]["maxSessionTurns"] == 7
    assert agent.TURN_LIMIT
    uncapped = agent_for("gemini-3.8-flash", max_turns=None)
    assert "maxSessionTurns" not in uncapped.settings().get("model", {})


def test_grok_47_has_exact_route_and_current_pin_and_build_flags():
    cfg = run.load_config(MODELS, "grok-4.7")
    assert cfg["model"] == "openrouter/x-ai/grok-4.7"
    assert cfg["api_base_url"] == "https://gateway.example/v1"
    assert cfg["api_key_env"] == "LITELLM_API_KEY"
    assert cfg["harness"] == {"name": "grok-build", "version": "1.0.41"}
    agent = agent_for("grok-4.7", max_turns=7)
    payload = json.loads(agent.blob(row={"id": "probe"}, prompt="probe", files=[],
                                    skip_dirs=(), stage_src="", timeout_s=1, env={}))
    argv = agent._preview_argv(payload["task"]["grok"])
    assert payload["task"]["grok"]["model_id"] == "openrouter/x-ai/grok-4.7"
    assert argv[argv.index("--model") + 1] == "openrouter/x-ai/grok-4.7"
    assert argv[argv.index("--max-turns") + 1] == "7"
    assert argv[argv.index("--reasoning-effort") + 1] == "high"
    assert "--always-approve" in argv


@pytest.mark.parametrize("name", AgentFactory.names())
def test_permission_override_is_delivered_or_rejected(name):
    agent = make_agent(name, model="openai/test", permission_mode="imaginary")
    assert any("permission mode" in error for error in agent.config_problems())
    default = make_agent(name, model="openai/test", permission_mode=None)
    assert default.permission_mode == "bypassPermissions"


@pytest.mark.parametrize("name", [name for name in yaml.safe_load(MODELS.read_text()) if name != "default"])
def test_every_shipped_harness_config_passes_preflight(name):
    cfg = run.load_config(MODELS, name)
    agent = agent_for(name, version=cfg["harness"]["version"])
    assert agent.setup() == []
    assert load_chat_config(MODELS, name)["unsupported"] == {}
    # Exercise the shared resolver through the actual SDK constructor, without HTTP.
    client = judge.make_judge_agent(name, MODELS)
    assert "reasoning_summary" not in client.generation_config


def test_model_pin_and_explicit_alternative_version_are_independent():
    cfg = run.load_config(MODELS, "gpt-6-sol")
    selected = resolve_version("codex", model_harness=cfg["harness"], models_path=MODELS)
    assert selected.version == "0.156.1"
    assert selected.source == "models.yaml:harness"
    override = resolve_version("codex", model_harness=cfg["harness"], models_path=MODELS,
                               cli_version="0.154.0")
    assert override.version == "0.154.0"


@pytest.mark.parametrize("model,version", [
    ("claude-fable-5-1", "2.1.270"), ("claude-opus-5", "2.1.270"),
    ("gpt-5.6-sol", "0.154.0"), ("gpt-6-astra", "0.154.0"),
    ("grok-4.6", "1.0.30"), ("grok-4.7", "1.0.41"),
    ("claude-opus-5-5", "2.1.281"), ("gpt-6-sol", "0.156.1"),
    ("muse-spark-1.3", "1.3.0-R3401.1"),
])
def test_model_release_pins_do_not_follow_shared_defaults(model, version):
    cfg = run.load_config(MODELS, model)
    selected = resolve_version(cfg["harness"]["name"], model_harness=cfg["harness"], models_path=MODELS,
                               task_config={cfg["harness"]["name"]: {"version": "9.9.9"}})
    assert selected.version == version
    assert selected.source == "models.yaml:harness"


def test_cli_runtime_defaults_and_harness_alias():
    parser = run.build_parser()
    args = parser.parse_args(["chess", "--model", "gpt-6-sol", "--harness", "terminus-2"])
    assert args.agent == "terminus-2"
    assert args.max_turns == 150
    assert args.permission_mode is None
    assert parser.parse_args(["chess", "--model", "gpt-6-sol", "--agent", "codex"]).agent == "codex"
    assert agent_for("gpt-6-sol", max_turns=150).TURN_LIMIT is None


@pytest.mark.parametrize("config", [
    {"reasoning_effort": "high", "effort": "low"},
    {"reasoning_effort": "high", "output_config": {"effort": "low"}},
    {"unknown_sdk_argument": True},
])
def test_conflicting_or_unsupported_api_settings_fail_before_calls(config):
    with pytest.raises((ValueError, TypeError)):
        api_generation("anthropic/test", config)


def test_summary_cannot_leak_into_raw_chat_factory():
    from core.llm_agents import get_llm_agent_class
    with pytest.raises(ValueError, match="harness.options"):
        get_llm_agent_class("openai/test", {"reasoning_summary": "detailed"})
