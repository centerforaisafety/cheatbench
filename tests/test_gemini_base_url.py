"""Gemini upstream configuration reaches both client implementations."""
import pytest


@pytest.mark.parametrize("explicit,env,expected", [
    ("https://explicit.example/v1", "https://env.example/v1", "https://explicit.example/v1"),
    (None, "https://env.example/v1", "https://env.example/v1"),
    (None, None, "https://generativelanguage.googleapis.com/v1beta/openai/"),
])
def test_sdk_url_precedence(monkeypatch, explicit, env, expected):
    from core.llm_agents import GeminiAgent, OpenAIAgent
    if env:
        monkeypatch.setenv("GEMINI_BASE_URL", env)
    else:
        monkeypatch.delenv("GEMINI_BASE_URL", raising=False)
    captured = {}
    monkeypatch.setattr(OpenAIAgent, "__init__", lambda self, **kw: captured.update(kw))
    GeminiAgent("gemini-test", api_base_url=explicit)
    assert captured["api_base_url"] == expected
    assert captured["api_key_env"] == "GEMINI_API_KEY"


@pytest.mark.parametrize("base,expected", [
    ("https://gateway.example", "https://gateway.example/v1/chat/completions"),
    ("https://gateway.example/v1/", "https://gateway.example/v1/chat/completions"),
    ("https://generativelanguage.googleapis.com/v1beta/openai/",
     "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"),
])
def test_shim_preserves_explicit_api_path(monkeypatch, base, expected):
    from core.agents import gemini_cli_runner as runner
    monkeypatch.setattr(runner._LoopbackServer, "__init__", lambda *args: None)
    shim = runner.GatewayShim(base_url=base, api_key="test", model_id="test",
                              model_slug="test", exclude_tools=[])
    assert shim.upstream_url == expected


@pytest.mark.parametrize("route,default_key", [
    ("gateway", "OPENAI_API_KEY"), ("native", "GEMINI_API_KEY"),
])
def test_cli_preserves_configured_credential(monkeypatch, route, default_key):
    import json
    from core.agents.gemini_cli import GeminiCLIAgent
    monkeypatch.setenv("GEMINI_CLI_ROUTE", route)
    plain = GeminiCLIAgent(model="gemini/test")
    assert plain.API_KEY_ENV == default_key
    agent = GeminiCLIAgent(model="gemini/test", api_key_env="CUSTOM_GATEWAY_KEY",
                           api_base_url="https://gateway.example/v1")
    agent.apply_tool_policy({"web_search": False, "web_fetch": False})
    payload = json.loads(agent.blob(row={"id": "test"}, prompt="test", files=[],
        stage_src="", timeout_s=60, skip_dirs=[], env={}))
    assert agent.API_KEY_ENV == "CUSTOM_GATEWAY_KEY"
    assert agent.container_key_envs() == ("CUSTOM_GATEWAY_KEY",)
    assert payload["task"]["gemini"]["api_key_env"] == "CUSTOM_GATEWAY_KEY"


def test_shipped_gemini_entry_resolves_url_from_env(monkeypatch):
    from pathlib import Path
    import run
    monkeypatch.setenv("GEMINI_BASE_URL", "https://gateway.example/v1")
    cfg = run.load_config(Path(__file__).resolve().parents[1] / "configs/models.yaml",
                          "gemini-3.8-flash")
    assert cfg["api_base_url"] == "https://gateway.example/v1"
    assert cfg["api_key_env"] == "LITELLM_API_KEY"
    monkeypatch.delenv("GEMINI_BASE_URL")
    with pytest.raises(SystemExit, match="GEMINI_BASE_URL"):
        run.load_config(Path(__file__).resolve().parents[1] / "configs/models.yaml",
                        "gemini-3.8-flash")


@pytest.mark.parametrize("key_env", ["CUSTOM_GATEWAY_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"])
def test_gateway_cli_receives_only_shim_credential(monkeypatch, key_env):
    from types import SimpleNamespace
    from core.agents import gemini_cli_runner as runner
    monkeypatch.setenv(key_env, "test-upstream-secret")
    shim = SimpleNamespace(url="http://127.0.0.1:1234", token="test-shim-token",
                           upstream_url="https://gateway.example/v1/chat/completions",
                           model_id="test-model", start=lambda: None)
    monkeypatch.setattr(runner, "GatewayShim", lambda **kwargs: shim)
    monkeypatch.setattr(runner, "write_settings", lambda *args: None)
    monkeypatch.setattr(runner, "compose_command", lambda *args: "test-cli")

    class Captured(Exception):
        pass

    def capture(command, env, timeout):
        assert env["GEMINI_API_KEY"] == "test-shim-token"
        assert env["GOOGLE_GEMINI_BASE_URL"] == shim.url
        assert "test-upstream-secret" not in env.values()
        raise Captured

    monkeypatch.setattr(runner, "run_cli", capture)
    with pytest.raises(Captured):
        runner.run({"content": "test", "gemini": {
            "route": "gateway", "api_key_env": key_env,
            "base_url": "https://gateway.example/v1"}}, install=None)
