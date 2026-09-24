"""Gemini-specific MCP-only handling; no paid model calls."""
import hashlib
import json
import urllib.error
import urllib.request

import pytest

from core.agents.gemini_cli import GeminiCLIAgent
from core.agents import gemini_cli_runner as runner
from core.sandbox import native_inference as bridge
from tests.test_chess_api_tool import rpc
from tests.test_chess_tools import NAMES, SCHEMA_HASH, FIXTURE
from tasks.chess import build, grader

POLICY = {"mcp_only": True, "web_search": False, "web_fetch": False}
WIRE_NAMES = {"mcp_chess_" + name for name in NAMES}


def test_empty_core_allowlist_preserves_same_named_mcp_tools():
    agent = GeminiCLIAgent(model="gemini/test", api_base_url="https://gateway.example/v1")
    assert agent.apply_tool_policy(POLICY) == []
    settings = agent.settings()
    assert settings["tools"]["core"] == []
    assert not set(NAMES) & set(settings["tools"]["exclude"])
    payload = json.loads(agent.blob(row={"id": "test", "tool_surface": "chess"},
        prompt="unchanged", files=[], skip_dirs=(), stage_src="", grader_src="test",
        timeout_s=3600, env={}))
    task = payload["task"]
    assert task["tools"] == [] and task["gemini"]["mcp_only"]
    assert task["row"]["episode_timeout_s"] == 3600
    assert task["content"] == "unchanged"


def test_unrestricted_gemini_keeps_its_native_tools():
    agent = GeminiCLIAgent(model="gemini/test")
    tools = agent.apply_tool_policy({"web_search": False, "web_fetch": False})
    assert "run_shell_command" in tools and "write_file" in tools
    assert "core" not in agent.settings()["tools"]


def test_runtime_allowlist_uses_observed_fully_qualified_mcp_names():
    from types import SimpleNamespace
    server = SimpleNamespace(grader_server_name="chess", grader_tools=dict.fromkeys(NAMES),
                             url="http://127.0.0.1:1/test/mcp")
    original = {"tools": {"core": [], "exclude": ["web_fetch"]}}
    settings = runner.mcp_settings(original, server, mcp_only=True)
    assert set(settings["tools"]["core"]) == WIRE_NAMES
    assert settings["mcpServers"]["chess"]["includeTools"] == NAMES
    assert not set(settings["tools"]["core"]) & set(GeminiCLIAgent.BASE_TOOLS)
    assert original["tools"]["core"] == []
    normal = runner.mcp_settings({}, server)
    assert "tools" not in normal and "includeTools" not in normal["mcpServers"]["chess"]


def test_unverified_direct_route_is_rejected_for_mcp_only():
    agent = GeminiCLIAgent(model="gemini/test", route="native")
    agent.apply_tool_policy(POLICY)
    assert any("gateway" in problem for problem in agent.setup())


def test_gemini_serves_exact_chess_schemas_and_dispatches(tmp_path):
    for rel, content in build.checkout(FIXTURE, "chess").items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    row = {"id": "LR4pA", "tool_surface": "chess", "restricted_tools": True,
           "episode_timeout_s": 300, "token": "test-harness-token"}
    server = runner.serve_grader(vars(grader), row, {}, str(tmp_path))
    try:
        tools = rpc(server, "tools/list")["tools"]
        assert [tool["name"] for tool in tools] == NAMES
        assert hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest() == SCHEMA_HASH
        assert not rpc(server, "tools/call", {"name": "write_notes", "arguments": {"text": "test"}}).get("isError")
        assert "test" in rpc(server, "tools/call", {"name": "read_notes"})["content"][0]["text"]
        for invalid in ("run_shell_command", [], None):
            assert rpc(server, "tools/call", {"name": invalid})["isError"]
    finally:
        server.close()


@pytest.fixture
def shim():
    value = runner.GatewayShim(base_url="https://gateway.example/v1", api_key="test",
        model_id="gemini/test", model_slug="test", exclude_tools=["web_fetch"],
        allowed_tools=WIRE_NAMES)
    yield value
    value.close()


def body(names):
    return {"tools": [{"functionDeclarations": [{"name": name, "parameters": {}}
                                               for name in names]}]}


def test_shim_accepts_only_chess_names_and_keeps_utility_calls(shim):
    shim.observe_request(body(sorted(WIRE_NAMES)), "test", "gemini/test", [])
    assert set(shim.declared_tools) == WIRE_NAMES and not shim.breaches
    shim.observe_request({}, "flash-lite", "gemini/flash-lite", [])


@pytest.mark.parametrize("tool", ["run_shell_command", "read_file", "write_file", "agent", "chess_exec", "other_play_move"])
def test_shim_refuses_unexpected_tools(shim, tool):
    with pytest.raises(ValueError, match="policy breach"):
        shim.observe_request(body([tool]), "test", "gemini/test", [])
    assert shim.breaches


def test_shim_blocks_server_side_web_tools(shim):
    with pytest.raises(ValueError, match="policy breach"):
        shim.observe_request({"tools": [{"googleSearch": {}}]}, "test", "gemini/test", [])


def test_http_policy_breach_never_contacts_upstream(shim, monkeypatch):
    from unittest.mock import Mock
    upstream = Mock(side_effect=AssertionError("must not contact model"))
    shim.start()
    request = urllib.request.Request(shim.url + "/v1beta/models/test:generateContent",
        data=json.dumps(body(["run_shell_command"])).encode(),
        headers={"x-goog-api-key": shim.token, "Content-Type": "application/json"})
    # Use an opener captured before patching the shim's upstream function.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    monkeypatch.setattr(runner.urllib.request, "urlopen", upstream)
    with pytest.raises(urllib.error.HTTPError) as error:
        opener.open(request, timeout=5)
    assert error.value.code == 403
    upstream.assert_not_called()


def test_chat_tools_survive_bridge_without_native_tools():
    allowed = [{"type": "function", "function": {"name": n}} for n in sorted(WIRE_NAMES)]
    doc = {"model": "gemini/test", "messages": [{"role": "user", "content": "unchanged"}],
           "tools": [*allowed, {"type": "function", "function": {"name": "run_shell_command"}}]}
    result = bridge.restrict_tools(doc)
    assert result["tools"] == allowed
    assert result["model"] == "gemini/test" and result["messages"][0]["content"] == "unchanged"
    assert "/v1/chat/completions" in bridge.POST_PATHS
    assert "/v1/chat/completions?url=https://example.com" not in bridge.POST_PATHS


def test_gemini_only_reroute_preserves_model_and_settings():
    task = {"tools": [], "gemini": {"route": "gateway", "mcp_only": True,
            "model_id": "gemini/test", "settings": {"tools": {"core": []}},
            "base_url": "https://gateway.example/v1"}}
    local = "http://127.0.0.1:8000/v1"
    result = json.loads(bridge.reroute_blob(json.dumps({"task": task}), local))["task"]
    assert result["gemini"]["base_url"] == local
    assert result["gemini"]["model_id"] == "gemini/test"
    assert result["gemini"]["settings"] == task["gemini"]["settings"]
    task["gemini"]["mcp_only"] = False
    with pytest.raises(ValueError):
        bridge.reroute_blob(json.dumps({"task": task}), local)
