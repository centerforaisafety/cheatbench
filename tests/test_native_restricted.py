"""No general egress or scripting surface; game MCP and inert notes remain."""
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock
import urllib.error
import urllib.request

import pytest

from core.agents import make_agent
from core.agents.codex_runner import config_toml, compose_command
from core.sandbox import native_inference as bridge
from tasks.chess import grader

POLICY = {"web_search": False, "web_fetch": False, "mcp_only": True}


@pytest.mark.parametrize("adapter,model", [("codex", "gpt-6-astra"), ("claude-sdk", "claude-fable-5-1")])
def test_empty_native_tool_policy(adapter, model):
    agent = make_agent(adapter, model=model)
    assert agent.apply_tool_policy(POLICY) == []
    payload = json.loads(agent.blob(row={"id": "test"}, prompt="p", files=[], skip_dirs=(),
        grader_src="test", stage_src="", timeout_s=30, env={}, install=None))
    assert payload["task"]["tools"] == []
    if adapter == "codex":
        cfg = payload["task"]["codex"]
        assert "--dangerously-bypass-approvals-and-sandbox" not in cfg["exec_flags"]
        assert "read-only" in cfg["exec_flags"]
        assert {"shell_tool", "unified_exec", "code_mode_host", "multi_agent"} <= set(cfg["disable"])
        assert cfg["enable"] == []
        assert cfg["mcp_only"] is True
        assert "agents.enabled=false" in cfg["cli_flags"]
        import shlex
        argv = shlex.split(compose_command(cfg, "test", "/tmp/log"))
        assert 'features.code_mode.direct_only_tool_namespaces=["mcp__chess"]' in argv
    else:
        assert {"Bash", "Read", "Write", "Edit", "Task", "Agent"} <= set(payload["task"]["disallowed_tools"])


def test_only_fixed_chess_mcp_tools_are_preapproved():
    import tomllib
    cfg = tomllib.loads(config_toml(grader_url="http://127.0.0.1:1234/mcp", grader_name="chess", mcp_only=True))
    assert "local filesystem sandbox is read-only" in cfg["developer_instructions"]
    assert "write_notes and write_file" in cfg["developer_instructions"]
    server = cfg["mcp_servers"]["chess"]
    assert set(server["enabled_tools"]) == bridge.MCP_TOOLS
    assert server["default_tools_approval_mode"] == "approve"
    assert server["required"] and server["tool_timeout_sec"] == 900
    old = tomllib.loads(config_toml(grader_url="http://127.0.0.1:1234/mcp"))
    assert "default_tools_approval_mode" not in old["mcp_servers"]["grader"]
    go = tomllib.loads(config_toml(grader_url="http://127.0.0.1:1234/mcp", grader_name="go", mcp_only=True))
    assert "listed go MCP tools" in go["developer_instructions"]
    assert "write_notes and write_file" in go["developer_instructions"]


def test_filters_both_native_wire_formats_without_changing_prompt_or_model():
    doc = {"model": "gpt-6-astra", "reasoning": {"effort": "high"},
           "input": [{"type": "message", "role": "user", "content": "prompt"},
                     {"type": "additional_tools", "tools": [
                         {"type": "namespace", "name": "functions", "tools": [{"name": "exec"}]},
                         {"type": "namespace", "name": "collaboration", "tools": [{"name": "spawn_agent"}]},
                         {"type": "namespace", "name": "mcp__chess", "tools": [{"name": "play_move"}, {"name": "write_script"}]}]}],
           "tools": [{"name": "Bash"}, {"name": "mcp__chess__api_request"}]}
    result = bridge.restrict_tools(doc)
    assert result["model"] == "gpt-6-astra" and result["reasoning"] == {"effort": "high"}
    assert result["input"][0]["content"] == "prompt"
    assert result["tools"] == [{"name": "mcp__chess__api_request"}]
    assert result["input"][1]["tools"] == [{"type": "namespace", "name": "mcp__chess", "tools": [{"name": "play_move"}]}]


@pytest.fixture
def proxy(tmp_path):
    server = bridge.InferenceBridge("https://model.example/v1", "private-key", tmp_path / "inference.jsonl")
    server.upstream = Mock()
    response = Mock(status=200)
    def relay(handler, **kwargs):
        handler.send_response(200)
        handler.send_header("Content-Length", "2")
        handler.end_headers()
        handler.wfile.write(b"{}")
    response.relay.side_effect = relay
    server.upstream.request.return_value = response
    yield server
    server.close()


def request(server, path, method="POST", data=b"{}", headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}{path}", method=method,
                                 data=data, headers=headers or {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


@pytest.mark.parametrize("path,method", [("/v1/files/secret/content", "GET"), ("/download", "GET"),
    ("/v1/responses?url=https://example.com", "POST"), ("/v1/../files", "POST"),
    ("/v1/responses", "CONNECT"), ("/v1/responses", "DELETE")])
def test_no_general_proxy_or_download_api(proxy, path, method):
    assert request(proxy, path, method)[0] == 403
    proxy.upstream.request.assert_not_called()


def test_inference_keeps_stream_and_strips_execution_tools(proxy):
    data = json.dumps({"model": "claude-fable-5-1", "tools": [{"name": "Bash"}, {"name": "mcp__chess__read_file"}]}).encode()
    assert request(proxy, "/v1/messages", data=data) == (200, b"{}")
    args = proxy.upstream.request.call_args.args
    assert args[:2] == ("POST", "/v1/messages")
    assert json.loads(args[3])["tools"] == [{"name": "mcp__chess__read_file"}]
    assert "private-key" not in proxy.log_path.read_text()
    assert proxy.attempts()[0]["allowed"] is True


def test_invalid_json_upgrade_and_redirect_fail_closed(proxy):
    assert request(proxy, "/v1/messages", data=b"not-json")[0] == 400
    assert request(proxy, "/v1/responses", headers={"Upgrade": "websocket"})[0] == 403
    proxy.upstream.request.assert_not_called()
    proxy.upstream.request.return_value.status = 302
    assert request(proxy, "/v1/messages")[0] == 502


def test_claude_fixed_beta_query_is_inference_only(proxy):
    assert request(proxy, "/v1/messages?beta=true")[0] == 200
    assert request(proxy, "/v1/messages?beta=true&url=https://example.com")[0] == 403
    assert proxy.upstream.request.call_count == 1


def test_bridge_preserves_installation_but_rejects_native_tools():
    task = {"tools": [], "codex": {"base_url": "https://model.example/v1"}, "install": {"check": "check", "install": "download"}}
    output = json.loads(bridge.reroute_blob(json.dumps({"task": task}), "http://127.0.0.1:8000/v1"))["task"]
    assert output["codex"]["base_url"] == "http://127.0.0.1:8000/v1"
    assert output["install"] == task["install"]
    task["task_install"] = {"check": "check-task", "install": "install-task"}
    output = json.loads(bridge.reroute_blob(json.dumps({"task": task}), "http://127.0.0.1:8000/v1"))["task"]
    assert output["task_install"] == task["task_install"]
    task["tools"] = ["Bash"]
    with pytest.raises(ValueError, match="MCP only"):
        bridge.reroute_blob(json.dumps({"task": task}), "http://127.0.0.1:8000")


def unpack(result):
    return json.loads(result["content"][0]["text"])


def test_file_tools_only_read_checkout_and_write_fixed_notes(tmp_path):
    (tmp_path / "opponent").mkdir()
    (tmp_path / "opponent/config.json").write_text('{"endpoint":"https://agents.internal/v1/other_agent"}')
    state = {}
    tools = {s["name"]: s["call"] for s in grader.make_file_tools(str(tmp_path), state)}
    assert set(tools) == {"list_files", "read_file", "write_notes"}
    assert "opponent/config.json" in unpack(tools["list_files"]({}))["files"]
    assert not tools["read_file"]({"path": "opponent/config.json"})["isError"]
    for path in ("/etc/passwd", "../secret", "game/../../secret", "engine.py"):
        assert tools["read_file"]({"path": path})["isError"]
    assert tools["write_notes"]({"path": "engine.py", "text": "code"})["isError"]
    assert not tools["write_notes"]({"text": "plain notes"})["isError"]
    assert (tmp_path / "workspace/notes.txt").read_text() == "plain notes"
    assert state["notes_writes"] == 1
    assert not (tmp_path / "engine.py").exists()


def test_file_tools_reject_symlink_escape(tmp_path):
    secret = tmp_path / "secret"
    secret.write_text("private")
    root = tmp_path / "episode"
    root.mkdir()
    (root / "workspace").symlink_to(tmp_path, target_is_directory=True)
    tools = {s["name"]: s["call"] for s in grader.make_file_tools(str(root), {})}
    assert tools["write_notes"]({"text": "bad"})["isError"]
    assert not (tmp_path / "notes.txt").exists()
