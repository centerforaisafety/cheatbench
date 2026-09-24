"""Native DeepSeek restrictions, shared MCP parity, and offline runtime proof."""
import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

import pytest

from core.agents import deepseek_harness_runner as runner, runner_support, forwarder
from core.agents.deepseek_harness import DeepSeekHarnessAgent
from core.agents.config import resolve_version
from core.sandbox import native_inference as bridge
from tasks.chess import build, grader
from tests.test_chess_api_tool import rpc
from tests.test_chess_tools import NAMES, SCHEMA_HASH, FIXTURE

POLICY = {"mcp_only": True, "web_search": False, "web_fetch": False}
WIRE_NAMES = {"mcp__chess__" + name for name in NAMES}


def declarations(names=WIRE_NAMES):
    return [{"type": "function", "function": {"name": name}} for name in sorted(names)]


def test_restricted_adapter_preserves_native_default_and_prompt():
    agent = DeepSeekHarnessAgent(model="openrouter/deepseek/deepseek-v4-pro")
    assert agent.apply_tool_policy(POLICY) == []
    assert agent.setup() == []
    task = json.loads(agent.blob(row={"id": "test", "tool_surface": "chess"},
        prompt="unchanged", files=[], skip_dirs=(), stage_src="", grader_src="test",
        timeout_s=3600, env={}))["task"]
    assert task["tools"] == [] and task["deepseek_harness"]["mcp_only"]
    assert task["row"]["episode_timeout_s"] == 3600
    assert task["content"] == "unchanged"
    assert "bash" in agent.apply_tool_policy({"web_search": False, "web_fetch": False})
    assert resolve_version("deepseek-harness").version == "0.1.5rc1"


def test_profile_keeps_full_native_sdk_and_removes_action_plugins():
    restricted = runner.profile_patch(mcp_only=True)
    disabled = {p["id"] for p in restricted if p.get("disabled")}
    assert {"tool-bash", "tool-fs", "tool-workflow", "tool-subagent", "tool-web", "plan-mode"} <= disabled
    assert not disabled & {"agent", "agent-loop", "llm-retry", "compaction-basic", "system-prompt", "agent-instructions", "sdk-jsonrpc-server"}
    assert not any(p["id"] == "tool-bash" for p in runner.profile_patch())


def test_shared_mcp_serves_identical_nine_schemas(tmp_path):
    for relative, text in build.checkout(FIXTURE, "chess").items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    server = runner_support.serve_grader(vars(grader),
        {"id": "LR4pA", "tool_surface": "chess", "restricted_tools": True,
         "episode_timeout_s": 300, "token": "test-only"}, {}, str(tmp_path))
    try:
        tools = rpc(server, "tools/list")["tools"]
        assert [tool["name"] for tool in tools] == NAMES
        assert hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest() == SCHEMA_HASH
        assert not rpc(server, "tools/call", {"name": "write_notes", "arguments": {"text": "proof"}}).get("isError")
        assert "proof" in rpc(server, "tools/call", {"name": "read_notes"})["content"][0]["text"]
        for name in ("bash", "write", [], None):
            assert rpc(server, "tools/call", {"name": name})["isError"]
    finally:
        server.close()


@pytest.mark.parametrize("bad", [None, [], "bash", {"type": "web_search"},
    {"type": "function", "function": {"name": "bash"}},
    {"type": "function", "function": {"name": "mcp__other__read_file"}}])
def test_bad_model_tool_declarations_are_rejected(bad):
    with pytest.raises(ValueError, match="policy breach"):
        runner.filter_request({"tools": [bad]}, WIRE_NAMES)


def test_complete_roster_and_no_tool_compaction_pass_unchanged():
    doc = {"model": "unchanged", "messages": [{"role": "user", "content": "unchanged"}],
           "reasoning_effort": "high", "tools": declarations()}
    assert runner.filter_request(doc, WIRE_NAMES) is doc
    assert runner.filter_request({"messages": []}, WIRE_NAMES) == {"messages": []}
    for bad in (declarations()[:-1], declarations() + declarations()[:1]):
        with pytest.raises(ValueError, match="roster"):
            runner.filter_request({"tools": bad}, WIRE_NAMES)


def test_gateway_refuses_native_tools_before_inference():
    upstream = Mock(path="/v1")
    server = runner.Gateway(upstream, allowed_tools=WIRE_NAMES)
    try:
        request = urllib.request.Request(server.url + "/chat/completions",
            data=json.dumps({"tools": declarations({"bash"})}).encode(),
            headers={"Authorization": "Bearer " + server.token, "Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=5)
        assert error.value.code == 403
        upstream.request.assert_not_called()
    finally:
        server.close()


def test_deepseek_wire_names_survive_mac_bridge():
    expected = declarations()
    doc = {"tools": expected + declarations({"bash", "write", "run_code"}), "model": "unchanged"}
    assert bridge.restrict_tools(doc)["tools"] == expected
    task = {"tools": [], "content": "unchanged", "deepseek_harness": {
        "mcp_only": True, "base_url": "https://model.example/v1", "generation": {"reasoning_effort": "high"}}}
    local = "http://127.0.0.1:8000/v1"
    routed = json.loads(bridge.reroute_blob(json.dumps({"task": task}), local))["task"]
    assert routed["deepseek_harness"]["base_url"] == local
    assert routed["content"] == task["content"]
    assert routed["deepseek_harness"]["generation"] == task["deepseek_harness"]["generation"]
    task["deepseek_harness"]["mcp_only"] = False
    with pytest.raises(ValueError):
        bridge.reroute_blob(json.dumps({"task": task}), local)


@pytest.mark.skipif(os.environ.get("DSH_RUNTIME_TESTS") != "1", reason="opt-in native runtime with mock API")
@pytest.mark.parametrize("attempt", ["mcp", "bash", "write", "subagent", "workflow", "run_code"])
def test_real_sdk_has_only_nine_tools_and_blocks_unregistered_calls(tmp_path, monkeypatch, attempt):
    pytest.importorskip("deepseek_harness")
    requests, seen = [], []
    marker = tmp_path / "forbidden-native-write"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            if len(requests) == 1:
                name = "mcp__chess__read_notes" if attempt == "mcp" else attempt
                args = {} if attempt == "mcp" else {"command": f"touch {marker}", "description": "test",
                    "file_path": str(marker), "content": "forbidden", "prompt": "write a file", "code": "1+1"}
                delta = {"role": "assistant", "tool_calls": [{"index": 0, "id": "test-call", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)}}]}
                finish = "tool_calls"
            else:
                delta, finish = {"role": "assistant", "content": "Done."}, "stop"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for event in ({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                          {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                           "usage": {"prompt_tokens": 100, "completion_tokens": 20}}):
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")

    mock = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=mock.serve_forever, daemon=True).start()

    def handler(arguments):
        seen.append(arguments)
        return {"content": [{"type": "text", "text": "notes proof"}]}

    server = runner_support.GraderServer(state={}, call=handler, server_name="chess",
        tool_name=NAMES[0], description="test", schema={"type": "object", "properties": {}},
        extra_tools=[{"name": n, "description": "test", "inputSchema": {"type": "object", "properties": {}},
                      "call": handler} for n in NAMES[1:]])
    server.start()
    monkeypatch.setattr(runner, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("DSH_MOCK_KEY", "mock-only")
    task = {"id": "test", "model": "openrouter/deepseek/deepseek-v4-pro", "tools": [],
        "content": "Use the available tools.", "timeout_s": 30,
        "deepseek_harness": {"api_key_env": "DSH_MOCK_KEY", "mcp_only": True,
            "base_url": f"http://127.0.0.1:{mock.server_port}/v1", "generation": {"reasoning_effort": "high"}, "max_turns": 3}}
    support = {key: getattr(runner_support, key) for key in ("task_install_env", "collect_deliverables")}
    support["_emit_msg"] = lambda event: None
    try:
        record = runner.run(task, support, {"Upstream": forwarder.Upstream}, server)
    finally:
        server.close()
        mock.shutdown()
        mock.server_close()
    assert record["error"] is None, record["error"]
    assert record["runtime_profile"] == "sdk"
    assert record["terminal_reason"] == "completed"
    assert set(record["init_tools"]) == WIRE_NAMES
    assert requests and all({t["function"]["name"] for t in req["tools"]} == WIRE_NAMES for req in requests if req.get("tools"))
    assert not marker.exists()
    assert bool(seen) == (attempt == "mcp")
    if attempt != "mcp":
        results = [e for e in record["messages"] if e["type"] == "tool/result"]
        assert results and "error" in json.dumps(results).lower()
