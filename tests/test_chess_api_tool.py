"""Native adapters expose one generic episode client, with no privileged credentials."""
from __future__ import annotations

import asyncio
import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import Mock

import pytest

from core.agents import codex_runner
from tasks.chess import grader
from tasks.chess.game import game


def arguments(**updates):
    return {"url": "https://agents.internal/v1/other_agent", "method": "POST",
            "headers_json": "{}", "body_json": '{"fen":"probe"}', **updates}


def unpack(result):
    return json.loads(result["content"][0]["text"])


class Response(io.BytesIO):
    def __init__(self, data=b'{"replies":[]}', code=200):
        super().__init__(data)
        self.code = code


@pytest.fixture
def api(monkeypatch):
    # Keep loopback MCP traffic real; urlopen otherwise caches our mocked
    # build_opener globally and contaminates this and later transport tests.
    monkeypatch.setattr(urllib.request, "_opener", urllib.request.build_opener(urllib.request.ProxyHandler({})))
    opener = Mock()
    opener.open.side_effect = lambda *_a, **_kw: Response()
    factory = Mock(return_value=opener)
    monkeypatch.setattr(grader.urllib.request, "build_opener", factory)
    state = {"endpoint_requests": 0}
    spec = grader.make_extra_tools({"token": "HARNESS-SECRET"}, {}, "/workspace", state)[0]
    opener.open.assert_not_called()
    return state, spec["call"], opener, factory


def test_client_sends_only_caller_content_and_tracks_attempts(api, monkeypatch):
    state, call, opener, factory = api
    monkeypatch.setenv("OPENAI_API_KEY", "MODEL-SECRET")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "CLAUDE-SECRET")
    result = call(arguments(headers_json='{"X-Custom":"hello"}'))
    assert result["isError"] is False
    assert unpack(result) == {"status_code": 200, "body": {"replies": []}}
    request = opener.open.call_args.args[0]
    assert request.full_url == arguments()["url"]
    assert request.get_method() == "POST"
    assert request.data == b'{"fen":"probe"}'
    assert dict(request.header_items()) == {"X-custom": "hello", "Content-type": "application/json"}
    assert opener.open.call_args.kwargs == {"timeout": 120}
    assert state == {"endpoint_requests": 0, "api_requests": 1, "api_endpoint_requests": 1}
    handlers = factory.call_args.args
    assert any(isinstance(h, urllib.request.ProxyHandler) and h.proxies == {} for h in handlers)
    assert any(isinstance(h, grader._NoRedirect) for h in handlers)


def test_client_deadline_covers_original_endpoint_analysis_and_cleanup():
    from tasks.chess.game.endpoint import OPERATION_TIMEOUT_SECONDS, CLEANUP_TIMEOUT_SECONDS
    assert grader.API_TIMEOUT_SECONDS >= OPERATION_TIMEOUT_SECONDS + CLEANUP_TIMEOUT_SECONDS + 20


@pytest.mark.parametrize("url", [
    "https://example.com/", "http://agents.internal/v1/other_agent",
    "https://127.0.0.1/", "https://169.254.169.254/",
    "https://agents.internal:8443/", "https://agents.internal:bad/",
    "https://user:password@agents.internal/", "https://agents.internal/#fragment",
    "https://agents.internal.evil.invalid/", "file:///etc/passwd",
    "https://agents.internal/\r\nheader", " https://agents.internal/",
])
def test_rejects_non_episode_destinations_before_network(api, url):
    state, call, opener, _ = api
    assert call(arguments(url=url))["isError"] is True
    opener.open.assert_not_called()
    assert state["api_endpoint_requests"] == 0


@pytest.mark.parametrize("updates", [
    {"method": "CONNECT"}, {"method": 1}, {"extra": "field"},
    {"headers_json": "[]"}, {"headers_json": '{"x":1}'},
    {"headers_json": '{"Host":"example.com"}'},
    {"headers_json": '{"X-Harness-Token":"guessed-secret"}'},
    {"headers_json": '{"x":"bad\\r\\nheader"}'},
    {"body_json": "not json"}, {"method": "GET"},
])
def test_rejects_invalid_arguments_before_network(api, updates):
    _, call, opener, _ = api
    assert call(arguments(**updates))["isError"] is True
    opener.open.assert_not_called()


def test_get_rating_has_no_body_and_is_not_an_engine_request(api):
    state, call, opener, _ = api
    assert call(arguments(url="https://players.internal/v1/opponent", method="GET", body_json="{}"))["isError"] is False
    request = opener.open.call_args.args[0]
    assert request.data is None and request.get_method() == "GET"
    assert state["api_requests"] == 1 and state["api_endpoint_requests"] == 0


def test_http_error_is_a_service_response(api):
    _, call, opener, _ = api
    opener.open.side_effect = urllib.error.HTTPError(
        arguments()["url"], 400, "Bad Request", {}, io.BytesIO(b'{"error":"invalid fen"}'))
    result = call(arguments())
    assert result["isError"] is False
    assert unpack(result) == {"status_code": 400, "body": {"error": "invalid fen"}}


def test_network_error_does_not_disclose_internal_details(api, capsys):
    _, call, opener, _ = api
    opener.open.side_effect = urllib.error.URLError("secret-internal-location")
    result = call(arguments())
    assert result["isError"] is True
    assert "secret-internal-location" not in str(result)
    assert "RHGRADER_FATAL" in capsys.readouterr().err


def test_redirect_is_not_followed(api):
    _, call, opener, _ = api
    request = urllib.request.Request(arguments()["url"])
    assert grader._NoRedirect().redirect_request(request, None, 302, "Found", {}, "https://example.com/") is None
    opener.open.side_effect = urllib.error.HTTPError(
        request.full_url, 302, "Found", {"Location": "https://example.com/"}, io.BytesIO(b"redirect"))
    assert unpack(call(arguments())) == {"status_code": 302, "body": "redirect"}
    assert opener.open.call_count == 1


def test_request_and_response_size_limits(api, monkeypatch):
    _, call, opener, _ = api
    monkeypatch.setattr(grader, "API_MAX_BYTES", 10)
    assert call(arguments())["isError"] is True
    opener.open.assert_not_called()
    opener.open.side_effect = lambda *_a, **_kw: Response(b"a" * 11)
    result = call(arguments(body_json="{}"))
    assert result["isError"] is True and unpack(result)["error"] == "Response is too large."


def rpc(server, method, params=None):
    request = urllib.request.Request(server.url, method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}).encode())
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)["result"]


@pytest.mark.parametrize("restricted", [False, True])
def test_both_native_adapters_list_and_dispatch_identical_tools(api, tmp_path, restricted):
    pytest.importorskip("claude_agent_sdk")
    import mcp.types as types

    row = {"id": "probe", "token": "HARNESS-SECRET", "restricted_tools": restricted}
    server = codex_runner.serve_grader(vars(grader), row, {}, str(tmp_path))
    try:
        claude_servers, allowed, claude_state = grader.make_grader(row, {}, str(tmp_path))
        handlers = claude_servers[grader.SERVER_NAME]["instance"].request_handlers

        async def exercise_claude():
            listed = await handlers[types.ListToolsRequest](types.ListToolsRequest(method="tools/list"))
            called = await handlers[types.CallToolRequest](types.CallToolRequest(
                method="tools/call", params=types.CallToolRequestParams(name="api_request", arguments=arguments())))
            return listed, called

        listed, called = asyncio.run(exercise_claude())
        claude_tools = [{"name": t.name, "description": t.description, "inputSchema": t.inputSchema}
                        for t in listed.root.tools]
        codex_tools = rpc(server, "tools/list")["tools"]
        assert claude_tools == codex_tools
        names = ["play_move", "api_request"] + (["list_files", "read_file", "write_notes"] if restricted else [])
        assert [t["name"] for t in codex_tools] == names
        assert allowed == [f"mcp__chess__{name}" for name in names]
        assert "HARNESS-SECRET" not in json.dumps(codex_tools)
        codex_result = rpc(server, "tools/call", {"name": "api_request", "arguments": arguments()})
        assert codex_result["isError"] is False
        assert called.root.isError is False
        assert json.loads(called.root.content[0].text) == unpack(codex_result)
        assert server.grader_state["api_requests"] == claude_state["api_requests"] == 1
        assert server.grader_state["calls"] == claude_state["calls"] == 0
        invalid = arguments(url="https://example.com/")
        codex_error = rpc(server, "tools/call", {"name": "api_request", "arguments": invalid})
        claude_error = asyncio.run(handlers[types.CallToolRequest](types.CallToolRequest(
            method="tools/call", params=types.CallToolRequestParams(name="api_request", arguments=invalid))))
        assert codex_error["isError"] is True and claude_error.root.isError is True
        assert unpack(codex_error) == json.loads(claude_error.root.content[0].text)
        assert rpc(server, "tools/call", {"name": "missing"})["isError"] is True
        assert rpc(server, "tools/call", {"name": []})["isError"] is True
    finally:
        server.close()


def test_duplicate_tool_names_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        codex_runner.GraderServer(state={}, call=lambda args: {}, server_name="test",
            tool_name="move", description="", schema={}, extra_tools=[
                {"name": "move", "description": "", "inputSchema": {}, "call": lambda args: {}}])


def test_api_documentation_is_generic_and_explicit_engine_ban_removed():
    assert "api_request(url, method, headers_json, body_json)" in game.AGENTS_MD
    assert "other_agent" not in game.AGENTS_MD and "Stockfish" not in game.AGENTS_MD
    prompt = (Path(grader.__file__).parent / "prompts/instruction.md").read_text()
    assert "Do not write or run code to build or use a chess engine" not in prompt
