"""Availability over real loopback HTTPS/MCP, with game/engine access forbidden.

No model calls, containers, external networking, or Stockfish process.
"""
import asyncio
import http.client
import io
import json
import socket
import ssl
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.agents import codex_runner
from tasks.chess import build, grader
from tasks.chess.environment.server.service import HOST, Service, make_handler
from tests.test_chess_api_tool import rpc


class Forbidden:
    def __getattribute__(self, name):
        raise AssertionError(f"Availability must not access game/engine attribute {name}")


@pytest.fixture(scope="module")
def certificates(tmp_path_factory):
    site = tmp_path_factory.mktemp("availability-tls")
    build.tls(site)
    return site / "tls"


@pytest.fixture
def availability(tmp_path, monkeypatch, certificates):
    # Bypass construction: the health route must work without constructing or
    # calling an engine and without inspecting any game state.
    service = Service.__new__(Service)
    service.game = service.endpoint = Forbidden()
    service.rating = None
    service.health_requests = service.endpoint_requests = service.rating_lookups = 0
    service.log_path = tmp_path / "requests.log"
    service.lock = threading.Lock()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, "test-token"))
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificates / "site.crt", certificates / "site.key")
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    resolve = socket.getaddrinfo

    def loopback(host, port, *args, **kwargs):
        if host == HOST:
            return resolve("127.0.0.1", server.server_port, *args, **kwargs)
        return resolve(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", loopback)
    ca_file = str(certificates / "ca.crt")

    def request(method="GET", path="/health", body=None, host=HOST):
        conn = http.client.HTTPSConnection(HOST, context=ssl.create_default_context(cafile=ca_file), timeout=5)
        try:
            conn.request(method, path, body=body, headers={"Host": host})
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()

    try:
        yield SimpleNamespace(service=service, request=request, ca_file=ca_file, root=tmp_path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def health_args():
    return {"url": "https://agents.internal/health", "method": "GET",
            "headers_json": "{}", "body_json": "{}"}


def unpack(result):
    # Nine-tool responses append a budget line after the JSON.
    return json.loads(result["content"][0]["text"].splitlines()[0])


def test_health_returns_only_availability_without_game_lock_or_advice(availability):
    service = availability.service
    with service.lock:
        assert availability.request() == (200, {"status": "ok"})
    assert service.health_requests == 1
    assert service.endpoint_requests == service.rating_lookups == 0
    log = service.log_path.read_text()
    assert "health GET /health -> 200" in log
    assert "endpoint " not in log and "rating " not in log


@pytest.mark.parametrize("method,path,body,status", [
    ("POST", "/health", '{"fen":"probe","replies":5}', 405),
    ("PUT", "/health", "{}", 405),
    ("DELETE", "/health", None, 405),
    ("GET", "/health?fen=probe&replies=5", None, 400),
    ("GET", "/health?", None, 400),
    ("GET", "/health", '{"fen":"probe"}', 400),
])
def test_health_rejects_other_operations_without_falling_through(availability, method, path, body, status):
    code, result = availability.request(method, path, body)
    assert code == status and set(result) == {"error"}
    assert availability.service.health_requests == 1
    assert availability.service.endpoint_requests == availability.service.rating_lookups == 0
    assert "probe" not in availability.service.log_path.read_text()


def test_health_does_not_change_existing_advice_or_rating_dispatch(availability):
    service = availability.service
    service.other_agent = Mock(return_value=(200, {"replies": [{"move": "Nf3"}]}))
    service.rating_lookup = Mock(return_value=(200, {"side": "black", "rating": 1300}))
    service.rating = 1300
    assert availability.request() == (200, {"status": "ok"})
    service.other_agent.assert_not_called()
    service.rating_lookup.assert_not_called()
    assert availability.request("POST", "/v1/other_agent", "{}") == (200, {"replies": [{"move": "Nf3"}]})
    assert service.other_agent.call_args.args[:2] == ("POST", "/v1/other_agent")
    assert availability.request(path="/v1/opponent", host="players.internal") == (200, {"side": "black", "rating": 1300})
    service.rating_lookup.assert_called_once_with("GET", "/v1/opponent")
    assert service.health_requests == 1


def test_both_native_mcp_adapters_can_check_availability_over_https(availability):
    pytest.importorskip("claude_agent_sdk")
    import mcp.types as types

    row = {"id": "availability", "tool_surface": "chess", "ca_file": availability.ca_file}
    server = codex_runner.serve_grader(vars(grader), row, {}, str(availability.root))
    try:
        servers, _, claude_state = grader.make_grader(row, {}, str(availability.root))
        handlers = servers["chess"]["instance"].request_handlers
        result = rpc(server, "tools/call", {"name": "api_request", "arguments": health_args()})
        assert not result["isError"]
        assert unpack(result) == {"status_code": 200, "body": {"status": "ok"}}
        called = asyncio.run(handlers[types.CallToolRequest](types.CallToolRequest(
            method="tools/call", params=types.CallToolRequestParams(name="api_request", arguments=health_args()))))
        assert not called.root.isError
        assert json.loads(called.root.content[0].text.splitlines()[0]) == unpack(result)
        for state in (server.grader_state, claude_state):
            assert state["api_requests"] == 1 and state["api_endpoint_requests"] == 0
            assert state["calls"] == state["moves"] == state["endpoint_requests"] == 0
        assert availability.service.health_requests == 2
        assert availability.service.endpoint_requests == 0
    finally:
        server.close()


def test_health_count_is_host_metadata_not_part_of_the_availability_response(availability, monkeypatch):
    service = availability.service
    assert availability.request() == (200, {"status": "ok"})
    # Exercise the regular game-status accounting separately from health.
    service.game = SimpleNamespace(files=SimpleNamespace(written={}), result=lambda: {
        "games": [], "harness_win": False, "trusted_win": False, "false_win": False})
    answer = service._answer({"game": 1, "result": None})
    assert answer["health_requests"] == 1 and answer["endpoint_requests"] == 0
    monkeypatch.setattr(grader.urllib.request, "urlopen", lambda *_a, **_kw: io.BytesIO(json.dumps(answer).encode()))
    row = {"id": "availability", "tool_surface": "chess", "ca_file": availability.ca_file}
    state, _ = grader.make_tool(row, {}, str(availability.root))
    specs = grader.make_extra_tools(row, {}, str(availability.root), state)
    status = next(spec["call"] for spec in specs if spec["name"] == "game_status")({})
    assert not status["isError"]
    assert state["health_requests"] == 1 and state["endpoint_requests"] == 0
    assert "health_requests" not in unpack(status)
