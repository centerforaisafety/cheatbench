"""Native Responses survive the Muse catalog proxy and shared HTTP relay.

Local HTTP peers exercise the actual wire, including fragmented SSE. These
tests verify transport compatibility, not availability of a live Meta model.
"""
from __future__ import annotations

import http.client
import http.server
import json
import shlex
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run as run_py  # noqa: E402
from core.agents import make_agent, forwarder  # noqa: E402
from core.agents import muse_code_runner as runner  # noqa: E402

MODEL = "meta/muse-spark-1.3"
CLI_TOKEN = "episode-test-token"
GATEWAY_KEY = "gateway-test-key"


@pytest.fixture
def gateway():
    state = SimpleNamespace(seen=[], status=200, body=b'{"object":"response"}',
                            content_type="application/json", release=None,
                            first=b"", truncate=False)

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            state.seen.append({
                "path": self.path, "headers": dict(self.headers),
                "body": self.rfile.read(int(self.headers["Content-Length"]))})
            self.send_response(state.status)
            self.send_header("Content-Type", state.content_type)
            self.send_header("x-request-id", "native-request-1")
            self.send_header("Retry-After", "7")
            if state.release is None:
                self.send_header("Content-Length", str(len(state.body)))
            else:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            if state.release is None:
                self.wfile.write(state.body)
                return
            self.wfile.write(b"%x\r\n" % len(state.first) + state.first + b"\r\n")
            self.wfile.flush()
            if not state.release.wait(10) or state.truncate:
                self.close_connection = True
                return
            self.wfile.write(b"%x\r\n" % len(state.body) + state.body + b"\r\n0\r\n\r\n")
            self.wfile.flush()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        if state.release is not None:
            state.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def proxy(gateway):
    shim = runner.start_shim(
        vars(forwarder), {"base_url": gateway.url + "/deployment/v1",
                          "model_slug": MODEL}, GATEWAY_KEY, CLI_TOKEN)
    try:
        yield shim
    finally:
        if gateway.release is not None:
            gateway.release.set()
        shim.close()


def request(url, body=b"{}", *, path="/responses", method="POST", token=CLI_TOKEN):
    parts = urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=5)
    conn.request(method, path, body, {
        "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        "Accept": "text/event-stream", "OpenAI-Beta": "responses=v1"})
    return conn, conn.getresponse()


def test_native_request_preserves_reasoning_images_namespaces_and_history(gateway, proxy):
    raw = json.dumps({
        "model": MODEL, "stream": True,
        "reasoning": {"effort": "high", "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
        "input": [
            {"role": "user", "content": [
                {"type": "input_text", "text": "Read this image"},
                {"type": "input_image", "image_url": "data:image/png;base64,AA=="}]},
            {"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque",
             "summary": [{"type": "summary_text", "text": "Prior reasoning"}]},
            {"type": "function_call", "name": "bash", "namespace": "muse",
             "call_id": "call_1", "arguments": '{"command":"pwd && ls"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": "OK"}],
        "tools": [{"type": "namespace", "name": "muse", "tools": [
            {"type": "function", "name": "bash", "parameters": {"type": "object"}}]}],
        "parallel_tool_calls": True, "provider_extension": {"future": "preserved"},
    }, indent=2).encode()
    conn, resp = request(proxy.url, raw, path="/responses?trace=test")
    try:
        assert resp.status == 200
        assert resp.read() == gateway.body
    finally:
        conn.close()
    seen = gateway.seen[0]
    assert seen["path"] == "/deployment/v1/responses?trace=test"
    assert seen["body"] == raw  # No conversion, sampling knobs or output cap.
    assert seen["headers"]["Authorization"] == f"Bearer {GATEWAY_KEY}"
    assert seen["headers"]["OpenAI-Beta"] == "responses=v1"
    assert CLI_TOKEN not in json.dumps(seen["headers"])
    assert CLI_TOKEN.encode() not in seen["body"]
    assert GATEWAY_KEY not in json.dumps(proxy.calls)
    assert "Prior reasoning" not in json.dumps(proxy.calls)


@pytest.mark.parametrize("status", [200, 400, 429])
def test_native_json_and_provider_errors_keep_status_headers_and_body(gateway, proxy, status):
    gateway.status = status
    gateway.body = (b'{"error":{"type":"provider_error","message":"route unavailable"}}'
                    if status != 200 else
                    b'{"output":[{"type":"reasoning","content":[{"text":"native"}]}]}')
    conn, resp = request(proxy.url, b'{"stream":false}')
    try:
        assert resp.status == status
        assert resp.getheader("Content-Type") == "application/json"
        assert resp.getheader("x-request-id") == "native-request-1"
        assert resp.getheader("Retry-After") == "7"
        assert resp.read() == gateway.body
    finally:
        conn.close()


@pytest.mark.parametrize("proxy_kind", ["muse", "shared"])
def test_native_sse_fragments_arrive_before_completion(gateway, proxy_kind):
    gateway.content_type = "text/event-stream"
    gateway.first = b'event: response.reasoning_text.delta\ndata: {"type":"response.reasoning_'
    gateway.body = (
        b'text.delta","delta":"native reasoning"}\n\n'
        b'event: response.function_call_arguments.delta\ndata: {"delta":"pwd && ls"}\n\n'
        b'event: response.completed\ndata: {"response":{"usage":{"output_tokens":8}}}\n\n')
    gateway.release = threading.Event()
    if proxy_kind == "muse":
        proxy = runner.start_shim(vars(forwarder),
                                  {"base_url": gateway.url, "model_slug": MODEL},
                                  GATEWAY_KEY, CLI_TOKEN)
    else:
        proxy = forwarder.Forwarder(base_url=gateway.url, api_key=GATEWAY_KEY,
                                    log=lambda _: None).start()
    conn, resp = request(proxy.url)
    try:
        assert resp.status == 200
        assert resp.getheader("Content-Type") == "text/event-stream"
        # Upstream blocks until this succeeds. A buffered relay times out.
        assert resp.read(len(gateway.first)) == gateway.first
        gateway.release.set()
        assert resp.read() == gateway.body
    finally:
        gateway.release.set()
        conn.close()
        proxy.close()


def test_truncated_stream_is_not_replaced_by_a_success_event(gateway, proxy):
    gateway.first = b'data: {"type":"response.created"}\n\n'
    gateway.release = threading.Event()
    gateway.truncate = True
    conn, resp = request(proxy.url)
    try:
        assert resp.read(len(gateway.first)) == gateway.first
        gateway.release.set()
        with pytest.raises(http.client.IncompleteRead):
            resp.read()
    finally:
        conn.close()


def test_catalog_requires_episode_token_and_lists_configured_id(gateway, proxy):
    for token, status in [("wrong", 401), (CLI_TOKEN, 200)]:
        conn, resp = request(proxy.url, method="GET", path="/muse-code/models", token=token)
        try:
            assert resp.status == status
            doc = json.loads(resp.read())
            if status == 200:
                assert [m["id"] for m in doc["data"]] == [MODEL]
        finally:
            conn.close()
    assert gateway.seen == []


def test_alternate_yaml_controls_route_model_effort_and_extra_body(tmp_path, monkeypatch, gateway):
    monkeypatch.setenv("MUSE_TEST_GATEWAY", gateway.url + "/custom/v1/")
    path = tmp_path / "alternate.yaml"
    path.write_text(yaml.safe_dump({"models": {"experiment": {
        "model": "meta/custom-deployment", "api_key_env": "MUSE_TEST_KEY",
        "api_base_url": "${MUSE_TEST_GATEWAY}",
        "generation_config": {"reasoning_effort": "medium"},
        "extra_body": {"reasoning": {"summary": "detailed"}, "metadata": {"run": "test"}},
    }}}))
    cfg = run_py.load_config(path, "experiment")
    cfg.pop("harness", None)
    cfg.pop("harness_options", None)
    agent = make_agent("muse-code", max_turns=3, **cfg)
    agent.apply_tool_policy({"web_search": False, "web_fetch": False})
    assert agent.setup() == []
    assert agent.api_key_env == "MUSE_TEST_KEY"
    payload = json.loads(agent.blob(row={"id": "test"}, prompt="p", files=[],
                                   skip_dirs=(), stage_src="", timeout_s=60, env={}))
    muse = payload["task"]["muse"]
    command = runner.compose_command(muse, "/tmp/prompt", "/tmp/log")
    assert "--model meta/custom-deployment" in command
    assert "--reasoning-effort medium" in command
    assert "--max-model-steps 3" in command
    # Exercise the same stdlib source delivered to the container.
    ns = {"__name__": "_test_forwarder"}
    exec(payload["modules"]["forwarder"], ns)
    proxy = runner.start_shim(ns, muse, GATEWAY_KEY, CLI_TOKEN)
    conn, resp = request(proxy.url, json.dumps({
        "model": muse["model_slug"], "reasoning": {"effort": "medium", "summary": "auto"},
        "max_output_tokens": 321, "metadata": {"cli": "kept"}}).encode())
    try:
        assert resp.status == 200
        resp.read()
        assert gateway.seen[0]["path"] == "/custom/v1/responses"
        assert json.loads(gateway.seen[0]["body"]) == {
            "model": "meta/custom-deployment",
            "reasoning": {"effort": "medium", "summary": "detailed"},
            "max_output_tokens": 321, "metadata": {"cli": "kept", "run": "test"}}
    finally:
        conn.close()
        proxy.close()


@pytest.mark.parametrize("base_url", ["", "https://api.meta.ai/v1"])
def test_native_body_overrides_use_proxy_with_bare_model_and_one_base_url(
        tmp_path, monkeypatch, gateway, base_url):
    monkeypatch.delenv("MUSE_BASE_URL", raising=False)
    monkeypatch.setenv("META_API_KEY", GATEWAY_KEY)
    agent = make_agent("muse-code", model=MODEL, api_base_url=base_url,
                       extra_body={"reasoning": {"summary": "detailed"}})
    agent.apply_tool_policy({"web_search": False, "web_fetch": False})
    payload = json.loads(agent.blob(row={"id": "test"}, prompt="p", files=[],
                                   skip_dirs=(), stage_src="", timeout_s=60, env={}))
    cfg = payload["task"]["muse"]
    assert cfg["route"] == "native"
    assert cfg["model_slug"] == "muse-spark-1.3"
    assert cfg["base_url"] == "https://api.meta.ai/v1"
    assert "--base-url" not in cfg["cli_flags"]
    for name in ("data_home", "config_home", "prompt_dir"):
        cfg[name] = str(tmp_path / name)
    ns = {"__name__": "_test_native_forwarder"}
    exec(payload["modules"]["forwarder"], ns)
    upstream_class = ns["Upstream"]

    def local_upstream(base, key, extra, timeout):
        assert base == "https://api.meta.ai"
        assert key == GATEWAY_KEY
        return upstream_class(gateway.url, key, extra, timeout=timeout)

    ns["Upstream"] = local_upstream

    def fake_cli(command, env, timeout_s):
        args = shlex.split(command)
        assert args.count("--base-url") == 1
        assert env["META_API_KEY"] != GATEWAY_KEY
        conn, resp = request(args[args.index("--base-url") + 1],
                             b'{"model":"muse-spark-1.3","reasoning":{"effort":"high"}}',
                             token=env["META_API_KEY"])
        try:
            assert resp.status == 200
            resp.read()
        finally:
            conn.close()
        session = Path(cfg["data_home"]) / "muse/sessions/2026/09/11/test/session.jsonl"
        session.parent.mkdir(parents=True)
        session.write_bytes((ROOT / "tests/fixtures/muse_code_session.jsonl").read_bytes())
        return 0, None

    monkeypatch.setattr(runner, "run_muse", fake_cli)
    monkeypatch.setattr(runner, "WORKDIR", str(tmp_path))
    record = runner.run(payload["task"], None, None, forwarder_ns=ns)
    assert record["error"] is None
    assert gateway.seen[0]["path"] == "/v1/responses"
    assert json.loads(gateway.seen[0]["body"]) == {
        "model": "muse-spark-1.3", "reasoning": {"effort": "high", "summary": "detailed"}}
