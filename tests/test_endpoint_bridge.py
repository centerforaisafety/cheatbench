"""core/sandbox/endpoint_bridge.py: which endpoints are bridged, and that both
relay modes carry an HTTP exchange to the remote server unchanged."""
import http.server
import json
import os
import ssl
import threading
import urllib.request

import pytest

from core.sandbox.endpoint_bridge import EndpointBridge, bridge_mode


@pytest.mark.parametrize("url,mcp,egress,expected", [
    ("http://10.0.0.5:8000/v1", True, True, "tls"),
    ("http://10.0.0.5:8000/v1", False, True, "tcp"),
    ("http://10.0.0.5:8000/v1", False, False, None),      # open networking reaches it directly
    ("https://api.openai.com/v1", True, True, None),      # provider endpoints keep their routing
    ("http://127.0.0.1:8000/v1", True, True, None),       # already loopback
    ("http://localhost:8000/v1", False, True, None),
    (None, True, True, None),
])
def test_bridge_mode(url, mcp, egress, expected):
    assert bridge_mode(url, mcp_only=mcp, egress_enforced=egress) == expected


@pytest.fixture
def upstream():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            out = json.dumps({"path": self.path, "auth": self.headers.get("Authorization"),
                              "echo": json.loads(body)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *_):
            pass

    # Bound to a non-loopback-looking name is not possible in a unit test, so the
    # bridge is constructed directly (bridge_mode is what declines loopback).
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def _post(url, context=None):
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps({"x": 1}).encode(),
                                 headers={"Authorization": "Bearer k", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10, context=context) as r:
        return json.loads(r.read())


def test_tcp_relay(upstream):
    with EndpointBridge(upstream, "tcp") as bridge:
        assert bridge.local_base_url.startswith("http://127.0.0.1:")
        assert bridge.local_base_url.endswith("/v1")
        assert _post(bridge.local_base_url) == {"path": "/v1/chat/completions", "auth": "Bearer k", "echo": {"x": 1}}


def test_tls_relay_is_trusted_through_ssl_cert_file(upstream):
    before = os.environ.get("SSL_CERT_FILE")
    with EndpointBridge(upstream, "tls") as bridge:
        assert bridge.local_base_url.startswith("https://127.0.0.1:")
        # A default context, as the inference bridge's Upstream builds it.
        assert _post(bridge.local_base_url, ssl.create_default_context())["echo"] == {"x": 1}
    assert os.environ.get("SSL_CERT_FILE") == before


def test_upstream_url_is_host_only():
    with EndpointBridge("http://user:pw@127.0.0.1:9/v1?x=1", "tcp") as bridge:
        assert bridge.upstream_url == "http://127.0.0.1:9/v1"
