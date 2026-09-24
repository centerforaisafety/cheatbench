"""Synthetic HTTP/SSE tests: no model calls or credentials are used."""
import gzip
import http.client
import http.server
import json
import os
import socket
import threading
import time
from contextlib import contextmanager

import pytest

from core.agents import forwarder
from core.agents.claude_sdk_runner import _slim
from core.sandbox.native_inference import InferenceBridge

PING = forwarder._SSE_KEEPALIVE


@contextmanager
def stream_bridge(tmp_path, pieces, *, interval=0.02, status=200,
                  content_type="text/event-stream", terminate=True,
                  content_encoding=None, request_headers=None):
    """pieces are (delay, bytes); delay models an upstream thinking pause."""
    finish = threading.Event()

    class Provider(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            if request_headers is not None:
                request_headers.update(self.headers.items())
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Retry-After", "7")
            if content_encoding:
                self.send_header("Content-Encoding", content_encoding)
            self.end_headers()
            try:
                for delay, part in pieces:
                    if finish.wait(delay):
                        break
                    self.wfile.write(b"%x\r\n" % len(part) + part + b"\r\n")
                    self.wfile.flush()
                if terminate:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
            except OSError:
                pass
            self.close_connection = True

    provider = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=provider.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    bridge = InferenceBridge("https://model.example/v1", "dummy-key", tmp_path / "events.jsonl",
                             keepalive_interval=interval)
    # Only the test substitutes a loopback provider. Production rejects HTTP origins.
    bridge.upstream = forwarder.Upstream(f"http://127.0.0.1:{provider.server_port}", "dummy-key", timeout=900)
    try:
        yield bridge
    finally:
        finish.set()
        bridge.close()
        provider.shutdown()
        provider.server_close()
        thread.join(1)


def begin(bridge, *, timeout=0.1, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", bridge.server_port, timeout=timeout)
    conn.request("POST", "/v1/messages?beta=true", body=b'{"model":"test","stream":true}',
                 headers={"Content-Type": "application/json", **(headers or {})})
    return conn, conn.getresponse()


def completion(bridge):
    for _ in range(100):
        events = bridge.attempts()
        if events and events[-1].get("event") in {"transport_complete", "transport_error"}:
            return events[-1]
        time.sleep(0.01)
    raise AssertionError("relay did not record a terminal transport event")


@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r"])
def test_sse_framing_survives_every_possible_chunk_split(ending):
    event = b'event: message' + ending + 'data: {"text":"♞"}'.encode() + ending * 2
    for split in range(len(event) + 1):
        parser = forwarder._SSEFrames()
        result = parser.feed(event[:split]) + parser.feed(event[split:])
        result += parser.feed(b"", final=True)
        assert result == [event]


def test_partial_and_oversized_events_fail_without_exposing_payload(monkeypatch):
    parser = forwarder._SSEFrames()
    parser.feed(b"data: private incomplete response")
    with pytest.raises(http.client.IncompleteRead) as error:
        parser.feed(b"", final=True)
    assert "private" not in str(error.value)
    monkeypatch.setattr(forwarder, "_MAX_SSE_EVENT", 20)
    with pytest.raises(ValueError, match="buffer limit"):
        forwarder._SSEFrames().feed(b"data: " + b"x" * 21)


def test_keepalive_survives_silence_longer_than_downstream_timeout(tmp_path):
    payload = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    with stream_bridge(tmp_path, [(0.3, payload)]) as bridge:
        conn, response = begin(bridge)
        try:
            received = response.read()
        finally:
            conn.close()
        assert received.replace(PING, b"") == payload
        assert received.count(PING) >= 2
        result = completion(bridge)
        assert result["event"] == "transport_complete"
        assert result["bytes_out"] == len(payload)
        assert result["keepalives"] >= 2
        assert "dummy-key" not in bridge.log_path.read_text()


def test_keepalive_never_splits_provider_json_or_dispatches_partial_event(tmp_path):
    payload = 'data: {"text":"♞ unchanged"}\r\n\r\n'.encode()
    with stream_bridge(tmp_path, [(0, payload[:12]), (0.25, payload[12:])]) as bridge:
        conn, response = begin(bridge)
        try:
            received = response.read()
        finally:
            conn.close()
        assert payload in received
        assert received.replace(PING, b"") == payload
        assert completion(bridge)["event"] == "transport_complete"


@pytest.mark.parametrize("terminate,payload", [
    (False, b"data: incomplete"),
    (True, b"data: incomplete"),
])
def test_interrupted_http_or_sse_is_not_reported_as_success(tmp_path, terminate, payload):
    with stream_bridge(tmp_path, [(0.05, payload)], terminate=terminate) as bridge:
        conn, response = begin(bridge)
        assert response.status == 200  # Headers are not a completed response.
        try:
            with pytest.raises(http.client.IncompleteRead):
                response.read()
        finally:
            conn.close()
        result = completion(bridge)
        assert result["event"] == "transport_error"
        assert result["error_type"] == "IncompleteRead"
        assert not any(e.get("event") == "transport_complete" for e in bridge.attempts())


@pytest.mark.parametrize("status", [403, 429, 503])
def test_upstream_errors_and_retry_headers_are_not_hidden(tmp_path, status):
    payload = b'{"error":{"type":"blocked","message":"provider rejection"}}'
    with stream_bridge(tmp_path, [(0, payload)], status=status, content_type="application/json") as bridge:
        conn, response = begin(bridge)
        try:
            assert response.status == status
            assert response.getheader("Retry-After") == "7"
            assert response.read() == payload
        finally:
            conn.close()
        # Transport completion means bytes arrived, not provider/model success.
        assert completion(bridge)["keepalives"] == 0


def test_provider_safeguard_sse_is_passed_through_unchanged(tmp_path):
    payload = b'event: error\ndata: {"type":"error","error":{"type":"refusal"}}\n\n'
    with stream_bridge(tmp_path, [(0.2, payload)]) as bridge:
        conn, response = begin(bridge)
        try:
            assert response.read().replace(PING, b"") == payload
        finally:
            conn.close()


def test_bridge_requests_identity_encoding_for_keepalives(tmp_path):
    observed = {}
    payload = b"data: unchanged\n\n"
    with stream_bridge(tmp_path, [(0, payload)], request_headers=observed) as bridge:
        conn, response = begin(bridge, headers={"Accept-Encoding": "gzip, br"})
        try:
            assert response.read().replace(PING, b"") == payload
        finally:
            conn.close()
    assert observed["Accept-Encoding"] == "identity"


def test_unexpected_compressed_sse_is_not_corrupted_by_keepalives(tmp_path):
    payload = gzip.compress(b"data: unchanged\n\n")
    with stream_bridge(tmp_path, [(0.1, payload)], content_encoding="gzip") as bridge:
        conn, response = begin(bridge, timeout=1)
        try:
            assert response.getheader("Content-Encoding") == "gzip"
            assert response.read() == payload
        finally:
            conn.close()
        assert completion(bridge)["keepalives"] == 0


def test_downstream_disconnect_releases_the_upstream_reader(tmp_path):
    with stream_bridge(tmp_path, [(20, b"data: never needed\n\n")]) as bridge:
        before = {id(t) for t in threading.enumerate() if t.name == "native-sse-reader"}
        conn, response = begin(bridge)
        assert response.read(len(PING)) == PING
        conn.sock.shutdown(socket.SHUT_RDWR)
        conn.close()
        response.close()
        for _ in range(200):
            readers = {id(t) for t in threading.enumerate() if t.name == "native-sse-reader"}
            if not readers - before:
                break
            time.sleep(0.01)
        assert not readers - before
        assert completion(bridge)["event"] == "transport_error"


def test_retry_telemetry_survives_without_raw_error_or_credentials():
    rec = {"_type": "SystemMessage", "subtype": "api_error", "data": {
        "retryAttempt": 1, "maxRetries": 10, "retryInMs": 582,
        "error": {"headers": {"Authorization": "secret"}}, "body": "private"}}
    assert _slim(rec) == {"_type": "SystemMessage", "subtype": "api_error", "data": {
        "retryAttempt": 1, "maxRetries": 10, "retryInMs": 582}}


@pytest.mark.skipif(os.environ.get("RH_TEST_LONG_STREAM") != "1", reason="explicit 125-second integration check")
def test_real_response_over_120_seconds(tmp_path):
    payload = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    started = time.monotonic()
    with stream_bridge(tmp_path, [(125, payload)], interval=1) as bridge:
        conn, response = begin(bridge, timeout=3)
        try:
            received = response.read()
        finally:
            conn.close()
        assert time.monotonic() - started >= 125
        assert received.replace(PING, b"") == payload
        assert completion(bridge)["keepalives"] >= 100
