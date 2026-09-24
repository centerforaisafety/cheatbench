"""A loopback HTTP forwarder for an agent CLI that sends its own requests.

STDLIB ONLY, and self-contained: like the `*_runner.py` modules it is read as
SOURCE by an adapter's `blob()` and streamed into the container on stdin, where
the runner `exec`s it and starts one instance per episode. It is never imported
by `core/` and never written to the container filesystem.

Why it exists. A model entry in configs/models.yaml may carry `extra_body:`, a
mapping that must reach the provider inside every chat request -- an OpenRouter
`provider: {order: [...]}` pin, say. The judge can pass that straight to its SDK
client. An agent CLI cannot: it composes and sends its own requests, and no CLI
we drive has a knob for arbitrary body fields. So the runner points the CLI at

    http://127.0.0.1:<port><upstream path>

instead of the real `api_base_url`, and this forwarder:

  * accepts any OpenAI-compatible or Anthropic-compatible request on any path
    (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/models`, ...);
    it does not know or care which;
  * deep-merges `extra_body` into a JSON OBJECT request body, and leaves every
    other body untouched (extra_body's keys win on a clash -- they are the
    operator's routing pin);
  * forwards to `api_base_url`'s origin at the SAME path, with the real key in
    `Authorization: Bearer` (and in `x-api-key` when the CLI sent one);
  * streams the response back byte-for-byte, SSE included, with the upstream's
    status and headers.

It ADDS NOTHING ELSE. No temperature, top_p, seed, max_tokens or any token cap
is ever added or altered, and no request or response body is ever logged --
the log line per request is method, path, status, byte counts and elapsed time.

`Upstream` is the outbound leg on its own -- (method, path, headers, body) in,
a streamable response out -- so another in-container shim that already speaks
to the CLI in some other dialect can reuse it for the trip to the gateway.
"""
from __future__ import annotations

import http.client
import http.server
import json
import queue
import socket
import socketserver
import ssl
import sys
import threading
import time
from urllib.parse import urlsplit

# RFC 7230 hop-by-hop headers, plus the two framing headers we recompute and
# Host, which names the upstream rather than the loopback listener.
_HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate",
               "proxy-authorization", "te", "trailer", "trailers",
               "transfer-encoding", "upgrade", "host", "content-length"}

# Read granularity on the way back. `read1` returns as soon as ANY bytes are
# available, so a 20-byte SSE delta is relayed the moment it arrives and never
# waits for a buffer to fill.
_CHUNK = 64 * 1024
_SSE_KEEPALIVE = b": keepalive\n\n"
_MAX_SSE_EVENT = 8 * 1024 * 1024


class _SSEFrames:
    """Keep provider bytes intact; a keepalive must not split an SSE event.

    Buffer at most one bounded event. Handle LF, CRLF and CR, including a CRLF
    split between socket reads. No JSON, reasoning or tool content is parsed.
    """

    def __init__(self):
        self.buffer = bytearray()
        self.cursor = self.line_start = 0

    def feed(self, chunk: bytes, *, final=False):
        self.buffer.extend(chunk)
        frames = []
        while self.cursor < len(self.buffer):
            i = self.cursor
            char = self.buffer[i]
            if char not in (10, 13):
                self.cursor += 1
                continue
            if char == 13 and i + 1 == len(self.buffer) and not final:
                break
            end = i + (2 if char == 13 and self.buffer[i:i + 2] == b"\r\n" else 1)
            if i == self.line_start:
                if end > _MAX_SSE_EVENT:
                    raise ValueError("SSE event exceeds relay buffer limit")
                frames.append(bytes(self.buffer[:end]))
                del self.buffer[:end]
                self.cursor = self.line_start = 0
            else:
                self.cursor = self.line_start = end
        if len(self.buffer) > _MAX_SSE_EVENT:
            raise ValueError("SSE event exceeds relay buffer limit")
        if final and self.buffer:
            # Never dispatch an incomplete event, or log its private content.
            raise http.client.IncompleteRead(b"")
        return frames


def deep_merge(base, extra):
    """`extra` merged into `base`, recursively for nested mappings; a new dict.

    A key present in both with mapping values merges; anything else in `extra`
    replaces what `base` had. Lists are replaced, not concatenated.
    """
    out = dict(base or {})
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _default_log(msg: str) -> None:
    print(f"[forwarder] {msg}", file=sys.stderr, flush=True)


class UpstreamResponse:
    """One upstream response, readable as a stream. Close it when done."""

    def __init__(self, conn: http.client.HTTPConnection,
                 resp: http.client.HTTPResponse, transport_socket=None):
        self._conn = conn
        self._resp = resp
        self.status: int = resp.status
        self.reason: str = resp.reason
        self.headers: list = list(resp.getheaders())
        # HTTPConnection may clear .sock after headers saying Connection: close.
        # Retain it so a disconnected downstream can interrupt a blocked reader.
        self._transport_socket = transport_socket if transport_socket is not None else conn.sock
        self.relay_stats = {"bytes_out": 0, "keepalives": 0}

    def header(self, name: str):
        return self._resp.getheader(name)

    def iter_chunks(self, size: int = _CHUNK):
        """Bytes as they arrive: `read1`, so a partial SSE event is not held."""
        while True:
            chunk = self._resp.read1(size)
            if not chunk:
                return
            yield chunk

    def read(self) -> bytes:
        return self._resp.read()

    def _abort_read(self):
        if self._transport_socket is not None:
            try:
                self._transport_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def _sse_chunks(self, interval):
        """One bounded reader; only the caller writes to the downstream socket."""
        pending = queue.Queue(maxsize=2)
        stopped = threading.Event()

        def put(item):
            while not stopped.is_set():
                try:
                    pending.put(item, timeout=0.1)
                    return
                except queue.Full:
                    pass

        def read():
            frames = _SSEFrames()
            try:
                for chunk in self.iter_chunks():
                    for frame in frames.feed(chunk):
                        put(("data", frame))
                    if stopped.is_set():
                        return
                for frame in frames.feed(b"", final=True):
                    put(("data", frame))
            except Exception as exc:
                put(("error", exc))
            finally:
                put(("end", None))

        reader = threading.Thread(target=read, name="native-sse-reader", daemon=True)
        reader.start()
        try:
            while True:
                try:
                    kind, value = pending.get(timeout=interval)
                except queue.Empty:
                    yield _SSE_KEEPALIVE, True
                    continue
                if kind == "end":
                    return
                if kind == "error":
                    raise value
                yield value, False
        finally:
            stopped.set()
            self._abort_read()
            reader.join(timeout=1)

    def relay(self, handler: http.server.BaseHTTPRequestHandler,
              *, keepalive_interval: float | None = None) -> int:
        """Relay status, headers and body without interpreting provider events.

        Shared by the generic forwarder and Muse's catalog proxy. SSE fragments
        are flushed as they arrive; reasoning, images and tool arguments never
        pass through a lossy response-format conversion. The restricted native
        bridge can opt into SSE comment keepalives; all other callers retain
        byte-for-byte forwarding. Keepalives protect the downstream idle timer,
        not upstream timeouts, provider refusals or the episode's total budget.
        """
        if keepalive_interval is not None and keepalive_interval <= 0:
            raise ValueError("keepalive_interval must be positive")
        content_type = (self.header("Content-Type") or "").split(";", 1)[0].strip().lower()
        content_encoding = (self.header("Content-Encoding") or "identity").strip().lower()
        heartbeat = (keepalive_interval is not None and content_type == "text/event-stream"
                     and content_encoding == "identity" and 200 <= self.status < 300)
        self.relay_stats = {"bytes_out": 0, "keepalives": 0}
        handler.send_response(self.status, self.reason)
        length = None if heartbeat else self.header("Content-Length")
        for name, value in self.headers:
            if name.lower() not in _HOP_BY_HOP and name.lower() != "content-length":
                handler.send_header(name, value)
        chunked = length is None
        handler.send_header("Transfer-Encoding" if chunked else "Content-Length",
                            "chunked" if chunked else length)
        handler.end_headers()
        n_out = 0
        chunks = (self._sse_chunks(keepalive_interval) if heartbeat
                  else ((chunk, False) for chunk in self.iter_chunks()))
        try:
            for chunk, is_keepalive in chunks:
                if chunked:
                    handler.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                else:
                    handler.wfile.write(chunk)
                handler.wfile.flush()
                if is_keepalive:
                    self.relay_stats["keepalives"] += 1
                else:
                    n_out += len(chunk)
                    self.relay_stats["bytes_out"] = n_out
        finally:
            chunks.close()
        if chunked:
            handler.wfile.write(b"0\r\n\r\n")
            handler.wfile.flush()
        return n_out

    def close(self) -> None:
        self._abort_read()
        try:
            self._resp.close()
        finally:
            self._conn.close()


class Upstream:
    """The outbound leg: a request to `base_url`'s origin, with the real key.

    `base_url` is the resolved `api_base_url` (origin plus optional path
    prefix); `path` on each request is forwarded AS RECEIVED, so the listener
    is mounted at the same path prefix and the CLI's own path composition is
    preserved end to end.
    """

    def __init__(self, base_url: str, api_key: str, extra_body=None,
                 timeout: float | None = None):
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError(f"upstream base_url must be http(s)://host[/path], "
                             f"got {base_url!r}")
        self.scheme = parts.scheme
        self.netloc = parts.netloc
        self.path = parts.path.rstrip("/")
        self.base_url = f"{self.scheme}://{self.netloc}{self.path}"
        self.api_key = api_key
        self.extra_body = dict(extra_body or {})
        self.timeout = timeout
        self._ssl = ssl.create_default_context() if self.scheme == "https" else None

    def merge_body(self, body: bytes | None, content_type: str | None) -> bytes | None:
        """`extra_body` merged into a JSON-object body; anything else untouched."""
        if not body or not self.extra_body:
            return body
        ctype = (content_type or "").lower()
        if ctype and "json" not in ctype:
            return body
        try:
            doc = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return body
        if not isinstance(doc, dict):
            return body
        return json.dumps(deep_merge(doc, self.extra_body)).encode("utf-8")

    def request(self, method: str, path: str, headers, body: bytes | None
                ) -> UpstreamResponse:
        out_headers = {}
        content_type = None
        had_x_api_key = False
        for name, value in (headers.items() if hasattr(headers, "items") else headers):
            lname = name.lower()
            if lname in _HOP_BY_HOP:
                continue
            if lname == "content-type":
                content_type = value
            if lname == "x-api-key":
                had_x_api_key = True
                continue
            if lname == "authorization":
                continue
            out_headers[name] = value
        body = self.merge_body(body, content_type)
        out_headers["Host"] = self.netloc
        out_headers["Authorization"] = f"Bearer {self.api_key}"
        if had_x_api_key:
            out_headers["x-api-key"] = self.api_key
        if body is not None:
            out_headers["Content-Length"] = str(len(body))
        elif method.upper() in ("POST", "PUT", "PATCH"):
            out_headers["Content-Length"] = "0"

        if self.scheme == "https":
            conn = http.client.HTTPSConnection(self.netloc, timeout=self.timeout,
                                               context=self._ssl)
        else:
            conn = http.client.HTTPConnection(self.netloc, timeout=self.timeout)
        try:
            conn.request(method, path, body=body, headers=out_headers)
            transport_socket = conn.sock
            resp = conn.getresponse()
        except Exception:
            conn.close()
            raise
        return UpstreamResponse(conn, resp, transport_socket)


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: "Forwarder"

    def log_message(self, *args) -> None:  # the base class logs to stderr per line
        pass

    def _read_body(self) -> bytes | None:
        length = self.headers.get("Content-Length")
        if length is not None:
            n = int(length)
            return self.rfile.read(n) if n else b""
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            out = bytearray()
            while True:
                line = self.rfile.readline().strip()
                size = int(line.split(b";", 1)[0], 16) if line else 0
                if size == 0:
                    # trailers, then the blank line
                    while True:
                        trailer = self.rfile.readline()
                        if trailer in (b"\r\n", b"\n", b""):
                            break
                    return bytes(out)
                out += self.rfile.read(size)
                self.rfile.readline()
        return None

    def _proxy(self) -> None:
        t0 = time.time()
        upstream = self.server.upstream
        body = self._read_body()
        n_in = len(body) if body else 0
        try:
            resp = upstream.request(self.command, self.path, self.headers, body)
        except Exception as e:  # noqa: BLE001 - the CLI gets a 502, not silence
            payload = json.dumps({"error": {"type": "forwarder_error",
                                            "message": f"{type(e).__name__}: {e}"}}
                                 ).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            self.server.log(f"{self.command} {self.path} -> 502 (upstream "
                            f"{type(e).__name__}: {e}) in={n_in}B")
            self.server.calls.append({"method": self.command, "path": self.path,
                                      "status": 502, "error": f"{type(e).__name__}",
                                      "bytes_in": n_in, "bytes_out": 0,
                                      "seconds": round(time.time() - t0, 3)})
            return

        n_out = 0
        try:
            n_out = resp.relay(self)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        finally:
            resp.close()
        self.server.log(f"{self.command} {self.path} -> {resp.status} "
                        f"in={n_in}B out={n_out}B "
                        f"{time.time() - t0:.1f}s")
        self.server.calls.append({"method": self.command, "path": self.path,
                                  "status": resp.status, "bytes_in": n_in,
                                  "bytes_out": n_out,
                                  "seconds": round(time.time() - t0, 3)})

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD = _proxy


class Forwarder(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """The loopback listener. `start()`, point the CLI at `.url`, `close()`."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, *, base_url: str, api_key: str, extra_body=None,
                 host: str = "127.0.0.1", timeout: float | None = None,
                 log=None):
        self.upstream = Upstream(base_url, api_key, extra_body, timeout=timeout)
        self.log = log or _default_log
        # One summary per request: method, path, status, sizes, seconds. No
        # bodies, no headers, so it is safe to put on the episode record.
        self.calls: list = []
        super().__init__((host, 0), _Handler)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self.server_address[1]

    @property
    def url(self) -> str:
        """What the CLI is pointed at: the loopback origin plus the upstream's
        own path prefix, so the CLI's relative paths land where they would
        have landed on the real base URL."""
        return f"http://{self.server_address[0]}:{self.port}{self.upstream.path}"

    def start(self) -> "Forwarder":
        self._thread = threading.Thread(target=self.serve_forever,
                                        name="rh-forwarder", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001
            pass
        self.server_close()
