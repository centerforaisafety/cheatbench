"""The Kimi Code CLI episode runner that executes INSIDE the container.

Same contract as `core/agents/codex_runner.py`, and for the same reasons: this
file lives in the repo, is never baked into the image, is streamed in on
**stdin** with `core/stage.py`, the staged bytes and `core/agents/forwarder.py`,
and writes exactly ONE line of JSON on stdout. Everything else -- the CLI's own
output, tracebacks, turn streaming -- goes to stderr, so the host parses stdout
without heuristics.

MODELLED ON HARBOR: `harbor/src/harbor/agents/installed/kimi_code.py`, which
installs `@moonshot-ai/kimi-code` with npm, configures the model through
`KIMI_MODEL_*` environment variables, registers MCP servers in
`$KIMI_CODE_HOME/mcp.json` and runs `kimi --prompt ... --output-format
stream-json </dev/null`. What is different here, and why:

  * EVERY MODEL REQUEST GOES THROUGH A LOOPBACK FORWARDER, always, not only
    when the model entry carries `extra_body`. Kimi Code composes and sends its
    own chat requests, so the model entry's OpenRouter provider pin can only be
    injected by a shim in front of it; that is what `core/agents/forwarder.py`
    is for and its `Upstream` class does the outbound leg here. But the shim is
    ALSO this adapter's only source of the model's reasoning: measured on
    kimi-code 0.42.0, the `--output-format stream-json` writer's
    `writeThinkingDelta` is a no-op and the session's `wire.jsonl` records no
    `think` parts either, while the SSE stream from the gateway carries
    `reasoning_content` deltas on every call. So `KimiForwarder` below relays
    each response byte-for-byte AND assembles, per request, the response id,
    the reasoning text, the visible text, the tool calls and the usage. The
    adapter's `to_trajectory` joins those to the CLI's own steps by response id
    (`step.end.messageId` in wire.jsonl == the chat completion `id`).
  * THE TRAJECTORY IS NOT ON STDOUT. `stream-json` prints assistant/tool lines
    for a human (and drops thinking, see above). The file of record is
    `$KIMI_CODE_HOME/sessions/<workspace>/<session>/agents/<agent>/wire.jsonl`
    -- the CLI's append-only event journal (`context.append_loop_event` with
    `step.begin` / `content.part` / `tool.call` / `tool.result` / `step.end`,
    plus `turn.prompt`, `profile.bind`, `usage.record`, ...). Harbor would copy
    that tree out over a bind mount; there is none here, so this runner READS
    every wire.jsonl and returns the records inside the one JSON line it owes.
  * The CLI never sees the real credential. `KIMI_MODEL_API_KEY` is set to a
    placeholder (the CLI refuses to start without one) and the host-side key
    variable is REMOVED from the CLI's environment; only the forwarder in this
    process holds the key, so the agent's Bash cannot read it out of
    /proc/self/environ.
  * The grader is hosted over loopback streamable-HTTP MCP, exactly as the
    Codex runner does it (that code is duplicated here on purpose: each runner
    is streamed into a container on its own). Kimi Code takes MCP servers from
    `$KIMI_CODE_HOME/mcp.json` in `{"mcpServers": {name: {"url": ...}}}` shape
    and prefixes the tool `mcp__<server>__<tool>`, which is the same name the
    Claude and Codex adapters produce. Verified against kimi-code 0.42.0 on a
    real call: `initialize` -> `tools/list` -> `tools/call` on this server.
  * Print mode (`-p`) REJECTS `--yolo` and `--auto` ("Cannot combine --prompt
    with --auto"); it runs in `auto` permission mode by itself (wire.jsonl
    records `permission.set_mode auto`), tools execute without prompting, and
    the model is told not to call `AskUserQuestion`. So there is no permission
    flag on the command line at all.
"""
from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from socketserver import ThreadingMixIn

# ---------------------------------------------------------------------------
# stdout hygiene, established before anything else can print. fd 1 is duplicated
# to a private handle and then replaced by fd 2, so any stray write to stdout
# lands on stderr instead of corrupting the single JSON line the host parses.
# ---------------------------------------------------------------------------
_OUT = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)


def _log(msg: str) -> None:
    print(f"[runner] {msg}", file=sys.stderr, flush=True)


# Live turn streaming. The host tees these to messages.jsonl / turns.log.
MSG_PREFIX = "\x1eRHMSG "


def _emit_msg(rec: dict) -> None:
    try:
        sys.stderr.write(MSG_PREFIX + json.dumps(rec) + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 - streaming is best effort, never fatal
        pass


WORKDIR = "/workspace"

# The formats a deliverable can take, and the export cap. Identical to the
# other runners: this is a property of the TASK, not of the agent.
DELIVERABLE_EXTS = (".xlsx", ".xls", ".xlsm", ".docx", ".doc", ".pptx", ".ppt",
                    ".pdf", ".csv", ".svg")
_DELIVERABLE_MAX_BYTES = 64 * 1024 * 1024

INSTALL_TIMEOUT_S = 600


def collect_deliverables(root: str, skip_dirs: tuple, include_files: tuple = ()) -> list:
    """Export office files and exact task-requested outputs for the host."""
    import base64
    import stat
    requested = {
        p for p in include_files
        if isinstance(p, str) and p and not os.path.isabs(p)
        and not any(part in ("", ".", "..") for part in p.split("/"))
    }
    out, total = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        top = rel_dir.split(os.sep)[0]
        if top in skip_dirs:
            dirnames[:] = []
            continue
        for name in sorted(filenames):
            relative = os.path.relpath(os.path.join(dirpath, name), root)
            if (os.path.splitext(name)[1].lower() not in DELIVERABLE_EXTS
                    and relative not in requested):
                continue
            q = os.path.join(dirpath, name)
            try:
                info = os.lstat(q)
                if not stat.S_ISREG(info.st_mode):
                    continue
                remaining = _DELIVERABLE_MAX_BYTES - total
                if info.st_size > remaining:
                    continue
                with open(q, "rb") as f:
                    data = f.read(remaining + 1)
            except OSError:
                continue
            if total + len(data) > _DELIVERABLE_MAX_BYTES:
                _log(f"deliverable export cap hit; skipping {q}")
                continue
            total += len(data)
            out.append({"name": os.path.relpath(q, root),
                        "b64": base64.b64encode(data).decode()})
    return out


# ---------------------------------------------------------------------------
# the adapter's runtime: `check || install`, same contract as the other runners
# ---------------------------------------------------------------------------
def _sh(cmd: str, timeout: float, env: dict | None = None):
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                          timeout=timeout, env=env)


# Task-authored installation commands receive an environment without model
# credentials or routing variables. Keep package-manager and proxy settings.
_SECRET_ENV_SUBSTRINGS = ("API_KEY", "APIKEY", "TOKEN", "SECRET", "PASSWORD",
                          "CREDENTIAL", "AUTH", "BASE_URL")
_SECRET_ENV_PREFIXES = ("ANTHROPIC_", "OPENAI_", "CLAUDE_", "CODEX_",
                        "GEMINI_", "GOOGLE_", "VERTEX_", "XAI_", "GROK_",
                        "KIMI_", "MOONSHOT_", "MUSE_", "LITELLM_", "AZURE_",
                        "AWS_", "HF_", "HUGGINGFACE_", "HUGGING_FACE_")


def task_install_env() -> dict:
    """`os.environ` with the model credential removed, for a task's own shell."""
    out = {}
    for k, v in os.environ.items():
        up = k.upper()
        if any(s in up for s in _SECRET_ENV_SUBSTRINGS):
            continue
        if up.startswith(_SECRET_ENV_PREFIXES):
            continue
        out[k] = v
    return out


def install_agent(spec: dict | None,
                  env: dict | None = None) -> dict | None:
    """Put the adapter's binary in place. Returns what happened, or None.

    Never raises: the host is owed exactly one JSON line whatever happens here.

    `env` is the environment the install shell gets, and None means "inherit
    ours". That is what the ADAPTER's own runtime install wants -- it is the
    harness's own command in the harness's own environment. A TASK's `install:`
    is handed `task_install_env()` instead.
    """
    if not spec:
        return None

    t0 = time.time()
    out = {"name": spec.get("name"), "status": "present",
           "pinned_version": spec.get("pinned_version") or "",
           "version": ""}

    def _version() -> str:
        cmd = spec.get("version_cmd")
        if not cmd:
            return ""
        try:
            res = _sh(cmd, 60, env)
            return (res.stdout or "").strip().split("\n")[-1] if \
                res.returncode == 0 else ""
        except Exception:  # noqa: BLE001
            return ""

    try:
        if _sh(spec.get("check") or "true", 60, env).returncode == 0:
            out["version"] = _version()
            out["seconds"] = round(time.time() - t0, 1)
            _log(f"{out['name']}: runtime already present "
                 f"({out['seconds']}s to check), version "
                 f"{out['version'] or 'unknown'}")
            return out
        res = _sh(spec["install"], INSTALL_TIMEOUT_S, env)
        out["status"] = "installed" if res.returncode == 0 else "failed"
        if res.returncode != 0:
            # NO TRUNCATION: whatever npm said, in full.
            out["error"] = (res.stderr or res.stdout or "").strip()
        else:
            out["version"] = _version()
    except Exception as e:  # noqa: BLE001
        out["status"] = "failed"
        out["error"] = f"{type(e).__name__}: {e}"

    out["seconds"] = round(time.time() - t0, 1)
    _log(f"{out['name']}: runtime {out['status']} in {out['seconds']}s, "
         f"version {out['version'] or 'unknown'}")
    if out.get("error"):
        _log(f"{out['name']}: install said: {out['error']}")
    return out


# ---------------------------------------------------------------------------
# the task's grader, as an MCP server the CLI reaches by URL
#
# Duplicated from `core/agents/codex_runner.py` (see the note there on why HTTP
# and not stdio, where it binds, and what the model sees). Each runner is
# streamed into a container on its own, so it carries its own copy. Verified
# against kimi-code 0.42.0: the CLI reads `{"mcpServers": {"grader": {"url":
# ...}}}` from `$KIMI_CODE_HOME/mcp.json`, infers the streamable-http transport
# from the `url` key, negotiates on this server and calls the tool as
# `mcp__grader__grade_deliverable`.
# ---------------------------------------------------------------------------
MCP_PROTOCOL_VERSION = "2025-06-18"


class _GraderMCPHandler(BaseHTTPRequestHandler):
    """Streamable-HTTP MCP, the methods the CLI actually calls."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        """Silence. The default writes to stderr, which the host parses."""

    def _authorised(self) -> bool:
        path = self.path.split("?", 1)[0]
        return hmac.compare_digest(path, self.server.grader_path)

    def _empty(self, code: int) -> None:
        self.send_response(code)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send(self, body: bytes) -> None:
        if "text/event-stream" in (self.headers.get("Accept") or ""):
            payload = b"event: message\ndata: " + body + b"\n\n"
            ctype = "text/event-stream"
        else:
            payload, ctype = body, "application/json"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:       # noqa: N802
        self._empty(404 if not self._authorised() else 405)

    def do_DELETE(self) -> None:    # noqa: N802
        self._empty(404 if not self._authorised() else 405)

    def do_POST(self) -> None:      # noqa: N802
        if not self._authorised():
            self._empty(404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._empty(400)
            return
        raw = self.rfile.read(length) if length else b""
        try:
            request = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            self._empty(400)
            return
        if not isinstance(request, dict):
            self._empty(400)
            return
        if request.get("id") is None:
            self._empty(202)
            return
        self._send(json.dumps(self._handle(request)).encode())

    def _handle(self, request: dict) -> dict:
        rid, method = request.get("id"), request.get("method")
        params = request.get("params")
        params = params if isinstance(params, dict) else {}
        server = self.server

        def ok(result):
            return {"jsonrpc": "2.0", "id": rid, "result": result}

        if method == "initialize":
            server.connected = True
            return ok({
                "protocolVersion": params.get("protocolVersion")
                or MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": server.grader_server_name,
                               "version": "1.0.0"},
            })
        if method == "ping":
            return ok({})
        if method == "tools/list":
            server.listed = True
            return ok({"tools": [
                {key: spec[key] for key in ("name", "description", "inputSchema")}
                for spec in server.grader_tools.values()]})
        if method == "tools/call":
            name = params.get("name")
            spec = server.grader_tools.get(name) if isinstance(name, str) else None
            if spec is None:
                return ok({"content": [{"type": "text",
                                        "text": f"no such tool: {name}"}],
                           "isError": True})
            args = params.get("arguments")
            try:
                return ok(spec["call"](args if isinstance(args, dict) else {}))
            except Exception as e:  # noqa: BLE001 - a grader bug is not a crash
                _log(f"grader raised: {type(e).__name__}: {e}")
                server.grader_errors.append(f"{type(e).__name__}: {e}")
                return ok({"content": [{"type": "text",
                                        "text": "the grader is unavailable"}],
                           "isError": True})
        return {"jsonrpc": "2.0", "id": rid,
                "error": {"code": -32601, "message": f"unknown method {method}"}}


class GraderServer(ThreadingHTTPServer):
    """The running grader endpoint, and the handle the episode holds it by."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, *, state: dict, call, server_name: str, tool_name: str,
                 description: str, schema: dict, extra_tools=()):
        tools = [{"name": tool_name, "description": description,
                  "inputSchema": schema, "call": call}, *extra_tools]
        names = [spec["name"] for spec in tools]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate grader tool names")
        if any(not callable(spec["call"]) for spec in tools):
            raise ValueError("Each grader tool needs a callable handler")
        super().__init__(("127.0.0.1", 0), _GraderMCPHandler)
        self.grader_state = state
        self.grader_call = call
        self.grader_server_name = server_name
        self.grader_tool_name = tool_name
        self.grader_description = description
        self.grader_schema = schema
        self.grader_tools = {spec["name"]: spec for spec in tools}
        self.grader_errors: list = []
        self.grader_path = "/" + secrets.token_hex(16) + "/mcp"
        self.connected = False
        self.listed = False
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self.server_address[0], self.server_address[1]
        return f"http://{host}:{port}{self.grader_path}"

    def start(self) -> None:
        self._thread = threading.Thread(target=self.serve_forever,
                                        kwargs={"poll_interval": 0.2},
                                        daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop serving. Idempotent, for the same reason as in codex_runner."""
        if self._thread is not None:
            try:
                self.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._thread.join(timeout=10)
            self._thread = None
        try:
            self.server_close()
        except Exception:  # noqa: BLE001
            pass

    def report(self) -> dict:
        entry = {"name": self.grader_server_name,
                 "transport": "streamable-http",
                 "status": "connected" if self.connected else "configured",
                 "tools_listed": self.listed}
        if self.grader_errors:
            entry["errors"] = list(self.grader_errors)
        return entry


def serve_grader(module: dict, row: dict, staged: dict, workdir: str):
    """Start the task's grader as an MCP endpoint. Returns the running server."""
    state, call = module["make_tool"](row, staged, workdir)
    primary = module["tool_spec"](row) if callable(module.get("tool_spec")) else {}
    extras = module["make_extra_tools"](row, staged, workdir, state) if callable(module.get("make_extra_tools")) else ()
    server = GraderServer(
        state=state, call=call,
        server_name=module.get("SERVER_NAME", "grader"),
        tool_name=primary.get("name", module.get("TOOL_BASENAME", "grade_deliverable")),
        description=primary.get("description", module.get("DESCRIPTION", "")),
        schema=primary.get("inputSchema", module.get("INPUT_SCHEMA"))
        or {"type": "object", "properties": {}}, extra_tools=extras)
    server.start()
    return server


# ---------------------------------------------------------------------------
# the model forwarder: core/agents/forwarder.py's `Upstream` for the outbound
# leg, plus a response tap that assembles what the model said
#
# What the tap keeps, per request: the chat completion `id`, the finish reason,
# the reasoning text (from `delta.reasoning_content`, the key Kimi Code itself
# defaults to; `delta.reasoning` is the fallback), the visible text, the tool
# calls (index-merged from the streamed fragments), the final `usage` object,
# and `provider` when the response carries one. NOTHING is truncated and no
# request body is ever kept -- the request's `model` and `stream` flags are the
# only fields read from it.
#
# `provider` is how the routing pin is witnessed. OpenRouter puts the serving
# provider's name on every response; measured through the litellm gateway it
# SURVIVES on a non-streaming response and is STRIPPED from streamed chunks,
# and the CLI always streams. So the forwarder makes ONE deliberate
# non-streaming probe request through the same `Upstream` (same extra_body,
# same key) before the CLI starts and records what came back; that probe is
# reported separately from the agent's own calls and its tokens are not
# counted in the episode's usage.
# ---------------------------------------------------------------------------
PROBE_PROMPT = "Say OK."


class _SSEAssembler:
    """Chat-completion chunks (or one JSON body) -> one assembled message."""

    def __init__(self) -> None:
        self.buf = b""
        self.id = None
        self.model = None
        self.created = None
        self.provider = None
        self.finish_reason = None
        self.reasoning: list = []
        self.content: list = []
        self.tool_calls: dict = {}
        self.usage = None
        self.error = None

    def feed(self, data: bytes) -> None:
        self.buf += data
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            self._line(line.rstrip(b"\r"))

    def finish_sse(self) -> None:
        if self.buf:
            self._line(self.buf)
            self.buf = b""

    def _line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            return
        try:
            obj = json.loads(payload)
        except ValueError:
            return
        if isinstance(obj, dict):
            self.take(obj, streamed=True)

    def take(self, obj: dict, *, streamed: bool) -> None:
        if obj.get("id"):
            self.id = obj["id"]
        if obj.get("model"):
            self.model = obj["model"]
        if obj.get("created") is not None:
            self.created = obj["created"]
        if obj.get("provider"):
            self.provider = obj["provider"]
        if isinstance(obj.get("usage"), dict):
            self.usage = obj["usage"]
        if obj.get("error") and self.error is None:
            self.error = json.dumps(obj["error"])
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
            part = choice.get("delta" if streamed else "message")
            if not isinstance(part, dict):
                continue
            if "reasoning_content" in part:
                if isinstance(part["reasoning_content"], str):
                    self.reasoning.append(part["reasoning_content"])
            elif isinstance(part.get("reasoning"), str):
                self.reasoning.append(part["reasoning"])
            if isinstance(part.get("content"), str):
                self.content.append(part["content"])
            for tc in part.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                idx = tc.get("index", len(self.tool_calls))
                slot = self.tool_calls.setdefault(
                    idx, {"id": None, "name": None, "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if isinstance(fn.get("arguments"), str):
                    slot["arguments"] += fn["arguments"]

    def result(self) -> dict:
        return {
            "id": self.id, "model": self.model, "created": self.created,
            "provider": self.provider, "finish_reason": self.finish_reason,
            "reasoning": "".join(self.reasoning),
            "content": "".join(self.content),
            "tool_calls": [self.tool_calls[k] for k in sorted(
                self.tool_calls, key=lambda x: (str(type(x)), x))],
            "usage": self.usage, "error": self.error,
        }


MCP_NAMES = frozenset({"play_move", "game_status", "reset_game", "write_notes", "read_notes",
                       "api_request", "list_files", "read_file", "write_file"})


def restricted_request(document, allowed_tools):
    """Validate native declarations without changing the prompt or inference settings."""
    if not isinstance(document, dict):
        raise ValueError("Kimi MCP-only policy breach: expected a request object")
    declared = document.get("tools", [])
    if not isinstance(declared, list):
        raise ValueError("Kimi MCP-only policy breach: invalid tool declarations")
    names = []
    for tool in declared:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or tool.get("type") != "function":
            raise ValueError("Kimi MCP-only policy breach: non-function tool")
        name = function.get("name")
        if not isinstance(name, str) or name not in allowed_tools:
            raise ValueError("Kimi MCP-only policy breach: unexpected tool")
        names.append(name)
    # Auxiliary/compaction requests may have no tools. An agent roster must be complete.
    if names and (len(names) != len(allowed_tools) or set(names) != set(allowed_tools)):
        raise ValueError("Kimi MCP-only policy breach: incomplete tool roster")
    return document


class _KimiForwardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def _read_body(self) -> bytes | None:
        length = self.headers.get("Content-Length")
        if length is not None:
            n = int(length)
            return self.rfile.read(n) if n else b""
        return None

    def _proxy(self) -> None:
        srv = self.server
        t0 = time.time()
        body = self._read_body()
        if srv.allowed_tools is not None:
            try:
                restricted_request(json.loads(body or b"{}"), srv.allowed_tools)
            except (ValueError, UnicodeError) as exc:
                srv.policy_error = str(exc)
                payload = json.dumps({"error": {"message": srv.policy_error, "type": "adapter_error"}}).encode()
                self.send_response(403)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
        call = {"type": "forwarder.call", "index": None, "method": self.command,
                "path": self.path, "status": None, "request_model": None,
                "stream": None, "bytes_in": len(body or b""), "bytes_out": 0,
                "seconds": None}
        if body:
            try:
                req = json.loads(body)
                if isinstance(req, dict):
                    call["request_model"] = req.get("model")
                    call["stream"] = bool(req.get("stream"))
            except ValueError:
                pass
        try:
            resp = srv.upstream.request(self.command, self.path, self.headers, body)
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
            call.update({"status": 502, "error": f"{type(e).__name__}: {e}",
                         "seconds": round(time.time() - t0, 3)})
            srv.record(call)
            return

        tap = _SSEAssembler()
        ctype = (resp.header("Content-Type") or "").lower()
        is_sse = "text/event-stream" in ctype
        # A failed request's body is kept WHATEVER its content type says. A
        # gateway 5xx keeps the `text/event-stream` header of the request it
        # was answering and then sends an HTML page, which the SSE assembler
        # drops on the floor (no `data:` lines) -- so the one call that ended
        # the episode used to be recorded with `error: null`.
        is_error = resp.status >= 400
        raw_body = b""
        n_out = 0
        try:
            self.send_response(resp.status, resp.reason)
            length = resp.header("Content-Length")
            for name, value in resp.headers:
                if name.lower() in _HOP_BY_HOP:
                    continue
                self.send_header(name, value)
            chunked = length is None
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Content-Length", length)
            self.end_headers()
            for chunk in resp.iter_chunks():
                if chunked:
                    self.wfile.write(b"%x\r\n" % len(chunk))
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                else:
                    self.wfile.write(chunk)
                self.wfile.flush()
                n_out += len(chunk)
                if is_sse:
                    tap.feed(chunk)
                if is_error or not is_sse:
                    raw_body += chunk
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        finally:
            resp.close()
        if is_sse:
            tap.finish_sse()
        elif raw_body:
            try:
                obj = json.loads(raw_body)
                if isinstance(obj, dict):
                    tap.take(obj, streamed=False)
            except ValueError:
                pass
        if is_error and tap.error is None and raw_body:
            # Whatever the gateway said, whole and verbatim -- it is the only
            # account of why the episode ended, and nothing else records it.
            tap.error = raw_body.decode("utf-8", "replace")
        call.update(tap.result())
        call.update({"status": resp.status, "bytes_out": n_out,
                     "seconds": round(time.time() - t0, 3)})
        srv.record(call)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD = _proxy


# RFC 7230 hop-by-hop, plus the framing headers we recompute and Host.
_HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate",
               "proxy-authorization", "te", "trailer", "trailers",
               "transfer-encoding", "upgrade", "host", "content-length"}


class KimiForwarder(ThreadingMixIn, HTTPServer):
    """The loopback listener the CLI is pointed at. `start()`, `.base_url`, `close()`."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, *, upstream, host: str = "127.0.0.1", allowed_tools=None):
        self.upstream = upstream          # core/agents/forwarder.py Upstream
        self.allowed_tools = frozenset(allowed_tools) if allowed_tools is not None else None
        self.policy_error = None
        self.calls: list = []
        self._lock = threading.Lock()
        super().__init__((host, 0), _KimiForwardHandler)
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return (f"http://{self.server_address[0]}:{self.server_address[1]}"
                f"{self.upstream.path}")

    @property
    def base_url(self) -> str:
        """What `KIMI_MODEL_BASE_URL` gets: the listener, with `/v1` where the
        CLI expects to append `/chat/completions` to it. The upstream's own
        path prefix is preserved, so `https://gw` becomes `http://lo/v1` ->
        `https://gw/v1/...`, and `https://gw/v1` becomes `http://lo/v1` too."""
        url = self.url.rstrip("/")
        return url if url.endswith("/v1") else url + "/v1"

    def record(self, call: dict) -> None:
        with self._lock:
            call["index"] = len(self.calls)
            self.calls.append(call)
        tokens = ""
        if isinstance(call.get("usage"), dict):
            u = call["usage"]
            tokens = (f" tokens in={u.get('prompt_tokens')} "
                      f"out={u.get('completion_tokens')}")
        _log(f"forwarder: {call['method']} {call['path']} -> {call['status']} "
             f"id={call.get('id')} finish={call.get('finish_reason')} "
             f"reasoning_chars={len(call.get('reasoning') or '')} "
             f"tool_calls={len(call.get('tool_calls') or [])}{tokens} "
             f"{call['seconds']}s"
             + (f" error={call['error']}" if call.get("error") else ""))

    def start(self) -> "KimiForwarder":
        self._thread = threading.Thread(target=self.serve_forever,
                                        name="rh-kimi-forwarder", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        if self._thread is not None:
            try:
                self.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._thread.join(timeout=10)
            self._thread = None
        try:
            self.server_close()
        except Exception:  # noqa: BLE001
            pass

    def probe(self, model: str, path: str = "/v1/chat/completions") -> dict:
        """ONE non-streaming request, to read the `provider` the pin resolved to.

        Sent through the same `Upstream` as everything else, so the body gets
        the same extra_body merge and the same key. No temperature, no token
        cap, nothing but the model and one short user message.
        """
        body = json.dumps({"model": model,
                           "messages": [{"role": "user", "content": PROBE_PROMPT}]}
                          ).encode()
        out = {"path": path, "status": None, "provider": None, "id": None,
               "model": None, "usage": None, "error": None, "content": None}
        t0 = time.time()
        try:
            resp = self.upstream.request(
                "POST", path, {"Content-Type": "application/json",
                               "Accept": "application/json"}, body)
        except Exception as e:  # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {e}"
            out["seconds"] = round(time.time() - t0, 3)
            return out
        try:
            raw = resp.read()
        finally:
            resp.close()
        out["status"] = resp.status
        try:
            obj = json.loads(raw)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            out["provider"] = obj.get("provider")
            out["id"] = obj.get("id")
            out["model"] = obj.get("model")
            out["usage"] = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
            if obj.get("error"):
                out["error"] = json.dumps(obj["error"])
            choices = obj.get("choices") or []
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
                out["content"] = msg.get("content")
        elif resp.status != 200:
            out["error"] = raw.decode("utf-8", "replace")
        out["seconds"] = round(time.time() - t0, 3)
        return out


def usage_from_calls(calls: list) -> dict | None:
    """The four INCLUSIVE categories summed over the forwarder's calls.

    OpenAI-shaped usage: `prompt_tokens` already includes the cache legs
    (measured: 19747 = inputOther 19491 + cached 256 on the same call in the
    CLI's own tally), so the sum is passed through as-is.
    """
    out = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
           "cache_write_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0,
           "calls_with_usage": 0, "calls": len(calls)}
    for call in calls:
        u = call.get("usage")
        if not isinstance(u, dict):
            continue
        out["calls_with_usage"] += 1
        pd = u.get("prompt_tokens_details") or {}
        cd = u.get("completion_tokens_details") or {}
        out["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        out["completion_tokens"] += int(u.get("completion_tokens") or 0)
        out["total_tokens"] += int(u.get("total_tokens") or 0)
        out["cached_tokens"] += int(pd.get("cached_tokens") or 0)
        out["cache_write_tokens"] += int(pd.get("cache_write_tokens") or 0)
        out["reasoning_tokens"] += int(cd.get("reasoning_tokens") or 0)
    return out if out["calls_with_usage"] else None


# ---------------------------------------------------------------------------
# Kimi Code's home: config.toml (tool policy, step cap) and mcp.json (grader)
# ---------------------------------------------------------------------------
def mcp_json(grader_url: str, grader_name: str = "grader", *, mcp_only=False) -> str:
    """`{"mcpServers": {name: {"url": ...}}}` -- the url shape, never a command.

    Same reasoning as the Codex runner's config.toml: a `command` in this file
    would be a path to the grader's source and answer key; a URL is an endpoint
    that only ever speaks verdicts.
    """
    server = {"url": grader_url}
    if mcp_only:
        if grader_name not in {"chess", "go"}:
            raise ValueError("Restricted Kimi requires a chess or Go namespace")
        server.update(enabledTools=sorted(MCP_NAMES), deferred=False, toolTimeoutMs=900_000)
    return json.dumps({"mcpServers": {grader_name: server}}, indent=1)


def prepare_home(cfg: dict, grader_url: str = "", grader_name: str = "grader") -> None:
    """Write the CLI's home before it starts: config.toml, and mcp.json when
    there is a grader. Both are composed by the adapter and logged; the agent
    may read them anyway, and neither holds anything but policy and a URL."""
    home = cfg["kimi_home"]
    os.makedirs(home, exist_ok=True)
    text = cfg.get("config_toml") or ""
    if text:
        with open(os.path.join(home, "config.toml"), "w") as f:
            f.write(text)
    if grader_url:
        with open(os.path.join(home, "mcp.json"), "w") as f:
            f.write(mcp_json(grader_url, grader_name, mcp_only=cfg.get("mcp_only", False)))
    _log(f"wrote {home}/config.toml ({len(text)} bytes), mcp_servers="
         f"{grader_name if grader_url else 'none'}")


def compose_command(cfg: dict, instruction: str) -> str:
    """Harbor's invocation, minus the tee: `kimi --prompt <instruction>
    --output-format stream-json`, stdin closed. The instruction is ARGV,
    shell-quoted; there is no permission flag because print mode refuses
    `--yolo`/`--auto` and runs in `auto` by itself."""
    parts = ["kimi", *list(cfg.get("cli_flags") or []),
             "--prompt", shlex.quote(instruction),
             "--output-format", "stream-json"]
    return ("if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
            + " ".join(parts) + " </dev/null")


def run_kimi(command: str, env: dict, timeout_s: int) -> tuple:
    """Run the CLI, keep its stdout (stream-json lines) AND its stderr.

    Returns (returncode, stream_json_lines, stdout_text, stderr_text, error).
    Never raises. Its own session so the timeout kills the CLI and everything
    it spawned.

    Stderr is RETAINED, not only relayed. It is where the CLI puts the thing
    that actually killed it -- `error: failed to run prompt: <reason>` -- while
    stdout ends on whatever stream-json line happened to be written last,
    which is normally a tool result and never the reason.
    """
    lines: list = []
    raw_out: list = []
    raw_err: list = []
    try:
        proc = subprocess.Popen(
            ["bash", "-c", command], cwd=WORKDIR, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
            start_new_session=True)
    except Exception as e:  # noqa: BLE001
        return None, lines, "", "", f"kimi launch failed: {type(e).__name__}: {e}"

    def _pump_out() -> None:
        try:
            for line in proc.stdout:
                raw_out.append(line)
                txt = line.rstrip("\n")
                sys.stderr.write("[kimi] " + txt + "\n")
                sys.stderr.flush()
                try:
                    obj = json.loads(txt)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    lines.append(obj)
        except Exception as e:  # noqa: BLE001
            _log(f"stdout pump: {type(e).__name__}: {e}")

    def _pump_err() -> None:
        try:
            for line in proc.stderr:
                raw_err.append(line)
                sys.stderr.write("[kimi:err] " + line.rstrip("\n") + "\n")
                sys.stderr.flush()
        except Exception as e:  # noqa: BLE001
            _log(f"stderr pump: {type(e).__name__}: {e}")

    t_out = threading.Thread(target=_pump_out, daemon=True)
    t_err = threading.Thread(target=_pump_err, daemon=True)
    t_out.start()
    t_err.start()

    deadline = time.time() + timeout_s
    error = None

    def _kill(sig) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

    try:
        rc = proc.wait(timeout=max(1.0, deadline - time.time()))
    except subprocess.TimeoutExpired:
        error = f"timeout after {timeout_s}s"
        _kill(signal.SIGKILL)
        try:
            rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            rc = None
    t_out.join(timeout=10)
    t_err.join(timeout=10)
    return rc, lines, "".join(raw_out), "".join(raw_err), error


# ---------------------------------------------------------------------------
# the wire journal, which is the actual record
# ---------------------------------------------------------------------------
def find_wires(kimi_home: str) -> list:
    """Every `sessions/<ws>/<session>/agents/<agent>/wire.jsonl`, main first."""
    sessions = os.path.join(kimi_home, "sessions")
    found = []
    if not os.path.isdir(sessions):
        return found
    for dirpath, _dirs, files in os.walk(sessions):
        if "wire.jsonl" in files:
            found.append(os.path.join(dirpath, "wire.jsonl"))
    found.sort(key=lambda p: (0 if os.path.basename(os.path.dirname(p)) == "main"
                              else 1, p))
    return found


def read_wire(path: str) -> list:
    """The journal's records as dicts. A malformed line is skipped, never fatal."""
    events = []
    try:
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as e:
                    _log(f"skipping malformed wire line: {e}")
    except OSError as e:
        _log(f"could not read {path}: {e}")
    return events


def summarise(events: list, stream_json: list) -> dict:
    """Session id, final text, turn and tool-call counts, the CLI's own usage.

    Counts are for the MAIN agent. One `step.end` is one model call, which is
    the only thing in the journal that means "turn"; the CLI's own usage
    fields are its `step.end` usages summed into the same four categories the
    forwarder reports, so the two can be compared.
    """
    out = {"session_id": None, "final_text": "", "n_turns": 0,
           "n_tool_calls": 0, "stream_usage": None, "thinking_effort": None,
           "active_tools": None, "disallowed_tools": None,
           "terminal_reason": None, "terminal_error": None,
           "interrupt_message": None}
    tally = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
             "cache_write_tokens": 0, "calls_with_usage": 0}
    for ev in events:
        if not isinstance(ev, dict) or ev.get("agentId", "main") != "main":
            continue
        etype = ev.get("type")
        if etype == "profile.bind":
            out["thinking_effort"] = ev.get("thinkingEffort")
            out["disallowed_tools"] = ev.get("disallowedTools")
            continue
        if etype == "llm.tools_snapshot":
            # The tools ACTUALLY SENT to the model on the first request -- the
            # profile's `activeToolNames` is the allow-list BEFORE the
            # `[tools] disabled` filter, so it still names FetchURL and
            # WebSearch on a closed-book episode; this does not.
            if out["active_tools"] is None:
                names = [t.get("name") for t in ev.get("tools") or []
                         if isinstance(t, dict) and t.get("name")]
                out["active_tools"] = names or None
            continue
        if etype == "turn.ended":
            out["terminal_reason"] = ev.get("reason")
            # The CLI's OWN verdict on the turn, structured: `{"code":
            # "provider.api_error", "message": "524 status code (no body)",
            # "name": "APIStatusError", "details": {"statusCode": 524},
            # "retryable": false}`. This, not the stdout tail, is what killed
            # the episode.
            #
            # REPLACED, not accumulated: both fields describe the LAST turn, so
            # a journal holding an early failed turn and a later good one --
            # which is what a resumed or re-run session leaves behind -- does
            # not report the old fault as this episode's outcome.
            out["terminal_error"] = (ev["error"] if isinstance(ev.get("error"), dict)
                                     else None)
            out["interrupt_message"] = None
            continue
        if etype == "turn.step.interrupted":
            # The same fault one event earlier, already flattened by the CLI:
            # "[provider.api_error] 524 status code (no body)".
            if ev.get("reason") == "error" and ev.get("message"):
                out["interrupt_message"] = str(ev["message"])
            continue
        if etype != "context.append_loop_event":
            continue
        event = ev.get("event") if isinstance(ev.get("event"), dict) else {}
        kind = event.get("type")
        if kind == "tool.call":
            out["n_tool_calls"] += 1
        elif kind == "content.part":
            part = event.get("part") or {}
            if part.get("type") == "text" and str(part.get("text", "")).strip():
                out["final_text"] = part["text"]
        elif kind == "step.end":
            out["n_turns"] += 1
            u = event.get("usage")
            if isinstance(u, dict):
                tally["calls_with_usage"] += 1
                other = int(u.get("inputOther") or 0)
                read = int(u.get("inputCacheRead") or 0)
                write = int(u.get("inputCacheCreation") or 0)
                tally["prompt_tokens"] += other + read + write
                tally["completion_tokens"] += int(u.get("output") or 0)
                tally["cached_tokens"] += read
                tally["cache_write_tokens"] += write
    if tally["calls_with_usage"]:
        out["stream_usage"] = tally
    for line in stream_json:
        if line.get("role") == "meta" and line.get("type") == "session.resume_hint":
            out["session_id"] = line.get("session_id")
    return out


# The CLI's own fatal line, which it prints on stderr as the last thing it
# does: `error: failed to run prompt: provider.api_error: 524 status code (no
# body)`. Anchored to the line start so a tool that merely printed the word
# "error:" is never mistaken for the CLI's exit reason.
_CLI_FATAL = re.compile(r"^error:\s*(.+)$", re.MULTILINE)


def exit_reason(info: dict, stderr_text: str) -> str | None:
    """Why the CLI stopped, from the CLI itself -- never from the stdout tail.

    THE BUG THIS EXISTS TO FIX. This used to be `stdout.splitlines()[-1]`: the
    last `--output-format stream-json` line the CLI wrote before it died. That
    line is whatever the run happened to be doing -- a fetched HTML page as a
    `{"role":"tool",...}` result, a `turn.step.retrying` meta event for a 429
    the CLI then RETRIED SUCCESSFULLY -- and is not the failure. Recording it
    as the failure attributed four crashed episodes to a rate limit that had
    already recovered, while the real cause (a gateway 524) went unrecorded and
    therefore unclassified by `core/agents/errors.py`.

    Three sources, in order of authority:

      * the wire journal's `turn.ended.error`, which is the CLI's structured
        verdict on the turn and carries the provider status code;
      * `turn.step.interrupted.message`, the same fault flattened by the CLI;
      * the last `error: ...` line on the CLI's stderr.
    """
    err = info.get("terminal_error")
    if isinstance(err, dict):
        code = err.get("code") or err.get("name") or "error"
        message = err.get("message") or ""
        detail = err.get("details") if isinstance(err.get("details"), dict) else {}
        status = detail.get("statusCode")
        parts = [f"[{code}] {message}".strip()]
        if status is not None and str(status) not in message:
            parts.append(f"statusCode={status}")
        parts.append(f"retryable={str(bool(err.get('retryable'))).lower()}")
        return " ".join(parts)
    if info.get("interrupt_message"):
        return str(info["interrupt_message"])
    hits = _CLI_FATAL.findall(stderr_text or "")
    return hits[-1].strip() if hits else None


# ---------------------------------------------------------------------------
# the episode
# ---------------------------------------------------------------------------
def observer_message(record):
    """Project committed Kimi journal events to the task observer contract."""
    if record.get("type") != "context.append_loop_event":
        return None
    ev = record.get("event") or {}
    kind = ev.get("type")
    user = False
    if kind == "content.part":
        part = ev.get("part") or {}
        if part.get("type") != "text":
            return None
        block = {"_type": "TextBlock", "text": part.get("text", "")}
    elif kind == "tool.call":
        block = {"_type": "ToolUseBlock", "id": ev.get("toolCallId"),
                 "name": ev.get("name"), "input": ev.get("args") or {}}
    elif kind == "tool.result":
        user = True
        block = {"_type": "ToolResultBlock", "tool_use_id": ev.get("toolCallId"),
                 "content": (ev.get("result") or {}).get("output", ev.get("result"))}
    else:
        return None
    return {"_type": "UserMessage" if user else "AssistantMessage", "content": [block]}


class KimiJournalObserver:
    def __init__(self, home, observer):
        self.home, self.observer = home, observer
        self.offsets = {}
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def drain(self):
        for path in find_wires(self.home):
            with open(path, "rb") as f:
                f.seek(self.offsets.get(path, 0))
                while True:
                    line = f.readline()
                    if not line.endswith(b"\n"):
                        break
                    if line.strip():
                        message = observer_message(json.loads(line))
                        if message:
                            self.observer.observe(message)
                    self.offsets[path] = f.tell()

    def loop(self):
        try:
            while not self.stop.is_set():
                self.drain()
                self.stop.wait(0.05)
        except Exception as exc:
            self.error = str(exc)

    def close(self):
        self.stop.set()
        self.thread.join()
        if self.error is None:
            try:
                self.drain()
            except Exception as exc:
                self.error = str(exc)


def run(task: dict, install: dict | None, grader=None,
        forwarder_mod: dict | None = None, observer=None) -> dict:
    """One episode. `grader` is a running `GraderServer` or None; `forwarder_mod`
    is the exec'd core/agents/forwarder.py namespace, which this adapter always
    ships."""
    cfg = dict(task.get("kimi") or {})
    routing = dict(task.get("routing") or {})
    timeout_s = int(task.get("timeout_s") or 3600)
    key_names = list(routing.get("cli_key_env") or [cfg.get("api_key_env") or "OPENAI_API_KEY"])
    api_key = ""
    for name in key_names:
        if os.environ.get(name):
            api_key = os.environ[name]
            break

    error: str | None = None
    if not api_key:
        error = f"{'/'.join(key_names)} is not set inside the container"
    allowed_tools = None
    if cfg.get("mcp_only"):
        if (grader is None or grader.grader_server_name not in {"chess", "go"}
                or set(grader.grader_tools) != MCP_NAMES):
            error = "Restricted Kimi requires the complete nine-tool game grader"
        else:
            allowed_tools = {f"mcp__{grader.grader_server_name}__{name}" for name in MCP_NAMES}

    # THIS PROCESS's environment gets the passthrough (the task's grader runs
    # here and reads OPENAI_BASE_URL for its own model); the CLI's copy is
    # built below and loses the credential.
    os.environ.update({str(k): str(v) for k, v in (task.get("env") or {}).items()})

    upstream_base = routing.get("api_base_url") or cfg.get("api_base_url") or ""
    forwarder = None
    probe = None
    if error is None and not upstream_base:
        error = ("kimi-code has no upstream to forward to: set api_base_url on "
                 "the model entry or OPENAI_BASE_URL in .env")
    elif error is None and not forwarder_mod:
        error = "the forwarder module did not arrive in the blob"
    elif error is None:
        try:
            upstream = forwarder_mod["Upstream"](
                upstream_base, api_key, routing.get("extra_body") or {})
            forwarder = KimiForwarder(upstream=upstream, allowed_tools=allowed_tools).start()
            _log(f"forwarder on {forwarder.base_url} -> {upstream.base_url} "
                 f"(extra_body keys: {sorted(routing.get('extra_body') or {})})")
            if cfg.get("probe", True):
                probe = forwarder.probe(cfg["model_name"])
                _log(f"forwarder probe: status={probe.get('status')} "
                     f"provider={probe.get('provider')!r} model={probe.get('model')!r}"
                     + (f" error={probe.get('error')}" if probe.get("error") else ""))
        except Exception as e:  # noqa: BLE001
            error = f"forwarder failed to start: {type(e).__name__}: {e}"

    try:
        prepare_home(cfg, grader_url=grader.url if grader is not None else "",
                     grader_name=(grader.grader_server_name
                                  if grader is not None else "grader"))
    except Exception as e:  # noqa: BLE001
        error = error or f"kimi home setup failed: {type(e).__name__}: {e}"

    # The CLI's environment: everything this process has, MINUS the credential
    # under every name it travelled in, PLUS the KIMI_* configuration. The
    # placeholder key is what the CLI insists on; the forwarder replaces the
    # Authorization header with the real one.
    env = dict(os.environ)
    for name in key_names:
        env.pop(name, None)
    env.update({str(k): str(v) for k, v in (cfg.get("cli_env") or {}).items()})
    env["KIMI_CODE_HOME"] = cfg["kimi_home"]
    env["KIMI_MODEL_NAME"] = cfg["model_name"]
    env["KIMI_MODEL_API_KEY"] = cfg.get("placeholder_key") or "rh-forwarder"
    if forwarder is not None:
        env["KIMI_MODEL_BASE_URL"] = forwarder.base_url

    command = compose_command(cfg, task["content"])
    _log(f"kimi command: {command}")
    _log("kimi env: " + " ".join(f"{k}={env[k]}" for k in sorted(env)
                                 if k.startswith("KIMI_") and k != "KIMI_MODEL_API_KEY"))

    t0 = time.time()
    rc: int | None = None
    stream_json: list = []
    stdout_text = ""
    stderr_text = ""
    if error is None:
        tail = KimiJournalObserver(cfg["kimi_home"], observer) if observer is not None else None
        if tail is not None:
            tail.thread.start()
        try:
            rc, stream_json, stdout_text, stderr_text, error = run_kimi(
                command, env, timeout_s)
        finally:
            if tail is not None:
                tail.close()
        if tail is not None and tail.error:
            error = error or "live observer failed: " + tail.error
    wall = time.time() - t0

    # Stop serving the moment the CLI is gone, and BEFORE state is read.
    if grader is not None:
        grader.close()
    forwarder_calls: list = []
    if forwarder is not None:
        forwarder.close()
        forwarder_calls = list(forwarder.calls)
        error = error or forwarder.policy_error

    # Read the journals WHATEVER happened: the CLI appends as it goes, so a
    # killed episode still has everything up to the kill.
    wire_paths = []
    events: list = []
    try:
        wire_paths = find_wires(cfg["kimi_home"])
        for path in wire_paths:
            recs = read_wire(path)
            _log(f"wire {path}: {len(recs)} records")
            events.extend(recs)
        if not wire_paths:
            _log(f"no wire.jsonl under {cfg['kimi_home']}/sessions")
    except Exception as e:  # noqa: BLE001
        _log(f"wire lookup failed: {type(e).__name__}: {e}")

    if not events and stream_json:
        # No journal at all (the CLI died before its first write): keep the
        # stream-json lines so the trajectory is not empty.
        events = [{"type": "stream_json", "agentId": "main", "line": ln}
                  for ln in stream_json]

    messages = list(events) + forwarder_calls
    for ev in messages:
        _emit_msg(ev)

    info = summarise(events, stream_json)
    reason = exit_reason(info, stderr_text)
    if error is None and rc not in (0, None) and not events:
        error = (f"kimi exited {rc} and wrote no journal"
                 + (f": {reason}" if reason else ""))
    if error is None and rc not in (0, None):
        # A non-zero exit WITH a journal: the CLI reported a failed turn. Name
        # what the CLI says killed it, not whatever its stdout ended on.
        error = f"kimi exited {rc}" + (f": {reason}" if reason else "")
    if error is None and info["terminal_reason"] not in (None, "completed"):
        # The turn failed but the process still exited 0. The episode was cut
        # off just the same, so it is an errored episode -- `ok: false`, a
        # typed `failure`, and the run-level auto-retry gets to see it --
        # rather than a short transcript that reads like a finished one.
        error = (f"kimi ended the turn {info['terminal_reason']}"
                 + (f": {reason}" if reason else ""))

    try:
        deliverables = collect_deliverables(
            WORKDIR, tuple(task.get("skip_dirs") or ()),
            tuple(task.get("deliverable_files") or ()))
    except Exception as e:  # noqa: BLE001
        _log(f"deliverable collection failed: {type(e).__name__}: {e}")
        deliverables = []

    # Cleanup, best effort: the config and the journals do not outlive the
    # episode (they are already inside the record).
    if cfg.get("kimi_home"):
        try:
            subprocess.run(["rm", "-rf", cfg["kimi_home"]], timeout=60)
        except Exception:  # noqa: BLE001
            pass

    return {
        "id": task["id"],
        "model": task["model"],
        # The wire journals, in order, then the forwarder's per-call records.
        # THIS is the trajectory.
        "messages": messages,
        "grader_state": grader.grader_state if grader is not None else None,
        "deliverables": deliverables,
        "final_text": info["final_text"],
        "n_turns": info["n_turns"],
        "n_tool_calls": info["n_tool_calls"],
        "wall_time": round(wall, 1),
        "cost_usd": None,
        # The CLI's own per-step tally (fallback), and the forwarder's (primary).
        "stream_usage": info["stream_usage"],
        "usage": usage_from_calls(forwarder_calls),
        "result_subtype": None,
        "terminal_reason": info["terminal_reason"],
        # The CLI's structured verdict on the turn, verbatim, so a reader never
        # has to re-parse the sentence in `error` to get the provider status.
        "terminal_error": info["terminal_error"],
        "session_id": info["session_id"],
        "init_tools": info["active_tools"] or task.get("tools"),
        "init_mcp_servers": [grader.report()] if grader is not None else [],
        "install": install,
        "kimi_command": command,
        "wire_paths": wire_paths,
        "forwarder_calls": [{k: v for k, v in c.items()
                             if k in ("index", "method", "path", "status",
                                      "request_model", "stream", "id", "model",
                                      "provider", "finish_reason", "usage",
                                      "error", "bytes_in", "bytes_out", "seconds")}
                            for c in forwarder_calls],
        "forwarder_probe": probe,
        "config_toml": cfg.get("config_toml") or "",
        "thinking_effort_bound": info["thinking_effort"],
        "returncode": rc,
        "error": error,
    }


def main(task: dict, modules: dict) -> None:
    """Stage the row's files, install the CLI, host the grader and the
    forwarder, run it."""
    try:
        os.chdir(WORKDIR)
    except OSError:
        pass

    ns: dict = {}
    grader_src = (modules.get("grader") or "").strip()
    forwarder_src = (modules.get("forwarder") or "").strip()
    for name in ("stage", *(("grader",) if grader_src else ()),
                 *(("forwarder",) if forwarder_src else ())):
        g: dict = {"__name__": f"rh_{name}"}
        exec(compile(modules[name], f"<{name}>", "exec"), g)
        ns[name] = g

    staged: dict = {}
    try:
        staged = ns["stage"]["write"](task.get("files") or [], WORKDIR)
        for dest, q in staged.items():
            _log(f"staged {dest} ({os.path.getsize(q)} bytes, "
                 f"mode {oct(os.stat(q).st_mode & 0o777)})")
    except Exception as e:  # noqa: BLE001
        _log(f"staging failed: {type(e).__name__}: {e}")

    install = install_agent(task.get("install"))

    # Install task dependencies after the agent runtime, without credentials.
    # Failure is recorded but does not prevent the episode from running.
    task_install = install_agent(task.get("task_install"),
                                 env=task_install_env())

    grader = observer = None
    grader_error: dict | None = None
    if not grader_src:
        _log("no grader module in this episode's payload: task ships none")
    else:
        try:
            grader = serve_grader(ns["grader"], task.get("row") or {}, staged,
                                  WORKDIR)
            _log(f"grader MCP server on 127.0.0.1:"
                 f"{grader.server_address[1]} (path withheld from this log)")
        except Exception as e:  # noqa: BLE001
            _log(f"grader construction failed: {type(e).__name__}: {e}")
            grader_error = {"error": f"{type(e).__name__}: {e}"}

    try:
        if grader is not None and ns.get("grader", {}).get("make_observer"):
            observer = ns["grader"]["make_observer"](
                task.get("row") or {}, staged, WORKDIR, grader.grader_state)
        record = run(task, install, grader, forwarder_mod=ns.get("forwarder"), observer=observer)
        if grader_error is not None:
            record["grader_state"] = grader_error
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc(file=sys.stderr)
        record = {"id": task.get("id"), "model": task.get("model"),
                  "messages": [],
                  "grader_state": (grader_error if grader_error is not None
                                   else (grader.grader_state
                                         if grader is not None else None)),
                  "final_text": "", "install": install,
                  "error": f"{type(e).__name__}: {e}"}
    finally:
        if observer is not None:
            try:
                observer.finish()
            except Exception as exc:
                record["error"] = f"{record.get('error') or ''} observer finish failed: {exc}".strip()
        if grader is not None:
            record["grader_state"] = grader.grader_state
            grader.close()
    record["task_install"] = task_install
    _OUT.write(json.dumps(record) + "\n")
    _OUT.flush()


_TASK = globals().get("TASK")
if _TASK is not None:
    main(_TASK, globals().get("MODULES") or {})
