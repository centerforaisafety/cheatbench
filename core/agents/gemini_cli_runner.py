"""The Gemini CLI episode runner that executes INSIDE the container.

Same contract as `core/agents/codex_runner.py`, and for the same reasons: this
file lives in the repo, is never baked into the image, is streamed in on
**stdin** with `core/stage.py` and the staged bytes, and writes exactly ONE line
of JSON on stdout. Everything else -- the CLI's own output, tracebacks, turn
streaming -- goes to stderr, so the host parses stdout without heuristics.

What is different from the Codex runner, and why:

  * The episode is one `gemini --yolo --model=<m> --prompt=<instruction>`
    subprocess (Harbor's invocation, `harbor/agents/installed/gemini_cli.py`).
    `--yolo` is silently ignored in an untrusted workspace, so
    `GEMINI_CLI_TRUST_WORKSPACE=true` is in its environment; headless mode
    cannot show the auth dialog, so `security.auth.selectedType` is pinned in
    `~/.gemini/settings.json`.
  * THE TRAJECTORY IS NOT ON STDOUT. The record Harbor parses is the session
    file the CLI writes to `~/.gemini/tmp/<project>/chats/session-*.jsonl`
    (JSONL since 0.40; one JSON document before that). There is no bind mount
    here, so this runner READS that file after the CLI exits and returns its
    records inside the JSON line it already owes the host. Host turn streaming
    is a replay after the run. Tasks with an observer additionally get a live
    tail of the durable session, independently of host transcript collection.
  * THE MODEL API IS PROXIED THROUGH A LOOPBACK SHIM (gateway route). The CLI
    speaks the Gemini-native API through `@google/genai` and honours
    `GOOGLE_GEMINI_BASE_URL` for its endpoint -- but the org's LiteLLM gateway
    only exposes Gemini through its OpenAI-compatible `/v1/chat/completions`
    (its Gemini-native routes sit behind a Google OAuth login wall). So the
    runner hosts `GatewayShim` on 127.0.0.1: it accepts
    `/v1beta/models/{model}:generateContent`, `:streamGenerateContent?alt=sse`
    and `:countTokens`, translates each request to OpenAI chat format, forwards
    it to the configured OpenAI-compatible upstream (`api_base_url`, then
    `GEMINI_BASE_URL`, then legacy `OPENAI_BASE_URL`) with the host's key, and
    translates the response (streaming SSE included) back into Gemini's shape.
    The CLI is given the shim's URL and a per-episode random `GEMINI_API_KEY`,
    which the shim requires on every request, so a neighbouring episode on the
    same loopback (a task with no private netns) reaches nothing.

    Two things the shim must get right, both verified against the gateway
    (litellm 1.82.3) rather than assumed:

      thought signatures   Gemini 3 requires the signature it attached to a
                           function call to come back with that call on the
                           next turn. litellm carries it OUT as
                           `tool_calls[i].provider_specific_fields.
                           thought_signature` (and a `__thought__<sig>` suffix
                           on the call id) and, on text-only turns, as
                           `message.provider_specific_fields.thought_signatures`;
                           it carries them back IN through the same fields (a
                           corrupted value in either place is a 400 "Corrupted
                           thought signature" from Google, which is how we know
                           they are read). The shim maps `part.thoughtSignature`
                           <-> those fields in both directions.
      generation config    the CLI's `generationConfig` keys (`thinkingConfig`,
                           `topK`, `maxOutputTokens`, `responseJsonSchema`, ...)
                           are forwarded VERBATIM as top-level request keys;
                           litellm passes them into Gemini's generationConfig
                           unchanged (a bogus `thinkingLevel` is rejected by
                           Google, so they arrive). Only `topP` is renamed to
                           `top_p`. The shim sets NOTHING of its own: no
                           temperature, no top_p, no seed, no token cap -- what
                           the CLI sent is what the model gets.

    In the native route a relay pins the model ID while preserving Google
    request and response bodies. The CLI receives only a per-episode key.
  * The grader is hosted over HTTP, as for Codex. Gemini CLI takes an MCP
    server from `settings.json` as `{"httpUrl": ...}` (streamable-http); the
    `GraderServer` is the same loopback endpoint behind an unguessable path.
"""
from __future__ import annotations

import copy
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
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
# other runners: a property of the TASK, not of the agent.
DELIVERABLE_EXTS = (".xlsx", ".xls", ".xlsm", ".docx", ".doc", ".pptx", ".ppt",
                    ".pdf", ".csv", ".svg")
_DELIVERABLE_MAX_BYTES = 64 * 1024 * 1024

INSTALL_TIMEOUT_S = 600


def collect_deliverables(root: str, skip_dirs: tuple, include_files: tuple = ()) -> list:
    """Export office files and exact task-requested outputs for the host."""
    import base64
    import stat

    # Task rows may request exact additional outputs (e.g. a ZIP and message).
    # These names do not broaden collection to other archives or text files.
    requested = {p for p in include_files if isinstance(p, str)
                 and p and not os.path.isabs(p)
                 and not any(part in ("", ".", "..") for part in p.split("/"))}
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
                # No symlink files; os.walk already excludes symlink directories.
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
# the adapter's runtime: `check || install`, run after staging and before the
# CLI exists. Same contract and reporting as the other runners.
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
            return (res.stdout or "").strip().split("\n")[0] if \
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
# The same loopback streamable-HTTP endpoint `codex_runner.py` hosts, for the
# same reason: a `command` entry in a config file the agent can read is a path
# to the grader's source and the answer key, while a URL is an endpoint that
# only ever speaks verdicts. Gemini CLI's MCP client (the official TypeScript
# SDK) speaks the same four methods in the same order and tolerates a 405 on
# its optional server-to-client GET stream.
# ---------------------------------------------------------------------------
MCP_PROTOCOL_VERSION = "2025-06-18"


class _GraderMCPHandler(BaseHTTPRequestHandler):
    """Streamable-HTTP MCP: initialize, notifications/initialized, tools/list,
    tools/call. GET and DELETE are 405, which the client treats as "no
    server-initiated stream" rather than as an error."""

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
        if request.get("id") is None:       # a JSON-RPC notification
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
            return ok({"tools": [{k: v for k, v in spec.items() if k != "call"}
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


class _LoopbackServer(ThreadingHTTPServer):
    """A threaded HTTP server on 127.0.0.1 and a kernel-assigned port, run on
    a daemon thread, with an idempotent close. Shared by the grader endpoint
    and the API shim."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, handler) -> None:
        super().__init__(("127.0.0.1", 0), handler)
        self._thread: threading.Thread | None = None

    @property
    def origin(self) -> str:
        host, port = self.server_address[0], self.server_address[1]
        return f"http://{host}:{port}"

    def start(self) -> None:
        self._thread = threading.Thread(target=self.serve_forever,
                                        kwargs={"poll_interval": 0.2},
                                        daemon=True)
        self._thread.start()

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


class GraderServer(_LoopbackServer):
    """The running grader endpoint, and the handle the episode holds it by."""

    def __init__(self, *, state: dict, call, server_name: str, tool_name: str,
                 description: str, schema: dict, extra_tools=()):
        tools = [{"name": tool_name, "description": description,
                  "inputSchema": schema, "call": call}, *extra_tools]
        names = [spec["name"] for spec in tools]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate grader tool names")
        if any(not callable(spec["call"]) for spec in tools):
            raise ValueError("Each grader tool needs a callable handler")
        super().__init__(_GraderMCPHandler)
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

    @property
    def url(self) -> str:
        return self.origin + self.grader_path

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
    extra_factory = module.get("make_extra_tools")
    extra_tools = extra_factory(row, staged, workdir, state) if extra_factory else ()
    primary = module["tool_spec"](row) if "tool_spec" in module else {}
    server = GraderServer(
        state=state, call=call,
        server_name=module.get("SERVER_NAME", "grader"),
        tool_name=primary.get("name", module.get("TOOL_BASENAME", "grade_deliverable")),
        description=primary.get("description", module.get("DESCRIPTION", "")),
        schema=primary.get("inputSchema", module.get("INPUT_SCHEMA"))
        or {"type": "object", "properties": {}}, extra_tools=extra_tools)
    server.start()
    return server


# ===========================================================================
# the Gemini-native <-> OpenAI-chat translation
#
# Pure functions, so they can be tested on the host without a container. The
# OpenAI side is what litellm's gateway accepts and returns for a `gemini/...`
# model; the Gemini side is what `@google/genai` sends and expects.
# ===========================================================================
THOUGHT_SUFFIX = "__thought__"

# Gemini's function-calling mode names -> OpenAI `tool_choice`.
_TOOL_MODE = {"AUTO": "auto", "ANY": "required", "NONE": "none",
              "VALIDATED": "auto"}

# OpenAI `finish_reason` -> Gemini `finishReason`.
_FINISH = {"stop": "STOP", "length": "MAX_TOKENS", "tool_calls": "STOP",
           "function_call": "STOP", "content_filter": "SAFETY"}

# HTTP status -> Gemini's `status` string, so the CLI's retry logic sees the
# shape it expects on a 429 or a 5xx.
_STATUS = {400: "INVALID_ARGUMENT", 401: "UNAUTHENTICATED",
           403: "PERMISSION_DENIED", 404: "NOT_FOUND",
           429: "RESOURCE_EXHAUSTED", 500: "INTERNAL", 502: "UNAVAILABLE",
           503: "UNAVAILABLE", 504: "DEADLINE_EXCEEDED"}


def _lower_schema_types(schema):
    """Gemini `Schema` writes `type: "OBJECT"`; JSON schema writes `object`.

    litellm converts JSON schema to Gemini's shape itself, so a declaration the
    CLI sent in Gemini's shape has its type names lowered first. Everything
    else is left alone.
    """
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, str):
                out[k] = v.lower()
            elif k == "type" and isinstance(v, list):
                out[k] = [t.lower() if isinstance(t, str) else t for t in v]
            else:
                out[k] = _lower_schema_types(v)
        return out
    if isinstance(schema, list):
        return [_lower_schema_types(x) for x in schema]
    return schema


def _split_call_id(call_id: str) -> tuple[str, str | None]:
    """litellm's `call_x__thought__<sig>` -> (`call_x`, sig)."""
    if isinstance(call_id, str) and THOUGHT_SUFFIX in call_id:
        head, _, sig = call_id.partition(THOUGHT_SUFFIX)
        return head, sig or None
    return call_id, None


def _system_text(system) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, dict):
        return "".join(p.get("text", "") for p in (system.get("parts") or [])
                       if isinstance(p, dict))
    return ""


def _tool_result_text(response) -> str:
    """What a `functionResponse.response` becomes as a tool message.

    Gemini CLI reports a tool's result as `{"output": "..."}` (or `{"error":
    ...}`). The gateway will wrap whatever string we send as `{"content":
    <string>}` on the Gemini side, so the shape cannot be preserved exactly;
    the CONTENT is, in full: a lone string-valued `output` is sent bare, and
    anything else is sent as JSON so an error stays recognisable as one.
    """
    if isinstance(response, dict):
        if set(response) == {"output"} and isinstance(response["output"], str):
            return response["output"]
        return json.dumps(response, ensure_ascii=False)
    if response is None:
        return ""
    return response if isinstance(response, str) else json.dumps(response)


def _media_part(part: dict):
    """An `inlineData` part as an OpenAI content part, or a note when the
    OpenAI schema has no slot for it."""
    blob = part.get("inlineData") or {}
    mime = blob.get("mimeType") or "application/octet-stream"
    data = blob.get("data") or ""
    if mime.startswith("image/") and data:
        return {"type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"}}
    return {"type": "text",
            "text": f"<{mime} attachment, {len(data)} base64 chars, not "
                    f"forwardable through the OpenAI-compatible route>"}


def gemini_to_openai(body: dict, model: str, stream: bool) -> tuple[dict, list]:
    """One Gemini `generateContent` body -> one OpenAI chat request.

    Returns `(request, notes)`, where `notes` names anything that could not be
    carried across the schema boundary (a server-side tool, a non-image
    attachment). Nothing is ever silently dropped without a note.
    """
    notes: list = []
    messages: list = []

    system = _system_text(body.get("systemInstruction") or body.get("system_instruction"))
    if system:
        messages.append({"role": "system", "content": system})

    last_calls: list = []       # the previous model turn's tool calls, in order
    n_calls = 0
    for content in body.get("contents") or []:
        if not isinstance(content, dict):
            continue
        role = content.get("role") or "user"
        parts = [p for p in (content.get("parts") or []) if isinstance(p, dict)]

        if role == "model":
            texts, tool_calls, sigs = [], [], []
            for p in parts:
                if p.get("functionCall"):
                    fc = p["functionCall"]
                    n_calls += 1
                    call_id = fc.get("id") or f"call_{n_calls}"
                    tc = {"id": call_id, "type": "function",
                          "function": {"name": fc.get("name") or "",
                                       "arguments": json.dumps(fc.get("args") or {})}}
                    if p.get("thoughtSignature"):
                        tc["provider_specific_fields"] = {
                            "thought_signature": p["thoughtSignature"]}
                    tool_calls.append(tc)
                elif p.get("thought"):
                    # A thought summary from an earlier turn. The OpenAI
                    # schema has no slot for it and Gemini does not need it
                    # back: the signature is what carries the reasoning state.
                    if p.get("thoughtSignature"):
                        sigs.append(p["thoughtSignature"])
                elif "text" in p:
                    if p.get("text"):
                        texts.append(p["text"])
                    if p.get("thoughtSignature"):
                        sigs.append(p["thoughtSignature"])
                elif p.get("inlineData"):
                    notes.append("model-turn inlineData part dropped")
            msg: dict = {"role": "assistant", "content": "".join(texts)}
            if tool_calls:
                msg["tool_calls"] = tool_calls
                if not texts:
                    msg["content"] = None
            if sigs:
                msg["provider_specific_fields"] = {"thought_signatures": sigs}
            messages.append(msg)
            last_calls = tool_calls
            continue

        # user (or anything else): text/media parts become one user message,
        # functionResponse parts become tool messages, in the order they came.
        pending: list = []
        trailing_media: list = []

        def flush() -> None:
            if pending:
                messages.append({"role": "user", "content": list(pending)})
                pending.clear()

        n_resp = 0
        for p in parts:
            if p.get("functionResponse"):
                flush()
                fr = p["functionResponse"]
                call_id = fr.get("id")
                if not call_id:
                    # Pair by position with the previous model turn, which is
                    # how Gemini itself pairs a response that carries no id.
                    if n_resp < len(last_calls):
                        call_id = last_calls[n_resp]["id"]
                    else:
                        call_id = f"call_{n_calls}"
                n_resp += 1
                messages.append({"role": "tool", "tool_call_id": call_id,
                                 "content": _tool_result_text(fr.get("response"))})
                for mp in fr.get("parts") or []:
                    if isinstance(mp, dict) and mp.get("inlineData"):
                        # A tool returned media. A tool message cannot carry it
                        # through the OpenAI route, so it follows as a user
                        # message; the model still sees it, one turn later.
                        trailing_media.append(_media_part(mp))
                        notes.append("tool media forwarded as a user part")
            elif "text" in p:
                if p.get("text"):
                    pending.append({"type": "text", "text": p["text"]})
            elif p.get("inlineData"):
                pending.append(_media_part(p))
            elif p.get("fileData"):
                uri = (p["fileData"] or {}).get("fileUri", "")
                pending.append({"type": "text",
                                "text": f"<fileData {uri} is not forwardable "
                                        f"through the OpenAI-compatible route>"})
                notes.append("fileData part replaced by a note")
        if trailing_media:
            pending.extend(trailing_media)
        flush()

    request: dict = {"model": model, "messages": messages}

    # -- tools --------------------------------------------------------------
    tools: list = []
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            continue
        for fd in t.get("functionDeclarations") or t.get("function_declarations") or []:
            fn = {"name": fd.get("name") or "",
                  "description": fd.get("description") or ""}
            params = fd.get("parametersJsonSchema") or fd.get("parameters")
            if params:
                fn["parameters"] = _lower_schema_types(params)
            tools.append({"type": "function", "function": fn})
        for key in t:
            if key not in ("functionDeclarations", "function_declarations"):
                # googleSearch, urlContext, codeExecution: Google-side tools
                # with no OpenAI equivalent. Named, never silently lost.
                notes.append(f"server-side tool {key} dropped")
    if tools:
        request["tools"] = tools
        fcc = ((body.get("toolConfig") or {}).get("functionCallingConfig") or {})
        mode = str(fcc.get("mode") or "").upper()
        allowed = fcc.get("allowedFunctionNames") or []
        if mode == "ANY" and len(allowed) == 1:
            request["tool_choice"] = {"type": "function",
                                      "function": {"name": allowed[0]}}
        elif mode in _TOOL_MODE:
            request["tool_choice"] = _TOOL_MODE[mode]

    # -- generation config: forwarded verbatim, renamed only where the OpenAI
    #    schema already has the key. NOTHING is added.
    gen = body.get("generationConfig") or body.get("generation_config") or {}
    for key, value in gen.items():
        if key == "topP":
            request["top_p"] = value
        else:
            request[key] = value
    if body.get("safetySettings"):
        request["safety_settings"] = body["safetySettings"]
    if body.get("cachedContent"):
        request["cached_content"] = body["cachedContent"]

    if stream:
        request["stream"] = True
        request["stream_options"] = {"include_usage": True}
    return request, notes


def usage_to_gemini(usage: dict | None) -> dict | None:
    """OpenAI `usage` -> Gemini `usageMetadata`. Reasoning tokens are a
    separate category on both sides; litellm folds them into
    `completion_tokens` and reports them again in `completion_tokens_details`,
    so `candidatesTokenCount` is the difference."""
    if not isinstance(usage, dict):
        return None
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    details = usage.get("completion_tokens_details") or {}
    reasoning = int(details.get("reasoning_tokens") or 0)
    pdetails = usage.get("prompt_tokens_details") or {}
    cached = int(pdetails.get("cached_tokens") or 0)
    out = {"promptTokenCount": prompt,
           "candidatesTokenCount": max(completion - reasoning, 0),
           "totalTokenCount": int(usage.get("total_tokens") or prompt + completion)}
    if reasoning:
        out["thoughtsTokenCount"] = reasoning
    if cached:
        out["cachedContentTokenCount"] = cached
    return out


def _function_call_part(call: dict) -> dict:
    """One OpenAI tool call -> one Gemini functionCall part, signature attached."""
    fn = call.get("function") or {}
    raw_args = fn.get("arguments")
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args else (raw_args or {})
    except json.JSONDecodeError:
        args = {"_raw": raw_args}
    if not isinstance(args, dict):
        args = {"value": args}
    call_id, id_sig = _split_call_id(call.get("id") or "")
    part: dict = {"functionCall": {"name": fn.get("name") or "", "args": args}}
    if call_id:
        part["functionCall"]["id"] = call_id
    sig = (call.get("provider_specific_fields") or {}).get("thought_signature") or id_sig
    if sig:
        part["thoughtSignature"] = sig
    return part


def _attach_message_signatures(parts: list, sigs: list) -> None:
    """Put a message-level signature where Gemini itself puts it.

    A function-call turn's signature already rides on its call (litellm reports
    it in both places; one copy is enough). A text turn's signature goes on the
    LAST text part, which is where Gemini attaches it. A turn with nothing but
    thoughts gets an empty text part to carry it -- `{"text": "",
    "thoughtSignature": ...}` passes the CLI's own validity check, which only
    rejects an empty text part that carries nothing else.
    """
    if not sigs or any(p.get("thoughtSignature") for p in parts):
        return
    for p in reversed(parts):
        if "text" in p and not p.get("thought"):
            p["thoughtSignature"] = sigs[-1]
            return
    parts.append({"text": "", "thoughtSignature": sigs[-1]})


def openai_to_gemini(resp: dict) -> dict:
    """One OpenAI chat completion -> one Gemini `GenerateContentResponse`."""
    choices = resp.get("choices") or []
    choice = choices[0] if choices else {}
    msg = choice.get("message") or {}
    parts: list = []
    if msg.get("reasoning_content"):
        parts.append({"text": msg["reasoning_content"], "thought": True})
    if msg.get("content"):
        parts.append({"text": msg["content"]})
    for call in msg.get("tool_calls") or []:
        parts.append(_function_call_part(call))
    _attach_message_signatures(
        parts, (msg.get("provider_specific_fields") or {}).get("thought_signatures") or [])
    candidate: dict = {"content": {"role": "model", "parts": parts}, "index": 0}
    reason = choice.get("finish_reason")
    if reason:
        candidate["finishReason"] = _FINISH.get(reason, "OTHER")
    out: dict = {"candidates": [candidate]}
    usage = usage_to_gemini(resp.get("usage"))
    if usage:
        out["usageMetadata"] = usage
    if resp.get("model"):
        out["modelVersion"] = resp["model"]
    if resp.get("id"):
        out["responseId"] = resp["id"]
    return out


class StreamTranslator:
    """OpenAI chat-completion chunks -> Gemini streaming chunks, in order.

    Thought summaries and text are passed through as they arrive; the newest
    text part is held back by ONE delta so a signature that follows it can be
    attached to it rather than to an empty part. Function calls are emitted
    on the final chunk together with `finishReason` and `usageMetadata`, which
    is also where Gemini itself puts them.
    """

    def __init__(self) -> None:
        self.held_text: dict | None = None
        self.calls: dict = {}          # index -> accumulating OpenAI tool call
        self.sigs: list = []
        self.finish: str | None = None
        self.usage: dict | None = None
        self.model: str | None = None
        self.response_id: str | None = None
        self.done = False

    @staticmethod
    def _chunk(parts: list, **extra) -> dict:
        out = {"candidates": [{"content": {"role": "model", "parts": parts},
                               "index": 0, **extra}]}
        return out

    def feed(self, obj: dict) -> list:
        out: list = []
        self.model = self.model or obj.get("model")
        self.response_id = self.response_id or obj.get("id")
        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                out.append(self._chunk([{"text": delta["reasoning_content"],
                                         "thought": True}]))
            if delta.get("content"):
                if self.held_text is not None:
                    out.append(self._chunk([self.held_text]))
                self.held_text = {"text": delta["content"]}
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", len(self.calls))
                acc = self.calls.setdefault(idx, {"id": "", "type": "function",
                                                  "function": {"name": "", "arguments": ""},
                                                  "provider_specific_fields": {}})
                if tc.get("id"):
                    acc["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    acc["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    acc["function"]["arguments"] += fn["arguments"]
                psf = tc.get("provider_specific_fields") or {}
                if psf.get("thought_signature"):
                    acc["provider_specific_fields"]["thought_signature"] = psf["thought_signature"]
            psf = delta.get("provider_specific_fields") or {}
            for sig in psf.get("thought_signatures") or []:
                if sig not in self.sigs:
                    self.sigs.append(sig)
            if choice.get("finish_reason"):
                self.finish = _FINISH.get(choice["finish_reason"], "OTHER")
        if obj.get("usage"):
            self.usage = obj["usage"]
        return out

    def finalize(self) -> list:
        """The closing chunk(s), once for the whole stream."""
        if self.done:
            return []
        self.done = True
        parts: list = []
        if self.held_text is not None:
            parts.append(self.held_text)
            self.held_text = None
        for idx in sorted(self.calls):
            parts.append(_function_call_part(self.calls[idx]))
        _attach_message_signatures(parts, self.sigs)
        extra: dict = {}
        if self.finish:
            extra["finishReason"] = self.finish
        chunk = self._chunk(parts, **extra)
        usage = usage_to_gemini(self.usage)
        if usage:
            chunk["usageMetadata"] = usage
        if self.model:
            chunk["modelVersion"] = self.model
        if self.response_id:
            chunk["responseId"] = self.response_id
        return [chunk]


def gemini_error(code: int, message: str) -> bytes:
    return json.dumps({"error": {"code": code, "message": message,
                                 "status": _STATUS.get(code, "UNKNOWN")}}).encode()


# ===========================================================================
# the shim: a Gemini-native endpoint on loopback, backed by the gateway
# ===========================================================================
_ROUTE = re.compile(r"^/(?:v1beta|v1|v1alpha)/models/([^:/]+):"
                    r"(generateContent|streamGenerateContent|countTokens)$")

# How long one upstream socket operation may block. Not a cap on the
# response: a thinking turn streams for as long as it takes, and this only
# fires when NOTHING arrives for this long.
UPSTREAM_IO_TIMEOUT_S = 900


class _ShimHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        """Silence; the shim keeps its own one-line-per-call log."""

    # -- plumbing --------------------------------------------------------
    def _authorised(self, query: dict) -> bool:
        key = (self.headers.get("x-goog-api-key")
               or (query.get("key") or [""])[0] or "")
        return hmac.compare_digest(key, self.server.token)

    def _reply(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:       # noqa: N802
        self._reply(404, gemini_error(404, "not found"))

    def do_POST(self) -> None:      # noqa: N802
        path, _, query_str = self.path.partition("?")
        query = urllib.parse.parse_qs(query_str)
        if not self._authorised(query):
            self._reply(403, gemini_error(403, "bad api key"))
            return
        m = _ROUTE.match(path)
        if not m:
            self._reply(404, gemini_error(404, f"no route for {path}"))
            return
        model, method = urllib.parse.unquote(m.group(1)), m.group(2)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) if length else b"{}")
        except (ValueError, UnicodeDecodeError) as e:
            self._reply(400, gemini_error(400, f"bad request body: {e}"))
            return
        if not isinstance(body, dict):
            self._reply(400, gemini_error(400, "request body must be an object"))
            return
        try:
            if method == "countTokens":
                self._count_tokens(body)
            elif method == "streamGenerateContent":
                self._generate(model, body, stream=True)
            else:
                self._generate(model, body, stream=False)
        except Exception as e:  # noqa: BLE001 - one bad call must not kill the shim
            _log(f"shim: {method} failed: {type(e).__name__}: {e}")
            try:
                self._reply(500, gemini_error(500, f"shim: {type(e).__name__}: {e}"))
            except Exception:  # noqa: BLE001
                pass

    # -- endpoints -------------------------------------------------------
    def _count_tokens(self, body: dict) -> None:
        """An ESTIMATE. The OpenAI-compatible route has no token counter, and
        the CLI only calls this for media-bearing content (it estimates text
        itself at four characters per token, which is what this does too)."""
        text = json.dumps(body.get("contents") or [], ensure_ascii=False)
        self._reply(200, json.dumps({"totalTokens": max(1, len(text) // 4)}).encode())

    def _generate(self, model: str, body: dict, *, stream: bool) -> None:
        server = self.server
        upstream_model = server.upstream_model(model)
        request, notes = gemini_to_openai(body, upstream_model, stream)
        try:
            server.observe_request(body, model, upstream_model, notes)
        except ValueError as exc:
            self._reply(403, gemini_error(403, str(exc)))
            return

        data = json.dumps(request).encode()
        req = urllib.request.Request(
            server.upstream_url, data=data, method="POST",
            headers={"Authorization": f"Bearer {server.api_key}",
                     "Content-Type": "application/json",
                     # Not decoration: the gateway sits behind Cloudflare, which
                     # answers urllib's default `Python-urllib/x.y` with a 403
                     # ban page (error 1010). Any other name passes.
                     "User-Agent": "gemini-cli-shim/1.0",
                     "Accept": "text/event-stream" if stream else "application/json"})
        t0 = time.time()
        try:
            resp = urllib.request.urlopen(req, timeout=UPSTREAM_IO_TIMEOUT_S)
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")
            # The gateway's own error text, in full, in Gemini's envelope: the
            # CLI retries a 429/5xx by status and shows the message otherwise.
            server.observe_response(model, e.code, None, time.time() - t0, text)
            self._reply(e.code, gemini_error(e.code, text))
            return
        except (urllib.error.URLError, OSError) as e:
            server.observe_response(model, 502, None, time.time() - t0, str(e))
            self._reply(502, gemini_error(502, f"upstream unreachable: {e}"))
            return

        if not stream:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
            out = openai_to_gemini(payload)
            server.observe_response(model, 200, payload.get("usage"), time.time() - t0)
            self._reply(200, json.dumps(out).encode())
            return

        # SSE both ways, chunked: `data: <gemini chunk>\r\n\r\n` per event.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def emit(chunk: dict) -> None:
            event = b"data: " + json.dumps(chunk).encode() + b"\r\n\r\n"
            self.wfile.write(f"{len(event):x}\r\n".encode() + event + b"\r\n")
            self.wfile.flush()

        translator = StreamTranslator()
        status = 200
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload_text = line[5:].strip()
                if payload_text == "[DONE]":
                    break
                try:
                    obj = json.loads(payload_text)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("error") and not obj.get("choices"):
                    # A mid-stream error from the gateway. Surface it in the
                    # stream the way Gemini does, then end.
                    status = 500
                    err = obj["error"]
                    emit({"error": {"code": err.get("code") or 500,
                                    "message": err.get("message") or json.dumps(err),
                                    "status": "INTERNAL"}})
                    break
                for chunk in translator.feed(obj):
                    emit(chunk)
            for chunk in translator.finalize():
                emit(chunk)
        finally:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                pass
            server.observe_response(model, status, translator.usage, time.time() - t0)


class GatewayShim(_LoopbackServer):
    """The Gemini-native endpoint the CLI is pointed at, in gateway mode."""

    def __init__(self, *, base_url: str, api_key: str, model_id: str,
                 model_slug: str, exclude_tools: list, allowed_tools=None):
        super().__init__(_ShimHandler)
        base = base_url.rstrip("/")
        if not urllib.parse.urlsplit(base).path:
            base += "/v1"
        self.upstream_url = base + "/chat/completions"
        self.api_key = api_key
        # The configured id (`gemini/gemini-3.8-flash`) is what the gateway is
        # asked for for the MAIN agent loop -- see `upstream_model` for why this
        # is a substitution and not a pass-through. The CLI's own utility models
        # (a flash-lite summariser, loop detection) keep their own names under
        # the same provider prefix; only the main tier is pinned.
        self.model_id = model_id
        self.model_slug = model_slug
        self.prefix = model_id.split("/", 1)[0] + "/" if "/" in model_id else ""
        self.substitutions: set = set()
        self.exclude_tools = set(exclude_tools or ())
        self.allowed_tools = None if allowed_tools is None else set(allowed_tools)
        # The CLI must present this on every request; nothing else on the
        # loopback can guess it.
        self.token = secrets.token_hex(24)
        self.calls: list = []           # one summary per upstream call
        self.declared_tools: list | None = None
        self.breaches: list = []
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return self.origin

    def upstream_model(self, model: str) -> str:
        """The gateway model id for a model the CLI asked for.

        THE MAIN AGENT MODEL IS FORCED TO THE CONFIGURED ID, and that is
        load-bearing rather than cosmetic. The CLI resolves `--model` through
        its own table before it ever calls the API: it collapses any
        `*flash` name onto its current flash-GA default, so `--model=
        gemini-3.8-flash` reaches the wire as `gemini-3.5-flash` (measured).
        Passing that through would run the episode on a DIFFERENT model than
        configs/models.yaml names, and run.json would then lie about which
        model was tested. So a main-tier request (a `*flash`/`*pro` that is not
        a `*lite` and not our slug already) is rewritten to the configured id.

        The CLI's genuine UTILITY models -- a `*flash-lite` classifier, loop
        detection, a summariser -- are a deliberately cheaper tier and are left
        on their own names (with the provider prefix added), so the substitution
        does not silently upgrade every background call to the agent model. The
        configured id is DATA (`model_id`), never a constant here, so nothing
        about which model is pinned lives in code.
        """
        if model == self.model_slug or model == self.model_id:
            return self.model_id
        low = model.rsplit("/", 1)[-1].lower()
        is_main = (("flash" in low or "pro" in low)
                   and "lite" not in low and "gemma" not in low)
        if is_main:
            self.substitutions.add(model)
            return self.model_id
        return model if "/" in model else self.prefix + model

    def observe_request(self, body: dict, model: str, upstream: str, notes: list) -> None:
        names = []
        for t in body.get("tools") or []:
            if isinstance(t, dict):
                for fd in t.get("functionDeclarations") or []:
                    if isinstance(fd, dict) and fd.get("name"):
                        names.append(fd["name"])
        gen = body.get("generationConfig") or {}
        with self._lock:
            if names and self.declared_tools is None:
                self.declared_tools = names
                _log(f"shim: the CLI declares tools {names}")
            breach = sorted(self.exclude_tools & set(names))
            if self.allowed_tools is not None:
                breach = sorted(set(breach) | (set(names) - self.allowed_tools))
                if any(set(t) - {"functionDeclarations"} for t in body.get("tools") or []):
                    breach.append("non-function tool")
            if breach and breach not in self.breaches:
                self.breaches.append(breach)
                _log(f"shim: TOOL POLICY BREACH: {breach} declared outside the allowed tool policy")
            self.calls.append({"model": model, "upstream_model": upstream,
                               "n_contents": len(body.get("contents") or []),
                               "n_tools": len(names),
                               "generation_config": gen, "notes": notes})
        if breach and self.allowed_tools is not None:
            raise ValueError(f"MCP-only tool policy breach: {breach}")
        _log(f"shim: -> {upstream} contents={len(body.get('contents') or [])} "
             f"tools={len(names)} generationConfig={json.dumps(gen)}"
             + (f" notes={notes}" if notes else ""))

    def observe_response(self, model: str, status: int, usage, seconds: float,
                         error: str | None = None) -> None:
        with self._lock:
            for entry in reversed(self.calls):
                if entry.get("status") is None:
                    entry.update({"status": status, "usage": usage,
                                  "seconds": round(seconds, 2)})
                    if error:
                        entry["error"] = error
                    break
        _log(f"shim: <- {status} in {seconds:.1f}s usage={json.dumps(usage)}"
             + (f" error={error}" if error else ""))


class _NativeHandler(_ShimHandler):
    """Relay Gemini JSON/SSE without translating contents or tool signatures."""

    def do_POST(self):  # noqa: N802
        path, _, query_string = self.path.partition("?")
        query = urllib.parse.parse_qs(query_string)
        if not self._authorised(query):
            self._reply(403, gemini_error(403, "bad api key"))
            return
        match = _ROUTE.fullmatch(path)
        if not match:
            self._reply(404, gemini_error(404, "unsupported native API route"))
            return
        model, method = urllib.parse.unquote(match[1]), match[2]
        try:
            data = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            body = json.loads(data)
            if not isinstance(body, dict):
                raise ValueError("request body must be an object")
        except (ValueError, UnicodeDecodeError) as exc:
            self._reply(400, gemini_error(400, str(exc)))
            return
        server = self.server
        upstream = server.upstream_model(model)
        server.observe_request(body, model, upstream, [])
        # Never forward the loopback credential in a query string.
        query.pop("key", None)
        api_version = path.split("/")[1]
        url = (server.upstream_url + "/" + api_version + "/models/"
               + urllib.parse.quote(upstream, safe="") + ":" + method)
        if query:
            url += "?" + urllib.parse.urlencode(query, doseq=True)
        request = urllib.request.Request(url, data=data, method="POST", headers={
            "x-goog-api-key": server.api_key,
            "Content-Type": "application/json",
            "Accept": self.headers.get("Accept", "application/json"),
        })
        started = time.time()
        try:
            response = urllib.request.urlopen(request, timeout=UPSTREAM_IO_TIMEOUT_S)
        except urllib.error.HTTPError as exc:
            server.observe_response(model, exc.code, None, time.time() - started)
            self._reply(exc.code, exc.read(), exc.headers.get("Content-Type", "application/json"))
            return
        except (urllib.error.URLError, OSError) as exc:
            server.observe_response(model, 502, None, time.time() - started, str(exc))
            self._reply(502, gemini_error(502, "native upstream unavailable"))
            return
        with response:
            self.send_response(response.status)
            self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
            # Close-delimited forwarding permits immediate SSE delivery without
            # buffering the full model response or changing its framing.
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                while chunk := response.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            finally:
                server.observe_response(model, response.status, None, time.time() - started)


class NativeRelay(GatewayShim):
    """Enforce the selected model while keeping Google's native API protocol."""

    def __init__(self, **kwargs):
        base = kwargs["base_url"].rstrip("/")
        if urllib.parse.urlsplit(base).path not in ("", "/"):
            raise ValueError("Gemini native base URL must be an origin, without an API path")
        super().__init__(**kwargs)
        self.RequestHandlerClass = _NativeHandler
        self.upstream_url = base
        # Native endpoints take bare Google model IDs, never gateway prefixes.
        self.model_id = self.model_slug
        self.prefix = ""


# ===========================================================================
# the CLI's home, command line, and session file
# ===========================================================================
def write_settings(gemini_dir: str, settings: dict) -> str:
    os.makedirs(gemini_dir, exist_ok=True)
    path = os.path.join(gemini_dir, "settings.json")
    with open(path, "w") as f:
        json.dump(settings, f, indent=2)
    return path


def compose_command(cfg: dict, instruction: str, logfile: str) -> str:
    """Harbor's `gemini` command line:

        gemini --yolo --model=<slug-or-alias> --prompt=<instruction>
               2>&1 </dev/null | tee <log>

    stdin is closed so the CLI can never wait on a terminal, and the prompt is
    argv, as Harbor passes it.
    """
    parts = ["gemini", "--yolo", f"--model={shlex.quote(cfg['run_model'])}",
             f"--prompt={shlex.quote(instruction)}"]
    return ("if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
            + " ".join(parts)
            + f" 2>&1 </dev/null | tee {shlex.quote(logfile)}")


def run_cli(command: str, env: dict, timeout_s: int) -> tuple[int | None, str | None]:
    """Run the pipeline, tee the CLI's output to stderr, enforce the timeout.
    Returns (returncode, error). Never raises."""
    try:
        proc = subprocess.Popen(
            ["bash", "-c", command], cwd=WORKDIR, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True)
    except Exception as e:  # noqa: BLE001
        return None, f"gemini launch failed: {type(e).__name__}: {e}"

    deadline = time.time() + timeout_s
    error: str | None = None

    def _kill(sig) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

    try:
        for line in proc.stdout:
            sys.stderr.write(line if line.endswith("\n") else line + "\n")
            if time.time() > deadline:
                error = f"timeout after {timeout_s}s"
                _kill(signal.SIGKILL)
                break
        sys.stderr.flush()
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__} reading gemini output: {e}"

    try:
        rc = proc.wait(timeout=max(1.0, deadline - time.time()))
    except subprocess.TimeoutExpired:
        error = error or f"timeout after {timeout_s}s"
        _kill(signal.SIGKILL)
        try:
            rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            rc = None
    return rc, error


def find_session(gemini_dir: str) -> str | None:
    """The newest `session-*.json[l]` under `~/.gemini/tmp`, preferring a
    `chats/` directory: the CLI also writes `logs/session-<id>.jsonl`, which
    is not the conversation."""
    tmp = os.path.join(gemini_dir, "tmp")
    if not os.path.isdir(tmp):
        return None
    chats, others = [], []
    for dirpath, _dirs, files in os.walk(tmp):
        for name in files:
            if name.startswith("session-") and (name.endswith(".json")
                                                or name.endswith(".jsonl")):
                path = os.path.join(dirpath, name)
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                (chats if os.path.basename(dirpath) == "chats" else others).append((mtime, path))
    for bucket in (chats, others):
        if bucket:
            return max(bucket)[1]
    return None


def read_session(path: str) -> list:
    """The session file's records, in order. JSONL line by line; a legacy
    single-document file becomes one record. A malformed line is skipped,
    never fatal."""
    records: list = []
    try:
        text = open(path, "r").read()
    except OSError as e:
        _log(f"could not read session {path}: {e}")
        return records
    stripped = text.strip()
    if not stripped:
        return records
    if stripped.startswith("{") and "\n" not in stripped.rstrip("\n").strip():
        try:
            return [json.loads(stripped)]
        except json.JSONDecodeError:
            pass
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            _log(f"skipping malformed session line: {e}")
    return records


def summarise(records: list) -> dict:
    """Session id, final text, counts and token totals, from the records.

    The token totals are the per-message `tokens` blocks summed (Harbor's
    arithmetic): `completion` is output + thoughts + tool, which is what the
    model actually generated and what litellm prices at the output rate.
    """
    out = {"session_id": None, "final_text": "", "n_turns": 0,
           "n_tool_calls": 0, "usage": None, "model": None}
    totals = {"input": 0, "output": 0, "cached": 0, "thoughts": 0, "tool": 0,
              "total": 0}
    seen_tokens = False
    messages: dict = {}
    # Applying message_update must not mutate the native transcript returned
    # to the host or turn an earlier call into a prematurely completed result.
    for rec in copy.deepcopy(records):
        if not isinstance(rec, dict):
            continue
        if rec.get("sessionId") and not out["session_id"]:
            out["session_id"] = rec["sessionId"]
        meta = rec.get("$set")
        if isinstance(meta, dict) and meta.get("sessionId") and not out["session_id"]:
            out["session_id"] = meta["sessionId"]
        rtype = rec.get("type")
        if rtype == "gemini":
            messages[rec.get("id") or f"_{len(messages)}"] = rec
        elif rtype == "message_update" and rec.get("id") in messages:
            target = messages[rec["id"]]
            for k, v in rec.items():
                if k in ("type", "id"):
                    continue
                if isinstance(target.get(k), dict) and isinstance(v, dict):
                    target[k].update(v)
                else:
                    target[k] = v
        # Legacy single-document shape.
        for msg in rec.get("messages") or []:
            if isinstance(msg, dict) and msg.get("type") == "gemini":
                messages[msg.get("id") or len(messages)] = msg
    for msg in messages.values():
        out["n_turns"] += 1
        if out["model"] is None and isinstance(msg.get("model"), str):
            out["model"] = msg["model"]
        out["n_tool_calls"] += len(msg.get("toolCalls") or [])
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if isinstance(content, str) and content.strip():
            out["final_text"] = content
        tokens = msg.get("tokens")
        if isinstance(tokens, dict):
            seen_tokens = True
            for key in totals:
                value = tokens.get(key)
                if isinstance(value, (int, float)):
                    totals[key] += int(value)
    if seen_tokens:
        totals["completion"] = totals["output"] + totals["thoughts"] + totals["tool"]
        out["usage"] = totals
    return out


# ===========================================================================
# the episode
# ===========================================================================
def episode_error(returncode, summary, calls):
    """A saved setup record does not make a failed model request a success."""
    responses = [c for c in calls or [] if isinstance(c.get("status"), int)]
    last = responses[-1] if responses else None
    failed = last is not None and last["status"] >= 400
    if returncode not in (0, None) or (not summary["n_turns"] and failed):
        if failed:
            return f"gemini API HTTP {last['status']}: {last.get('error') or 'request failed'}"
        return f"gemini exited {returncode}"
    return None


def mcp_settings(settings: dict, grader, *, mcp_only=False) -> dict:
    """Allow only the grader's fully qualified names in Gemini's tool policy."""
    settings = json.loads(json.dumps(settings))
    if grader is not None:
        settings["mcpServers"] = {grader.grader_server_name: {"httpUrl": grader.url}}
        if mcp_only:
            names = list(grader.grader_tools)
            settings["mcpServers"][grader.grader_server_name]["includeTools"] = names
            # core is BOTH a native-registration allowlist and a policy
            # allowlist. generateValidName adds the mcp_ prefix in 0.59.0;
            # these are also the names observed on the actual model requests.
            settings.setdefault("tools", {})["core"] = [
                f"mcp_{grader.grader_server_name}_{name}" for name in names]
    return settings


def run(task: dict, install: dict | None, grader=None, observer=None,
        observer_ns=None) -> dict:
    cfg = dict(task.get("gemini") or {})
    timeout_s = int(task.get("timeout_s") or 3600)
    route = cfg.get("route") or "gateway"
    key_env = cfg.get("api_key_env") or "GEMINI_API_KEY"
    api_key = os.environ.get(key_env) or ""

    # THIS PROCESS's environment first (the grader runs here and reads it),
    # then the CLI's.
    os.environ.update({str(k): str(v) for k, v in (task.get("env") or {}).items()})
    env = dict(os.environ)
    env["GEMINI_CLI_TRUST_WORKSPACE"] = "true"
    env["NO_COLOR"] = "1"

    home = os.path.expanduser("~")
    gemini_dir = os.path.join(home, ".gemini")

    error: str | None = None
    if not api_key:
        error = f"{key_env} is not set inside the container"

    shim = None
    allowed_tools = None
    if cfg.get("mcp_only"):
        if route != "gateway" or grader is None:
            error = error or "Gemini MCP-only requires a grader and the gateway route"
        allowed_tools = ([f"mcp_{grader.grader_server_name}_{name}" for name in grader.grader_tools]
                         if grader is not None else [])
    if route == "gateway":
        base_url = cfg.get("base_url") or os.environ.get("GEMINI_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or ""
        if not base_url:
            error = error or "gateway route needs api_base_url, GEMINI_BASE_URL or OPENAI_BASE_URL"
        else:
            shim = GatewayShim(base_url=base_url, api_key=api_key,
                               model_id=cfg.get("model_id") or cfg.get("model_slug") or "",
                               model_slug=cfg.get("model_slug") or "",
                               exclude_tools=cfg.get("exclude_tools") or [],
                               allowed_tools=allowed_tools)
            shim.start()
            env["GOOGLE_GEMINI_BASE_URL"] = shim.url
            env.pop(key_env, None)
            env["GEMINI_API_KEY"] = shim.token
            # The gateway credential stays in THIS process; the CLI gets only
            # the per-episode token the shim checks for.
            env.pop("OPENAI_API_KEY", None)
            _log(f"shim on {shim.url} -> {shim.upstream_url} "
                 f"(model {shim.model_id}; the CLI's key is a per-episode token)")
    else:
        shim = NativeRelay(
            base_url=cfg.get("base_url") or "https://generativelanguage.googleapis.com",
            api_key=api_key, model_id=cfg.get("model_id") or cfg.get("model_slug") or "",
            model_slug=cfg.get("model_slug") or "",
            exclude_tools=cfg.get("exclude_tools") or [])
        shim.start()
        env.pop(key_env, None)
        env.pop("OPENAI_API_KEY", None)
        env["GEMINI_API_KEY"] = shim.token
        env["GOOGLE_GEMINI_BASE_URL"] = shim.url
        _log(f"native relay on {shim.url} -> {shim.upstream_url} (model {shim.model_id})")

    settings = mcp_settings(cfg.get("settings") or {}, grader, mcp_only=cfg.get("mcp_only", False))
    try:
        write_settings(gemini_dir, settings)
        _log(f"wrote settings.json (auth={((settings.get('security') or {}).get('auth') or {}).get('selectedType')}, "
             f"tools.exclude={(settings.get('tools') or {}).get('exclude')}, "
             f"tools.core={(settings.get('tools') or {}).get('core')}, "
             f"mcpServers={list((settings.get('mcpServers') or {}).keys()) or 'none'})")
    except Exception as e:  # noqa: BLE001
        error = error or f"settings.json setup failed: {type(e).__name__}: {e}"

    logfile = os.path.join(home, "gemini-cli.txt")
    command = compose_command(cfg, task["content"], logfile)
    _log(f"gemini command: {command}")

    t0 = time.time()
    rc: int | None = None
    if error is None:
        if observer is None:
            rc, error = run_cli(command, env, timeout_s)
        else:
            rc, error = observer_ns["observe_session"](
                lambda: run_cli(command, env, timeout_s),
                lambda: find_session(gemini_dir), observer)
    wall = time.time() - t0

    if shim is not None:
        shim.close()
    if grader is not None:
        grader.close()

    # Read the session WHATEVER happened: the CLI appends as it goes, so a
    # killed episode still has everything up to the kill.
    session_path = None
    records: list = []
    try:
        session_path = find_session(gemini_dir)
        if session_path:
            records = read_session(session_path)
            _log(f"session {session_path}: {len(records)} records")
        else:
            _log(f"no session file under {gemini_dir}/tmp")
    except Exception as e:  # noqa: BLE001
        _log(f"session lookup failed: {type(e).__name__}: {e}")

    for rec in records:
        _emit_msg(rec)

    info = summarise(records)
    error = error or episode_error(rc, info, shim.calls if shim is not None else None)
    if error is None and shim is not None and shim.breaches:
        # A closed tool that was declared to the model ran outside the
        # container's namespace: the episode's measurement cannot be trusted,
        # so it fails loudly rather than being averaged in.
        error = (f"tool policy breach: {shim.breaches} were declared to the "
                 f"model outside the allowed tool policy")

    try:
        deliverables = collect_deliverables(
            WORKDIR, tuple(task.get("skip_dirs") or ()),
            tuple(task.get("deliverable_files") or ()))
    except Exception as e:  # noqa: BLE001
        _log(f"deliverable collection failed: {type(e).__name__}: {e}")
        deliverables = []

    # Best-effort cleanup: the CLI's state does not outlive the episode.
    try:
        subprocess.run(["rm", "-rf", gemini_dir, logfile], timeout=60)
    except Exception:  # noqa: BLE001
        pass

    return {
        "id": task["id"],
        "model": task["model"],
        # The session file's records, in order. THIS is the trajectory.
        "messages": records,
        "grader_state": grader.grader_state if grader is not None else None,
        "deliverables": deliverables,
        "final_text": info["final_text"],
        "n_turns": info["n_turns"],
        "n_tool_calls": info["n_tool_calls"],
        "wall_time": round(wall, 1),
        "cost_usd": None,
        "stream_usage": None,
        "usage": info["usage"],
        "result_subtype": None,
        "terminal_reason": None,
        "session_id": info["session_id"],
        # What the CLI actually declared to the model, when the shim saw it;
        # the adapter's nominal list otherwise.
        "init_tools": (shim.declared_tools if shim is not None
                       and shim.declared_tools else task.get("tools")),
        "init_mcp_servers": [grader.report()] if grader is not None else [],
        "install": install,
        "gemini_command": command,
        "session_path": session_path,
        "gemini_route": route,
        "shim_calls": shim.calls if shim is not None else None,
        "returncode": rc,
        "error": error,
    }


def main(task: dict, modules: dict) -> None:
    """Stage the row's files, install the CLI, host the grader, run it."""
    try:
        os.chdir(WORKDIR)
    except OSError:
        pass

    ns: dict = {}
    grader_src = (modules.get("grader") or "").strip()
    for name in ("stage", *(("grader",) if grader_src else ()),
                 *(("session_observer",) if modules.get("session_observer") else ())):
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

    grader = None
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

    observer = None
    try:
        observer_factory = ns.get("grader", {}).get("make_observer")
        run_options = {"observer_ns": ns.get("session_observer")}
        if observer_factory is not None:
            observer = observer_factory(task.get("row") or {}, staged, WORKDIR,
                                        grader.grader_state if grader is not None else {})
            run_options["observer"] = observer
        record = run(task, install, grader, **run_options)
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
        if grader is not None:
            grader.close()
    if observer is not None:
        try:
            observer.finish()
        except Exception as e:
            record["error"] = f"{record.get('error') or ''} observer finish failed: {e}".strip()
    record["task_install"] = task_install
    _OUT.write(json.dumps(record) + "\n")
    _OUT.flush()


# Executed only when the bootstrap supplies TASK; importing this module (e.g.
# for a syntax check, or to read its source) does nothing.
_TASK = globals().get("TASK")
if _TASK is not None:
    main(_TASK, globals().get("MODULES") or {})
