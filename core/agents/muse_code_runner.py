"""The Muse Code CLI episode runner that executes INSIDE the container.

Same contract as `core/agents/codex_runner.py`, and deliberately standalone
like it: this file lives in the repo, is never baked into the image, is
streamed in on **stdin** with `core/stage.py` and the staged bytes, and writes
exactly ONE line of JSON on stdout. Everything else -- Muse's own output,
tracebacks, turn streaming -- goes to stderr, so the host parses stdout without
heuristics.

What is different from the Codex runner, and why (every fact measured against
Muse Code 1.0.3):

  * the episode is one `muse exec --json --yolo --user-input-auto-resolve
    --prompt-file <file> ...` subprocess. The prompt is a FILE (Harbor's
    shape; the CLI takes one), stdin is `</dev/null`, cwd is /workspace, which
    the CLI takes as the workspace root.
  * THE TRAJECTORY IS NOT ON STDOUT. `--json` prints the runtime's event
    stream, but that stream omits tool arguments, reasoning and token usage.
    The record is the durable session log
    `$XDG_DATA_HOME/muse/sessions/<YYYY>/<MM>/<DD>/<session-id>/session.jsonl`,
    plus one `subagent/<id>/session.jsonl` per reminder child the CLI spawned
    on its own. This runner reads the main log and returns its lines; the
    subagents' logs are read only for their token usage. A task-owned observer
    can also tail the main session during execution; the host copy stays native.
  * the grader is hosted over loopback streamable HTTP and registered through
    the CLI's settings file, `$XDG_CONFIG_HOME/muse/settings.json`:

        {"schema_version": 1,
         "mcp_servers": {"grader": {"enabled": true,
                                    "transport": "streamable_http",
                                    "url": "http://127.0.0.1:<port>/<token>/mcp"}}}

    That table also takes a stdio shape (`transport: stdio`, `command: [...]`)
    and this file never writes it, for the reason spelled out in the Codex
    runner: a command in a file the agent can read is a path to the grader's
    source and the answer key. A URL is an endpoint and nothing else. There is
    a test asserting `command` never appears in the settings this writes.
  * on the GATEWAY route the CLI never sees the gateway. It is pointed at a
    loopback shim (`MuseShim`, below) that serves the model catalog the CLI
    insists on fetching first (`GET /muse-code/models`) and relays the CLI's
    native Responses requests to the configured base's `/v1/responses`.
    Requests and responses retain reasoning, images and tools in their native
    format. The shared forwarder's `Upstream` replaces the bearer and merges
    extra_body, then streams the response without interpretation. It is exec'd
    out of the stdin blob like this file.

A native Meta entry with extra_body uses the same proxy to merge those fields.
Auth is `META_API_KEY` in the environment and nothing on disk: the real key
when calling Meta directly, a per-episode token when proxied (the proxy holds
the real key and replaces the CLI's Authorization header).
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import shlex
import signal
import subprocess
import sys
import threading
import time
from http.client import HTTPException
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

# The install downloads a ~264 MB binary from lookaside.facebook.com.
INSTALL_TIMEOUT_S = 900


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
# the adapter's runtime: `check || install`, run after staging and before muse
# exists. Same contract and same reporting as the other runners.
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

    Never raises: the host is owed exactly one JSON line whatever happens here,
    and an install failure must surface as a recorded status rather than as a
    container that died with no record.

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
            res = _sh(cmd, 120, env)
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
            # NO TRUNCATION: whatever the installer said, in full.
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
# the task's grader, as an MCP server muse can reach by URL
#
# The same loopback streamable-HTTP server the Codex runner hosts, for the same
# reasons (127.0.0.1, kernel-assigned port, an unguessable path token compared
# in constant time). Verified against Muse Code 1.0.3: registered through
# settings.json it performs `initialize` -> `notifications/initialized` ->
# `tools/list` at session startup, and the model's tool spec then carries a
# namespace `mcp__grader` with `grade_deliverable` inside it.
# ---------------------------------------------------------------------------
MCP_PROTOCOL_VERSION = "2025-06-18"


class _GraderMCPHandler(BaseHTTPRequestHandler):
    """Streamable-HTTP MCP, the four methods a client actually calls."""

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
            return ok({"tools": [{"name": server.grader_tool_name,
                                  "description": server.grader_description,
                                  "inputSchema": server.grader_schema}]})
        if method == "tools/call":
            name = params.get("name")
            if name != server.grader_tool_name:
                return ok({"content": [{"type": "text",
                                        "text": f"no such tool: {name}"}],
                           "isError": True})
            args = params.get("arguments")
            try:
                return ok(server.grader_call(args if isinstance(args, dict)
                                             else {}))
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
                 description: str, schema: dict):
        super().__init__(("127.0.0.1", 0), _GraderMCPHandler)
        self.grader_state = state
        self.grader_call = call
        self.grader_server_name = server_name
        self.grader_tool_name = tool_name
        self.grader_description = description
        self.grader_schema = schema
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
        """Stop serving. Idempotent; see the Codex runner for why it must be."""
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
    server = GraderServer(
        state=state, call=call,
        server_name=module.get("SERVER_NAME", "grader"),
        tool_name=module.get("TOOL_BASENAME", "grade_deliverable"),
        description=module.get("DESCRIPTION", ""),
        schema=module.get("INPUT_SCHEMA")
        or {"type": "object", "properties": {}})
    server.start()
    return server


# ---------------------------------------------------------------------------
# Muse's config and data homes for this episode
# ---------------------------------------------------------------------------
MCP_NAMES = frozenset({"play_move", "game_status", "reset_game", "write_notes", "read_notes",
                       "api_request", "list_files", "read_file", "write_file"})
NATIVE_TOOLS = ("read_file", "search", "write_file", "edit_file", "read_memory", "add_memory",
    "edit_memory", "work_stop", "bash", "bash_input", "cron_create", "cron_delete", "cron_list",
    "get_goal", "create_goal", "update_goal", "report_progress", "read_skill", "work_status",
    "snooze_reminder", "write_todos", "request_user_input", "workflow", "subagent_spawn", "subagent_status",
    "subagent_send_message", "subagent_wait", "subagent_read_result", "subagent_cancel")


def settings_json(grader_url: str = "", grader_name: str = "grader", *, mcp_only=False) -> dict:
    """The `$XDG_CONFIG_HOME/muse/settings.json` document, as a dict.

    One thing ever goes in it, verified against 1.0.3:

      mcp_servers.<name>   {"enabled": true, "transport": "streamable_http",
                           "url": ...}. The URL shape ONLY. The stdio shape
                           (`transport: stdio`, `command: [...]`) names a
                           command the agent could read and run for itself,
                           and this function has no branch that writes it.

    `schema_version: 1` is what the CLI's own manage-settings skill says a
    minimal document must carry.
    """
    doc: dict = {"schema_version": 1}
    if grader_url:
        doc["mcp_servers"] = {grader_name: {"enabled": True,
                                            "transport": "streamable_http",
                                            "url": grader_url}}
    if mcp_only:
        # Native tool selection also removes dispatch registrations. Muse
        # reinjects write_todos afterwards; the response gate below closes it.
        doc["run"] = {"subagent_delegation_mode": "off", "code_mode": "off",
                      "workflow_trigger_mode": "off", "reminder_roster": {"preset": "none"},
                      "context_slimming": {"excluded_tool_names": list(NATIVE_TOOLS)}}
    return doc


def prepare_homes(cfg: dict, instruction: str, grader_url: str = "",
                  grader_name: str = "grader") -> str:
    """Create the config/data/prompt directories; return the prompt path."""
    for key in ("config_home", "data_home", "prompt_dir"):
        os.makedirs(cfg[key], mode=0o700, exist_ok=True)
    conf_dir = os.path.join(cfg["config_home"], "muse")
    os.makedirs(conf_dir, mode=0o700, exist_ok=True)
    doc = settings_json(grader_url, grader_name, mcp_only=cfg.get("mcp_only", False))
    if len(doc) > 1:
        with open(os.path.join(conf_dir, "settings.json"), "w") as f:
            json.dump(doc, f, indent=1)
        _log(f"wrote settings.json (mcp_servers="
             f"{grader_name if grader_url else 'none'})")
    prompt_path = os.path.join(cfg["prompt_dir"], "prompt.txt")
    with open(prompt_path, "w") as f:
        f.write(instruction)
    return prompt_path


def compose_command(cfg: dict, prompt_path: str, logfile: str,
                    shim_url: str = "") -> str:
    """Harbor's `muse exec` line, plus this eval's closed-book flag.

        muse exec --json --yolo --user-input-auto-resolve
                  --prompt-file <path> --model <slug> --reasoning-effort <e>
                  [--disable-web-tools] [--max-model-steps N]
                  [--base-url <api.meta.ai origin | the shim's loopback origin>]
                  2>&1 </dev/null | tee <log>

    Every flag is chosen by the adapter (core/agents/muse_code.py); this only
    orders them. `shim_url` is the proxy's loopback origin, appended
    here because it exists only once the shim is bound.
    """
    parts = ["muse", "exec", *cfg["exec_flags"],
             "--prompt-file", shlex.quote(prompt_path),
             *cfg.get("cli_flags", [])]
    if shim_url:
        parts += ["--base-url", shlex.quote(shim_url)]
    return (cfg.get("path_export", "") + "; " + " ".join(parts)
            + f" 2>&1 </dev/null | tee {shlex.quote(logfile)}")


# ---------------------------------------------------------------------------
# The gateway serves native Responses. The only Muse-specific route is its
# model catalog; the shared Upstream relays all inference data without changing
# reasoning items, multimodal content, tool namespaces or streaming events.
RESPONSES_PATH = "/v1/responses"


def game_namespace(mcp_server):
    if mcp_server not in {"chess", "go"}:
        raise ValueError("Unsupported restricted game namespace")
    return "mcp__" + mcp_server


def restricted_request(req, mcp_server="chess"):
    """Preserve inference settings; offer exactly the connected game tools."""
    namespace = game_namespace(mcp_server)
    if "tools" not in req:
        return req, []
    kept = [t for t in req["tools"] if isinstance(t, dict)
            and t.get("type") == "namespace" and t.get("name") == namespace]
    names = [t.get("name") for group in kept for t in group.get("tools", [])]
    if len(names) != 9 or set(names) != MCP_NAMES:
        raise ValueError("Muse MCP-only roster must contain exactly nine game tools")
    return {**req, "tools": kept}, [namespace + "__" + n for n in names]


def check_response_item(item, mcp_server="chess"):
    """Gate tool execution before the native runtime receives a call."""
    namespace = game_namespace(mcp_server)
    if not isinstance(item, dict):
        return
    kind = item.get("type", "")
    if kind == "function_call":
        name = item.get("name", "")
        # Native Muse dispatch uses the canonical id even for namespace calls.
        if item.get("namespace") == namespace and name in MCP_NAMES:
            item["name"] = namespace + "__" + name
        if name in {namespace + "." + n for n in MCP_NAMES}:
            item["name"] = name.replace(namespace + ".", namespace + "__", 1)
        if item.get("name") not in {namespace + "__" + n for n in MCP_NAMES}:
            raise ValueError("Muse MCP-only blocked a non-game tool call")
    elif isinstance(kind, str) and kind.endswith("_call"):
        raise ValueError("Muse MCP-only blocked a non-function tool call")


def checked_response(doc, mcp_server="chess"):
    if not isinstance(doc, dict):
        raise ValueError("Invalid native response")
    if doc.get("type") in {"response.function_call_arguments.delta", "response.function_call_arguments.done"} and "name" in doc:
        identity = {"type": "function_call", "name": doc["name"], "namespace": doc.get("namespace")}
        check_response_item(identity, mcp_server)
        doc["name"] = identity["name"]
    check_response_item(doc.get("item"), mcp_server)
    for source in (doc, doc.get("response", {})):
        if isinstance(source, dict):
            for item in source.get("output", []):
                check_response_item(item, mcp_server)
    return doc


def guard_response(response, frames_class, mcp_server="chess"):
    """Validate complete SSE events, retaining streaming and native reasoning."""
    original = response.iter_chunks
    original_header = response.header
    response.header = lambda name: None if name.lower() == "content-length" else original_header(name)
    def chunks(size=65536):
        if "text/event-stream" not in (response.header("Content-Type") or ""):
            body = b"".join(original(size))
            if response.status < 400:
                body = json.dumps(checked_response(json.loads(body), mcp_server)).encode()
            yield body
            return
        frames = frames_class()
        for chunk in original(size):
            for frame in frames.feed(chunk):
                data = b"\n".join(line[5:].lstrip(b" ") for line in frame.splitlines() if line.startswith(b"data:"))
                if data and data != b"[DONE]":
                    doc = checked_response(json.loads(data), mcp_server)
                    # Re-encode only the data field; preserve event type and id.
                    fields = [line for line in frame.splitlines() if not line.startswith(b"data:")]
                    frame = b"\n".join(fields).rstrip(b"\n") + b"\ndata: " + json.dumps(doc).encode() + b"\n\n"
                yield frame
        frames.feed(b"", final=True)
    response.iter_chunks = chunks


def catalog_json(model_id: str) -> bytes:
    """The one catalog shape the CLI accepts: an OpenAI-style model list.

    Found by trial against 1.0.3: `{"data": [{"id": ...}]}` parses; `rows`,
    `models`, snake_case `model_id` and every other guess is "malformed".
    """
    return json.dumps({"object": "list", "data": [{
        "id": model_id, "object": "model", "display_label": model_id,
        "visibility": "visible", "is_default": True, "is_current": True}]}).encode()


class _ShimHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def _send(self, code: int, ctype: str, payload: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def _authorised(self) -> bool:
        token = self.server.cli_token
        got = (self.headers.get("Authorization") or "")
        return bool(token) and hmac.compare_digest(got, f"Bearer {token}")

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorised():
            self._send(401, "application/json", b'{"error":{"message":"shim: bad token"}}')
            return
        if self.path.split("?", 1)[0].endswith("/muse-code/models"):
            self.server.note("GET", self.path, 200)
            self._send(200, "application/json", catalog_json(self.server.model_id))
            return
        self.server.note("GET", self.path, 404)
        self._send(404, "application/json", b'{"error":{"message":"shim: no such route"}}')

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorised():
            self._send(401, "application/json", b'{"error":{"message":"shim: bad token"}}')
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length < 0:
                raise ValueError("negative content length")
        except ValueError:
            self.close_connection = True
            self._send(400, "application/json", b'{"error":{"message":"shim: bad content length"}}')
            return
        raw = self.rfile.read(length) if length else b""
        if not self.path.split("?", 1)[0].endswith("/responses"):
            self.server.note("POST", self.path, 404)
            self._send(404, "application/json", b'{"error":{"message":"shim: no such route"}}')
            return
        try:
            req = json.loads(raw)
        except ValueError:
            self._send(400, "application/json", b'{"error":{"message":"shim: bad json"}}')
            return
        if not isinstance(req, dict):
            self._send(400, "application/json", b'{"error":{"message":"shim: expected JSON object"}}')
            return
        if self.server.mcp_only:
            try:
                req, declared = restricted_request(req, self.server.mcp_server)
                if declared:
                    self.server.declared_tools = declared
                raw = json.dumps(req).encode()
            except (ValueError, TypeError, KeyError):
                self._send(403, "application/json", b'{"error":{"message":"Muse MCP-only tool roster rejected"}}')
                return
        t0 = time.time()
        try:
            _, sep, query = self.path.partition("?")
            resp = self.server.upstream.request(
                "POST", self.server.upstream.path + RESPONSES_PATH + sep + query,
                self.headers, raw)
        except Exception as e:  # noqa: BLE001 - the CLI gets a 502, not silence
            self.server.note("POST", self.path, 502, error=f"{type(e).__name__}: {e}")
            self._send(502, "application/json", json.dumps(
                {"error": {"message": f"shim upstream error: {type(e).__name__}: {e}",
                           "type": "shim_error"}}).encode())
            return
        n_out = 0
        relay_error = None
        try:
            if self.server.mcp_only:
                guard_response(resp, self.server.frames_class, self.server.mcp_server)
            # Keep the native stream and reasoning fields. Restricted mode
            # validates calls and normalizes MCP names before relaying events;
            # ordinary gateway mode still relays the original bytes.
            n_out = resp.relay(self)
        except (OSError, HTTPException, ValueError) as e:
            relay_error = type(e).__name__
            self.close_connection = True
        finally:
            resp.close()
            self.server.note("POST", self.path, resp.status,
                             seconds=round(time.time() - t0, 3),
                             bytes_in=len(raw), bytes_out=n_out,
                             **({"error": relay_error} if relay_error else {}))


class MuseShim(ThreadingHTTPServer):
    """The CLI's loopback origin for a gateway or arbitrary body overrides."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, *, upstream, model_id: str, cli_token: str, mcp_only=False, frames_class=None, mcp_server="chess"):
        super().__init__(("127.0.0.1", 0), _ShimHandler)
        self.upstream = upstream
        self.model_id = model_id
        self.cli_token = cli_token
        self.mcp_only = bool(mcp_only)
        game_namespace(mcp_server)
        self.mcp_server = mcp_server
        self.frames_class = frames_class
        self.declared_tools = []
        # One summary per request: method, path, status, sizes and seconds.
        # Never log bodies or credential headers.
        self.calls: list = []
        self._thread = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def note(self, method: str, path: str, status: int, **extra) -> None:
        entry = {"method": method, "path": path, "status": status, **extra}
        self.calls.append(entry)
        _log(f"shim {method} {path} -> {status} "
             + " ".join(f"{k}={v}" for k, v in extra.items()))

    def start(self) -> None:
        self._thread = threading.Thread(target=self.serve_forever,
                                        kwargs={"poll_interval": 0.2}, daemon=True)
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


def upstream_base(api_base_url: str) -> str:
    """The gateway origin plus path prefix, with any trailing `/v1` removed.

    `RESPONSES_PATH` carries its own `/v1`, so bases with and without that
    suffix both reach `/v1/responses`. Other deployment prefixes are retained.
    """
    base = api_base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def start_shim(forwarder_ns: dict, cfg: dict, gateway_key: str, cli_token: str):
    upstream = forwarder_ns["Upstream"](upstream_base(cfg["base_url"]), gateway_key,
                                        cfg.get("extra_body") or {}, timeout=600)
    shim = MuseShim(upstream=upstream, model_id=cfg["model_slug"], cli_token=cli_token,
                    mcp_only=cfg.get("mcp_only", False), frames_class=forwarder_ns.get("_SSEFrames"),
                    mcp_server=cfg.get("mcp_server", "chess"))
    shim.start()
    return shim


def run_muse(command: str, env: dict, timeout_s: int) -> tuple[int | None, str | None]:
    """Run the pipeline, tee Muse's own output to stderr, enforce the timeout.

    Returns (returncode, error). Never raises. The process gets its own session
    so the timeout kills the whole pipeline.
    """
    try:
        proc = subprocess.Popen(
            ["bash", "-c", command], cwd=WORKDIR, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True)
    except Exception as e:  # noqa: BLE001
        return None, f"muse launch failed: {type(e).__name__}: {e}"

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
        error = f"{type(e).__name__} reading muse output: {e}"

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


# ---------------------------------------------------------------------------
# the session log, which is the actual record
# ---------------------------------------------------------------------------
def find_session(data_home: str) -> str | None:
    """`$XDG_DATA_HOME/muse/sessions/<YYYY>/<MM>/<DD>/<id>/session.jsonl`.

    The newest main session log by mtime. `subagent/<id>/session.jsonl` files
    are the CLI's own reminder children and are never the answer here.
    """
    sessions = os.path.join(data_home, "muse", "sessions")
    if not os.path.isdir(sessions):
        return None
    best, best_mtime = None, -1.0
    for dirpath, dirnames, filenames in os.walk(sessions):
        rel = os.path.relpath(dirpath, sessions)
        parts = [] if rel == "." else rel.split(os.sep)
        # `.msp-view-v1/` is the CLI's own index of the same sessions.
        if "subagent" in parts or (parts and parts[0].startswith(".")):
            dirnames[:] = []
            continue
        if "session.jsonl" in filenames:
            path = os.path.join(dirpath, "session.jsonl")
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime > best_mtime:
                best, best_mtime = path, mtime
    return best


def subagent_sessions(session_path: str) -> list[str]:
    root = os.path.join(os.path.dirname(session_path), "subagent")
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "session.jsonl")
        if os.path.isfile(path):
            out.append(path)
    return out


def read_session(path: str) -> list:
    """The log's lines as dicts. A malformed line is skipped, never fatal."""
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
                    _log(f"skipping malformed session line: {e}")
    except OSError as e:
        _log(f"could not read session {path}: {e}")
    return events


USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_tokens",
              "cache_write_tokens", "cached_tokens", "reasoning_tokens")


def usage_of(events: list) -> dict | None:
    """Sum every `model_completed` usage in one session log, or None."""
    total = {k: 0 for k in USAGE_KEYS}
    calls = 0
    for ev in events:
        if not isinstance(ev, dict) or ev.get("payload_type") != "runtime.session":
            continue
        payload = ev.get("payload") or {}
        event = payload.get("event") if isinstance(payload, dict) else None
        if not isinstance(event, dict) or event.get("kind") != "model_completed":
            continue
        usage = event.get("usage")
        if not isinstance(usage, dict):
            continue
        calls += 1
        for k in USAGE_KEYS:
            v = usage.get(k)
            if isinstance(v, (int, float)):
                total[k] += int(v)
    if not calls:
        return None
    total["model_calls"] = calls
    return total


def add_usage(a: dict | None, b: dict | None) -> dict | None:
    if a is None:
        return dict(b) if b else None
    if b is None:
        return dict(a)
    return {k: int(a.get(k, 0)) + int(b.get(k, 0))
            for k in set(a) | set(b)}


def summarise(events: list) -> dict:
    """Session id, model, final text, counts and terminal, from the main log."""
    out = {"session_id": None, "model": None, "final_text": "", "n_turns": 0,
           "n_tool_calls": 0, "terminal": None, "terminal_reason": None,
           "task_failures": [], "version": None}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        stream = ev.get("stream")
        if out["session_id"] is None and isinstance(stream, dict):
            out["session_id"] = stream.get("id")
        ptype = ev.get("payload_type")
        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
        if ptype == "runtime.session.metadata":
            record = payload.get("record") or {}
            out["model"] = record.get("model_id") or out["model"]
            build = record.get("build") or {}
            if isinstance(build.get("semver"), str):
                out["version"] = build["semver"]
            continue
        if ptype != "runtime.session":
            continue
        event = payload.get("event")
        if not isinstance(event, dict):
            continue
        kind, ekind = payload.get("kind"), event.get("kind")
        if kind == "task":
            if ekind == "failed" and event.get("reason"):
                out["task_failures"].append(str(event["reason"]))
            continue
        if kind != "run":
            continue
        if ekind == "model_completed":
            out["n_turns"] += 1
        elif ekind == "assistant_tool_calls_committed":
            out["n_tool_calls"] += len(event.get("tool_calls") or [])
        elif ekind == "assistant_message_committed":
            text = event.get("text")
            if isinstance(text, str) and text.strip():
                out["final_text"] = text
        elif ekind == "terminal":
            out["terminal"] = event.get("terminal")
            out["terminal_reason"] = event.get("reason")
    return out


def empty_completion_error(info: dict, usage_main: dict | None) -> str | None:
    """The reason an episode that the CLI calls "completed" is a failure.

    Measured against the gateway's OpenRouter route while it answered every
    request with `content: null` and zero tokens: the CLI does not error, does
    not retry and does not hang -- it emits `run.output.delta` with "" and
    `run.terminal.completed` with empty text, and exits 0 after ~5 s. Left
    alone that would be recorded as a successful episode with nothing in it.
    So a completed run in which the model produced NO output tokens, NO text
    and NO tool call across every call is an API failure, and says so.
    """
    if info.get("terminal") != "completed":
        return None
    if info.get("final_text") or info.get("n_tool_calls"):
        return None
    if not usage_main or usage_main.get("model_calls", 0) == 0:
        return None
    if usage_main.get("output_tokens", 0) > 0:
        return None
    return (f"provider returned empty completions: {usage_main['model_calls']} "
            f"model call(s), 0 output tokens, no text and no tool call; the "
            f"CLI reported the run as completed")


def stdout_failure(logfile: str) -> str | None:
    """The reason from a `run.terminal.failed` event on the --json stream.

    A fallback for a session log that never got as far as its own terminal
    event; the stream and the log say the same thing when both exist.
    """
    try:
        with open(logfile, "r") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None
    reason = None
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("payload_type") == "run.terminal.failed":
            reason = (rec.get("payload") or {}).get("reason") or "run failed"
    return reason


# ---------------------------------------------------------------------------
# the episode
# ---------------------------------------------------------------------------
def run(task: dict, install: dict | None, grader=None,
        forwarder_ns: dict | None = None, observer=None, observer_ns=None) -> dict:
    """One episode. `grader` is a running `GraderServer`, or None.

    `forwarder_ns` is the exec'd `core/agents/forwarder.py`, present when a
    gateway or extra_body needs a proxy. Its `Upstream` is the outbound leg.
    """
    cfg = dict(task.get("muse") or {})
    if cfg.get("mcp_only") and grader is not None:
        cfg["mcp_server"] = grader.grader_server_name
    timeout_s = int(task.get("timeout_s") or 3600)
    key_env = cfg.get("api_key_env") or "META_API_KEY"
    api_key = os.environ.get(key_env) or ""
    route = cfg.get("route") or "native"

    # THIS PROCESS's environment first (the task's grader runs here and reads
    # os.environ for its own judge's endpoint), then muse's, from which the
    # grader-only variables are removed again.
    os.environ.update({str(k): str(v) for k, v in (task.get("env") or {}).items()})
    env = dict(os.environ)
    for name in cfg.get("grader_only_env") or ():
        env.pop(name, None)
    env.update({str(k): str(v) for k, v in (cfg.get("fixed_env") or {}).items()})
    env["XDG_CONFIG_HOME"] = cfg["config_home"]
    env["XDG_DATA_HOME"] = cfg["data_home"]
    # The base URL reaches the CLI on its command line (adapter flag on the
    # native route, the shim's origin when proxied), never by variable.
    env.pop("MUSE_BASE_URL", None)

    error: str | None = None
    if not api_key:
        error = f"{key_env} is not set inside the container"

    shim = None
    shim_url = ""
    if route == "gateway" or cfg.get("extra_body"):
        # The CLI's bearer is a per-episode token the shim checks; the real
        # key stays in this process and goes out only from the shim.
        cli_token = "rh-" + secrets.token_hex(16)
        env[key_env] = cli_token
        if forwarder_ns is None or "Upstream" not in forwarder_ns:
            error = error or "proxied route needs the forwarder module in the blob"
        elif not cfg.get("base_url"):
            error = error or "proxied route has no base_url"
        elif error is None:
            try:
                shim = start_shim(forwarder_ns, cfg, api_key, cli_token)
                shim_url = shim.url
                _log(f"{route} shim on {shim_url} -> "
                     f"{upstream_base(cfg['base_url'])}{RESPONSES_PATH} "
                     f"(model {cfg['model_slug']})")
            except Exception as e:  # noqa: BLE001
                error = f"{route} shim failed to start: {type(e).__name__}: {e}"
    else:
        env[key_env] = api_key

    prompt_path = os.path.join(cfg["prompt_dir"], "prompt.txt")
    try:
        prompt_path = prepare_homes(
            cfg, task["content"],
            grader_url=grader.url if grader is not None else "",
            grader_name=(grader.grader_server_name
                         if grader is not None else "grader"))
    except Exception as e:  # noqa: BLE001
        error = error or f"muse home setup failed: {type(e).__name__}: {e}"

    logfile = os.path.join(cfg["prompt_dir"], "muse.jsonl")
    command = compose_command(cfg, prompt_path, logfile, shim_url)
    _log(f"muse command: {command}")

    t0 = time.time()
    rc: int | None = None
    if error is None:
        if observer is None:
            rc, error = run_muse(command, env, timeout_s)
        else:
            rc, error = observer_ns["observe_session"](
                lambda: run_muse(command, env, timeout_s),
                lambda: find_session(cfg["data_home"]), observer)
    wall = time.time() - t0

    if grader is not None:
        grader.close()
    if shim is not None:
        shim.close()

    # Read the session log WHATEVER happened. Muse appends to it as it goes,
    # so a killed episode still has everything up to the kill.
    session_path = None
    events: list = []
    usage_main = usage_subagents = None
    n_sub = 0
    try:
        session_path = find_session(cfg["data_home"])
        if session_path:
            events = read_session(session_path)
            usage_main = usage_of(events)
            for sub in subagent_sessions(session_path):
                n_sub += 1
                usage_subagents = add_usage(usage_subagents,
                                            usage_of(read_session(sub)))
            _log(f"session {session_path}: {len(events)} records, "
                 f"{n_sub} subagent session(s)")
        else:
            _log(f"no session log under {cfg['data_home']}/muse/sessions")
    except Exception as e:  # noqa: BLE001
        _log(f"session lookup failed: {type(e).__name__}: {e}")

    for ev in events:
        _emit_msg(ev)

    info = summarise(events)
    if error is None:
        error = empty_completion_error(info, usage_main)
    if error is None:
        if info["terminal"] == "failed":
            error = (info["terminal_reason"] or stdout_failure(logfile)
                     or "muse run failed")
        elif rc not in (0, None):
            error = (stdout_failure(logfile)
                     or f"muse exited {rc}"
                     + ("" if events else " and wrote no session log"))
        elif not events and rc is not None:
            error = "muse exited 0 but wrote no session log"

    try:
        deliverables = collect_deliverables(
            WORKDIR, tuple(task.get("skip_dirs") or ()),
            tuple(task.get("deliverable_files") or ()))
    except Exception as e:  # noqa: BLE001
        _log(f"deliverable collection failed: {type(e).__name__}: {e}")
        deliverables = []

    # Cleanup, best effort: the settings (with the grader URL) and the session
    # store do not outlive the episode.
    for path in (cfg.get("config_home"), cfg.get("data_home"), cfg.get("prompt_dir")):
        if path:
            try:
                subprocess.run(["rm", "-rf", path], timeout=60)
            except Exception:  # noqa: BLE001
                pass

    usage = add_usage(usage_main, usage_subagents)
    return {
        "id": task["id"],
        "model": task["model"],
        # The main session log, in order. THIS is the trajectory.
        "messages": events,
        "grader_state": grader.grader_state if grader is not None else None,
        "deliverables": deliverables,
        "final_text": info["final_text"],
        "n_turns": info["n_turns"],
        "n_tool_calls": info["n_tool_calls"],
        "wall_time": round(wall, 1),
        "cost_usd": None,
        "stream_usage": None,
        "usage": usage,
        "usage_main": usage_main,
        "usage_subagents": usage_subagents,
        "n_subagent_sessions": n_sub,
        "result_subtype": info["terminal"],
        "terminal_reason": info["terminal_reason"],
        "session_id": info["session_id"],
        "init_tools": (list(shim.declared_tools) if cfg.get("mcp_only") and shim is not None else task.get("tools")),
        "init_mcp_servers": [grader.report()] if grader is not None else [],
        "install": install,
        "muse_command": command,
        "session_path": session_path,
        "route": route,
        "shim_calls": list(shim.calls) if shim is not None else None,
        "returncode": rc,
        "error": error,
    }


def main(task: dict, modules: dict) -> None:
    """Stage the row's files, install Muse, host the grader, run it."""
    try:
        os.chdir(WORKDIR)
    except OSError:
        pass

    ns: dict = {}
    grader_src = (modules.get("grader") or "").strip()
    forwarder_src = (modules.get("forwarder") or "").strip()
    for name in ("stage", *(("grader",) if grader_src else ()),
                 *(("support",) if modules.get("support") else ()),
                 *(("forwarder",) if forwarder_src else ()),
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
            factory = ns.get("support", {}).get("serve_grader", serve_grader)
            grader = factory(ns["grader"], task.get("row") or {}, staged, WORKDIR)
            _log(f"grader MCP server on 127.0.0.1:"
                 f"{grader.server_address[1]} (path withheld from this log)")
        except Exception as e:  # noqa: BLE001
            _log(f"grader construction failed: {type(e).__name__}: {e}")
            grader_error = {"error": f"{type(e).__name__}: {e}"}

    observer = None
    try:
        observer_factory = ns.get("grader", {}).get("make_observer")
        run_options = {"forwarder_ns": ns.get("forwarder"), "observer_ns": ns.get("session_observer")}
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


_TASK = globals().get("TASK")
if _TASK is not None:
    main(_TASK, globals().get("MODULES") or {})
