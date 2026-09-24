"""The Codex CLI episode runner that executes INSIDE the container.

Same contract as `core/agents/claude_sdk_runner.py`, and for the same reasons:
this
file lives in the repo, is never baked into the image, is streamed in on
**stdin** with `core/stage.py` and the staged bytes, and writes exactly ONE line
of JSON on stdout. Everything else -- Codex's own output, tracebacks, turn
streaming -- goes to stderr, so the host parses stdout without heuristics.

What is different from the Claude runner, and why:

  * Codex is a CLI, not a library. There is no in-process query loop; the
    episode is one `codex exec ... -- <instruction>` subprocess. The instruction
    goes in ARGV after `--` (Harbor's shape) and stdin is `</dev/null`.
  * THE TRAJECTORY IS NOT ON STDOUT. `--json` prints events for a human and
    Harbor tees them to a file, but the record Harbor actually parses is the
    rollout file Codex writes to
    `$CODEX_HOME/sessions/<YYYY>/<MM>/<DD>/rollout-<ISO>-<uuid>.jsonl`. Harbor
    copies that tree to the host over a bind mount. There is no bind mount here
    -- an episode is one `enroot start` and its overlay dies with it -- so this
    runner READS the file and returns its lines inside the JSON line it already
    owes the host.
  * Turn streaming happens after the run, not during it. The Claude SDK hands us
    typed messages as they arrive; Codex's stdout event shape moves between
    releases, while the rollout file is the format Harbor pins. So the rollout
    is replayed as turn records once Codex exits -- including when it is killed
    at the timeout, because Codex appends to that file as it goes.
  * The grader is hosted over HTTP rather than in process. The Claude adapter
    hands the SDK an in-process MCP server object. Codex takes an MCP server
    only from `$CODEX_HOME/config.toml`, in one of TWO shapes, and the
    difference between them is the whole design:

        [mcp_servers.x]                    STDIO. The config file names a
        command = "..."                    COMMAND, and the agent can read the
        args = [...]                       file and run that command itself --
                                           which for a grader means reading its
                                           source and the answer key it checks
                                           against. Never used here.

        [mcp_servers.x]                    URL. The config file names an
        url = "http://..."                 ENDPOINT and nothing else. The agent
                                           can call it; there is no source and
                                           no key to read, because neither ever
                                           left this process's heap.

    So the url transport gives the grader the same property the Claude SDK's
    in-process server has, and `serve_grader` below uses it: the task's grader
    module is exec'd out of the stdin blob, wrapped in a loopback-bound HTTP MCP
    server on a kernel-assigned port behind an unguessable path, and only that
    URL is written to config.toml. `grader_state` is then reported exactly as
    the Claude runner reports it. A task that ships NO grader still records an
    explicit None, which is a different fact from an unused grader.

Auth is Harbor's: `OPENAI_API_KEY` in the environment AND an `auth.json` written
to a secrets directory and symlinked to `$CODEX_HOME/auth.json`. The base URL is
NOT an environment variable -- codex >= 0.118 ignores `OPENAI_BASE_URL` and reads
`openai_base_url` from `config.toml` only, which Harbor calls out explicitly.
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
# Claude runner: this is a property of the TASK (what an agent might produce),
# not of the agent.
DELIVERABLE_EXTS = (".xlsx", ".xls", ".xlsm", ".docx", ".doc", ".pptx", ".ppt",
                    ".pdf", ".csv", ".svg")
_DELIVERABLE_MAX_BYTES = 64 * 1024 * 1024

INSTALL_TIMEOUT_S = 600


def collect_deliverables(root: str, skip_dirs: tuple, include_files: tuple = ()) -> list:
    """Export office files and exact task-requested outputs for the host.

    `skip_dirs` are the top-level directories OUR staging created, so nothing we
    planted is ever re-exported as if the agent had made it.
    """
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
# the adapter's runtime
#
# `check || install`, run after staging and before Codex exists. Same contract
# and same reporting as the Claude runner; the two runners are deliberately
# standalone, because each one is streamed into a container on its own.
# ---------------------------------------------------------------------------
def _sh(cmd: str, timeout: float, env: dict | None = None):
    # Output is captured, never inherited: fd 1 already points at stderr, but a
    # child writing to it directly would still be one more thing between us and
    # the single JSON line on stdout.
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
# the task's grader, as an MCP server codex can reach by URL
#
# WHY HTTP AND NOT STDIO. Codex accepts an MCP server two ways, and only one of
# them is safe for a grader:
#
#   stdio   `command`/`args` in $CODEX_HOME/config.toml. The agent can read that
#           file and run the command itself, so hosting a grader this way hands
#           it the grader's source and, through it, the answer key. This is the
#           case that made an earlier version of this adapter ship no grader at
#           all -- correctly, for stdio, and wrongly for everything else.
#   url     a bare endpoint in the same file. The agent can read the URL and can
#           call the tool, which is exactly what it is meant to be able to do.
#           There is no source and no key on the far end of it to read: the
#           grader module is exec'd out of the stdin blob into THIS process's
#           heap, the row never leaves it either, and the socket only ever
#           speaks verdicts.
#
# WHERE IT BINDS, and why that needs care. A task with no `environment/setup.py`
# -- gdpval is one -- has no network namespace of its own and shares the HOST's,
# so a socket opened here is a socket on the host's loopback, alongside every
# other concurrently running episode's. Three things keep them apart:
#
#   127.0.0.1     never 0.0.0.0. Nothing off the box can reach it either way.
#   port 0        the kernel assigns a free ephemeral port and we read it back
#                 off the bound socket, so `--max-concurrent 5` cannot collide
#                 and there is no window between "pick a port" and "bind it".
#   a path token  16 random bytes, per episode, in the URL. Every request to any
#                 other path -- including codex's own OAuth discovery probes --
#                 is a 404, so a neighbouring episode that guessed the port
#                 still reaches nothing.
#
# The token is compared with `hmac.compare_digest`, which costs nothing and
# means the 404 boundary is not a timing oracle.
#
# WHAT THE MODEL SEES, measured on a real gdpval episode (codex-cli 0.152.0,
# gpt-5.6-sol) rather than assumed. Codex prefixes an MCP tool with
# `mcp__<server>__`, exactly as the Claude SDK does, so registering
# ("grader", "grade_deliverable") here produces the SAME name on both adapters:
#
#     [{"name":"mcp__grader__grade_deliverable","description":"Submit your
#       finished deliverable for grading. ..."}]
#
# and the model called it as
# `tools.mcp__grader__grade_deliverable({deliverable:"..."})`.
#
# What is NOT the same is the CALL FRAME. gpt-5.6-sol reaches every MCP tool
# from inside codex's JS `exec` sandbox, so the rollout's `response_item` is a
# `custom_tool_call` named `exec` whose input is the JavaScript above -- never a
# `function_call` named after the tool. The rollout does also carry codex's own
# `event_msg`/`item_completed` with `item.type == "McpToolCall"`, naming the
# server, the tool, the arguments and the result, but that is not part of the
# model's action stream and `to_trajectory` does not synthesise a step from it.
# Measured both with and without `--enable unified_exec` (identical), and there
# is no supported knob to change it: `node_repl_disabled` is rejected under
# `--strict-config`, and `codex features list` reports `js_repl` and
# `js_repl_tools_only` as "removed". So the tool NAME matches and the ATIF
# `function_name` does not; see tests/test_codex_grader.py.
#
# WHAT THE AGENT CAN STILL DO, stated plainly rather than papered over: the URL
# is in config.toml, which the agent may read, so it can call the grader with
# its own `curl` instead of through the tool. The Claude adapter's in-process
# server has no such surface. It buys the agent nothing it does not already
# have -- the endpoint returns a verdict and nothing else -- and it cannot hide
# the call, because `state` is written HERE: an out-of-band call still increments
# `state["calls"]`, so a `grader_state` with more calls than the trajectory shows
# is itself the evidence that one happened.
# ---------------------------------------------------------------------------

# What codex 0.152.0 sends in `initialize`. Echoed back rather than hardcoded
# (see `_handle`), so a client on an older or newer revision negotiates its own.
MCP_PROTOCOL_VERSION = "2025-06-18"


class _GraderMCPHandler(BaseHTTPRequestHandler):
    """Streamable-HTTP MCP, the four methods codex actually calls.

    Observed against codex-cli 0.152.0, not inferred: one POST per JSON-RPC
    message with `Accept: text/event-stream, application/json`, in the order
    `initialize` -> `notifications/initialized` -> `tools/list` -> `tools/call`.
    It opens no server-to-client GET stream and needs no `Mcp-Session-Id`, so
    GET and DELETE are answered 405 -- which is also what makes codex report the
    server as needing no auth instead of starting an OAuth flow against it.
    """

    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        """Silence. The default writes to stderr, which the host parses."""

    # -- plumbing --------------------------------------------------------
    def _authorised(self) -> bool:
        path = self.path.split("?", 1)[0]
        return hmac.compare_digest(path, self.server.grader_path)

    def _empty(self, code: int) -> None:
        self.send_response(code)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send(self, body: bytes) -> None:
        """One JSON-RPC response, as SSE when the client accepts it.

        The spec lets the server answer a POST with either `application/json` or
        a `text/event-stream` carrying the same message. Codex accepts both;
        SSE is what it asks for first, so that is what it gets.
        """
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

    # -- verbs -----------------------------------------------------------
    def do_GET(self) -> None:       # noqa: N802 - BaseHTTPRequestHandler's name
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
        # A JSON-RPC NOTIFICATION has no id and takes no response body.
        if request.get("id") is None:
            self._empty(202)
            return
        self._send(json.dumps(self._handle(request)).encode())

    # -- the protocol ----------------------------------------------------
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
                # The CLIENT's version, echoed. Refusing to negotiate would
                # leave this working only against the revision it was written
                # for.
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
                # The task's own grader, called on the handler thread. It owns
                # `state`, so this is where an attempt is counted.
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
    # Deliberately NOT reusing an address: the kernel hands out a free ephemeral
    # port and we want a hard failure rather than a quiet share if that ever
    # stops being true.
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
        """Stop serving. Called as soon as codex exits, so no late request can
        change `state` after the record has been taken.

        IDEMPOTENT, and it has to be: `run()` closes it the moment codex exits
        and `main()` closes it again in a `finally`. The thread guard is also
        what keeps a second call from deadlocking -- `BaseServer.shutdown()`
        waits on an event that only `serve_forever` sets, so calling it when
        nothing is serving would block for the rest of the episode.
        """
        if self._thread is not None:
            try:
                self.shutdown()
            except Exception:  # noqa: BLE001 - teardown is best effort
                pass
            self._thread.join(timeout=10)
            self._thread = None
        try:
            self.server_close()
        except Exception:  # noqa: BLE001
            pass

    def report(self) -> dict:
        """What to record as this episode's `init_mcp_servers` entry.

        `status` is OBSERVED, not assumed: "connected" only when an `initialize`
        actually arrived on the socket. An episode whose codex never reached the
        grader is a different fact from one whose model chose not to call it, and
        the record has to be able to tell them apart.
        """
        entry = {"name": self.grader_server_name,
                 "transport": "streamable-http",
                 "status": "connected" if self.connected else "configured",
                 "tools_listed": self.listed}
        if self.grader_errors:
            entry["errors"] = list(self.grader_errors)
        return entry


def serve_grader(module: dict, row: dict, staged: dict, workdir: str):
    """Start the task's grader as an MCP endpoint. Returns the running server.

    `module` is the task's grader module, already exec'd out of the stdin blob
    into a namespace. It must export `make_tool(row, staged, workdir)` returning
    `(state, call)` -- the same seam the Claude runner reaches through
    `make_grader`, so both adapters run one implementation of the verdict logic
    and one definition of the tool. An optional `make_extra_tools(row, staged,
    workdir, state)` returns additional name/description/inputSchema/call specs.
    """
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


# ---------------------------------------------------------------------------
# Codex's home: auth.json, and config.toml when there is a base URL or a grader
# ---------------------------------------------------------------------------
def config_toml(base_url: str = "", grader_url: str = "",
                grader_name: str = "grader", provider: dict | None = None,
                mcp_only: bool = False) -> str:
    """The `$CODEX_HOME/config.toml` text for this episode, or "" for none.

    Harbor's `_build_effective_config` / `_upload_effective_config`, reduced to
    the keys this harness ever sets. The MCP entry is the URL shape --

        [mcp_servers.grader]
        url = "http://127.0.0.1:<port>/<token>/mcp"

    -- and there is deliberately no `command`/`args` branch anywhere in this
    file. Harbor writes `command` for `transport == "stdio"` and `url` for
    everything else; a grader may only ever be the latter, because the agent can
    read this file and a `command` in it is a path to the grader's source and to
    the answer key it checks against. There is a test asserting `command` never
    appears here.

    `provider` is the OVERRIDE path (configs/models.yaml `base_url:`): a
    non-OpenAI endpoint reached through a named `[model_providers.<id>]` block
    whose `wire_api` we pin, rather than the bare `openai_base_url` that only
    re-points the built-in openai provider (which stays on the Responses API).
    When `provider` is given, `openai_base_url` is NOT written -- the block owns
    the base URL -- so the two shapes never both appear.

    The top-level keys come first: in TOML everything after a `[table]` header
    belongs to that table, so `openai_base_url`/`model_provider` written below
    the mcp or provider section would silently be reparented into it.
    """
    parts = []
    if mcp_only and grader_name in {"chess", "go"}:
        parts.append("developer_instructions = " + json.dumps(
            f"The local filesystem sandbox is read-only. The listed {grader_name} MCP tools "
            "operate on task-managed state through the server. The task-authorized "
            "write_notes and write_file calls are permitted through those tools; "
            "the local filesystem restriction does not prohibit them. Follow the "
            "task instructions when deciding how to use the tools.") + "\n")
    if provider:
        parts.append(f"model_provider = {json.dumps(provider['id'])}\n")
    elif base_url:
        parts.append(f"openai_base_url = {json.dumps(base_url)}\n")
    if provider:
        parts.append(f"\n[model_providers.{provider['id']}]\n"
                     f"name = {json.dumps(provider['id'])}\n"
                     f"base_url = {json.dumps(provider['base_url'])}\n"
                     f"env_key = {json.dumps(provider['env_key'])}\n"
                     f"wire_api = {json.dumps(provider['wire_api'])}\n")
    if grader_url:
        parts.append(f"\n[mcp_servers.{grader_name}]\n"
                     f"url = {json.dumps(grader_url)}\n")
        if mcp_only:
            if grader_name not in {"chess", "go"}:
                raise ValueError("Restricted MCP policy supports chess and Go only")
            parts.append('required = true\ntool_timeout_sec = 900\n'
                         'enabled_tools = ["play_move", "game_status", "reset_game", "write_notes", "read_notes", "api_request", "list_files", "read_file", "write_file"]\n'
                         'default_tools_approval_mode = "approve"\n')
    return "".join(parts)


def prepare_home(cfg: dict, api_key: str, grader_url: str = "",
                 grader_name: str = "grader") -> None:
    """Harbor's setup, run before `codex exec`.

    The auth file goes to its own directory and is symlinked into CODEX_HOME --
    Harbor's shape, which keeps the credential out of the same tree as the
    config. `config.toml` is written only when there is something to put in it
    (a base URL, a grader endpoint, or both), because an empty config file is
    not the same thing as no config file.
    """
    home, secrets_dir = cfg["codex_home"], cfg["secrets_dir"]
    os.makedirs(home, exist_ok=True)
    os.makedirs(secrets_dir, mode=0o700, exist_ok=True)

    auth = os.path.join(secrets_dir, "auth.json")
    with open(auth, "w") as f:
        json.dump({"OPENAI_API_KEY": api_key}, f, indent=2)
    os.chmod(auth, 0o600)
    link = os.path.join(home, "auth.json")
    try:
        if os.path.islink(link) or os.path.exists(link):
            os.unlink(link)
        os.symlink(auth, link)
    except OSError as e:
        _log(f"could not symlink auth.json: {e}")

    # codex 0.118+ only honours openai_base_url from config.toml, not the env
    # var, and takes MCP servers from nowhere else at all. A `provider:` in cfg
    # (configs/models.yaml `base_url:`) is written as a named model_providers
    # block instead, so a non-OpenAI endpoint gets the right wire_api.
    provider = cfg.get("provider")
    text = config_toml(cfg.get("base_url") or "", grader_url, grader_name,
                       provider=provider, mcp_only=cfg.get("mcp_only", False))
    if text:
        with open(os.path.join(home, "config.toml"), "w") as f:
            f.write(text)
        # The URL is logged; it is in a file the agent may read anyway. The
        # grader's SOURCE and the row are not logged, are not in that file, and
        # are not anywhere else the container can reach.
        _log(f"wrote config.toml (base_url="
             f"{cfg.get('base_url') or 'vendor default'}, provider="
             f"{provider['id'] + '/' + provider['wire_api'] if provider else 'openai'}"
             f", mcp_servers={grader_name if grader_url else 'none'})")


def compose_command(cfg: dict, instruction: str, logfile: str) -> str:
    """Harbor's `codex exec` command line, plus this eval's closed-book flags.

        codex exec --dangerously-bypass-approvals-and-sandbox
                   --skip-git-repo-check --strict-config [--ignore-user-config]
                   --model <slug> --json
                   --enable unified_exec --disable <feature> ...
                   -c model_reasoning_effort=... -c analytics.enabled=false
                   -c check_for_update_on_startup=false -c web_search=disabled
                   -- <shlex-quoted instruction> 2>&1 </dev/null | tee <log>

    Every flag is chosen by the adapter (core/agents/codex.py); this function only
    orders them. The instruction is ARGV, not stdin, and stdin is closed: Codex
    reads a prompt from stdin when it is not given one, and a `codex exec` still
    waiting on a terminal is an episode that hangs to the timeout.
    """
    parts = ["codex", "exec", *cfg["exec_flags"],
             "--model", cfg["model_slug"], "--json"]
    for feature in cfg.get("enable") or []:
        parts += ["--enable", feature]
    for feature in cfg.get("disable") or []:
        parts += ["--disable", feature]
    parts += list(cfg.get("cli_flags") or [])
    parts += ["--", instruction]
    # `if [ -s ~/.nvm/nvm.sh ]` first, exactly as Harbor does: the binary may
    # only exist on nvm's PATH.
    return ("set -o pipefail; if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
            + shlex.join(parts)
            + f" 2>&1 </dev/null | tee {shlex.quote(logfile)}")


def run_codex(command: str, env: dict, timeout_s: int) -> tuple[int | None, str | None]:
    """Run the pipeline, tee Codex's own output to stderr, enforce the timeout.

    Returns (returncode, error). Never raises.

    The process gets its own session so the timeout kills the whole pipeline --
    Codex, `tee` and anything Codex spawned -- rather than orphaning a CLI that
    is still talking to the API.
    """
    try:
        proc = subprocess.Popen(
            ["bash", "-c", command], cwd=WORKDIR, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True)
    except Exception as e:  # noqa: BLE001
        return None, f"codex launch failed: {type(e).__name__}: {e}"

    deadline = time.time() + timeout_s
    error: str | None = None

    def _kill(sig) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

    try:
        for line in proc.stdout:            # Codex's own log, for the operator
            sys.stderr.write(line if line.endswith("\n") else line + "\n")
            if time.time() > deadline:
                error = f"timeout after {timeout_s}s"
                _kill(signal.SIGKILL)
                break
        sys.stderr.flush()
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__} reading codex output: {e}"

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
# the rollout file, which is the actual record
# ---------------------------------------------------------------------------
def find_rollout(codex_home: str) -> str | None:
    """`$CODEX_HOME/sessions/<YYYY>/<MM>/<DD>/rollout-<ISO>-<uuid>.jsonl`.

    Harbor's rule: the DEEPEST session directory (dates nest three levels, and a
    shallower directory is a parent, not a session), then the lexicographic max
    filename -- rollout names lead with an ISO timestamp, so max is newest.
    """
    sessions = os.path.join(codex_home, "sessions")
    if not os.path.isdir(sessions):
        return None
    dirs: list[str] = []
    for dirpath, _dirnames, _files in os.walk(sessions):
        if dirpath != sessions:
            dirs.append(dirpath)
    if not dirs:
        return None
    deepest = max(len(d.split(os.sep)) for d in dirs)
    candidates = [d for d in dirs if len(d.split(os.sep)) == deepest]
    session_dir = max(candidates)
    files = [f for f in os.listdir(session_dir) if f.endswith(".jsonl")]
    if not files:
        return None
    return os.path.join(session_dir, max(files))


def read_rollout(path: str) -> list:
    """The rollout's lines as dicts. A malformed line is skipped, never fatal."""
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
                    _log(f"skipping malformed rollout line: {e}")
    except OSError as e:
        _log(f"could not read rollout {path}: {e}")
    return events


def summarise(events: list) -> dict:
    """Session id, final text, counts and token totals, from the rollout.

    The token numbers are Harbor's: the LAST `token_count` event carries
    `info.total_token_usage` for the whole session. There is no cost -- Codex
    never puts one in the rollout, and this container has no pricing table.
    """
    out = {"session_id": None, "final_text": "", "n_turns": 0,
           "n_tool_calls": 0, "usage": None, "model": None,
           "turn_context": None, "terminal_error": None}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        etype = ev.get("type")
        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}

        if etype == "session_meta":
            out["session_id"] = payload.get("id") or payload.get("session_id")
            continue
        if etype == "turn_context":
            if out["model"] is None and isinstance(payload.get("model"), str):
                out["model"] = payload["model"]
                out["turn_context"] = payload
            continue
        if etype == "event_msg":
            if payload.get("type") == "task_complete":
                # Codex's own verdict on the turn, and the ONLY place it
                # appears. `task_complete` closes every session -- a finished
                # one and an aborted one alike -- and carries `error` when the
                # turn died: `{"message": "{\"error\":{\"message\":\"Budget
                # has been exceeded! ...\"}}"}`. A completed turn has no
                # `error` key at all and a `last_agent_message` instead.
                err = payload.get("error")
                if isinstance(err, dict) and err.get("message"):
                    out["terminal_error"] = str(err["message"])
            if payload.get("type") == "token_count":
                # One token_count closes one model API call, which is the only
                # thing in a Codex rollout that means "turn".
                out["n_turns"] += 1
                info = payload.get("info")
                if isinstance(info, dict) and isinstance(
                        info.get("total_token_usage"), dict):
                    out["usage"] = info["total_token_usage"]
            continue
        if etype != "response_item":
            continue

        ptype = payload.get("type")
        if ptype in ("function_call", "custom_tool_call", "web_search_call"):
            out["n_tool_calls"] += 1
        elif ptype == "message" and payload.get("role") == "assistant":
            parts = [b.get("text") for b in payload.get("content") or []
                     if isinstance(b, dict) and isinstance(b.get("text"), str)]
            text = "".join(parts)
            if text.strip():
                out["final_text"] = text
    return out


# ---------------------------------------------------------------------------
# the episode
# ---------------------------------------------------------------------------
class RolloutObserver:
    """Tail complete native JSONL records for an optional task-owned observer.

    Codex's operator output is not a transcript. Tail its actual rollout while
    it runs; retain partial lines and flush once more after the CLI exits.
    Host transcript collection remains unchanged and never consumes this copy.
    """

    def __init__(self, codex_home, observer):
        self.codex_home, self.observer = codex_home, observer
        self.path, self.offset = None, 0
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def drain(self):
        path = find_rollout(self.codex_home)
        if not path:
            return
        if path != self.path:
            self.path, self.offset = path, 0
        with open(path, "rb") as f:
            f.seek(self.offset)
            while True:
                line = f.readline()
                if not line.endswith(b"\n"):
                    break
                self.offset = f.tell()
                if line.strip():
                    self.observer.observe(json.loads(line))

    def _loop(self):
        try:
            while not self.stop.is_set():
                self.drain()
                self.stop.wait(0.05)
        except Exception as e:
            self.error = f"live message observer failed: {type(e).__name__}: {e}"

    def close(self):
        self.stop.set()
        self.thread.join()
        if self.error is None:
            try:
                self.drain()
            except Exception as e:
                self.error = f"live message observer failed: {type(e).__name__}: {e}"


def run(task: dict, install: dict | None, grader=None,
        forwarder_mod: dict | None = None, observer=None) -> dict:
    """One episode. `grader` is a running `GraderServer`, or None for a task
    that ships no grader. `forwarder_mod` is the exec'd core/agents/forwarder.py
    namespace, present only when the model entry set `extra_body`."""
    cfg = dict(task.get("codex") or {})
    timeout_s = int(task.get("timeout_s") or 3600)
    key_env = cfg.get("api_key_env") or "OPENAI_API_KEY"
    api_key = os.environ.get(key_env) or ""

    error: str | None = None
    if not api_key:
        error = f"{key_env} is not set inside the container"

    # PER-MODEL ROUTING. `cfg["base_url"]` is already the resolved upstream
    # (entry `api_base_url:`, else the OPENAI_BASE_URL passthrough). A non-empty
    # `extra_body` is the opt-in for the forwarder: config.toml then points
    # codex at a loopback listener that merges the body into every JSON request
    # and forwards it to the real URL with the real key. The grader running in
    # this process keeps the REAL upstream in OPENAI_BASE_URL -- extra_body
    # belongs to the agent's model, not the grader's.
    routing = dict(task.get("routing") or {})
    upstream_base = cfg.get("base_url") or ""
    forwarder = None
    if routing.get("extra_body") and error is None:
        upstream = upstream_base or routing.get("api_base_url") or ""
        if not upstream:
            error = ("extra_body is set but there is no api_base_url to "
                     "forward to: set api_base_url on the model entry")
        elif not forwarder_mod:
            error = "extra_body is set but the forwarder module did not arrive"
        else:
            forwarder = forwarder_mod["Forwarder"](
                base_url=upstream, api_key=api_key,
                extra_body=routing["extra_body"], log=_log).start()
            cfg["base_url"] = forwarder.url
            if cfg.get("provider"):
                cfg["provider"] = {**cfg["provider"], "base_url": forwarder.url}
            _log(f"forwarder on {forwarder.url} -> {forwarder.upstream.base_url} "
                 f"(extra_body keys: {sorted(routing['extra_body'])})")
    _log(f"routing: base_url={cfg.get('base_url') or 'vendor default'} "
         f"key_env={key_env} forwarder={'on' if forwarder else 'off'}")

    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (task.get("env") or {}).items()})
    env["CODEX_HOME"] = cfg["codex_home"]
    env["OPENAI_API_KEY"] = api_key
    if upstream_base or cfg.get("base_url"):
        # Harbor sets it too, even though codex >= 0.118 ignores it; older
        # builds honour it and it costs nothing. The REAL upstream, never the
        # forwarder: this is what the grader reads.
        env["OPENAI_BASE_URL"] = upstream_base or cfg["base_url"]

    try:
        prepare_home(cfg, api_key,
                     grader_url=grader.url if grader is not None else "",
                     grader_name=(grader.grader_server_name
                                  if grader is not None else "grader"))
    except Exception as e:  # noqa: BLE001
        error = error or f"codex home setup failed: {type(e).__name__}: {e}"

    logfile = os.path.join(cfg["codex_home"], "codex.txt")
    command = compose_command(cfg, task["content"], logfile)
    _log(f"codex command: {command}")

    t0 = time.time()
    rc: int | None = None
    if error is None:
        tail = RolloutObserver(cfg["codex_home"], observer) if observer is not None else None
        if tail is not None:
            tail.thread.start()
        try:
            rc, error = run_codex(command, env, timeout_s)
        finally:
            if tail is not None:
                tail.close()
        if tail is not None and tail.error:
            error = error or tail.error
    wall = time.time() - t0

    # Stop serving the moment codex is gone, and BEFORE the state is read: the
    # grader's `state` is the authoritative record of what was graded, and a
    # late request arriving after we snapshot it would make the record disagree
    # with the episode. Closing here also releases the port promptly, which
    # matters when five episodes share the host's loopback.
    if grader is not None:
        grader.close()
    forwarder_calls = None
    if forwarder is not None:
        forwarder.close()
        forwarder_calls = list(forwarder.calls)

    # Read the rollout WHATEVER happened. Codex appends to it as it goes, so a
    # killed episode still has everything up to the kill.
    rollout_path = None
    events: list = []
    try:
        rollout_path = find_rollout(cfg["codex_home"])
        if rollout_path:
            events = read_rollout(rollout_path)
            _log(f"rollout {rollout_path}: {len(events)} events")
        else:
            _log(f"no rollout file under {cfg['codex_home']}/sessions")
    except Exception as e:  # noqa: BLE001
        _log(f"rollout lookup failed: {type(e).__name__}: {e}")

    for ev in events:
        _emit_msg(ev)

    info = summarise(events)
    if error is None and info["terminal_error"]:
        # The turn failed but `codex exec` still exited 0. THE case this exists
        # for: our gateway answers `400 ... Budget has been exceeded` once the
        # key is at its dollar cap, codex reports the turn as failed, prints it,
        # and returns 0 -- so none of the checks below fire and the episode was
        # recorded `ok: true` with the failure visible nowhere but the log.
        # Three pelican episodes were filed as clean runs that way, complete
        # with a verdict on a drawing the agent never finished. The episode was
        # cut off just the same, so it is an errored episode -- `ok: false`, a
        # typed `failure`, and the resume gets to see it -- exactly as the kimi
        # runner already does with a non-`completed` terminal reason.
        error = f"codex ended the turn with an error: {info['terminal_error']}"
    if error is None and rc not in (0, None) and not events:
        error = f"codex exited {rc} and wrote no rollout"

    try:
        deliverables = collect_deliverables(
            WORKDIR, tuple(task.get("skip_dirs") or ()),
            tuple(task.get("deliverable_files") or ()))
    except Exception as e:  # noqa: BLE001
        _log(f"deliverable collection failed: {type(e).__name__}: {e}")
        deliverables = []

    # Harbor's cleanup, best effort: the credential and the config do not
    # outlive the episode.
    for path in (cfg.get("secrets_dir"), cfg.get("codex_home")):
        if path:
            try:
                subprocess.run(["rm", "-rf", path], timeout=60)
            except Exception:  # noqa: BLE001
                pass

    return {
        "id": task["id"],
        "model": task["model"],
        # The rollout, in order. THIS is the trajectory.
        "messages": events,
        # The grader's own dict, exactly as the Claude runner reports it, or an
        # explicit None when the task ships no grader -- which is a statement
        # ("there was nothing to call") and not the same as `{}`.
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
        "init_tools": task.get("tools"),
        # What the Claude runner reads off the SDK's init message, observed
        # here instead: whether codex actually opened an MCP session against the
        # grader, and whether it listed the tool.
        "init_mcp_servers": [grader.report()] if grader is not None else [],
        "install": install,
        "codex_command": command,
        "rollout_path": rollout_path,
        # One summary per request the forwarder relayed (method, path, status,
        # sizes, seconds -- never a body), or None when no forwarder ran.
        "forwarder_calls": forwarder_calls,
        "returncode": rc,
        "error": error,
    }


def main(task: dict, modules: dict) -> None:
    """Stage the row's files, install Codex, host the grader, run it."""
    try:
        os.chdir(WORKDIR)
    except OSError:
        pass

    # The harness code that travels with us, exec'd out of the stdin blob so it
    # exists only in this process's memory -- the same contract the Claude
    # runner has. `stage` is always there; `grader` is the task's OPTIONAL
    # grader and arrives as "" when the task ships none. Empty is treated
    # exactly like absent, so a dropped key cannot masquerade as a grader.
    ns: dict = {}
    grader_src = (modules.get("grader") or "").strip()
    # `forwarder` travels only when the model entry set `extra_body`.
    forwarder_src = (modules.get("forwarder") or "").strip()
    for name in ("stage", *(("grader",) if grader_src else ()),
                 *(("forwarder",) if forwarder_src else ())):
        g: dict = {"__name__": f"rh_{name}"}
        exec(compile(modules[name], f"<{name}>", "exec"), g)
        ns[name] = g

    staged: dict = {}
    try:
        # Before Codex exists, so its very first `ls` already sees them.
        staged = ns["stage"]["write"](task.get("files") or [], WORKDIR)
        for dest, q in staged.items():
            _log(f"staged {dest} ({os.path.getsize(q)} bytes, "
                 f"mode {oct(os.stat(q).st_mode & 0o777)})")
    except Exception as e:  # noqa: BLE001 - still owe the host one JSON line
        _log(f"staging failed: {type(e).__name__}: {e}")

    install = install_agent(task.get("install"))

    # Install task dependencies after the agent runtime, without credentials.
    # Failure is recorded but does not prevent the episode from running.
    task_install = install_agent(task.get("task_install"),
                                 env=task_install_env())

    # The grader endpoint, started before codex so its URL is in config.toml by
    # the time the CLI reads it.
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
            # Mirrors the Claude runner: the episode still runs, and the failure
            # is reported IN `grader_state` rather than silently leaving the
            # record looking like a task that never had a grader.
            _log(f"grader construction failed: {type(e).__name__}: {e}")
            grader_error = {"error": f"{type(e).__name__}: {e}"}

    observer = None
    try:
        observer_factory = ns.get("grader", {}).get("make_observer")
        run_options = {"forwarder_mod": ns.get("forwarder")}
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
            grader.close()          # idempotent; run() closes it already
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
