"""The Grok Build episode runner that executes INSIDE the container.

MODELLED ON HARBOR: `harbor/src/harbor/agents/installed/grok_build.py`
(`_build_config_toml`, `_build_run_script`, `run`, `_parse_usage_report`), and
on the same contract as `core/agents/codex_runner.py`: this file lives in the
repo, is never baked into the image, is streamed in on **stdin** with
`core/stage.py` and the staged bytes, and writes exactly ONE line of JSON on
stdout. Everything else -- the CLI's own stderr, tracebacks, turn streaming --
goes to stderr, so the host parses stdout without heuristics.

What is different from the Codex runner, and why:

  * The CLI is a single static binary (`grok`, from https://x.ai/cli/install.sh)
    rather than an npm package. There is no Node to bring; `check || install`
    is a curl and a bash.
  * The episode is one `grok -p <instruction> --always-approve --output-format
    streaming-json ...` subprocess. The instruction is ARGV (Harbor's shape),
    stdin is `</dev/null` (headless grok never reads it), and STDOUT IS DATA:
    the streaming-json record carries the per-response `usage` events and the
    terminal `end` event with the session's totals and, when the server
    stamped one, its cost. So stdout goes to a file the runner parses, and
    only stderr is relayed as the operator's log.
  * THE TRAJECTORY IS ON DISK, NOT ON STDOUT. Harbor parses
    `$GROK_HOME/sessions/<url-encoded cwd>/<session-id>/chat_history.jsonl`
    -- "raw chat messages sent to the model" -- after copying the sessions
    tree to the host. There is no bind mount here, so this runner reads the
    file inside the container and returns its lines as `messages`. The file is
    appended as the session runs, which is what makes a killed episode
    recoverable, and is also what the live turn tail reads.
  * The grader is hosted over HTTP, exactly as codex_runner does it: grok takes
    an MCP server from `$GROK_HOME/config.toml` in the `url` shape, so the
    task's grader is exec'd out of the stdin blob, served on loopback behind an
    unguessable path, and only that URL is written to the config file. What
    the model sees is different from codex and Claude, and measured rather than
    assumed (see `serve_grader`): grok exposes MCP tools through two built-in
    meta-tools, `search_tool` (discover) and `use_tool` (call), so the
    trajectory shows `use_tool({tool_name: "grader__grade_deliverable", ...})`
    and never a call named after the grader.

Auth is the config file's: `[model.<id>] env_key = "<VAR>"` names the
environment variable the CLI reads the key from, and the sandbox passes exactly
that variable in by name. `base_url` in the same block is where the CLI sends
its requests; the model id is sent VERBATIM, with no prefix stripping anywhere,
which is what lets `openrouter/x-ai/grok-4.6` reach the gateway's OpenRouter
route unchanged.
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import shlex
import shutil
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
# other runners: this is a property of the TASK, not of the agent.
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
# `check || install`, run after staging and before grok exists. Same contract
# and same reporting as the other runners; each runner is deliberately
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
# the task's grader, as an MCP server grok can reach by URL
#
# The same server codex_runner hosts, for the same reasons (see the block
# comment there: url not stdio, loopback, kernel-assigned port, unguessable
# path, hmac-compared). Two things are grok-specific and were MEASURED against
# grok 1.0.24 rather than inferred:
#
#   * grok's MCP client (rmcp, streamable HTTP) POSTs `initialize`,
#     `notifications/initialized`, `tools/list` and `tools/call` exactly as
#     codex does, with `Accept: application/json, text/event-stream`. Answered
#     with an SSE body it logs "worker quit with fatal: unexpected server
#     response: empty sse stream" for its notification listener and carries on
#     -- the tool call still lands -- so `_send` answers with plain
#     `application/json` whenever the client accepts it, and SSE only when
#     that is all it accepts.
#   * the model does not see `grader__grade_deliverable` as a tool of its own.
#     grok exposes every MCP tool through two built-ins, `search_tool` (which
#     returned the grader's name, description and input schema under
#     `grader__grade_deliverable`) and `use_tool` (called with `tool_name:
#     "grader__grade_deliverable", tool_input: {...}`), and the server saw ONE
#     `tools/call` for it. So the tool name the server advertises is the same
#     on all three adapters; the call frame in the trajectory is grok's.
# ---------------------------------------------------------------------------
MCP_PROTOCOL_VERSION = "2025-06-18"


class _GraderMCPHandler(BaseHTTPRequestHandler):
    """Streamable-HTTP MCP, the methods grok actually calls."""

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
        """One JSON-RPC response: JSON when the client accepts it, else SSE.

        The spec lets the server answer a POST with either. grok accepts both
        and treats a one-event SSE reply as a broken listener stream (measured,
        see above), so JSON is preferred whenever it is on offer.
        """
        accept = self.headers.get("Accept") or ""
        if "application/json" in accept or "text/event-stream" not in accept:
            payload, ctype = body, "application/json"
        else:
            payload = b"event: message\ndata: " + body + b"\n\n"
            ctype = "text/event-stream"
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
        """Stop serving. Idempotent; see codex_runner.GraderServer.close."""
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
        actually arrived on the socket.
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

    `module` must export `make_tool(row, staged, workdir)` returning
    `(state, call)` -- the same seam the other runners reach, so every adapter
    runs one implementation of the verdict logic and one definition of the tool.
    """
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
# grok's home: config.toml, and nothing else of ours
# ---------------------------------------------------------------------------
# How long grok waits on one grader call. gdpval's grader renders with
# LibreOffice twice and then asks a vision model, ~25 s in the normal case;
# grok's own default for this key is 6000 s, so this is a tightening, not a
# rescue, and it keeps a wedged grader from eating the episode's wall clock.
GRADER_TOOL_TIMEOUT_S = 300


def _toml_str(value) -> str:
    """A TOML basic string. JSON's escaping is a valid subset of TOML's."""
    return json.dumps(str(value))


def _toml_bool(value) -> str:
    return "true" if value else "false"


def config_toml(cfg: dict, grader_url: str = "", grader_name: str = "grader") -> str:
    """The `$GROK_HOME/config.toml` text for this episode.

    Harbor's `_build_config_toml`, with three additions that are measured
    facts about grok 1.0.24 rather than preferences:

      * `[model."<id>"]` is what points the CLI at the gateway. `base_url` +
        `env_key` + `api_backend = "chat_completions"` is the documented BYOK
        shape (docs/user-guide/11-custom-models.md); the id is sent verbatim.
        `supports_reasoning_effort = true` is REQUIRED for `--reasoning-effort`
        to reach the wire on a custom model -- without it the CLI logs
        "model does not support reasoning effort; ignoring" and sends
        nothing. No `temperature`, `top_p` or `max_completion_tokens`: the
        request carries only `model`, `messages`, `tools`, `reasoning_effort`
        and `stream`, verified through a logging proxy.
      * The auxiliary models (`session_summary`, `image_description`,
        `web_search`) are pinned to the same entry, as Harbor does, so the CLI
        never falls back to its native `grok-4.6` catalog entry for a side
        call (which through the gateway is a 400). The session-title call the
        CLI makes at the start of every session is one such side call; it is
        the CLI's own bookkeeping, it carries the CLI's own sampling settings,
        and it is EXCLUDED from the `end` usage totals, so it is not in
        `cost_usd` either. See the adapter's docstring.
      * Everything that would phone home or generate on the side is switched
        off: telemetry, trace/codebase/workspace uploads, the remote model
        catalog, managed config, cross-session memory, session recap, turn
        summary, prompt suggestions, subagents (whose sessions the judge would
        never see), image/video generation (xAI-side tools), and the two
        web tools when the policy closes them.

    The scalar `disable_web_search` comes FIRST: in TOML everything after a
    `[table]` header belongs to that table. The MCP entry is the URL shape and
    there is deliberately no `command`/`args` branch (see codex_runner).
    """
    slug = cfg["model_slug"]
    lines = [
        f"disable_web_search = {_toml_bool(not cfg.get('web_search'))}",
        "",
        "[cli]",
        "auto_update = false",
        "session_registry = false",
        "use_leader = false",
        "",
        "[features]",
        "telemetry = false",
        "backend_tools = false",
        "remote_fetch = false",
        "managed_config = false",
        "campaigns = false",
        "image_gen = false",
        "video_gen = false",
        "ask_user_question = false",
        "feedback = false",
        "title_refresh = false",
        "session_recap = false",
        "turn_summary = false",
        f"web_fetch = {_toml_bool(cfg.get('web_fetch'))}",
        "",
        "[telemetry]",
        "trace_upload = false",
        "",
        "[harness]",
        "disable_codebase_upload = true",
        "disable_workspace_teleport = true",
        "",
        "[memory]",
        "enabled = false",
        "",
        "[subagents]",
        "enabled = false",
        "",
        "[ui]",
        "prompt_suggestions = false",
        "",
        "[models]",
        f"default = {_toml_str(slug)}",
        f"session_summary = {_toml_str(slug)}",
        f"image_description = {_toml_str(slug)}",
        f"web_search = {_toml_str(slug)}",
    ]
    if cfg.get("base_url"):
        lines += [
            "",
            f"[model.{_toml_str(slug)}]",
            f"name = {_toml_str(slug)}",
            f"model = {_toml_str(cfg['model_id'])}",
            f"base_url = {_toml_str(cfg['base_url'])}",
            f"env_key = {_toml_str(cfg['api_key_env'])}",
            'api_backend = "chat_completions"',
            "supports_reasoning_effort = true",
        ]
    if grader_url:
        lines += [
            "",
            f"[mcp_servers.{grader_name}]",
            f"url = {_toml_str(grader_url)}",
            f"tool_timeout_sec = {GRADER_TOOL_TIMEOUT_S}",
        ]
    return "\n".join(lines) + "\n"


def prepare_home(cfg: dict, grader_url: str = "",
                 grader_name: str = "grader") -> str:
    """Write `$GROK_HOME/config.toml`. Returns the path written.

    The credential is NOT written anywhere: the CLI reads it from the
    environment variable `env_key` names, which is the one variable the sandbox
    passed in by name. The config file the agent may read therefore holds a
    URL, a variable NAME and a loopback endpoint, and nothing else.
    """
    home = cfg["grok_home"]
    os.makedirs(home, exist_ok=True)
    path = os.path.join(home, "config.toml")
    with open(path, "w") as f:
        f.write(config_toml(cfg, grader_url, grader_name))
    _log(f"wrote config.toml (model={cfg['model_slug']}, base_url="
         f"{cfg.get('base_url') or 'vendor default'}, mcp_servers="
         f"{grader_name if grader_url else 'none'})")
    return path


def gateway_probe(cfg: dict, api_key: str, timeout: float = 120.0) -> dict | None:
    """ONE plain chat completion to the base URL, recording who answered.

    The CLI never surfaces the gateway's response metadata, and the thing this
    adapter exists to guarantee -- that `openrouter/x-ai/grok-4.6` reached the
    gateway's OpenRouter route and was served by xAI -- is stated by the
    response's `provider` field and nowhere else. So the runner asks once, with
    the same model id, key and base URL the CLI is about to use, and records
    `{model, provider, status, seconds}` on the episode. It is a probe, not a
    turn: it is not in the trajectory, its tokens are not in `usage`, and its
    body carries no temperature, top_p, seed or token cap. Never fatal: a
    failed probe is recorded as one, and the episode runs regardless.
    """
    import urllib.error
    import urllib.request

    base = (cfg.get("base_url") or "").rstrip("/")
    if not base or not api_key:
        return None
    body = json.dumps({
        "model": cfg["model_id"],
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
    }).encode()
    # An explicit User-Agent: the gateway's edge rejects urllib's default one
    # outright (Cloudflare "error code: 1010"), measured; any named agent
    # string passes.
    req = urllib.request.Request(
        base + "/chat/completions", data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json",
                 "User-Agent": "grok-build-runner/1.0"})
    t0 = time.time()
    out: dict = {"model": cfg["model_id"], "url": base + "/chat/completions"}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
            out["status"] = resp.status
    except urllib.error.HTTPError as e:
        out["status"] = e.code
        out["error"] = e.read().decode("utf-8", "replace")
        data = {}
    except Exception as e:  # noqa: BLE001 - never fatal
        out["error"] = f"{type(e).__name__}: {e}"
        data = {}
    out["seconds"] = round(time.time() - t0, 1)
    if isinstance(data, dict):
        for key in ("provider", "model"):
            if isinstance(data.get(key), str):
                out["response_" + key] = data[key]
        usage = data.get("usage")
        if isinstance(usage, dict):
            out["usage"] = {k: usage[k] for k in ("prompt_tokens",
                                                   "completion_tokens", "cost")
                            if k in usage}
    _log(f"gateway probe: status={out.get('status')} provider="
         f"{out.get('response_provider')} model={out.get('response_model')} "
         f"in {out['seconds']}s" + (f" error={out['error']}" if out.get("error")
                                    else ""))
    return out


def compose_argv(cfg: dict, instruction: str) -> list[str]:
    """Harbor's headless `grok` command line, as argv.

        grok --no-auto-update -p <instruction> --always-approve
             --output-format streaming-json --session-id <uuid>
             --model <id> [--max-turns N] --reasoning-effort <effort>
             [--disable-web-search] [--disallowed-tools a,b] --cwd /workspace

    Every flag is chosen by the adapter (core/agents/grok_build.py); this
    function only orders them. The instruction is ARGV, not stdin: headless
    grok does not read a prompt from stdin at all.
    """
    grok = shutil.which("grok") or os.path.join(
        os.environ.get("HOME", "/root"), ".grok", "bin", "grok")
    argv = [grok, "--no-auto-update", "-p", instruction,
            *list(cfg.get("permission_flags") or ["--always-approve"]),
            "--output-format", "streaming-json",
            "--session-id", cfg["session_id"],
            "--model", cfg["model_slug"]]
    if cfg.get("max_turns"):
        argv += ["--max-turns", str(int(cfg["max_turns"]))]
    if cfg.get("reasoning_effort"):
        argv += ["--reasoning-effort", str(cfg["reasoning_effort"])]
    if not cfg.get("web_search") and not cfg.get("web_fetch"):
        argv.append("--disable-web-search")
    if cfg.get("disallowed_tools"):
        argv += ["--disallowed-tools", ",".join(cfg["disallowed_tools"])]
    if cfg.get("mcp_only"):
        argv += ["--tools", "search_tool,use_tool", "--no-subagents", "--no-plan"]
    argv += ["--cwd", WORKDIR]
    return argv


# ---------------------------------------------------------------------------
# the session directory, which is the actual record
# ---------------------------------------------------------------------------
def find_session_dir(grok_home: str, session_id: str) -> str | None:
    """`$GROK_HOME/sessions/<url-encoded cwd>/<session-id>/`.

    The middle component is the URL-encoded cwd (or a slug+hash when that
    would exceed 255 bytes), so it is not composed but searched for: Harbor's
    `_find_chat_history_path` does the same walk.
    """
    sessions = os.path.join(grok_home, "sessions")
    if not os.path.isdir(sessions):
        return None
    for dirpath, dirnames, _files in os.walk(sessions):
        if session_id in dirnames:
            return os.path.join(dirpath, session_id)
    return None


class ChatHistoryTail:
    """Incremental reader of `chat_history.jsonl`, complete lines only.

    grok appends one JSON object per line as the session runs. Reading a line
    that is still being written would hand the host half a JSON object, so
    only text up to the last newline is consumed and the remainder waits.
    """

    def __init__(self, path: str, observer=None):
        self.path = path
        self.offset = 0
        self.records: list = []
        self.observer = observer
        self.error = None

    def poll(self) -> list:
        """New complete records since the last poll, also emitted live."""
        fresh: list = []
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                data = fh.read()
        except OSError:
            return fresh
        cut = data.rfind(b"\n")
        if cut < 0:
            return fresh
        chunk, self.offset = data[:cut + 1], self.offset + cut + 1
        for line in chunk.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                _log(f"skipping malformed chat_history line: {e}")
                continue
            if isinstance(rec, dict):
                fresh.append(rec)
                self.records.append(rec)
                _emit_msg(rec)
                if self.observer is not None and self.error is None:
                    try:
                        self.observer.observe(rec)
                    except Exception as e:
                        self.error = f"live message observer failed: {type(e).__name__}: {e}"
        return fresh


def read_stream(path: str) -> list:
    """The CLI's streaming-json stdout as dicts. Non-JSON lines are skipped."""
    events = []
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        _log(f"could not read stream {path}: {e}")
    return events


USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens",
              "cache_creation_input_tokens", "reasoning_tokens")


def _usage_dict(payload) -> dict | None:
    if not isinstance(payload, dict):
        return None
    out = {}
    for key in USAGE_KEYS:
        value = payload.get(key)
        out[key] = int(value) if isinstance(value, (int, float)) and not \
            isinstance(value, bool) else 0
    return out


def summarise_stream(events: list) -> dict:
    """Usage, cost, stop reason and the tool list, from the streaming-json.

    Harbor's `_parse_usage_report`: the terminal `end`/`error` payload is
    authoritative for the totals and for the reported cost; the per-response
    `usage` events are summed as the fallback for an episode that was killed
    before `end` (the host reads `stream_usage` when `usage` is absent).

    Token field policy, from the CLI's own docs and confirmed on a real run:
    `input_tokens` is the UNCACHED prompt count and the two cache buckets sit
    beside it (25152 fresh + 12928 cache reads on the smoke session, with
    `total_tokens` = all three + output). The adapter says so with
    PROMPT_TOKENS_INCLUDE_CACHE = False.

    `reported_cost_usd` is the server's figure and only when it is COMPLETE:
    `cost_is_partial` drops it, exactly as Harbor does, so a partial bill is
    never recorded as the whole one. Through the gateway it is absent, and the
    host prices the tokens itself.
    """
    out = {"usage": None, "stream_usage": None, "usage_calls": [],
           "reported_cost_usd": None, "stop_reason": None,
           "result_subtype": None, "session_id": None, "num_turns": None,
           "init_tools": None, "error": None, "usage_is_incomplete": False}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        etype = ev.get("type")
        if etype == "available_commands":
            tools = ev.get("tools")
            if isinstance(tools, list):
                out["available_tools"] = [str(t) for t in tools]
                if out["init_tools"] is None:
                    out["init_tools"] = out["available_tools"]
            continue
        if etype == "usage":
            u = _usage_dict(ev.get("usage"))
            if u:
                out["usage_calls"].append(u)
            continue
        if etype in ("end", "error"):
            out["result_subtype"] = etype
            if etype == "error":
                msg = ev.get("message")
                out["error"] = str(msg) if msg else "grok reported an error"
            if isinstance(ev.get("stopReason"), str):
                out["stop_reason"] = ev["stopReason"]
            if isinstance(ev.get("sessionId"), str):
                out["session_id"] = ev["sessionId"]
            if isinstance(ev.get("num_turns"), int):
                out["num_turns"] = ev["num_turns"]
            total = _usage_dict(ev.get("usage"))
            if total and any(total.values()):
                out["usage"] = total
            out["usage_is_incomplete"] = ev.get("usage_is_incomplete") is True
            if ev.get("cost_is_partial") is not True:
                ticks = ev.get("total_cost_usd_ticks")
                cost = ev.get("total_cost_usd")
                if isinstance(ticks, int) and not isinstance(ticks, bool) \
                        and ticks >= 0:
                    out["reported_cost_usd"] = ticks / 10_000_000_000
                elif isinstance(cost, (int, float)) and not \
                        isinstance(cost, bool) and cost >= 0:
                    out["reported_cost_usd"] = float(cost)
    if out["usage_calls"]:
        summed = {k: sum(u[k] for u in out["usage_calls"]) for k in USAGE_KEYS}
        out["stream_usage"] = summed
    return out


def summarise_chat(messages: list) -> dict:
    """Final text and counts, from chat_history. One assistant line = one
    model response, which is the only thing in the file that means "turn"."""
    out = {"final_text": "", "n_turns": 0, "n_tool_calls": 0, "model": None}
    for m in messages:
        if not isinstance(m, dict) or m.get("type") != "assistant":
            continue
        out["n_turns"] += 1
        calls = m.get("tool_calls")
        if isinstance(calls, list):
            out["n_tool_calls"] += len(calls)
        if out["model"] is None and isinstance(m.get("model_id"), str):
            out["model"] = m["model_id"]
        content = m.get("content")
        if isinstance(content, list):
            content = "".join(b.get("text") or "" for b in content
                              if isinstance(b, dict))
        if isinstance(content, str) and content.strip():
            out["final_text"] = content
    return out


# ---------------------------------------------------------------------------
# the episode
# ---------------------------------------------------------------------------
def run_grok(argv: list, env: dict, timeout_s: int, stdout_path: str,
             tail: ChatHistoryTail | None, home: str) -> tuple[int | None, str | None]:
    """Run the CLI, relay its stderr, tail its chat history, enforce the timeout.

    Returns (returncode, error). Never raises. The process gets its own session
    so the timeout kills everything grok spawned, not just grok.

    The tail is created lazily: the session directory does not exist until the
    CLI has created it, and its name has a URL-encoded cwd in the middle.
    """
    try:
        stdout_fh = open(stdout_path, "wb")
    except OSError as e:
        return None, f"cannot open {stdout_path}: {e}"
    try:
        proc = subprocess.Popen(
            argv, cwd=WORKDIR, env=env, stdin=subprocess.DEVNULL,
            stdout=stdout_fh, stderr=subprocess.PIPE, text=True, bufsize=1,
            start_new_session=True)
    except Exception as e:  # noqa: BLE001
        stdout_fh.close()
        return None, f"grok launch failed: {type(e).__name__}: {e}"

    def _relay() -> None:
        try:
            for line in proc.stderr:            # the CLI's own log, for the operator
                sys.stderr.write(line if line.endswith("\n") else line + "\n")
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass

    relay = threading.Thread(target=_relay, daemon=True)
    relay.start()

    deadline = time.time() + timeout_s
    error: str | None = None

    def _kill(sig) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

    rc: int | None = None
    while True:
        try:
            rc = proc.wait(timeout=0.05 if tail is not None and tail.observer is not None else 1.0)
        except subprocess.TimeoutExpired:
            rc = None
        if tail is not None and tail.path is None:
            session_dir = find_session_dir(home, tail.session_id)
            if session_dir:
                tail.path = os.path.join(session_dir, "chat_history.jsonl")
        if tail is not None and tail.path:
            tail.poll()
        if rc is not None:
            break
        if time.time() > deadline:
            error = f"timeout after {timeout_s}s"
            _kill(signal.SIGKILL)
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                rc = None
            break
    relay.join(timeout=10)
    stdout_fh.close()
    return rc, error


class _LazyTail(ChatHistoryTail):
    """A tail whose file is found once the CLI has created the session dir."""

    def __init__(self, session_id: str, observer=None):
        super().__init__(path=None, observer=observer)
        self.session_id = session_id


def run(task: dict, install: dict | None, grader=None, observer=None) -> dict:
    """One episode. `grader` is a running `GraderServer`, or None for a task
    that ships no grader."""
    cfg = dict(task.get("grok") or {})
    timeout_s = int(task.get("timeout_s") or 3600)
    key_env = cfg.get("api_key_env") or "OPENAI_API_KEY"
    api_key = os.environ.get(key_env) or ""
    home = cfg["grok_home"]

    # THIS PROCESS's environment first, then grok's: the task's grader runs in
    # this process and reads os.environ for its own endpoint (see codex_runner
    # for the episode that taught us this).
    os.environ.update({str(k): str(v) for k, v in (task.get("env") or {}).items()})
    env = dict(os.environ)
    env["GROK_HOME"] = home
    env["GROK_DISABLE_AUTOUPDATER"] = "1"
    env.setdefault("RUST_LOG", "warn")
    if api_key:
        env[key_env] = api_key

    error: str | None = None
    if not api_key:
        error = f"{key_env} is not set inside the container"

    try:
        prepare_home(cfg, grader_url=grader.url if grader is not None else "",
                     grader_name=(grader.grader_server_name
                                  if grader is not None else "grader"))
    except Exception as e:  # noqa: BLE001
        error = error or f"grok home setup failed: {type(e).__name__}: {e}"

    probe = None
    if error is None:
        try:
            probe = gateway_probe(cfg, api_key)
        except Exception as e:  # noqa: BLE001
            probe = {"error": f"{type(e).__name__}: {e}"}

    argv = compose_argv(cfg, task["content"])
    command = " ".join(shlex.quote(a) for a in argv)
    _log(f"grok command: {command}")

    stream_path = os.path.join(home, "grok-build.jsonl")
    tail = _LazyTail(cfg["session_id"], observer=observer)

    t0 = time.time()
    rc: int | None = None
    if error is None:
        rc, error = run_grok(argv, env, timeout_s, stream_path, tail, home)
    wall = time.time() - t0

    # Stop serving the moment grok is gone and BEFORE the state is read.
    if grader is not None:
        grader.close()

    # Read the session WHATEVER happened. grok appends chat_history as it goes,
    # so a killed episode still has everything up to the kill.
    session_dir = find_session_dir(home, cfg["session_id"])
    if session_dir and tail.path is None:
        tail.path = os.path.join(session_dir, "chat_history.jsonl")
    if tail.path:
        tail.poll()
    error = error or tail.error
    messages = list(tail.records)
    if session_dir:
        _log(f"session {session_dir}: {len(messages)} chat_history lines")
    else:
        _log(f"no session directory for {cfg['session_id']} under {home}/sessions")

    stream = summarise_stream(read_stream(stream_path))
    chat = summarise_chat(messages)
    if error is None and stream.get("error"):
        error = stream["error"]
    if error is None and rc not in (0, None) and not messages:
        error = f"grok exited {rc} and wrote no chat history"

    try:
        deliverables = collect_deliverables(
            WORKDIR, tuple(task.get("skip_dirs") or ()),
            tuple(task.get("deliverable_files") or ()))
    except Exception as e:  # noqa: BLE001
        _log(f"deliverable collection failed: {type(e).__name__}: {e}")
        deliverables = []

    # Harbor's cleanup, best effort: the config and the session do not outlive
    # the episode (the record already carries everything of theirs we keep).
    try:
        subprocess.run(["rm", "-rf", home], timeout=60)
    except Exception:  # noqa: BLE001
        pass

    return {
        "id": task["id"],
        "model": task["model"],
        # chat_history.jsonl, in order. THIS is the trajectory.
        "messages": messages,
        "grader_state": grader.grader_state if grader is not None else None,
        "deliverables": deliverables,
        "final_text": chat["final_text"],
        "n_turns": (stream["num_turns"] if stream["num_turns"] is not None
                    else chat["n_turns"]),
        "n_tool_calls": chat["n_tool_calls"],
        "wall_time": round(wall, 1),
        # Never an estimate: the adapter's `reported_cost_usd` reads this field
        # back, and the host's litellm estimate must never be relabelled as the
        # vendor's own figure on a re-cost. Set ONLY from the CLI's `end`.
        "cost_usd": None,
        "reported_cost_usd": stream["reported_cost_usd"],
        "usage": stream["usage"],
        "stream_usage": stream["stream_usage"],
        "usage_calls": stream["usage_calls"],
        "usage_is_incomplete": stream["usage_is_incomplete"],
        "result_subtype": stream["result_subtype"],
        "terminal_reason": stream["stop_reason"],
        "session_id": stream["session_id"] or cfg["session_id"],
        # What the CLI actually offered the model, from its own
        # `available_commands` event -- the ground truth the adapter's declared
        # tool list is checked against, not a copy of it.
        "init_tools": (stream.get("available_tools") if cfg.get("mcp_only") else
                       (stream["init_tools"] if stream["init_tools"] is not None else task.get("tools"))),
        "init_mcp_servers": [grader.report()] if grader is not None else [],
        "install": install,
        "grok_command": command,
        "session_dir": session_dir,
        # Who served the model id at the base URL, from one probe request
        # before the CLI started (see `gateway_probe`). Not part of `usage`.
        "gateway_probe": probe,
        "returncode": rc,
        "error": error,
    }


def main(task: dict, modules: dict) -> None:
    """Stage the row's files, install grok, host the grader, run it."""
    try:
        os.chdir(WORKDIR)
    except OSError:
        pass

    ns: dict = {}
    grader_src = (modules.get("grader") or "").strip()
    for name in ("stage", *(("grader",) if grader_src else ()),
                 *(("support",) if modules.get("support") else ())):
        g: dict = {"__name__": f"rh_{name}"}
        exec(compile(modules[name], f"<{name}>", "exec"), g)
        ns[name] = g

    staged: dict = {}
    try:
        # Before grok exists, so its very first `ls` already sees them.
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
        run_options = {}
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
