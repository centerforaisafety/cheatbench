"""Host-only inference bridge for network-denied native game episodes.

This bridge forwards only model API operations to a fixed origin and filters
native tool declarations. It does not choose network or installation policy;
the episode's shared egress guard independently handles those rules.
"""
from __future__ import annotations

import http.client
import json
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from core.agents.forwarder import Upstream

MAX_BODY = 64 * 1024 * 1024
POST_PATHS = frozenset({"/v1/responses", "/responses", "/v1/messages", "/messages",
                        "/v1/chat/completions", "/chat/completions",
                        "/v1/messages/count_tokens", "/messages/count_tokens",
                        "/v1/messages?beta=true", "/v1/messages/count_tokens?beta=true"})
GET_PATHS = frozenset({"/v1/models", "/models"})
MCP_TOOLS = frozenset({"play_move", "game_status", "reset_game", "write_notes", "read_notes",
                       "api_request", "list_files", "read_file", "write_file"})


def restrict_tools(document, *, grok_mcp=False, mcp_server="chess"):
    """Expose only the selected game's MCP functions, including additional_tools.

    Runtime controls separately disable execution. Filtering declarations is
    needed because the native CLI can still advertise a disabled code host.
    No instruction, model, sampling parameter, or conversation result changes.
    """
    if mcp_server not in {"chess", "go"}:
        raise ValueError("Unsupported restricted game namespace")
    namespace = "mcp__" + mcp_server

    def keep(tool):
        if tool.get("type") == "namespace":
            if tool.get("name") != namespace:
                return None
            children = tool.get("tools", [])
            if not isinstance(children, list):
                raise ValueError("Invalid namespace tools")
            return {**tool, "tools": [t for t in children if isinstance(t, dict) and t.get("name") in MCP_TOOLS]}
        name = tool.get("name", "")
        if name in {namespace + "__" + n for n in MCP_TOOLS}:
            return tool
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            # Grok's native MCP dispatcher discovers/calls the same nine
            # server tools. session_title is an auxiliary structured result.
            if grok_mcp and tool["function"].get("name") in {"search_tool", "use_tool", "session_title"}:
                return tool
            if tool["function"].get("name") in {prefix + n for prefix in ("mcp_" + mcp_server + "_", namespace + "__") for n in MCP_TOOLS}:
                return tool
        return None

    def filtered(items):
        if not isinstance(items, list):
            raise ValueError("Invalid tool declarations")
        return [value for tool in items if isinstance(tool, dict) and (value := keep(tool)) is not None]

    if "tools" in document:
        document["tools"] = filtered(document["tools"])
    for item in document.get("input", []) if isinstance(document.get("input"), list) else []:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            item["tools"] = filtered(item.get("tools", []))
    return document


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def reject(self, status=403, *, allowed=False):
        self.server.record(self.command, self.path, status, allowed)
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def do_CONNECT(self):
        self.reject()

    do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = do_CONNECT

    def do_GET(self):
        self.forward(GET_PATHS)

    def do_POST(self):
        self.forward(POST_PATHS)

    def forward(self, paths):
        # Compare the entire target, including Claude's fixed beta query.
        # Other query strings, absolute URLs and traversal are not normalized.
        if self.path not in paths or self.headers.get("Upgrade") or self.headers.get("Transfer-Encoding"):
            return self.reject()
        lengths = self.headers.get_all("Content-Length") or []
        try:
            if len(lengths) > 1:
                raise ValueError("duplicate length")
            size = int(lengths[0]) if lengths else 0
            if not 0 <= size <= MAX_BODY:
                raise ValueError("invalid length")
        except ValueError:
            return self.reject(400)
        body = self.rfile.read(size) if size else None
        encoding = (self.headers.get("Content-Encoding") or "identity").lower()
        if encoding != "identity":
            return self.reject(400)
        if self.command == "POST":
            try:
                document = json.loads(body or b"")
                if not isinstance(document, dict):
                    raise ValueError("expected object")
                body = json.dumps(restrict_tools(document, grok_mcp=self.server.grok_mcp,
                                                 mcp_server=self.server.mcp_server)).encode()
            except (ValueError, UnicodeError):
                return self.reject(400)
        elif self.command != "POST" and body:
            return self.reject(400)
        started = time.monotonic()
        try:
            # SSE comments cannot be inserted into a compressed response. Ask
            # for identity; the relay still safely passes compressed bytes
            # through unchanged if an upstream ignores this preference.
            headers = {name: value for name, value in self.headers.items()
                       if name.lower() != "accept-encoding"}
            headers["Accept-Encoding"] = "identity"
            response = self.server.upstream.request(self.command, self.path, headers, body)
        except Exception as exc:
            self.server.record(self.command, self.path, 502, True,
                               event="upstream_error", error_type=type(exc).__name__)
            return self.reject(502, allowed=True)
        try:
            # Do not send redirects back to a client that may follow them.
            if 300 <= response.status < 400:
                return self.reject(502, allowed=True)
            self.server.record(self.command, self.path, response.status, True, event="response_started")
            response.relay(self, keepalive_interval=self.server.keepalive_interval)
            stats = getattr(response, "relay_stats", {})
            self.server.record(self.command, self.path, response.status, True,
                               event="transport_complete", elapsed_s=round(time.monotonic() - started, 3),
                               **(stats if isinstance(stats, dict) else {}))
        except (OSError, ValueError, http.client.HTTPException) as exc:
            self.close_connection = True
            stats = getattr(response, "relay_stats", {})
            self.server.record(self.command, self.path, response.status, True,
                               event="transport_error", error_type=type(exc).__name__,
                               elapsed_s=round(time.monotonic() - started, 3),
                               **(stats if isinstance(stats, dict) else {}))
        finally:
            response.close()


class InferenceBridge(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, base_url, api_key, log_path, *, keepalive_interval=15, grok_mcp=False, mcp_server="chess"):
        if mcp_server not in {"chess", "go"}:
            raise ValueError("Unsupported restricted game namespace")
        if keepalive_interval <= 0:
            raise ValueError("keepalive_interval must be positive")
        parsed = urlsplit(base_url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or parsed.path.rstrip("/") not in ("", "/v1")):
            raise ValueError("Restricted native inference requires an HTTPS model origin, optionally ending in /v1")
        super().__init__(("127.0.0.1", 0), Handler)
        self.upstream = Upstream(base_url, api_key, timeout=900)
        self.keepalive_interval = keepalive_interval
        self.grok_mcp = bool(grok_mcp)
        self.mcp_server = mcp_server
        self.hosts = (parsed.hostname,)
        self.path_prefix = parsed.path.rstrip("/")
        self.log_path = Path(log_path)
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        self._thread.start()

    @property
    def local_base_url(self):
        return f"http://127.0.0.1:{self.server_port}{self.path_prefix}"

    def record(self, method, path, status, allowed, **details):
        # No prompts, bodies, headers or keys. Bound rejected targets too.
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(),
                 "host": self.hosts[0], "method": method, "path": path[:256],
                 "status": status, "allowed": allowed, **details}
        with self._lock:
            with self.log_path.open("a") as stream:
                stream.write(json.dumps(entry) + "\n")

    def attempts(self):
        if not self.log_path.exists():
            return []
        return [json.loads(line) for line in self.log_path.read_text().splitlines()]

    def close(self):
        self.shutdown()
        self.server_close()
        self._thread.join(5)


def reroute_blob(blob, local_base_url):
    """Change only the CLI transport; preserve model, prompt, tools and sampling."""
    payload = json.loads(blob)
    task = payload["task"]
    if task.get("tools"):
        raise ValueError("Restricted native games require an empty native tool list (MCP only)")
    task.setdefault("env", {})["ANTHROPIC_BASE_URL"] = local_base_url
    task["env"]["OPENAI_BASE_URL"] = local_base_url
    task.setdefault("routing", {})["api_base_url"] = local_base_url
    if "terminus_2" in task:
        if not task["terminus_2"].get("mcp_only"):
            raise ValueError("Restricted Terminus-2 requires MCP-only execution")
        task["api_base"] = local_base_url
    if "codex" in task:
        task["codex"]["base_url"] = local_base_url
        if task["codex"].get("provider"):
            task["codex"]["provider"]["base_url"] = local_base_url
    if "gemini" in task:
        if not task["gemini"].get("mcp_only") or task["gemini"].get("route") != "gateway":
            raise ValueError("Restricted native Gemini requires MCP-only gateway routing")
        task["gemini"]["base_url"] = local_base_url
        task["env"]["GEMINI_BASE_URL"] = local_base_url
    if "deepseek_harness" in task:
        if not task["deepseek_harness"].get("mcp_only"):
            raise ValueError("Restricted native DeepSeek requires MCP-only routing")
        task["deepseek_harness"]["base_url"] = local_base_url
        task["env"]["DEEPSEEK_BASE_URL"] = local_base_url
    if "grok" in task:
        if not task["grok"].get("mcp_only"):
            raise ValueError("Restricted native Grok requires MCP-only routing")
        task["grok"]["base_url"] = local_base_url
    if "kimi" in task:
        if not task["kimi"].get("mcp_only"):
            raise ValueError("Restricted native Kimi requires MCP-only routing")
        task["kimi"]["api_base_url"] = local_base_url
    if "muse" in task:
        if not task["muse"].get("mcp_only") or task["muse"].get("route") != "gateway":
            raise ValueError("Restricted native Muse requires MCP-only gateway routing")
        task["muse"]["base_url"] = local_base_url
    return json.dumps(payload).encode()
