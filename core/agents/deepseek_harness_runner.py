"""Drive the official SDK; retain native events including child-session tools."""
from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import tempfile

WORKDIR = "/workspace"

def capture_usage(response, call):
    """Observe streaming usage without changing any bytes sent to the CLI."""
    original = response.iter_chunks
    def chunks(size=65536):
        pending = b""
        for chunk in original(size):
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                if line.startswith(b"data:"):
                    try:
                        event = json.loads(line[5:].strip())
                    except ValueError:
                        continue
                    if isinstance(event, dict) and isinstance(event.get("usage"), dict):
                        call["usage"] = event["usage"]
            yield chunk
        # Non-streaming Chat Completions are also valid.
        if pending:
            try:
                event = json.loads(pending)
                if isinstance(event.get("usage"), dict):
                    call["usage"] = event["usage"]
            except (ValueError, AttributeError):
                pass
    response.iter_chunks = chunks


def filter_request(document, allowed_tools=None):
    if not isinstance(document, dict):
        raise ValueError("Invalid model request: expected an object")
    if allowed_tools is not None:
        declared = document.get("tools") or []
        if not isinstance(declared, list):
            raise ValueError("MCP-only tool policy breach: invalid declarations")
        names = []
        for tool in declared:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(function, dict):
                raise ValueError("MCP-only tool policy breach: non-function tool")
            name = function.get("name")
            if not isinstance(name, str) or name not in allowed_tools:
                raise ValueError("MCP-only tool policy breach: unexpected tool name")
            names.append(name)
        # Compaction requests carry no tools. Agent requests must expose the
        # complete configured grader, never a silently degraded partial roster.
        if names and (len(names) != len(set(names)) or set(names) != set(allowed_tools)):
            raise ValueError("MCP-only tool policy breach: incomplete tool roster")
        return document
    names = {(t.get("function") or {}).get("name") for t in document.get("tools") or []}
    if names & {"web_search", "web_fetch"}:
        raise ValueError("Disabled native web tools unexpectedly registered")
    return document


class GatewayHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def fail(self, status, message):
        data = json.dumps({"error": {"message": message, "type": "adapter_error"}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        server = self.server
        if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + server.token):
            return self.fail(401, "Invalid local credential")
        if self.path != server.upstream.path + "/chat/completions":
            return self.fail(403, "Only chat completions are enabled")
        try:
            doc = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            try:
                doc = filter_request(doc, server.allowed_tools)
            except ValueError as exc:
                server.error = str(exc)
                return self.fail(403, server.error)
            # Count model turns (requests carrying the agent's tools), excluding
            # auxiliary compaction requests. Failed retries still consume a trial.
            if doc.get("tools"):
                with server.lock:
                    if server.max_turns and server.turns >= server.max_turns:
                        server.limit_hit.set()
                        return self.fail(429, "max_turns reached")
                    server.turns += 1
                    server.tools = [t.get("function", {}).get("name") for t in doc["tools"]]
            body = json.dumps(doc).encode()
            start = time.monotonic()
            response = server.upstream.request("POST", self.path, self.headers, body)
            try:
                call = {"path": self.path, "status": response.status,
                        "model": doc.get("model"), "seconds_to_headers": round(time.monotonic()-start, 3)}
                server.calls.append(call)
                capture_usage(response, call)
                response.relay(self)
            finally:
                response.close()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            server.error = f"gateway failed: {type(exc).__name__}: {exc}"
            self.fail(502, server.error)


class Gateway(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, upstream, max_turns=None, allowed_tools=None):
        self.upstream, self.max_turns = upstream, max_turns
        self.allowed_tools = allowed_tools
        self.token = secrets.token_hex(24)
        self.turns, self.calls, self.tools = 0, [], []
        self.error = None
        self.lock = threading.Lock()
        self.limit_hit = threading.Event()
        super().__init__(("127.0.0.1", 0), GatewayHandler)
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_port}{self.upstream.path}"

    def close(self):
        self.shutdown()
        self.server_close()
        self.thread.join()


def usage_totals(entries):
    if not any(entry.get("usage") for entry in entries):
        return None
    out = {k: 0 for k in ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens")}
    for entry in entries:
        usage = entry.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens"):
            out[key] += int(usage.get(key) or 0)
        out["prompt_cache_hit_tokens"] += int(usage.get("prompt_cache_hit_tokens") or
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    return out


# Plugin IDs from the pinned dsh-v0.1.5-rc.1 full sdk composition. Keep the
# native agent, instructions, retries, compaction and persistence; remove the
# model-facing action plugins, including alternate delegation/code surfaces.
MCP_ONLY_DISABLED = (
    "tool-bash", "tool-pwsh", "tool-jobs", "tool-fs", "tool-fs-search",
    "tool-skill", "tool-subagent-control", "tool-subagent-list-agents",
    "tool-subagent", "tool-subagent-fork", "tool-workflow", "tool-todo",
    "tool-goal", "tool-ralph", "plan-mode",
)


def profile_patch(grader=None, *, mcp_only=False):
    patch = [{"id": name, "disabled": True} for name in
             ("tool-web", "web-search-deepseek", "web-fetch-http", "session-telemetry-otel")]
    if mcp_only:
        patch.extend({"id": name, "disabled": True} for name in MCP_ONLY_DISABLED)
    patch.append({"id": "sdk-jsonrpc-server", "config": {"maxTokensAsSuccess": False}})
    if grader:
        patch.append({"insert": [{"id": "task-mcp", "name": "@deepseek-ai/dsh-mcp-client",
            "config": {"transport": "streamable-http", "serverName": grader.grader_server_name,
                "url": grader.url, "headers": {}, "toolCallTimeoutMs": 900000 if mcp_only else 120000,
                "failOnStartupError": True}}]})
    return patch


def observer_messages(event):
    """Project native events into the live task observer's existing message contract."""
    kind, data = event.get("type"), event.get("data") or {}
    message = data.get("message") or data
    blocks = []
    for b in message.get("content") or []:
        typ = b.get("type")
        if kind == "assistant/message" and typ in ("text", "reasoning"):
            blocks.append({"_type": "TextBlock" if typ == "text" else "ThinkingBlock",
                           "text" if typ == "text" else "thinking": b.get("text", "")})
        elif kind == "assistant/message" and typ == "tool-call":
            args = b.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"raw_arguments": args}
            blocks.append({"_type": "ToolUseBlock", "id": b.get("id"), "name": b.get("name"), "input": args})
        elif kind == "tool/result" and typ == "tool-result":
            blocks.append({"_type": "ToolResultBlock", "tool_use_id": b.get("toolCallId"),
                           "content": b.get("content"), "is_error": b.get("isError", False)})
    if blocks:
        yield {"_type": "UserMessage" if kind == "tool/result" else "AssistantMessage", "content": blocks}
    if kind == "tool/ptc-dispatch":
        yield {"_type": "AssistantMessage", "content": [{"_type": "ToolUseBlock",
            "id": data.get("subCallId"), "name": data.get("name"), "input": data.get("arguments") or {}}]}
        yield {"_type": "UserMessage", "content": [{"_type": "ToolResultBlock",
            "tool_use_id": data.get("subCallId"), "content": data.get("content") or [], "is_error": data.get("isError", False)}]}


def run(task, support, forwarder, grader=None, observer=None):
    from deepseek_harness import DeepSeekHarness
    cfg = task["deepseek_harness"]
    mcp_only = bool(cfg.get("mcp_only"))
    if mcp_only and (grader is None or task.get("tools")):
        raise ValueError("DeepSeek MCP-only requires a grader and no native tools")
    allowed_tools = ({f"mcp__{grader.grader_server_name}__{name}" for name in grader.grader_tools}
                     if mcp_only else None)
    os.environ.update({str(k): str(v) for k, v in task.get("env", {}).items()})
    api_key = os.environ.get(cfg["api_key_env"], "")
    if not api_key:
        raise ValueError(f"{cfg['api_key_env']} is not set")
    upstream = forwarder["Upstream"](cfg["base_url"], api_key, cfg.get("extra_body"),
                                    timeout=900 if mcp_only else 120)
    gateway = Gateway(upstream, cfg.get("max_turns"), allowed_tools)
    home = Path(tempfile.mkdtemp(prefix="dsh-"))
    patch = home / "policy.patch.yml"
    patch.write_text(json.dumps(profile_patch(grader, mcp_only=mcp_only)))  # JSON is valid YAML.
    env = support["task_install_env"]()
    env.update({"DSH_PERMISSION_MODE": "danger-full-access", "DSH_TELEMETRY_DISABLED": "1",
                "DSH_MAX_TOKENS_AS_SUCCESS": "false"})
    messages, notifications = [], []
    session_id = "episode"
    def collect(notification):
        payload = notification.payload
        # Keep durable messages and tool events, not duplicate token streams.
        if notification.method == "session.event":
            event = payload.get("event") or {}
            kind = event.get("type")
            if kind in {"assistant/message", "user/message", "tool/result", "tool/ptc-dispatch", "turn/end"}:
                event = {**event, "session_id": payload.get("sessionId")}
                event["data"] = {k: v for k, v in (event.get("data") or {}).items() if k != "stream"}
                messages.append(event)
                support["_emit_msg"](event)
                if observer:
                    for record in observer_messages(event):
                        observer.observe(record)
        elif notification.method == "session.status":
            notifications.append({"method": notification.method, "payload": payload})
    started, error, final_text, terminal = time.monotonic(), None, "", "failed"
    harness = DeepSeekHarness(provider="deepseek-official", model=task["model"],
        reasoning_effort=cfg["generation"]["reasoning_effort"],
        max_tokens=cfg["generation"].get("max_tokens"),
        cwd=WORKDIR, runtime_cwd=WORKDIR, dsh_home=str(home), profile="sdk",
        patches=(str(patch),), env=env, base_url=gateway.url, api_key=gateway.token,
        initialize_timeout_seconds=min(30, task["timeout_s"]),
        request_timeout_seconds=task["timeout_s"], shutdown_timeout_seconds=2)
    # Bound wall time even if tool execution or SDK activity continues forever.
    stop = threading.Event()
    reason = []
    def watchdog():
        while not stop.wait(0.05):
            if gateway.limit_hit.is_set() or time.monotonic() - started >= task["timeout_s"]:
                reason.append("max_turns" if gateway.limit_hit.is_set() else "timeout")
                harness.close()
                return
    watcher = threading.Thread(target=watchdog, daemon=True)
    watcher.start()
    try:
        with harness:
            result = harness.run(task["content"], session_id=session_id, on_notification=collect)
            final_text, terminal = result.final_response, result.finish_reason
            if terminal != "completed":
                error = f"DeepSeek Harness ended with {terminal or 'missing finish reason'}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        stop.set()
        watcher.join(timeout=5)
        harness.close()
        gateway.close()
    if reason:
        terminal = reason[0]
        error = (f"timeout after {task['timeout_s']}s" if terminal == "timeout"
                 else "max_turns reached before confirmed task completion")
    if gateway.error:
        error = error or gateway.error
    if mcp_only and set(gateway.tools) != allowed_tools:
        error = error or "DeepSeek MCP-only did not advertise the complete grader tool roster"
    return {"id": task["id"], "model": task["model"], "messages": messages,
        "session_id": session_id, "notifications": notifications, "runtime_profile": "sdk",
        "final_text": final_text, "error": error, "terminal_reason": terminal,
        "n_turns": sum(m["type"] == "assistant/message" for m in messages),
        "n_tool_calls": sum(sum(b.get("type") == "tool-call" for b in
            (m.get("data", {}).get("message", {}).get("content") or [])) for m in messages)
            + sum(m["type"] == "tool/ptc-dispatch" for m in messages),
        "wall_time": round(time.monotonic() - started, 3),
        "usage": usage_totals(gateway.calls),
        "usage_complete": bool(gateway.calls) and all(c.get("usage") for c in gateway.calls if c["status"] == 200),
        "forwarder_calls": gateway.calls, "init_tools": gateway.tools,
        "init_mcp_servers": [grader.report()] if grader else [],
        "tool_policy": {"mcp_only": mcp_only, "disabled_plugins": [
            p["id"] for p in profile_patch(mcp_only=mcp_only) if p.get("disabled")]},
        "grader_state": grader.grader_state if grader else None,
        "deliverables": support["collect_deliverables"](WORKDIR, tuple(task.get("skip_dirs") or []),
                                                        tuple(task.get("deliverable_files") or []))}


def main(task, modules):
    output = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    namespaces = {}
    for name in ("support", "stage", "forwarder", "grader"):
        if modules.get(name):
            ns = {"__name__": f"runner_{name}"}
            exec(compile(modules[name], f"<{name}>", "exec"), ns)
            namespaces[name] = ns
    support = namespaces["support"]
    grader = observer = install = task_install = None
    record = {"id": task["id"], "model": task["model"], "messages": []}
    try:
        os.chdir(WORKDIR)
        staged = namespaces["stage"]["write"](task.get("files") or [], WORKDIR)
        install = support["install_agent"](task.get("install"))
        if install and install.get("status") == "failed":
            raise RuntimeError("DeepSeek Harness installation failed: " + install.get("error", ""))
        task_install = support["install_agent"](task.get("task_install"), env=support["task_install_env"]())
        if task_install and task_install.get("status") == "failed":
            raise RuntimeError("Task installation failed: " + task_install.get("error", ""))
        module = namespaces.get("grader") or {}
        if module:
            grader = support["serve_grader"](module, task.get("row") or {}, staged, WORKDIR)
        if module.get("make_observer"):
            observer = module["make_observer"](task.get("row") or {}, staged, WORKDIR,
                                                grader.grader_state if grader else {})
        record = run(task, support, namespaces["forwarder"], grader, observer)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if observer:
            try:
                observer.finish()
            except Exception as exc:
                record["error"] = f"{record.get('error') or ''} observer finish failed: {exc}".strip()
        if grader:
            record["grader_state"] = grader.grader_state
            grader.close()
    record.update(install=install, task_install=task_install)
    output.write(json.dumps(record) + "\n")
    output.flush()


if globals().get("TASK") is not None:
    main(TASK, MODULES)
