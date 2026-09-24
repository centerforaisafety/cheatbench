"""Offline native Codex tool-surface probe; run inside its prebuilt image.

No model credentials or API calls. A loopback Responses stub captures the
actual tool registry sent by the installed CLI, then returns a final message.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_error(405)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.path == "/mcp":
            self.server.mcp_requests.append(payload)
            if payload.get("id") is None:
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            method = payload.get("method")
            result = ({"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                       "serverInfo": {"name": "chess", "version": "1"}} if method == "initialize" else
                      {"tools": [{"name": "read_file", "description": "Read an episode file.",
                                  "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}},
                                                  "required": ["path"], "additionalProperties": False}}]} if method == "tools/list" else
                      {"content": [{"type": "text", "text": "READ-MCP-OK"}], "isError": False} if method == "tools/call" else {})
            data = json.dumps({"jsonrpc": "2.0", "id": payload["id"], "result": result}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.server.requests.append(payload)
        item = {"id": "msg_probe", "type": "message", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "Probe complete.", "annotations": []}]}
        if len(self.server.requests) == 1:
            item = {"id": "fc_probe", "type": "function_call", "call_id": "call_read", "name": "read_file", "namespace": "mcp__chess",
                    "arguments": '{"path":"opponent/config.json"}', "status": "completed"}
        elif len(self.server.requests) == 2:
            item = {"id": "fc_script", "type": "custom_tool_call", "call_id": "call_script", "name": "exec",
                    "namespace": "functions", "input": "text(137+29)", "status": "completed"}
        response = {"id": "resp_probe", "object": "response", "created_at": 1,
                    "status": "completed", "output": [item], "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
        events = [
            {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
            {"type": "response.output_item.added", "output_index": 0, "item": item},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": response},
        ]
        data = "".join("event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n" for e in events).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.requests = []
    server.mcp_requests = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix="restricted-cli-probe-") as directory:
            root = Path(directory)
            (root / "config.toml").write_text(
                f'openai_base_url = "http://127.0.0.1:{server.server_port}/v1"\n'
                '[features.code_mode]\nenabled = false\ndirect_only_tool_namespaces = ["mcp__chess"]\n'
                f'[mcp_servers.chess]\nurl = "http://127.0.0.1:{server.server_port}/mcp"\n'
                'required = true\nenabled_tools = ["read_file"]\ndefault_tools_approval_mode = "approve"\n')
            disabled = ["shell_tool", "unified_exec", "shell_snapshot", "multi_agent",
                        "code_mode_host", "plugins", "remote_plugin", "apps", "browser_use", "computer_use",
                        "image_generation", "view_image", "workspace_dependencies", "goals", "tool_suggest",
                        "enable_request_compression"]
            command = ["codex", "exec", "--sandbox", "read-only", "--skip-git-repo-check", "--strict-config",
                       "--model", "gpt-6-astra", "--json", "-c", 'approval_policy="never"',
                       "-c", "agents.enabled=false",
                       "-c", 'web_search="disabled"', "-c", "analytics.enabled=false",
                       "-c", "check_for_update_on_startup=false"]
            for feature in disabled:
                command += ["--disable", feature]
            command += ["--", "Respond with Probe complete."]
            result = subprocess.run(command, env={**os.environ, "CODEX_HOME": directory, "OPENAI_API_KEY": "offline-probe-key"},
                                    capture_output=True, text=True, timeout=45)
            print(json.dumps({"returncode": result.returncode,
                              "requests": [{"tools": p.get("tools", []), "model": p.get("model"), "keys": list(p),
                                            "additional_tools": [{"name": ns.get("name"), "tools": [t.get("name") for t in ns.get("tools", [])]}
                                                                 for item in p.get("input", []) if item.get("type") == "additional_tools" for ns in item.get("tools", [])],
                                            "results": [i for i in p.get("input", []) if i.get("type") in ("function_call_output", "custom_tool_call_output")],
                                            "other_tools": {k: v for k, v in p.items() if "tool" in k}} for p in server.requests],
                              "mcp_requests": server.mcp_requests,
                              "stderr": result.stderr[-4000:], "stdout": result.stdout[-3000:]}))
            assert result.returncode == 0
            assert any(p.get("method") == "tools/call" for p in server.mcp_requests)
            results = [i for p in server.requests for i in p.get("input", [])
                       if i.get("type") in ("function_call_output", "custom_tool_call_output")]
            assert "READ-MCP-OK" in json.dumps(results), results
            assert "code-mode host is disabled" in json.dumps(results), results
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
