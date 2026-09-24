"""Offline Kimi 0.43.0 dispatch proof, executed from stdin inside its native image."""
import json
import os
from pathlib import Path
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

runner_ns = {"__name__": "probe_kimi_runner"}
exec(MODULES["runner"], runner_ns)
runner = SimpleNamespace(**runner_ns)
forwarder_ns = {"__name__": "probe_forwarder"}
exec(MODULES["forwarder"], forwarder_ns)
reports = []

for case in TASK["cases"]:
    with tempfile.TemporaryDirectory(prefix="kimi-nine-") as directory:
        root = Path(directory)
        for rel, content in case["files"].items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        grader_ns = {"__name__": "probe_grader"}
        exec(MODULES[case["game"]], grader_ns)
        grader = runner.serve_grader(grader_ns, {"id": "probe", "tool_surface": case["game"]}, {}, directory)
        prefix = "mcp__" + case["game"] + "__"
        expected = {prefix + name for name in runner.MCP_NAMES}
        marker = root / "forbidden-native-write"
        requests = []
        proof = "Kimi nine-tool notes proof"
        if case["attempt"] == "mcp":
            steps = [(prefix + "write_notes", {"text": proof}),
                     (prefix + "read_file", {"path": str(root / "workspace/notes.txt"), "offset": 0, "limit": 16384}),
                     (prefix + "write_file", {"path": str(root / "workspace/notes.txt"), "text": proof + " updated"}),
                     (prefix + "read_notes", {})]
        else:
            steps = [(case["attempt"], {"command": "touch " + str(marker), "file_path": str(marker),
                      "path": str(marker), "content": "forbidden", "prompt": "Write " + str(marker),
                      "description": "canary", "subagent_type": "general-purpose", "text": "forbidden"}),
                     (prefix + "read_notes", {})]

        class Provider(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                doc = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(doc)
                n = len(requests)
                if n <= len(steps):
                    name, args = steps[n - 1]
                    delta = {"role": "assistant", "reasoning_content": "Keep native reasoning.", "tool_calls": [
                        {"index": 0, "id": f"call{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
                    finish = "tool_calls"
                else:
                    delta, finish = {"role": "assistant", "content": "Done."}, "stop"
                events = [{"id": f"probe-{n}", "object": "chat.completion.chunk", "created": 1,
                           "model": "kimi-k3", "choices": [{"index": 0, "delta": part, "finish_reason": reason}]}
                          for part, reason in [(delta, None), ({}, finish)]]
                events[-1]["usage"] = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
                data = ("".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
        threading.Thread(target=provider.serve_forever, daemon=True).start()
        task = case["task"]
        task["kimi"].update(kimi_home=str(root / "kimi-home"), probe=False)
        task["routing"]["api_base_url"] = f"http://127.0.0.1:{provider.server_port}/v1"
        task["routing"]["cli_key_env"] = ["KIMI_PROBE_KEY"]
        os.environ["KIMI_PROBE_KEY"] = "mock-only"
        runner_ns["WORKDIR"] = directory
        try:
            record = runner.run(task, None, grader, forwarder_ns)
        finally:
            grader.close()
            provider.shutdown()
            provider.server_close()
        assert record["error"] is None, record["error"]
        assert record["returncode"] == 0
        assert len(record["init_tools"]) == 9 and set(record["init_tools"]) == expected, record["init_tools"]
        assert len(requests) == len(steps) + 1, len(requests)
        assert all({t["function"]["name"] for t in doc["tools"]} == expected for doc in requests)
        assert requests[0]["thinking"]["effort"] == "high"
        assert "Keep native reasoning." in json.dumps(record["messages"])
        assert not marker.exists()
        assert not record["grader_state"].get("api_requests")
        assert not record["grader_state"].get("moves")
        assert record["grader_state"]["notes_reads"] == 1
        if case["attempt"] == "mcp":
            assert (root / "workspace/notes.txt").read_text() == proof + " updated"
            assert record["grader_state"]["notes_writes"] == 1 and record["grader_state"]["file_writes"] == 1
            assert proof in json.dumps(requests[2]["messages"])
        else:
            results = [m for m in requests[1]["messages"] if m.get("role") == "tool"]
            assert results and any(word in json.dumps(results).lower() for word in ("disabled", "unknown", "not found", "not available")), results
            assert not record["grader_state"]["file_writes"]
        reports.append({"game": case["game"], "attempt": case["attempt"], "passed": True,
                        "tools": record["init_tools"], "calls": len(requests), "effort": record["thinking_effort_bound"]})
print(json.dumps({"native_policy_cases": reports}))
