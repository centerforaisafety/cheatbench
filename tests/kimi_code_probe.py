"""Exercise the installed Kimi CLI against a deterministic local model and MCP grader."""

import json
import os
import sys
import threading
import tempfile
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.agents import make_agent  # noqa: E402
from core.agents import kimi_code_runner as runner  # noqa: E402

requests = []


class Provider(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        requests.append(body)
        assert self.headers["Authorization"] == "Bearer probe-key"
        n = len(requests)
        tools = [t["function"]["name"] for t in body.get("tools", [])]
        assert "FetchURL" not in tools and "WebSearch" not in tools
        if n == 1:
            delta = {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Compute six times seven.",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call1",
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "arguments": json.dumps(
                                {"command": "printf 42 > submission.txt"}
                            ),
                        },
                    }
                ],
            }
        elif n == 2:
            name = next(t for t in tools if "verify" in t)
            delta = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call2",
                        "type": "function",
                        "function": {"name": name, "arguments": '{"answer":42}'},
                    }
                ],
            }
        else:
            delta = {"role": "assistant", "content": "Verified 42."}
        chunks = [
            {
                "id": f"kimi-probe-{n}",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "kimi-k3",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {
                "id": f"kimi-probe-{n}",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "kimi-k3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if n < 3 else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            },
        ]
        data = (
            "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
            + "data: [DONE]\n\n"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
threading.Thread(target=server.serve_forever, daemon=True).start()
os.environ["OPENAI_API_KEY"] = "probe-key"
with tempfile.TemporaryDirectory(prefix="kimi-probe-") as temp:
    runner.WORKDIR = temp
    a = make_agent(
        "kimi-code",
        model="kimi-k3",
        version="0.43.0",
        api_base_url=f"http://127.0.0.1:{server.server_port}/v1",
        generation_config={"thinking_effort": "high"},
        max_turns=1 if "--cap" in sys.argv else 5,
    )
    a.apply_tool_policy({"web_search": False, "web_fetch": False})
    p = json.loads(
        a.blob(
            row={"id": "probe", "deliverable_files": ["submission.txt"]},
            prompt="Compute 6*7, write submission.txt, call verify, and finish.",
            files=[],
            skip_dirs=(),
            stage_src="",
            timeout_s=60,
            env={},
        )
    )
    task = p["task"]
    task["kimi"]["kimi_home"] = temp + "/kimi-home"
    task["kimi"]["probe"] = False
    fw = {"__name__": "probe_forwarder"}
    exec(p["modules"]["forwarder"], fw)
    grader_ns = {}
    exec((ROOT / "tests/terminus_2_probe_grader.py").read_text(), grader_ns)
    grader = runner.serve_grader(grader_ns, {}, {}, temp)
    try:
        raw = runner.run(task, None, grader, fw)
    finally:
        grader.close()
        server.shutdown()
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "record.json").write_text(json.dumps(raw, indent=2))
    (out / "requests.json").write_text(json.dumps(requests, indent=2))
    if "--cap" in sys.argv:
        assert raw["error"] and "max" in raw["error"].lower(), raw["error"]
        assert raw["n_turns"] == 1, raw["n_turns"]
        assert any(f["name"] == "submission.txt" for f in raw["deliverables"])
        sys.exit(0)
    assert raw["error"] is None, raw["error"]
    assert raw["grader_state"]["answer"] == 42, raw["grader_state"]
    assert any(f["name"] == "submission.txt" for f in raw["deliverables"])
    trajectory = a.to_trajectory(raw).to_json_dict()
    (out / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
    assert "Compute six times seven." in json.dumps(trajectory)
    assert raw["usage"]["prompt_tokens"] == 300, raw["usage"]
    assert requests[0]["thinking"]["effort"] == "high"
    assert raw["n_turns"] == 3 and raw["n_tool_calls"] == 2
    assert set(raw["init_tools"]) == set(a.BASE_TOOLS) | {"mcp__grader__verify"}
    assert "probe-key" not in json.dumps(raw)
    print(
        json.dumps(
            {
                "ok": True,
                "calls": len(requests),
                "tools": raw["init_tools"],
                "effort": raw["thinking_effort_bound"],
                "usage": raw["usage"],
            }
        ),
        file=sys.stderr,
    )
