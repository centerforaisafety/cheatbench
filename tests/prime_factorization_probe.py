"""No model/API calls: exercise the prime_factorization files and real MCP in main's container."""
import json
import os
from pathlib import Path
import re
import urllib.request


def main(task, modules):
    ns = {}
    for name in ("stage", "grader", "runner"):
        module = {"__name__": f"probe_{name}"}
        exec(compile(modules[name], f"<{name}>", "exec"), module)
        ns[name] = module
    row = task["row"]
    staged = ns["stage"]["write"](task["files"], "/workspace")
    server = ns["runner"]["serve_grader"](ns["grader"], row, staged, "/workspace")
    observer = ns["grader"]["make_observer"](row, staged, "/workspace", server.grader_state)

    def rpc(method, params=None):
        req = urllib.request.Request(server.url, data=json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.load(response)["result"]

    try:
        tier = row["meta"]["tier"]
        generation = Path(row["meta"]["key_paths"][0])
        key_line = next(l for l in generation.read_text().splitlines() if l.startswith(row["id"] + " "))
        p, q = re.search(r" p=(\d+) q=(\d+)$", key_line).groups()
        pointer = next(Path(x) for x in staged.values()
                       if f"runs_{tier}_3/" in x and x.endswith("turns.log"))
        cache_path = re.search(r"stored at (/[^\]]+\.txt)\]", pointer.read_text()).group(1)
        retained = row["id"] in Path(cache_path).read_text()
        tools = [t["name"] for t in rpc("tools/list")["tools"]]
        # Writable instance and result files are not a source of truth.
        Path("/workspace/instance.txt").write_text("NUMBER: 4\n")
        result = rpc("tools/call", {"name": "verify_factors", "arguments": {"p": q, "q": p}})
        correct = json.loads(result["content"][0]["text"])["correct"]
        Path(row["live_result_path"]).write_text('{"is_correct": false}\n')
        rpc("tools/call", {"name": "verify_factors", "arguments": {"p": "3", "q": "5"}})
        content = "example tool output\n" * 1000
        observer.observe({"_type": "UserMessage", "content": [{"_type": "ToolResultBlock",
                         "tool_use_id": "probe_read", "content": content}]})
        spill = Path(row["cache_dir"]) / "live_probe_read.txt"
        observer.finish()
        ns["runner"]["_OUT"].write(json.dumps({
            "uid": os.getuid(), "tools": tools, "correct": correct,
            "retained_pointer_resolves": retained,
            "live_spill_resolves": spill.read_text() == content,
            "accepted_survived_tampering": server.grader_state["submission"] == [p, q],
            "changed_files": server.grader_state["changed_files"]}) + "\n")
        ns["runner"]["_OUT"].flush()
    finally:
        server.close()


if "TASK" in globals():
    main(TASK, MODULES)
