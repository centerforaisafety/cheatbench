"""In-memory fake provider; the actual Harbor agent executes every response."""

import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Provider(BaseHTTPRequestHandler):
    calls = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.headers["Authorization"] == "Bearer probe-secret-key"
        assert not body.get(
            "tools"
        ), "T2 must use its own text parser, not native tool calling"
        Provider.calls += 1
        if Provider.calls == 1:
            prompt = json.dumps(body["messages"])
            url = re.search(r"http://127\.0\.0\.1:[0-9]+/[^\\\s\"]+", prompt).group(0)
            script = (
                "import urllib.request,json\n"
                "open('submission.txt','w').write('42')\n"
                f"url={url!r}\n"
                "for method,params in [('initialize',{}),('tools/list',{}),"
                "('tools/call',{'name':'verify','arguments':{'answer':6*7}})]:\n"
                " data=json.dumps({'jsonrpc':'2.0','id':1,'method':method,'params':params}).encode()\n"
                " print(urllib.request.urlopen(urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'})).read().decode())\n"
            )
            content = {
                "analysis": "I will calculate 6*7 and call the available verifier.",
                "plan": "Use the terminal and MCP endpoint.",
                "commands": [
                    {"keystrokes": "python - <<'PY'\n" + script + "PY\n", "duration": 1}
                ],
                "task_complete": False,
            }
        else:
            content = {
                "analysis": "The verifier returned verified 42.",
                "plan": "Finish.",
                "commands": [],
                "task_complete": True,
            }
        result = {
            "id": "probe",
            "object": "chat.completion",
            "created": 1,
            "model": "probe",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(content)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }
        data = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
threading.Thread(target=server.serve_forever, daemon=True).start()
TASK["api_base"] = f"http://127.0.0.1:{server.server_port}/v1"
os.environ[TASK["api_key_env"]] = "probe-secret-key"
exec(compile(MODULES["runner"], "<runner>", "exec"), {"TASK": TASK, "MODULES": MODULES})
