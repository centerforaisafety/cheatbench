"""Episode-local RPC bridge. Only game/tool dispatch is reused, never the model loop.

All nine capabilities retain Stephen's virtual checkout and game semantics. KataGo,
its weights, the fixtures and edited-policy workers stay outside the agent container.
"""
from pathlib import Path
import argparse
import asyncio
import hmac
import json
import os
import ssl
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tasks.go.episode import Episode
from tasks.go.katago_api import KataGoEngine
from tasks.go.engine_signature import verify


def enter_namespaces(pid):
    import ctypes
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    for namespace in ('user', 'net'):
        fd = os.open(f'/proc/{pid}/ns/{namespace}', os.O_RDONLY)
        try:
            if libc.setns(fd, 0) != 0:
                errno = ctypes.get_errno()
                raise OSError(errno, f'setns({namespace}): {os.strerror(errno)}')
        finally:
            os.close(fd)


class Service:
    def __init__(self, row, output):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.runner = None
        self.engine = None
        self.sequence = 0
        self.fatal_error = None
        self.lock = threading.Lock()
        self.row = row
        self.output = output
        try:
            self.wait(self.start())
        except BaseException:
            if self.engine is not None:
                self.wait(self.engine.close())
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)
            raise

    def wait(self, coroutine):
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout=900)
        except TimeoutError:
            future.cancel()
            raise

    async def start(self):
        self.engine = KataGoEngine(max_visits=400)
        await self.engine.start()
        signature = await verify(self.engine)
        self.runner = Episode(ROOT / "tasks/go/fixtures" / self.row["fixture"], self.output, self.engine)
        self.runner.engine_signature = signature
        (self.output / "engine_signature.json").write_text(json.dumps(signature, indent=2))
        (self.output / "prompt.md").write_text(self.runner.build_prompt() + "\n")

    async def dispatch(self, name, arguments, timeout_s):
        runner = self.runner
        if self.sequence == 0:
            runner.started = time.monotonic()
            runner.deadline = runner.started + min(3600, max(1, float(timeout_s)))
        self.sequence += 1
        call_id = str(self.sequence)
        runner.active_tool_call_id = call_id
        runner.event("shared_tool_requested", name=name, arguments=arguments, call_id=call_id)
        try:
            result = await runner.dispatch(name, arguments)
            runner.message({"role": "tool", "tool_call_id": call_id, "content": result})
            return {"content": [{"type": "text", "text": result}], "isError": False,
                    "game_state": runner.game.result()}
        finally:
            # Persist state even if a tool or episode is interrupted.
            (self.output / "game.json").write_text(json.dumps(runner.game.result(), indent=2))
            (self.output / "accounting.json").write_text(json.dumps({
                "api_requests": len(runner.api_requests),
                "endpoint_requests": len(runner.engine_api_requests),
                "health_requests": sum(r["service"] == "health" for r in runner.api_requests),
                "tool_calls": self.sequence}, indent=2))

    def call(self, payload):
        name, arguments = payload.get("name"), payload.get("arguments")
        names = {s["function"]["name"] for s in self.runner.build_tools()}
        if name not in names or not isinstance(arguments, dict):
            raise ValueError("Invalid tool call")
        with self.lock:
            if self.fatal_error:
                raise RuntimeError(self.fatal_error)
            return self.wait(self.dispatch(name, arguments, payload.get("timeout_s", 3600)))

    def close(self):
        self.wait(self.engine.close())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


def handler(service, token):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/harness/call" or not hmac.compare_digest(
                    self.headers.get("X-Harness-Token", ""), token):
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1048576:
                    raise ValueError("Invalid request size")
                answer = service.call(json.loads(self.rfile.read(length)))
            except (ValueError, TypeError) as exc:
                answer = {"content": [{"type": "text", "text": "Invalid Go tool request."}], "isError": True}
            except Exception as exc:
                # The authenticated bridge converts this into the shared fatal channel.
                service.fatal_error = f"{type(exc).__name__}: {exc}"
                print(f"Go infrastructure error: {service.fatal_error}", file=sys.stderr, flush=True)
                (service.output / "infrastructure_error.json").write_text(json.dumps({"error": service.fatal_error}))
                answer = {"infrastructure_error": "Go game/engine service failed; episode invalid."}
            data = json.dumps(answer).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return Handler


def main():
    p = argparse.ArgumentParser()
    for name in ("mirror", "cert", "key", "overrides", "only", "log", "ready-file"):
        p.add_argument("--" + name)
    p.add_argument("--port", type=int, default=443)
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--netns-pid", type=int, default=0)
    args = p.parse_args()
    row = json.loads(Path(args.overrides).read_text())[args.only]
    if args.netns_pid:
        enter_namespaces(args.netns_pid)
    output = Path(args.log).resolve().parent / ("go_" + args.only)
    service = Service(row, output)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(args.cert, args.key)
    httpd = ThreadingHTTPServer((args.bind, args.port), handler(service, row["token"]))
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    Path(args.ready_file).write_text(str(httpd.server_address[1]))
    def watch():
        while True:
            if args.netns_pid and not os.path.exists(f"/proc/{args.netns_pid}"):
                httpd.shutdown()
                return
            process = service.engine._proc
            if process is not None and process.returncode is not None:
                service.fatal_error = f"KataGo exited {process.returncode}"
                (service.output / "infrastructure_error.json").write_text(
                    json.dumps({"error": service.fatal_error}))
            if service.fatal_error:
                httpd.shutdown()
                return
            time.sleep(1)
    threading.Thread(target=watch, daemon=True).start()
    def stop(signum, frame):
        threading.Thread(target=httpd.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        service.close()


if __name__ == "__main__":
    main()
