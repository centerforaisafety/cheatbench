"""Installation, artifact collection and grader HTTP transport for streamed runners.

Shared contracts taken from the existing Muse runner; no vendor CLI behavior.
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
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def _log(msg: str) -> None:
    print(f'[runner] {msg}', file=sys.stderr, flush=True)
MSG_PREFIX = '\x1eRHMSG '

def _emit_msg(rec: dict) -> None:
    try:
        sys.stderr.write(MSG_PREFIX + json.dumps(rec) + '\n')
        sys.stderr.flush()
    except Exception:
        pass
DELIVERABLE_EXTS = ('.xlsx', '.xls', '.xlsm', '.docx', '.doc', '.pptx', '.ppt', '.pdf', '.csv', '.svg')
_DELIVERABLE_MAX_BYTES = 64 * 1024 * 1024
INSTALL_TIMEOUT_S = 900

def collect_deliverables(root: str, skip_dirs: tuple, include_files: tuple=()) -> list:
    """Export office files and exact task-requested outputs for the host."""
    import base64
    import stat
    requested = {p for p in include_files if isinstance(p, str) and p and (not os.path.isabs(p)) and (not any((part in ('', '.', '..') for part in p.split('/'))))}
    (out, total) = ([], 0)
    for (dirpath, dirnames, filenames) in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        top = rel_dir.split(os.sep)[0]
        if top in skip_dirs:
            dirnames[:] = []
            continue
        for name in sorted(filenames):
            relative = os.path.relpath(os.path.join(dirpath, name), root)
            if os.path.splitext(name)[1].lower() not in DELIVERABLE_EXTS and relative not in requested:
                continue
            q = os.path.join(dirpath, name)
            try:
                info = os.lstat(q)
                if not stat.S_ISREG(info.st_mode):
                    continue
                remaining = _DELIVERABLE_MAX_BYTES - total
                if info.st_size > remaining:
                    continue
                with open(q, 'rb') as f:
                    data = f.read(remaining + 1)
            except OSError:
                continue
            if total + len(data) > _DELIVERABLE_MAX_BYTES:
                _log(f'deliverable export cap hit; skipping {q}')
                continue
            total += len(data)
            out.append({'name': os.path.relpath(q, root), 'b64': base64.b64encode(data).decode()})
    return out

def _sh(cmd: str, timeout: float, env: dict | None=None):
    return subprocess.run(['bash', '-c', cmd], capture_output=True, text=True, timeout=timeout, env=env)
_SECRET_ENV_SUBSTRINGS = ('API_KEY', 'APIKEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL', 'AUTH', 'BASE_URL')
_SECRET_ENV_PREFIXES = ('ANTHROPIC_', 'OPENAI_', 'CLAUDE_', 'CODEX_', 'GEMINI_', 'GOOGLE_', 'VERTEX_', 'XAI_', 'GROK_', 'KIMI_', 'MOONSHOT_', 'MUSE_', 'LITELLM_', 'AZURE_', 'AWS_', 'HF_', 'HUGGINGFACE_', 'HUGGING_FACE_')

def task_install_env() -> dict:
    """`os.environ` with the model credential removed, for a task's own shell."""
    out = {}
    for (k, v) in os.environ.items():
        up = k.upper()
        if any((s in up for s in _SECRET_ENV_SUBSTRINGS)):
            continue
        if up.startswith(_SECRET_ENV_PREFIXES):
            continue
        out[k] = v
    return out

def install_agent(spec: dict | None, env: dict | None=None) -> dict | None:
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
    out = {'name': spec.get('name'), 'status': 'present', 'pinned_version': spec.get('pinned_version') or '', 'version': ''}

    def _version() -> str:
        cmd = spec.get('version_cmd')
        if not cmd:
            return ''
        try:
            res = _sh(cmd, 120, env)
            return (res.stdout or '').strip().split('\n')[0] if res.returncode == 0 else ''
        except Exception:
            return ''
    try:
        if _sh(spec.get('check') or 'true', 60, env).returncode == 0:
            out['version'] = _version()
            out['seconds'] = round(time.time() - t0, 1)
            _log(f"{out['name']}: runtime already present ({out['seconds']}s to check), version {out['version'] or 'unknown'}")
            return out
        res = _sh(spec['install'], INSTALL_TIMEOUT_S, env)
        out['status'] = 'installed' if res.returncode == 0 else 'failed'
        if res.returncode != 0:
            out['error'] = (res.stderr or res.stdout or '').strip()
        else:
            out['version'] = _version()
    except Exception as e:
        out['status'] = 'failed'
        out['error'] = f'{type(e).__name__}: {e}'
    out['seconds'] = round(time.time() - t0, 1)
    _log(f"{out['name']}: runtime {out['status']} in {out['seconds']}s, version {out['version'] or 'unknown'}")
    if out.get('error'):
        _log(f"{out['name']}: install said: {out['error']}")
    return out
MCP_PROTOCOL_VERSION = '2025-06-18'

class _GraderMCPHandler(BaseHTTPRequestHandler):
    """Streamable-HTTP MCP, the four methods a client actually calls."""
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args) -> None:
        """Silence. The default writes to stderr, which the host parses."""

    def _authorised(self) -> bool:
        path = self.path.split('?', 1)[0]
        return hmac.compare_digest(path, self.server.grader_path)

    def _empty(self, code: int) -> None:
        self.send_response(code)
        self.send_header('Content-Length', '0')
        self.end_headers()

    def _send(self, body: bytes) -> None:
        if 'text/event-stream' in (self.headers.get('Accept') or ''):
            payload = b'event: message\ndata: ' + body + b'\n\n'
            ctype = 'text/event-stream'
        else:
            (payload, ctype) = (body, 'application/json')
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        self._empty(404 if not self._authorised() else 405)

    def do_DELETE(self) -> None:
        self._empty(404 if not self._authorised() else 405)

    def do_POST(self) -> None:
        if not self._authorised():
            self._empty(404)
            return
        try:
            length = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            self._empty(400)
            return
        raw = self.rfile.read(length) if length else b''
        try:
            request = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            self._empty(400)
            return
        if not isinstance(request, dict):
            self._empty(400)
            return
        if request.get('id') is None:
            self._empty(202)
            return
        self._send(json.dumps(self._handle(request)).encode())

    def _handle(self, request: dict) -> dict:
        (rid, method) = (request.get('id'), request.get('method'))
        params = request.get('params')
        params = params if isinstance(params, dict) else {}
        server = self.server

        def ok(result):
            return {'jsonrpc': '2.0', 'id': rid, 'result': result}
        if method == 'initialize':
            server.connected = True
            return ok({'protocolVersion': params.get('protocolVersion') or MCP_PROTOCOL_VERSION, 'capabilities': {'tools': {'listChanged': False}}, 'serverInfo': {'name': server.grader_server_name, 'version': '1.0.0'}})
        if method == 'ping':
            return ok({})
        if method == 'tools/list':
            server.listed = True
            return ok({'tools': [{k: v for k, v in spec.items() if k != 'call'}
                                 for spec in server.grader_tools.values()]})
        if method == 'tools/call':
            name = params.get('name')
            spec = server.grader_tools.get(name) if isinstance(name, str) else None
            if spec is None:
                return ok({'content': [{'type': 'text', 'text': f'no such tool: {name}'}], 'isError': True})
            args = params.get('arguments')
            try:
                return ok(spec['call'](args if isinstance(args, dict) else {}))
            except Exception as e:
                _log(f'grader raised: {type(e).__name__}: {e}')
                server.grader_errors.append(f'{type(e).__name__}: {e}')
                return ok({'content': [{'type': 'text', 'text': 'the grader is unavailable'}], 'isError': True})
        return {'jsonrpc': '2.0', 'id': rid, 'error': {'code': -32601, 'message': f'unknown method {method}'}}

class GraderServer(ThreadingHTTPServer):
    """The running grader endpoint, and the handle the episode holds it by."""
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, *, state: dict, call, server_name: str, tool_name: str,
                 description: str, schema: dict, extra_tools=()):
        tools = [{'name': tool_name, 'description': description,
                  'inputSchema': schema, 'call': call}, *extra_tools]
        names = [spec['name'] for spec in tools]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError('Grader tool names must be nonempty strings')
        if len(names) != len(set(names)):
            raise ValueError('Duplicate grader tool names')
        if any(not callable(spec['call']) for spec in tools):
            raise ValueError('Each grader tool needs a callable handler')
        super().__init__(('127.0.0.1', 0), _GraderMCPHandler)
        self.grader_state = state
        self.grader_call = call
        self.grader_server_name = server_name
        self.grader_tool_name = tool_name
        self.grader_description = description
        self.grader_schema = schema
        self.grader_tools = {spec['name']: spec for spec in tools}
        self.grader_errors: list = []
        self.grader_path = '/' + secrets.token_hex(16) + '/mcp'
        self.connected = False
        self.listed = False
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        (host, port) = (self.server_address[0], self.server_address[1])
        return f'http://{host}:{port}{self.grader_path}'

    def start(self) -> None:
        self._thread = threading.Thread(target=self.serve_forever, kwargs={'poll_interval': 0.2}, daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop serving. Idempotent; see the Codex runner for why it must be."""
        if self._thread is not None:
            try:
                self.shutdown()
            except Exception:
                pass
            self._thread.join(timeout=10)
            self._thread = None
        try:
            self.server_close()
        except Exception:
            pass

    def report(self) -> dict:
        entry = {'name': self.grader_server_name, 'transport': 'streamable-http', 'status': 'connected' if self.connected else 'configured', 'tools_listed': self.listed}
        if self.grader_errors:
            entry['errors'] = list(self.grader_errors)
        return entry

def serve_grader(module: dict, row: dict, staged: dict, workdir: str):
    """Start the task's grader as an MCP endpoint. Returns the running server."""
    (state, call) = module['make_tool'](row, staged, workdir)
    extra_factory = module.get('make_extra_tools')
    extra_tools = extra_factory(row, staged, workdir, state) if extra_factory else ()
    primary = module['tool_spec'](row) if 'tool_spec' in module else {}
    server = GraderServer(state=state, call=call,
        server_name=module.get('SERVER_NAME', 'grader'),
        tool_name=primary.get('name') or module.get('TOOL_BASENAME', 'grade_deliverable'),
        description=primary.get('description') or module.get('DESCRIPTION', ''),
        schema=primary.get('inputSchema') or module.get('INPUT_SCHEMA') or {'type': 'object', 'properties': {}},
        extra_tools=extra_tools)
    server.start()
    return server
