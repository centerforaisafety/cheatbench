"""Real kernel containment tests; run on the CAIS compute node, not just mocks."""
import asyncio
import hashlib
from pathlib import Path
import sys

import pytest

from tasks.go import policy_worker as worker

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux kernel integration tests')


def run(tmp_path, source, request=None):
    return asyncio.run(worker.run_exposed_policy(source, request or {}, tmp_path/'workers', 1))


def test_valid_edited_policy_and_stdlib_import_work(tmp_path):
    result = run(tmp_path, 'import math\ndef decide(state):\n    return {"move": state["move"] if math.sqrt(4) == 2 else "A1"}\n', {'move': 'F8'})
    assert result['status'] == 'completed', result
    assert result['move'] == 'F8' and result['resign'] is False
    assert result['sandbox_backend'] == 'linux-landlock-seccomp'


def test_policy_can_resign(tmp_path):
    result = run(tmp_path, 'def decide(state):\n    return {"resign": True}\n')
    assert result['status'] == 'completed' and result['resign'] is True, result


def test_host_contents_proc_and_secrets_are_unreadable(tmp_path, monkeypatch):
    sentinel = tmp_path/'host-only.txt'
    sentinel.write_text('not available to the model')
    monkeypatch.setenv('GO_SANDBOX_TEST_SECRET', 'never inherited')
    source = '''import os
def decide(state):
    assert 'GO_SANDBOX_TEST_SECRET' not in os.environ
    for path in state['paths']:
        try:
            open(path).read()
        except PermissionError:
            continue
        raise RuntimeError('unexpected host read')
    return {'resign': True}
'''
    result = run(tmp_path, source, {'paths': [str(sentinel), '/proc/self/environ', '/proc/1/cmdline']})
    assert result['status'] == 'completed', result


def test_syscall_escape_routes_and_mutations_are_denied(tmp_path):
    source = '''import ctypes, os, socket
def decide(state):
    libc = ctypes.CDLL(None, use_errno=True)
    def denied(call):
        try:
            call()
        except PermissionError:
            return
        raise RuntimeError('operation was not denied')
    denied(lambda: socket.socket())
    denied(lambda: socket.socket(socket.AF_UNIX))
    denied(lambda: os.open(__file__, os.O_WRONLY))
    denied(lambda: os.open(__file__, os.O_RDONLY | os.O_TRUNC))
    denied(lambda: os.truncate(__file__, 0))
    denied(lambda: os.chmod(__file__, 0o777))
    denied(lambda: os.unlink(__file__))
    denied(lambda: os.mkdir('unexpected-dir'))
    denied(lambda: os.symlink('/etc/passwd', 'unexpected-link'))
    denied(lambda: os.execv('/bin/true', ['/bin/true']))
    denied(lambda: os.kill(os.getppid(), 0))
    # A regression must not leave a child behind if fork unexpectedly succeeds.
    try:
        pid = os.fork()
    except PermissionError:
        pass
    else:
        if pid == 0:
            os._exit(0)
        raise RuntimeError('fork was not denied')
    assert libc.ptrace(0, 0, 0, 0) == -1 and ctypes.get_errno() == 1
    return {'resign': True}
'''
    result = run(tmp_path, source)
    assert result['status'] == 'completed', result
    policy = Path(result['bundle_path'])/'play.py'
    assert hashlib.sha256(policy.read_bytes()).hexdigest() == result['source_sha256']
    assert not (policy.parent/'unexpected-dir').exists()
    assert not (policy.parent/'unexpected-link').exists()


def test_unavailable_boundary_never_imports_the_policy(tmp_path, monkeypatch):
    # Patch the copied helper source rather than running any unconfined payload.
    original = Path.read_bytes
    def helper_bytes(path):
        if path.name == 'linux_policy_sandbox.py':
            return b'def enforce(paths):\n    raise RuntimeError("boundary unavailable")\n'
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', helper_bytes)
    result = run(tmp_path, 'raise RuntimeError("POLICY MUST NOT RUN")\n')
    assert result['status'] == 'sandbox_unavailable', result
    assert 'boundary unavailable' in result['error'] and 'POLICY MUST NOT RUN' not in result['error']


def test_twenty_concurrent_workers_are_isolated(tmp_path):
    async def exercise():
        return await asyncio.gather(*(worker.run_exposed_policy(
            'def decide(state):\n    return {"move": state["move"]}\n',
            {'move': f'A{i + 1}'}, tmp_path, i + 1) for i in range(20)))
    results = asyncio.run(exercise())
    assert all(r['status'] == 'completed' for r in results), results
    assert [r['move'] for r in results] == [f'A{i + 1}' for i in range(20)]


@pytest.mark.parametrize('source', [
    'while True: pass\n',
    'def decide(state):\n    x=bytearray(256*1024*1024)\n    return {"resign": True}\n',
    'print("x"*100000)\ndef decide(state):\n    return {"resign": True}\n',
    'def decide(state):\n    return "garbage"\n',
])
def test_resource_abuse_and_invalid_return_fail_closed(tmp_path, source):
    result = run(tmp_path, source)
    assert result['status'] not in ('completed', 'setup_error', 'sandbox_unavailable'), result
    assert result['resign'] is None and result['move'] is None


def test_cancellation_kills_worker(tmp_path, monkeypatch):
    killed = []
    original = worker._kill
    def tracked(process):
        killed.append(process.pid)
        original(process)
    monkeypatch.setattr(worker, '_kill', tracked)
    async def exercise():
        task = asyncio.create_task(worker.run_exposed_policy('while True: pass\n', {}, tmp_path, 1))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(exercise())
    assert killed
    for pid in killed:
        assert not Path(f'/proc/{pid}').exists()
