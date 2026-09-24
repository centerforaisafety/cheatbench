import asyncio
import json
from types import SimpleNamespace
import sys

import pytest

from core.grader_guard import install_grader_guard
from core import judge


def test_guard_preserves_invalid_submission_but_reports_crash(capsys):
    ns = {}
    install_grader_guard('''
def make_tool(*args):
    def call(args):
        if args.get('crash'):
            raise RuntimeError('review backend failed')
        return {'isError': True, 'content': []}
    return {}, call
''', ns)
    _, call = ns['make_tool']()
    assert call({})['isError']
    assert not capsys.readouterr().err
    with pytest.raises(RuntimeError, match='backend failed'):
        call({'crash': True})
    assert '\x1eRHGRADER_FATAL ' in capsys.readouterr().err


def test_import_failure_is_reported_before_agent_start(capsys):
    with pytest.raises(ModuleNotFoundError):
        install_grader_guard('import missing_reviewer_dependency_734', {})
    assert 'RHGRADER_FATAL ' in capsys.readouterr().err


@pytest.mark.parametrize('reason', ['reviewer_infrastructure', 'batch_cancelled'])
def test_invalid_attempt_is_not_judged(reason):
    assert not judge.is_judgeable({'transcript_path': 'partial.json', 'failure': reason})


def test_batch_exits_nonzero_saves_failure_and_stops_queued_rows(tmp_path, monkeypatch, capsys):
    import run
    import yaml
    task = tmp_path / 'task'
    task.mkdir()
    (task / 'task.yaml').write_text('name: probe\ndata: data.jsonl\nprompts:\n  instruction: instruction.md\n')
    (task / 'instruction.md').write_text('Do the task.')
    rows = [{'id': 'one'}, {'id': 'two'}]
    (task / 'data.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    cfg = tmp_path / 'models.yaml'
    cfg.write_text(yaml.safe_dump({'models': {'probe': {
        'model': 'openai/probe', 'api_key_env': 'REVIEWER_TEST_KEY',
        'api_base_url': 'https://example.invalid/v1'}}}))
    monkeypatch.setenv('REVIEWER_TEST_KEY', 'test-only')
    monkeypatch.setattr(run.task_build, 'ensure_task_built', lambda *a, **kw: (rows, {}))
    monkeypatch.setattr(run.sandbox, 'ensure_path', lambda: None)
    monkeypatch.setattr(run.sandbox, 'preflight', lambda **kw: [])
    async def no_build(*a, **kw): pass
    monkeypatch.setattr(run.sandbox, 'ensure_image_async', no_build)
    original = run.make_agent
    def make(*a, **kw):
        agent = original(*a, **kw)
        agent.setup = lambda: []
        return agent
    monkeypatch.setattr(run, 'make_agent', make)
    monkeypatch.setattr(run.judging, 'load_judge_spec', lambda *a, **kw: SimpleNamespace(prompt_path=task/'judge.md', schema_path=task/'judge_schema.py', fields=[]))
    started = []
    async def fail(task, row, *args, **kwargs):
        started.append(row['id'])
        await asyncio.sleep(0)
        return {'id': row['id'], 'episode': row['id']+'#1', 'replicate': 1,
                'ok': False, 'failure': 'reviewer_infrastructure',
                'error': 'review backend unavailable'}
    monkeypatch.setattr(run.trial, 'run_trial', fail)
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['run.py', str(task), '--model', 'probe',
        '--judge', 'probe', '--models-config', str(cfg), '--agent', 'codex',
        '--no-judge', '--max-concurrent', '1', '--output-dir', str(out), '--run-id', 'test'])
    assert run.main() == 1
    assert started == ['one']
    records = [json.loads(line) for f in out.glob('*/episodes.jsonl') for line in f.read_text().splitlines()]
    assert len(records) == 1 and records[0]['failure'] == 'reviewer_infrastructure'
    assert 'FATAL: reviewer infrastructure failed' in capsys.readouterr().err

@pytest.mark.parametrize('cancel', [False, True])
def test_live_episode_stops_and_reaps_process(tmp_path, monkeypatch, cancel):
    from core import sandbox, trial
    from core.agents import make_agent
    task_dir = tmp_path / 'task'
    task_dir.mkdir()
    (task_dir / 'task.yaml').write_text('name: probe\nprompts:\n  instruction: instruction.md\n')
    (task_dir / 'instruction.md').write_text('Test task.')
    task = trial.load_task(str(task_dir))
    agent = make_agent('codex', model='probe')
    agent.apply_tool_policy({'web_search': False, 'web_fetch': False})
    code = "import sys,time; "
    if not cancel:
        code += "sys.stderr.write('\\x1eRHGRADER_FATAL {\\\"error\\\":\\\"backend crashed\\\"}\\n'); sys.stderr.flush(); "
    code += 'time.sleep(30)'
    monkeypatch.setattr(sandbox, 'container_argv', lambda *a, **kw: [sys.executable, '-c', code])
    async def no_build(*a, **kw): pass
    monkeypatch.setattr(sandbox, 'ensure_image_async', no_build)
    original = asyncio.create_subprocess_exec
    processes = []
    async def create(*args, **kwargs):
        process = await original(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', create)
    async def check():
        episode = asyncio.create_task(trial.run_trial(task, {'id': 'one', 'files': {}}, agent,
            image='unused', api_key='test', out_dir=tmp_path/'out', timeout_s=60))
        if cancel:
            while not processes:
                await asyncio.sleep(.01)
            await asyncio.sleep(.05)
            episode.cancel()
        result = await asyncio.wait_for(episode, timeout=5)
        assert result['failure'] == ('batch_cancelled' if cancel else 'reviewer_infrastructure')
        assert result['ok'] is False
        assert processes[0].returncode is not None
        assert not judge.is_judgeable(result)
    asyncio.run(check())

def test_svg_bad_submission_is_recoverable_but_renderer_failure_is_fatal(monkeypatch):
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    from tasks.svgs import leaderboard_server
    def fail(*args, **kwargs):
        raise RuntimeError('renderer unavailable')
    monkeypatch.setattr(leaderboard_server.cairosvg, 'svg2png', fail)
    rnd = leaderboard_server.Round({'id': 'test', 'input': 'Draw a circle',
                                   'files_abs': {'/rubric.json': 'unused'}})
    server, endpoints = leaderboard_server.start([rnd])
    endpoint = endpoints[0][1]
    def submit(svg):
        request = Request(endpoint['endpoint'], data=json.dumps({'submission': svg}).encode(),
                          headers={'Authorization': 'Bearer '+endpoint['token']})
        with pytest.raises(HTTPError) as exc:
            urlopen(request, timeout=5)
        return exc.value.code
    try:
        assert submit('not an SVG') == 400
        assert submit(None) == 400
        assert submit({'unexpected': 'object'}) == 400
        assert server.infrastructure_errors == []
        assert submit('<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"/>') == 503
        assert 'renderer unavailable' in server.infrastructure_errors[0]
    finally:
        server.shutdown()
        server.server_close()


def test_request_size_failures_are_infrastructure_even_in_old_records():
    from core.agents import errors
    message = 'BadRequestError: request_too_large: Request exceeds the maximum size'
    assert errors.classify(message) is errors.RequestTooLargeError
    assert errors.is_lost('request_too_large')
    assert not judge.is_judgeable({'transcript_path': 'partial.json', 'failure': 'unclassified', 'error': message})


@pytest.mark.skipif(__import__('os').environ.get('RH_TERMINUS2_CONTAINER_TEST') != '1', reason='real Enroot opt-in')
@pytest.mark.parametrize('cancel', [False, True])
def test_real_enroot_reviewer_failure_reaps_descendants(tmp_path, monkeypatch, cancel):
    import os
    import time
    import uuid
    from pathlib import Path
    from core import sandbox, trial
    from core.agents import make_agent
    token = 't2-reap-' + uuid.uuid4().hex
    task_dir = tmp_path / 'task'
    task_dir.mkdir()
    (task_dir/'task.yaml').write_text('name: reap\nprompts:\n  instruction: instruction.md\n')
    (task_dir/'instruction.md').write_text('Cleanup probe')
    task = trial.load_task(str(task_dir))
    agent = make_agent('codex', model='probe')
    agent.apply_tool_policy({'web_search': False, 'web_fetch': False})
    original = sandbox.container_argv
    code = "import sys,time,subprocess,json; json.load(sys.stdin); subprocess.Popen(['python','-c','import time; time.sleep(300)'," + repr(token) + "]); print('DESCENDANT_READY',file=sys.stderr,flush=True); time.sleep(2); "
    if not cancel:
        code += "print('\\x1eRHGRADER_FATAL {\\\"error\\\":\\\"injected reviewer failure\\\"}',file=sys.stderr,flush=True); "
    code += 'time.sleep(300)'
    monkeypatch.setattr(sandbox, 'container_argv', lambda image, bootstrap, **kw: original(image, code, **kw))
    sandbox.ensure_path()
    def descendants():
        found = []
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                args = p.read_bytes().split(b'\0')
                if token.encode() in args:
                    found.append(int(p.parent.name))
            except OSError:
                pass
        return found
    async def check():
        episode = asyncio.create_task(trial.run_trial(task, {'id':'one','files':{}}, agent,
            image=sandbox.DEFAULT_IMAGE, api_key='test', out_dir=tmp_path/'out', timeout_s=300))
        for _ in range(1200):
            if descendants(): break
            if episode.done(): pytest.fail('Container ended before descendant started: '+str(episode.result().get('error')))
            await asyncio.sleep(.1)
        assert descendants(), 'No real container descendant observed'
        before = descendants()
        if cancel: episode.cancel()
        result = await asyncio.wait_for(episode, 20)
        assert result['failure'] == ('batch_cancelled' if cancel else 'reviewer_infrastructure')
        assert not judge.is_judgeable(result)
        for _ in range(100):
            if not descendants(): break
            await asyncio.sleep(.1)
        assert not descendants(), before
        assert list((tmp_path/'out').rglob('container.stderr.log'))
        (tmp_path/'cleanup-evidence.json').write_text(json.dumps({'observed_host_pids':before,'remaining':descendants(),'failure':result['failure']}))
    asyncio.run(check())
