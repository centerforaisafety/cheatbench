"""Services must prepare real submissions and release resources on every exit."""
import asyncio
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import pytest

import run
from core import task_service, trial

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def svg_task(tmp_path, monkeypatch):
    root = tmp_path / 'svgs'
    shutil.copytree(ROOT / 'tasks/svgs', root,
                    ignore=shutil.ignore_patterns('results', 'environment', '__pycache__', '.build.lock'))
    return trial.load_task(str(root))


def assert_closed(endpoint):
    port = int(endpoint.split(':')[2].split('/')[0])
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', port)) != 0


@pytest.mark.parametrize('raw', [{'module': '../outside.py'}, {'module': '/tmp/no.py'},
                                 {'module': 'service.py', 'config': []}, {'other': 'x'}, 'service.py'])
def test_invalid_service(svg_task, raw):
    with pytest.raises(ValueError):
        task_service.parse_service(svg_task.root, raw)


def test_no_service_needs_no_files(tmp_path):
    task = trial.Task(root=tmp_path, name='ordinary')
    with task_service.open_service(task, rows=[], out_dir=tmp_path, load_model=None):
        pass
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('mode', ['success', 'cancel', 'retry', 'repeat'])
def test_root_runner_starts_svg_and_cleans_up(svg_task, tmp_path, monkeypatch, mode):
    """Exercise the public CLI with real build, hook, HTTP server and checker.

    Only the container/agent is substituted; the leaderboard needs no model client.
    """
    monkeypatch.setenv('SERVICE_TEST_KEY', 'offline-test')
    config = tmp_path / 'models.yaml'
    config.write_text('models:\n  default: test\n  test:\n    model: openai/test\n'
                      '    api_key_env: SERVICE_TEST_KEY\n'
                      '    api_base_url: http://127.0.0.1\n')
    agent = run.make_agent('codex', model='test', api_key_env='SERVICE_TEST_KEY',
                           api_base_url='http://127.0.0.1')
    monkeypatch.setattr(agent, 'setup', lambda: [])
    monkeypatch.setattr(run, 'make_agent', lambda *a, **kw: agent)
    monkeypatch.setattr(run.sandbox, 'ensure_path', lambda: None)
    monkeypatch.setattr(run.sandbox, 'preflight', lambda **kw: [])
    async def ensure_image(*a, **kw):
        assert kw.get("dockerfile") == svg_task.image_dockerfile
    monkeypatch.setattr(run.sandbox, 'ensure_image_async', ensure_image)
    assert svg_task.service['config'] == {}
    endpoints = []
    tokens = []
    async def episode(task, row, agent, **kw):
        stage = tmp_path / ('staged-' + str(len(endpoints)))
        stage.mkdir(exist_ok=True)
        board = stage / 'leaderboard'
        board.mkdir(exist_ok=True)
        for dest, source in row['files'].items():
            shutil.copyfile(task.environment / source, stage / dest)
        endpoint = json.loads((board / 'leaderboard_config.json').read_text())['endpoint']
        endpoints.append(endpoint)
        tokens.append(json.loads((board / 'leaderboard_config.json').read_text())['token'])
        if mode == 'cancel':
            raise asyncio.CancelledError()
        (stage / 'drawing.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600"><circle cx="400" cy="300" r="100"/></svg>')
        result = await asyncio.to_thread(subprocess.run,
            [sys.executable, str(board / 'leaderboard.pyc'), 'drawing.svg'],
            cwd=stage, capture_output=True, text=True, check=True)
        assert '6. drawing.svg' in result.stdout
        assert 'Submissions: 1' in result.stdout
        if mode == 'retry' and len(endpoints) == 1:
            return {'id': row['id'], 'episode': row['id'] + '#1', 'replicate': 1,
                    'ok': False, 'failure': next(iter(run.agent_errors.TRANSIENT_REASONS))}
        replicate = kw['replicate']
        return {'id': row['id'], 'episode': row['id'] + f'#{replicate}', 'replicate': replicate, 'ok': True}
    monkeypatch.setattr(run.trial, 'run_trial', episode)
    monkeypatch.setattr(sys, 'argv', ['run.py', str(svg_task.root), '--model', 'test',
        '--agent', 'codex', '--judge', 'test', '--no-judge', '--no-progress', '--limit', '1',
        '--models-config', str(config), '--run-id', 'test', '--output-dir', str(tmp_path / 'out')])
    if mode == 'repeat':
        sys.argv.extend(['--repeat', '2', '--max-concurrent', '2'])
    if mode == 'cancel':
        with pytest.raises(asyncio.CancelledError):
            run.main()
    else:
        assert run.main() == 0
    expected_calls = 2 if mode in {'retry', 'repeat'} else 1
    assert len(endpoints) == expected_calls
    assert len(set(tokens)) == expected_calls
    for endpoint in endpoints:
        assert_closed(endpoint)
    assert not list(svg_task.environment.glob('service_*'))
    logs = list((tmp_path / 'out').glob('*/leaderboard/*.json'))
    assert len(logs) == expected_calls
    calls = [call for log in logs for entries in json.loads(log.read_text()).values()
             for call in entries]
    assert len(calls) == (0 if mode == 'cancel' else expected_calls)
    assert all(call['number'] == 1 for call in calls)
    if mode != 'cancel':
        assert calls[0]['rank'] == 6
        assert set(calls[0]) == {'number', 'sha256', 'rank'}
        # Resume skips completed episodes without starting another service.
        assert run.main() == 0
        assert len(endpoints) == expected_calls
        sys.argv.append('--redo')
        assert run.main() == 0
        assert len(list((tmp_path / 'out').glob('*/leaderboard/*.json'))) == expected_calls + (2 if mode == 'repeat' else 1)
        assert_closed(endpoints[-1])
    # Persisted rows retain their placeholder, never the runtime token/URL.
    row = svg_task.rows()[0]
    original = json.loads((svg_task.environment / row['files']['leaderboard/leaderboard_config.json']).read_text())
    assert original['endpoint'] == 'RUNTIME_ENDPOINT'


def test_svg_startup_failure_closes_server(svg_task, tmp_path, monkeypatch):
    from core.task_build import ensure_task_built
    from tasks.svgs import service
    rows, _ = ensure_task_built(svg_task)
    endpoints = []
    real_start = service.start_server
    def capture(rounds):
        server, configs = real_start(rounds)
        endpoints.append(configs[0][1]['endpoint'])
        return server, configs
    monkeypatch.setattr(service, 'start_server', capture)
    def fail_staging(*args, **kwargs):
        raise OSError('staging unavailable')
    monkeypatch.setattr(service.tempfile, 'TemporaryDirectory', fail_staging)
    with pytest.raises(OSError, match='staging unavailable'):
        with service.start(task=svg_task, rows=rows[:1], out_dir=tmp_path / 'out',
                           config={}, load_model=None):
            pytest.fail('episodes must not start')
    assert_closed(endpoints[0])
    assert len(list((tmp_path / 'out/leaderboard').glob('*.json'))) == 1


def test_root_runner_builds_task_image_recipe(svg_task, tmp_path, monkeypatch):
    recipe = svg_task.root / 'Dockerfile'
    recipe.write_text('FROM python:3.11-slim\n')
    config = svg_task.root / 'task.yaml'
    config.write_text(config.read_text() + '\nimage_dockerfile: Dockerfile\n')
    task = trial.load_task(str(svg_task.root))
    assert task.image_dockerfile == recipe
    test_root_runner_starts_svg_and_cleans_up(task, tmp_path, monkeypatch, 'success')


def test_svg_service_never_resolves_a_model(svg_task, tmp_path):
    from core.task_build import ensure_task_built
    from tasks.svgs import service
    rows, _ = ensure_task_built(svg_task)
    def no_model(name):
        pytest.fail('Leaderboard must not resolve a reviewer model')
    with service.start(task=svg_task, rows=rows[:1], out_dir=tmp_path / 'out',
                       config={}, load_model=no_model) as server:
        assert server.infrastructure_errors == []
    assert len(list((tmp_path / 'out/leaderboard').glob('*.json'))) == 1
