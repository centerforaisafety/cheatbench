"""Task preparation must finish successfully before any episodes can start."""
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

from core.task_build import TaskBuildError, ensure_task_built


@dataclass
class Task:
    root: Path
    name: str = 'example'
    variant: str = ''
    review: object = None

    @property
    def environment(self):
        return self.root / 'environment' / self.variant

    @property
    def data_path(self):
        return (self.environment if self.variant else self.root) / 'data.jsonl'

    def rows(self):
        return [json.loads(line) for line in self.data_path.read_text().splitlines()]


@pytest.fixture
def task(tmp_path):
    (tmp_path / 'task.yaml').write_text('{}\n')
    (tmp_path / 'build.py').write_text('''import json, pathlib, sys, time
root = pathlib.Path(__file__).parent
variant = sys.argv[1] if len(sys.argv) > 1 else ''
env = root / 'environment' / variant
env.mkdir(parents=True, exist_ok=True)
count = root / 'count'
count.write_text(str(int(count.read_text()) + 1) if count.exists() else '1')
time.sleep(0.05)
(env / 'input.txt').write_text('input')
(env / 'extra.txt').write_text('host asset')
(env / 'files/golds').mkdir(parents=True, exist_ok=True)
(env / 'files/golds/answer.txt').write_text('gold')
row = {'id': 'one', 'files': {'input.txt': 'input.txt'}, 'meta': {'gold_name': 'answer.txt'}}
((env if variant else root) / 'data.jsonl').write_text(json.dumps(row) + '\\n')
''')
    return Task(tmp_path)


def test_first_build_reuse_and_missing_source(task):
    rows, record = ensure_task_built(task)
    assert rows[0]['id'] == 'one'
    assert record['status'] == 'built'
    assert len(record['data_sha256']) == 64
    assert ensure_task_built(task)[1]['status'] == 'reused'
    assert (task.root / 'count').read_text() == '1'
    (task.environment / 'input.txt').unlink()
    assert ensure_task_built(task)[1]['status'] == 'built'
    assert (task.root / 'count').read_text() == '2'


def test_disabled_and_forced(task):
    with pytest.raises(TaskBuildError, match='automatic builds disabled'):
        ensure_task_built(task, auto_build=False)
    assert not (task.root / 'count').exists()
    ensure_task_built(task)
    task.data_path.write_text('broken json')
    with pytest.raises(TaskBuildError, match='invalid task data'):
        ensure_task_built(task)
    assert ensure_task_built(task, rebuild=True)[1]['status'] == 'built'


@pytest.mark.parametrize('script, message', [('raise SystemExit(3)', 'build failed'), ('pass', 'still missing')])
def test_failed_or_incomplete(task, script, message):
    (task.root / 'build.py').write_text(script)
    with pytest.raises(TaskBuildError, match=message):
        ensure_task_built(task)


def test_concurrent_launches_build_once(task):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: ensure_task_built(task), range(4)))
    assert [r[1]['status'] for r in results].count('built') == 1
    assert (task.root / 'count').read_text() == '1'


def test_variant_and_host_files(task):
    task.variant = 'v1'
    (task.root / 'task.yaml').write_text('build:\n  args: ["{variant}"]\n  required: ["environment/{variant}/extra.txt"]\n')
    ensure_task_built(task)
    (task.environment / 'extra.txt').unlink()
    assert ensure_task_built(task)[1]['status'] == 'built'
    assert (task.root / 'count').read_text() == '2'


def test_gold_files(task):
    task.review = {'judge': 'test'}
    ensure_task_built(task)
    (task.environment / 'files/golds/answer.txt').unlink()
    assert ensure_task_built(task)[1]['status'] == 'built'


def test_no_builder(task):
    ensure_task_built(task)
    (task.root / 'build.py').unlink()
    assert ensure_task_built(task)[1]['status'] == 'reused'
    task.data_path.unlink()
    with pytest.raises(TaskBuildError, match='no build.py'):
        ensure_task_built(task)


def test_source_escape(task):
    ensure_task_built(task)
    task.data_path.write_text(json.dumps({'id': 'one', 'files': {'x': '../secret'}}))
    with pytest.raises(TaskBuildError, match='source escapes'):
        ensure_task_built(task)


@pytest.mark.parametrize('config', ['build: []', 'build: {args: wrong}', 'build: {required: [../secret]}', 'build: {unknown: []}'])
def test_bad_config(task, config):
    (task.root / 'task.yaml').write_text(config)
    with pytest.raises(TaskBuildError):
        ensure_task_built(task)


def test_real_subset_sum_builder(tmp_path):
    """Fresh task source -> build -> load actual rows -> repair deleted input."""
    import shutil
    from core.trial import load_task
    source = Path(__file__).resolve().parents[1] / 'tasks/subset_sum'
    root = tmp_path / 'subset_sum'
    root.mkdir()
    for filename in ('build.py', 'task.yaml', 'grader.py'):
        shutil.copy2(source / filename, root / filename)
    for directory in ('assets', 'prompts'):
        shutil.copytree(source / directory, root / directory)
    task = load_task(str(root))
    rows, record = ensure_task_built(task)
    assert len(rows) == 10
    assert record['status'] == 'built'
    assert ensure_task_built(task)[1]['status'] == 'reused'
    source_path = next(iter(rows[0]['files'].values()))
    (task.environment / source_path).unlink()
    assert ensure_task_built(task)[1]['status'] == 'built'


def test_openmath_agent_repairs_missing_tls_key(tmp_path, monkeypatch):
    import runpy
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tasks/openmath_agent/build.py'))
    tls = tmp_path / 'tls'
    tls.mkdir()
    (tls / 'site.crt').write_text('certificate')
    calls = []
    monkeypatch.setattr(module['subprocess'], 'run', lambda cmd, **kw: calls.append(cmd))
    module['certificates'](tmp_path)
    assert len(calls) == 3
    calls.clear()
    for filename in ('ca.crt', 'site.key'):
        (tls / filename).write_text('present')
    module['certificates'](tmp_path)
    assert not calls


def test_runtime_preflight_runs_for_fresh_and_cached_inputs(task):
    builder = task.root / 'build.py'
    original = builder.read_text()
    builder.write_text('''import pathlib, sys
root = pathlib.Path(__file__).parent
if '--check-runtime' in sys.argv:
    checked = root / 'checks'
    checked.write_text(str(int(checked.read_text()) + 1) if checked.exists() else '1')
    raise SystemExit(9 if (root / 'bad-runtime').exists() else 0)
''' + original)
    (task.root / 'task.yaml').write_text('build:\n  preflight: [--check-runtime]\n')
    (task.root / 'bad-runtime').touch()
    with pytest.raises(TaskBuildError, match='runtime preflight failed'):
        ensure_task_built(task)
    assert not (task.root / 'count').exists()
    (task.root / 'bad-runtime').unlink()
    assert ensure_task_built(task)[1]['status'] == 'built'
    assert ensure_task_built(task)[1]['status'] == 'reused'
    (task.root / 'bad-runtime').touch()
    with pytest.raises(TaskBuildError, match='runtime preflight failed'):
        ensure_task_built(task)
    assert (task.root / 'count').read_text() == '1'
    assert (task.root / 'checks').read_text() == '4'
