"""Task schemas and portable builds without archived evaluation results."""
import gzip
import json
from pathlib import Path

import pytest
from core.judge import load_judge_spec

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'chess', 'go', 'software_engineering'}


@pytest.mark.parametrize('task', [p.parent.name for p in (ROOT / 'tasks').glob('*/task.yaml')
                                  if p.parent.name not in EXCLUDED])
def test_all_included_tasks_use_shared_prohibition_fields(task):
    spec = load_judge_spec(ROOT / 'tasks' / task)
    assert spec.fields[0] == 'cheated'
    assert spec.verdict.model_fields['explicit_prohibition_violated'].annotation is bool
    assert 'explicit_prohibition_evidence' in spec.fields
    if task != 'sycophancy':
        assert spec.verdict.model_fields['honey_pots_seen'].annotation is bool


def test_included_builders_have_no_developer_paths_or_symlink_creation():
    paths = [p for p in (ROOT / 'tasks').glob('*/build.py') if p.parent.name not in EXCLUDED]
    paths.append(ROOT / 'tasks/creative_writings/_builder.py')
    for path in paths:
        source = path.read_text()
        for forbidden in ('/data/long_phan/', '/data/jason_lim/', '/Users/', '.symlink_to(', 'os.symlink('):
            assert forbidden not in source, (path, forbidden)


def test_sycophancy_default_build_does_not_read_developer_checkout(tmp_path, monkeypatch):
    from tasks.sycophancy import build
    monkeypatch.setattr(build, 'OUT', tmp_path / 'data.jsonl')
    monkeypatch.setattr('sys.argv', ['build.py'])
    assert build.main() == 0
    rows = [json.loads(line) for line in build.OUT.read_text().splitlines()]
    assert sum(r['id'].startswith('bubble__') for r in rows) == 7
    assert not any(r['id'].startswith('donation__') for r in rows)
    monkeypatch.setattr('sys.argv', ['build.py', '--thresholds', 'test'])
    with pytest.raises(SystemExit) as error:
        build.main()
    assert error.value.code == 2


def test_knowledge_download_records_portable_paths(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys
    from tasks.knowledge_work import build
    task_id = build.TASK_IDS[0]
    monkeypatch.setattr(build, 'TASK_IDS', (task_id,))
    item = dict(task_id=task_id, prompt='Example', sector='Test', occupation='Test',
                reference_files=['input.txt'], deliverable_files=['work.pdf'])
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=lambda *a, **k: {'train': [item]}))
    source = tmp_path / 'source'
    source.write_bytes(b'fixture')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=lambda *a, **k: str(source)))
    rows = build.download_source(tmp_path / 'built')
    assert rows
    for path in rows[0]['files'].values():
        assert not Path(path).is_absolute()
        assert '..' not in Path(path).parts
        assert (tmp_path / 'built/environment' / path).read_bytes() == b'fixture'
