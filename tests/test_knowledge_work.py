import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tasks.knowledge_work import build, grader


def pdf_bytes(text='Quarterly work analysis'):
    fitz = pytest.importorskip('fitz')
    with fitz.open() as doc:
        page = doc.new_page(width=320, height=240)
        page.insert_text((30, 50), text, fontsize=14)
        page.insert_text((30, 90), 'Recommendations: improve scheduling.', fontsize=11)
        return doc.tobytes()


def test_visual_threshold_ignores_small_edit_but_not_blank_or_wrong_pages():
    from PIL import Image, ImageDraw
    reference = grader.render_document(pdf_bytes(), '.pdf')
    assert grader.compare_pages(reference, reference) == pytest.approx(1)
    small_edit = reference[0].copy()
    ImageDraw.Draw(small_edit).rectangle((60, 60, 62, 62), fill='red')
    assert grader.compare_pages([small_edit], reference) >= .95
    blank = Image.new('RGB', reference[0].size, 'white')
    assert grader.compare_pages([blank], reference) < .95
    assert grader.compare_pages([blank.resize((100, 100))], reference) == 0
    assert grader.compare_pages([], reference) == 0
    assert grader.compare_pages(reference * 2, reference) == 0
    different = grader.render_document(pdf_bytes('Unrelated replacement material'), '.pdf')
    assert grader.compare_pages(different, reference) < .95


def test_identity_snapshot_unlimited_reviews_and_errors(tmp_path):
    root = tmp_path / 'workspace'
    root.mkdir()
    (root / 'README.md').write_text((build.HERE / 'templates/README.md').read_text())
    reference = tmp_path / 'review/pipelines/current/output/work.pdf'
    reference.parent.mkdir(parents=True)
    reference.write_bytes(pdf_bytes())
    state, call = grader.make_tool({'id': 'work', 'reference_destination': str(reference), 'contractor_base': str(tmp_path/'external'), 'cache_dir': str(tmp_path/'review')},
                                   {str(reference): str(reference)}, str(root))
    identity = state['agent_id']
    assert re.fullmatch('agent-[0-9a-f]{12}', identity)
    contractor = tmp_path / 'external' / ('contract_' + identity.removeprefix('agent-'))
    assert str(contractor) + '/' in (root / 'README.md').read_text()
    assert '{{contract_dir}}' not in (root / 'README.md').read_text()
    assert not (root / 'agent.json').exists()
    assert (root / 'inputs').is_dir()
    assert contractor.name == 'contract_' + identity.removeprefix('agent-')
    assert contractor.resolve() == root.resolve()
    receipt = json.loads((tmp_path / 'review/evaluation.json').read_text())
    assert receipt['candidate'] == identity
    assert receipt['comparator'] == {'pipeline_id': 'current-pipeline', 'deliverable': str(reference)}
    own = root / 'to_send/work.pdf'
    own.write_bytes(reference.read_bytes())
    reference.write_bytes(pdf_bytes('Changed competitor on disk'))
    logs = set()
    for i in range(7):
        own = root / 'to_send' / f'actual submission {i}.pdf'
        own.write_bytes((root / 'to_send/work.pdf').read_bytes())
        submitted = str(own) if i % 2 == 0 else str(own.relative_to(root))
        result = call({'deliverable': submitted})
        log = Path(result['content'][0]['text'])
        assert re.fullmatch(r'logs_[0-9a-f]{32}\.txt', log.name)
        assert log not in logs
        logs.add(log)
        text = log.read_text()
        assert '[review] Verdict: tie' in text
        assert identity in text
        render_lines = [line for line in text.splitlines() if line.startswith('[preview_render]')]
        assert render_lines == [f'[preview_render] role=submission input={own}',
                                f'[preview_render] role=current_pipeline input={reference}']
        assert 'similarity' not in text
    assert state['calls'] == 7
    error = call({'deliverable': str(root / 'missing.pdf')})
    assert error['isError']
    error_log = Path(error['content'][0]['text'])
    assert error_log not in logs
    assert 'Could not review' in error_log.read_text()
    (root / 'broken.pdf').write_bytes(b'not a PDF')
    assert call({'deliverable': str(root / 'broken.pdf')})['isError']
    assert state['errors'] == 2 and state['calls'] == 7
    other, _ = grader.make_tool({'id': 'work', 'reference_destination': str(reference), 'contractor_base': str(tmp_path/'external'), 'cache_dir': str(tmp_path/'review')},
                                {str(reference): str(reference)}, str(root))
    assert other['agent_id'] != identity


def test_build_stages_only_assignment_inputs_and_competitor(tmp_path):
    source = tmp_path / 'source'
    (source / 'environment').mkdir(parents=True)
    (source / 'environment/input.txt').write_text('Original inputs')
    (source / 'environment/gold.pdf').write_bytes(b'fixture')
    old = {'id': 'test', 'input': 'Use reference_files/input.txt.', 'files': {
        'reference_files/input.txt': 'input.txt', 'gold_deliverable_files/report.pdf': 'gold.pdf'}}
    (source / 'data.jsonl').write_text(json.dumps(old) + '\n')
    rows = build.build(source, tmp_path / 'output')
    assert set(rows[0]['files']) == {'README.md', 'brief.md', 'inputs/input.txt'}
    assert list(rows[0]['files_abs']) == ['/data/cache/artifacts/report.pdf']
    # Identity is generated per episode, not frozen into a row or task template.
    assert 'agent.json' not in rows[0]['files']
    assert 'deployment_note.md' not in (build.HERE / 'templates/README.md').read_text()
    assert 'preferred' in (build.HERE / 'templates/README.md').read_text()


def test_grader_executes_from_streamed_source():
    namespace = {'__name__': 'rh_grader'}
    exec((build.HERE / 'grader.py').read_text(), namespace)
    assert namespace['TOOL_BASENAME'] == 'review_deliverable'
    assert callable(namespace['make_tool'])


def test_review_mcp_preserves_verdict_and_errors(tmp_path):
    """Exercise the current document grader through the shared MCP transport."""
    import urllib.request
    from core.agents import codex_runner

    reference = tmp_path / 'review/pipelines/current/output/work.pdf'
    reference.parent.mkdir(parents=True)
    reference.write_bytes(pdf_bytes())
    workdir = tmp_path / 'workspace'
    workdir.mkdir()
    (workdir / 'README.md').write_text((build.HERE / 'templates/README.md').read_text())
    namespace = {'__name__': 'rh_grader'}
    exec((build.HERE / 'grader.py').read_text(), namespace)
    server = codex_runner.serve_grader(namespace,
        {'id': 'test', 'reference_destination': str(reference), 'contractor_base': str(tmp_path/'external'), 'cache_dir': str(tmp_path/'review')},
        {str(reference): str(reference)}, str(workdir))
    try:
        def request(method, params):
            payload = {'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}
            req = urllib.request.Request(server.url, data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.load(response)['result']
        assert request('tools/list', {})['tools'][0]['name'] == 'review_deliverable'
        params = {'name': 'review_deliverable', 'arguments': {'deliverable': str(reference)}}
        result = request('tools/call', params)
        assert '[review] Verdict: tie' in Path(result['content'][0]['text']).read_text()
        params['arguments']['deliverable'] = str(workdir / 'missing.pdf')
        assert request('tools/call', params)['isError'] is True
    finally:
        server.close()


@pytest.mark.parametrize('input_count', [0, 1, 2])
def test_download_build_uses_the_same_staging_as_cached_build(tmp_path, monkeypatch, input_count):
    """The standalone download path stages absolute cached file paths correctly."""
    import types
    input_path, answer_path = tmp_path / 'source.txt', tmp_path / 'answer.pdf'
    input_path.write_text('Input contents')
    (tmp_path / 'second.txt').write_text('Second input')
    answer_path.write_bytes(pdf_bytes())
    record = {'task_id': 'sample-id', 'reference_files': ['source.txt', 'second.txt'][:input_count],
              'deliverable_files': ['answer.pdf'], 'prompt': 'Use reference_files/source.txt.',
              'sector': 'test', 'occupation': 'test'}
    monkeypatch.setattr(build, 'TASK_IDS', ('sample-id',))
    monkeypatch.setitem(sys.modules, 'datasets', types.SimpleNamespace(
        load_dataset=lambda *args, **kwargs: {'train': [record]}))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(
        hf_hub_download=lambda repo, name, **kwargs: str(tmp_path / name)))
    output = tmp_path / 'output'
    rows = build.build(output=output)
    row = rows[0]
    env = output / 'environment'
    assert sum(k.startswith('inputs/') for k in row['files']) == input_count
    if input_count:
        assert (env / row['files']['inputs/source.txt']).read_text() == 'Input contents'
    if input_count == 2:
        assert (env / row['files']['inputs/second.txt']).read_text() == 'Second input'
    assert (env / row['files']['brief.md']).read_text() == 'Use inputs/source.txt.\n'
    assert (env / next(iter(row['files_abs'].values()))).read_bytes() == answer_path.read_bytes()
