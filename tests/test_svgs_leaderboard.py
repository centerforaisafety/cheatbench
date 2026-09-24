"""Exercise the real HTTP leaderboard and compiled submission client, without an LLM."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import llm_agents
from tasks.svgs import build, leaderboard_server

SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600"><circle cx="400" cy="300" r="100"/></svg>'


def test_rows_and_official_submission_without_model(tmp_path, monkeypatch):
    def no_model(*args, **kwargs):
        pytest.fail('The SVG leaderboard must never create an LLM client')
    monkeypatch.setattr(llm_agents, 'get_llm_agent_class', no_model)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    rows = build.build()
    assert len(rows) == 20
    assert all(len(row['files_abs']) == 1 for row in rows)
    rnd = leaderboard_server.Round(rows[0])
    server, configs = leaderboard_server.start([rnd])
    try:
        board = tmp_path / 'leaderboard'
        board.mkdir()
        (board / 'leaderboard.pyc').write_bytes((build.ENV / 'shared/leaderboard/leaderboard.pyc').read_bytes())
        (board / 'leaderboard_config.json').write_text(json.dumps(configs[0][1]))
        (tmp_path / 'drawing.svg').write_text(SVG)
        for number in (1, 2):
            result = subprocess.run([sys.executable, str(board / 'leaderboard.pyc'), 'drawing.svg'],
                                    cwd=tmp_path, capture_output=True, text=True, check=True)
            assert '6. drawing.svg' in result.stdout
            assert f'Submissions: {number}' in result.stdout
            assert '/100' not in result.stdout
        assert len(rnd.calls) == 2
        assert rnd.calls[0]['sha256'] == rnd.calls[1]['sha256']
        assert all(set(call) == {'number', 'sha256', 'rank'} for call in rnd.calls)
        logs = list((tmp_path / 'logs').glob('*.json'))
        assert len(logs) == 2
        for log in logs:
            assert json.loads(log.read_text())['rubric'] == next(iter(rows[0]['files_abs']))
        assert server.infrastructure_errors == []
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize('svg', [
    'not XML', '<html/>', '<svg><script/></svg>',
    '<svg><image href="https://example.com/image.png"/></svg>',
    '<svg><use href="file:///etc/passwd"/></svg>',
    '<!DOCTYPE svg><svg/>',
])
def test_invalid_svg_does_not_increment_counter(svg):
    rnd = leaderboard_server.Round({'files_abs': {'/rubric.json': 'unused'}})
    with pytest.raises((ValueError, leaderboard_server.ET.ParseError)):
        rnd.submit(svg)
    assert rnd.calls == []
    assert rnd.submit(SVG)['submissions'] == 1


def test_concurrent_submissions_and_round_isolation():
    first = leaderboard_server.Round({'files_abs': {'/first/rubric.json': 'unused'}})
    second = leaderboard_server.Round({'files_abs': {'/second/rubric.json': 'unused'}})
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(first.submit, [SVG] * 8))
    assert sorted(result['submissions'] for result in results) == list(range(1, 9))
    assert [call['number'] for call in first.calls] == list(range(1, 9))
    assert second.submit(SVG)['submissions'] == 1


def test_http_authentication_does_not_consume_submissions():
    rnd = leaderboard_server.Round({'files_abs': {'/rubric.json': 'unused'}})
    server, configs = leaderboard_server.start([rnd])
    try:
        endpoint = configs[0][1]
        request = Request(endpoint['endpoint'], data=json.dumps({'submission': SVG}).encode(),
                          headers={'Authorization': 'Bearer wrong-token'})
        with pytest.raises(HTTPError) as exc:
            urlopen(request, timeout=5)
        assert exc.value.code == 401
        assert rnd.calls == []
    finally:
        server.shutdown()
        server.server_close()
