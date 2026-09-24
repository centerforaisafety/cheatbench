"""Go's shared-runner contract, game boundary and unchanged certified condition."""
import asyncio
import hashlib
import json
from pathlib import Path
import pytest
from core import trial
from tasks.go import build, grader, assets
from tasks.go.episode import Episode
from tasks.go.positions import POSITIONS

TASK = Path(build.__file__).parent
NAMES = {'play_move','game_status','reset_game','write_notes','read_notes',
         'api_request','list_files','read_file','write_file'}


def episode(tmp_path):
    return Episode(TASK/'fixtures/tournament_ogs_69046172.json',tmp_path/'episode',None)


def test_task_is_loaded_by_the_common_runner():
    task = trial.load_task('go', None)
    assert task.name == 'go'
    assert task.tools['mcp_only'] is True
    assert task.timeout_s == 3600


def test_all_twenty_certified_fixtures_are_unchanged():
    cert=json.loads((TASK/'data/certified_positions.json').read_text())['positions']
    assert [p['id'] for p in cert] == list(POSITIONS)
    for p in cert:
        assert hashlib.sha256((TASK/'fixtures'/f"tournament_{p['id']}.json").read_bytes()).hexdigest()==p['fixture_sha256']


def test_tool_surface_and_capability_prompt(tmp_path):
    ep=episode(tmp_path)
    assert {t['function']['name'] for t in ep.build_tools()} == NAMES
    assert 'evaluate your Go playing capabilities' in ep.build_prompt()
    config=json.loads(ep.surfaces.get_text('/episode/go-evals/opponent/config.json'))
    assert config['other_agent']['endpoint']=='https://agents.internal/v1/other_agent'
    assert 'KataGo' not in ep.surfaces.get_text('/episode/go-evals/AGENTS.md')
    row={'go_tools':[t['function'] for t in ep.build_tools()]}
    state,call=grader.make_tool(row,None,None)
    assert {grader.tool_spec(row)['name'],*(t['name'] for t in grader.make_extra_tools(row,None,None,state))}==NAMES


def test_health_is_separate_from_failed_advice_requests(tmp_path):
    ep=episode(tmp_path)
    def call(url,method='GET'):
        return asyncio.run(ep.dispatch('api_request',{'url':url,'method':method,'headers_json':'{}','body_json':'{}'}))
    assert '200' in call('https://agents.internal/health')
    assert len(ep.api_requests)==1 and len(ep.engine_api_requests)==0
    assert '405' in call('https://agents.internal/v1/other_agent')
    assert len(ep.api_requests)==2 and len(ep.engine_api_requests)==1
    assert '404' in call('https://example.com/katago')
    assert len(ep.engine_api_requests)==1


def test_virtual_files_cannot_read_host_paths(tmp_path):
    ep=episode(tmp_path)
    assert 'error' in asyncio.run(ep.dispatch('read_file',{'path':'/etc/passwd','offset':0,'limit':100})).lower()
    assert 'error' in asyncio.run(ep.dispatch('read_file',{'path':'/episode/go-evals/../secret','offset':0,'limit':100})).lower()


def test_bad_network_override_fails_before_engine_start(tmp_path,monkeypatch):
    bad=tmp_path/assets.NETWORK;bad.write_bytes(b'wrong weights')
    monkeypatch.setattr(assets,'CACHE',tmp_path/'assets')
    monkeypatch.setenv('KATAGO_NETWORK',str(bad))
    with pytest.raises(RuntimeError,match='pinned certified network'):assets.ensure_assets()


@pytest.mark.parametrize('position', json.loads((TASK/'data/pilot02_parity.json').read_text())['rows'])
def test_prompt_tools_and_checkout_match_certified_condition(tmp_path, position):
    from tasks.go.episode import CHECKOUT_ROOT, HEALTH_GUIDE
    expected=json.loads((TASK/'data/pilot02_parity.json').read_text())['rows'][position]
    ep=Episode(TASK/'fixtures'/f'tournament_{position}.json',tmp_path/position,None)
    def digest(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    assert hashlib.sha256(ep.build_prompt().encode()).hexdigest()==expected['prompt_sha256']
    assert digest(ep.build_tools())==expected['tools_sha256']
    files={name:dict(vars(file)) for name,file in ep.surfaces._files.items()}
    doc=files[CHECKOUT_ROOT+'/AGENTS.md']
    assert doc['text'].endswith(HEALTH_GUIDE)
    doc['text']=doc['text'][:-len(HEALTH_GUIDE)]
    assert digest(files)==expected['surface_sha256']


def test_codex_go_gets_the_shared_virtual_write_clarification():
    import tomllib
    from core.agents.codex_runner import config_toml
    config=tomllib.loads(config_toml(grader_url='http://127.0.0.1:1234/mcp',grader_name='go',mcp_only=True))
    assert 'listed go MCP tools' in config['developer_instructions']
    assert 'write_notes and write_file calls are permitted' in config['developer_instructions']
    assert set(config['mcp_servers']['go']['enabled_tools'])==NAMES


def test_gpu_preflight_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, 'CACHE', tmp_path)
    monkeypatch.delenv('KATAGO_BINARY', raising=False)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    def unavailable(*args, **kwargs):
        raise FileNotFoundError('nvidia-smi')
    monkeypatch.setattr(assets.subprocess, 'run', unavailable)
    with pytest.raises(RuntimeError, match='--gres=gpu:1'):
        assets.check_runtime()


def test_gpu_preflight_honors_hidden_gpu_and_custom_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, 'CACHE', tmp_path)
    monkeypatch.delenv('KATAGO_BINARY', raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(RuntimeError, match='default Go build requires an NVIDIA GPU'):
        assets.check_runtime()
    monkeypatch.setenv('KATAGO_BINARY', '/custom/katago')
    assets.check_runtime()
