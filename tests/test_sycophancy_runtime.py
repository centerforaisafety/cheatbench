import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import yaml

from core.chat_config import load_chat_config
from core.llm_agents import LLMResponse, TokenUsage
from tasks.sycophancy import run_chat, score

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/sycophancy'
MODELS = ['claude-fable-5-1', 'claude-opus-5', 'gpt-6-astra', 'gpt-5.6-sol', 'gpt-6-sol',
          'gemini-3.8-flash', 'grok-4.6', 'kimi-k3', 'deepseek-v4-pro', 'muse-spark-1.3', 'glm-5.3']


@pytest.fixture
def endpoint_env(monkeypatch):
    for name in ['ANTHROPIC_BASE_URL', 'OPENAI_BASE_URL', 'GEMINI_BASE_URL',
                 'LITELLM_BASE_URL', 'DEEPSEEK_BASE_URL']:
        monkeypatch.setenv(name, 'https://example.invalid/v1')
    for name in ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'LITELLM_API_KEY',
                 'DEEPSEEK_API_KEY', 'META_API_KEY']:
        monkeypatch.setenv(name, 'offline-test')
    monkeypatch.setenv('CLAUDE_CODE_MAX_OUTPUT_TOKENS', '128000')


@pytest.mark.parametrize('name', MODELS)
def test_canonical_model_builds_valid_sdk_request(name, endpoint_env):
    cfg = run_chat.load_config(ROOT / 'configs/models.yaml', name)
    # Constructor validates provider/model compatibility too; no HTTP request.
    client, gen = run_chat.make_model(name, cfg)
    assert client.generation_config == gen
    if name.startswith('claude-'):
        assert gen['max_tokens'] == run_chat.ANTHROPIC_MAX_TOKENS
        assert gen['thinking']['type'] == 'adaptive'
    if name == 'claude-fable-5-1':
        assert gen['output_config']['effort'] == 'high'
    if name in ('gpt-6-astra', 'gpt-5.6-sol'):
        assert gen['reasoning_effort'] == 'high'
        assert cfg['unsupported'] == {}
    if name == 'kimi-k3':
        assert gen['extra_body']['thinking'] == {'type': 'enabled', 'effort': 'high'}
        assert gen['extra_body']['provider']['order'] == ['moonshotai']


def test_override_inherits_updated_reasoning_and_routing(tmp_path, endpoint_env):
    base = tmp_path / 'models.yaml'
    base.write_text('models:\n  claude-fable-5-1:\n    model: anthropic/claude-fable-5-1\n'
                    '    api_key_env: ANTHROPIC_API_KEY\n    api_base_url: https://example.invalid/changed\n'
                    '    generation_config: {effort: max, thinking: {type: adaptive}}\n')
    cfg = load_chat_config(base, 'claude-fable-5-1', TASK / 'models.yaml')
    assert cfg['generation_config']['output_config']['effort'] == 'max'
    assert cfg['api_base_url'] == 'https://example.invalid/changed'


def test_no_hidden_claude_budget(endpoint_env, monkeypatch):
    monkeypatch.delenv('CLAUDE_CODE_MAX_OUTPUT_TOKENS')
    with pytest.raises(ValueError, match='CLAUDE_CODE_MAX_OUTPUT_TOKENS'):
        load_chat_config(ROOT / 'configs/models.yaml', MODELS[0], TASK / 'models.yaml')


@pytest.fixture
def local_run(tmp_path, monkeypatch):
    task = tmp_path / 'sycophancy'
    shutil.copytree(TASK, task, ignore=shutil.ignore_patterns('environment', 'results', '__pycache__', '.build.lock'))
    config = tmp_path / 'models.yaml'
    config.write_text('models:\n  test:\n    model: openai/test\n'
                      '    api_key_env: TEST_KEY\n    generation_config: {reasoning_effort: high}\n')
    monkeypatch.setenv('TEST_KEY', 'offline')
    calls = []
    class Model:
        async def async_completions(self, messages):
            calls.append(messages)
            return LLMResponse(content='0.4', token_usage=TokenUsage(input_tokens=10, output_tokens=2, cost=0.001))
    monkeypatch.setattr(run_chat, 'make_model', lambda name, cfg: (Model(), cfg['generation_config']))
    monkeypatch.setattr(run_chat.judging, 'make_judge_agent', lambda *a: object())
    async def judge(transcript, *args):
        assert transcript['messages'][0]['content'][0]['text'] == '0.4'
        return {'parse_ok': True, 'verdict': {'cheated': False, 'estimate': 0.4, 'committed': True, 'summary': '0.4'},
                'output_mode': 'structured', 'judge_cost_usd': 0.001, 'judge_attempts': 1, 'judge_error': None}
    monkeypatch.setattr(run_chat.judging, 'judge_episode', judge)
    output = tmp_path / 'outside-repository'
    argv = ['run_chat.py', str(task), '--model', 'test', '--judge', 'test', '--repeat', '2',
            '--run-id', 'example', '--models-config', str(config),
            '--output-dir', str(output), '--no-progress']
    monkeypatch.setattr(sys, 'argv', argv)
    return SimpleNamespace(task=task, config=config, output=output, calls=calls, argv=argv)


def test_full_build_run_extract_score_and_resume(local_run):
    r = local_run
    assert not (r.task / 'results').exists()
    assert run_chat.main() == 0
    assert len(r.calls) == 56
    directories = [r.output / 'sycophancy_all_test_example']
    bubble, forecast = score.score_run(directories)
    assert bubble['baseline_n'] == 2 and bubble['exposed_n'] == 12
    assert len(forecast['questions']) == 7
    meta = json.loads((directories[0] / 'run.json').read_text())
    assert meta['generation_config'] == {'reasoning_effort': 'high'}
    assert all(meta[k] for k in ('prompt_sha256', 'row_sha256', 'judge_prompt_sha256', 'judge_schema_sha256'))
    assert run_chat.main() == 0
    assert len(r.calls) == 56


@pytest.mark.parametrize('change', ['prompt', 'data', 'model', 'judge_schema'])
def test_changed_content_is_refused_before_new_calls(local_run, change):
    r = local_run
    assert run_chat.main() == 0
    if change == 'prompt':
        p = r.task / 'prompts/instruction.md'
    elif change == 'data':
        p = r.task / 'environment/all/data.jsonl'
    elif change == 'judge_schema':
        p = r.task / 'judge_schema.py'
    else:
        p = r.config
    if change == 'model':
        p.write_text(p.read_text().replace('reasoning_effort: high', 'reasoning_effort: low'))
    else:
        if change == 'data':
            rows = [json.loads(line) for line in p.read_text().splitlines()]
            rows[0]['input'] += ' Changed belief.'
            p.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        else:
            p.write_text(p.read_text() + '\n')
    with pytest.raises(SystemExit, match='different inputs'):
        run_chat.main()
    assert len(r.calls) == 56


def test_failed_provider_returns_nonzero(local_run, monkeypatch):
    class Failed:
        async def async_completions(self, *args):
            raise RuntimeError('offline failure')
    monkeypatch.setattr(run_chat, 'make_model', lambda name, cfg: (Failed(), cfg['generation_config']))
    local_run.argv.extend(['--limit', '1'])
    assert run_chat.main() == 1


def test_partial_forecast_is_not_presented_as_complete(tmp_path):
    episodes = [{'episode': 'a', 'vars': {'question': 'agi', 'condition': 'baseline'}}]
    (tmp_path / 'episodes.jsonl').write_text(json.dumps(episodes[0]) + '\n')
    (tmp_path / 'judge.jsonl').write_text(json.dumps({'episode': 'a', 'estimate': 0.4, 'parse_ok': True}) + '\n')
    with pytest.raises(ValueError, match='Incomplete forecast'):
        score.score_run([tmp_path])


def test_latest_judgment_only(tmp_path):
    (tmp_path / 'episodes.jsonl').write_text(json.dumps({'episode': 'a', 'vars': {'condition': 'baseline'}}) + '\n')
    (tmp_path / 'judge.jsonl').write_text('\n'.join(json.dumps({'episode': 'a', 'estimate': n}) for n in [0.2, 0.4]))
    assert score.numbers(tmp_path)[('', 'baseline')] == [40.0]


@pytest.mark.parametrize('name', MODELS)
def test_real_sdk_serializes_request_with_offline_transport(name, endpoint_env):
    import httpx
    import anthropic
    if name.startswith('claude-'):
        from anthropic import _base_client
        httpx = getattr(_base_client, 'httpx2', httpx)
    import openai

    cfg = run_chat.load_config(ROOT / 'configs/models.yaml', name)
    client, gen = run_chat.make_model(name, cfg)
    requests = []
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        expected_content = [{'type': 'text', 'text': 'test'}] if cfg['model'].startswith('anthropic/') else 'test'
        assert body['messages'] == [{'role': 'user', 'content': expected_content}]
        assert body['model'] == cfg['factory_model'].split('/', 1)[1]
        for key, value in gen.items():
            if key == 'extra_body':
                for extra_key, extra_value in value.items():
                    assert body[extra_key] == extra_value
            else:
                assert body[key] == value
        if cfg['model'].startswith('anthropic/'):
            data = {'id': 'msg_test', 'type': 'message', 'role': 'assistant', 'model': body['model'],
                    'content': [{'type': 'text', 'text': '0.4'}], 'stop_reason': 'end_turn',
                    'usage': {'input_tokens': 2, 'output_tokens': 1}}
        else:
            data = {'id': 'chatcmpl-test', 'object': 'chat.completion', 'created': 1, 'model': body['model'],
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '0.4'}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3, 'cost': 0}}
        return httpx.Response(200, json=data)
    async def call():
        transport = httpx.MockTransport(respond)
        sdk = anthropic.AsyncAnthropic if cfg['model'].startswith('anthropic/') else openai.AsyncOpenAI
        original = client.async_client
        client.async_client = sdk(api_key='offline', base_url='https://example.invalid/v1',
                                  http_client=httpx.AsyncClient(transport=transport), timeout=10, max_retries=0)
        try:
            reply = await asyncio.wait_for(client.async_completions([{'role': 'user', 'content': 'test'}]), timeout=10)
            assert reply.content == '0.4'
        finally:
            await client.async_client.close()
            await original.close()
    asyncio.run(call())
    assert len(requests) == 1


def test_no_judge_does_not_require_judge_configuration(local_run, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('judge config should not be loaded')
    monkeypatch.setattr(run_chat.judging, 'load_judge_config', fail)
    local_run.argv.extend(['--no-judge', '--limit', '1'])
    assert run_chat.main() == 0
    assert len(local_run.calls) == 2


def test_failed_judge_returns_nonzero_and_is_not_hidden_on_resume(local_run, monkeypatch):
    async def failed(*args):
        raise RuntimeError('review service unavailable')
    monkeypatch.setattr(run_chat.judging, 'judge_episode', failed)
    local_run.argv.extend(['--limit', '1'])
    assert run_chat.main() == 1
    assert len(local_run.calls) == 2
    assert run_chat.main() == 1
    assert len(local_run.calls) == 2


def test_judge_initialization_failure_prevents_provider_calls(local_run, monkeypatch):
    def failed(*args):
        raise RuntimeError('missing judge credential')
    monkeypatch.setattr(run_chat.judging, 'make_judge_agent', failed)
    assert run_chat.main() == 1
    assert not local_run.calls


def test_manifest_rejects_partial_run_before_scoring(local_run):
    assert run_chat.main() == 0
    directory = local_run.output / 'sycophancy_all_test_example'
    path = directory / 'judge.jsonl'
    path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
    with pytest.raises(ValueError, match='missing or failed judgments'):
        score.validate_run(directory)
