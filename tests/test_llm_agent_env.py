"""Environment-selected routes reach both clients without becoming model parameters."""
from unittest.mock import Mock

import pytest
from core import llm_agents, routing
import run


@pytest.mark.parametrize('cls', [llm_agents.OpenAIAgent, llm_agents.GrokAgent,
                                llm_agents.GeminiAgent, llm_agents.OpenRouterAgent,
                                llm_agents.AnthropicAgent])
def test_custom_route_reaches_sync_and_async_clients(monkeypatch, cls):
    monkeypatch.setenv('TEST_ROUTE_KEY', 'test-key')
    monkeypatch.setenv('TEST_ROUTE_URL', 'https://gateway.example/v1')
    sdk = llm_agents.anthropic if cls is llm_agents.AnthropicAgent else llm_agents.openai
    names = ('Anthropic', 'AsyncAnthropic') if cls is llm_agents.AnthropicAgent else ('OpenAI', 'AsyncOpenAI')
    clients = [Mock(), Mock()]
    for name, client in zip(names, clients):
        monkeypatch.setattr(sdk, name, client)
    agent = cls('test-model', api_key_env='TEST_ROUTE_KEY',
                api_base_url_env='TEST_ROUTE_URL', api_base_url='https://fallback.example',
                temperature=0.4)
    for client in clients:
        assert client.call_args.kwargs['api_key'] == 'test-key'
        assert client.call_args.kwargs['base_url'] == 'https://gateway.example/v1'
    assert agent.generation_config == {'temperature': 0.4}
    monkeypatch.setenv('TEST_ROUTE_URL', 'https://second.example/v1')
    cls('test-model', api_key_env='TEST_ROUTE_KEY', api_base_url_env='TEST_ROUTE_URL')
    assert clients[0].call_args.kwargs['base_url'] == 'https://second.example/v1'


@pytest.mark.parametrize('value', [None, '', '   ', 'not-a-url'])
def test_missing_or_invalid_url_variable_does_not_fall_back(monkeypatch, value):
    monkeypatch.delenv('TEST_ROUTE_URL', raising=False)
    if value is not None:
        monkeypatch.setenv('TEST_ROUTE_URL', value)
    with pytest.raises(ValueError, match='TEST_ROUTE_URL'):
        routing.resolve_base_url('https://fallback.example', 'TEST_ROUTE_URL')
    with pytest.raises(SystemExit, match='TEST_ROUTE_URL'):
        routing.resolve('test', {'api_base_url_env': 'TEST_ROUTE_URL'}, where='test')


def test_model_config_resolves_url_env_for_native_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv('TEST_ROUTE_URL', 'https://gateway.example/v1')
    config = tmp_path / 'models.yaml'
    config.write_text('models:\n  test:\n    model: openrouter/z-ai/glm-5.3\n'
                      '    api_key_env: TEST_ROUTE_KEY\n'
                      '    api_base_url_env: TEST_ROUTE_URL\n')
    result = run.load_config(config, 'test')
    assert result['api_key_env'] == 'TEST_ROUTE_KEY'
    assert result['api_base_url'] == 'https://gateway.example/v1'
    assert 'api_base_url_env' not in result['generation_config']
