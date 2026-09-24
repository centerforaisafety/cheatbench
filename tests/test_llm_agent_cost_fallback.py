"""Deterministic cost fallback for direct OpenAI-compatible providers."""
from types import SimpleNamespace

import litellm
import pytest

from core.llm_agents import OpenAIAgent


def test_vendor_price_and_cache_reads_are_preserved(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'offline')
    monkeypatch.setattr(litellm, 'model_cost', {
        'demo': {'litellm_provider': 'vendor'},
        'vendor/demo': {'input_cost_per_token': 0.001, 'output_cost_per_token': 0.002,
                        'cache_read_input_token_cost': 0.0001},
        'reseller/vendor/demo': {'input_cost_per_token': 0.1, 'output_cost_per_token': 0.2},
    })
    def missing(*args, **kwargs):
        raise ValueError('provider-scoped lookup unavailable')
    monkeypatch.setattr(litellm.cost_calculator, 'completion_cost', missing)
    agent = OpenAIAgent(model='demo', api_base_url='https://example.invalid/v1')
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=40), cost=None)
    assert agent._calculate_cost(SimpleNamespace(model='demo', usage=usage)) == pytest.approx(0.084)
    usage.cost = 0.012
    assert agent._calculate_cost(SimpleNamespace(model='demo', usage=usage)) == 0.012


@pytest.mark.parametrize('asynchronous', [False, True])
def test_missing_usage_preserves_completion_without_inventing_cost(monkeypatch, asynchronous):
    import asyncio
    from openai.types.chat import ChatCompletion
    monkeypatch.setenv('OPENAI_API_KEY', 'offline')
    agent = OpenAIAgent(model='demo', api_base_url='https://example.invalid/v1')
    response = ChatCompletion.model_validate({
        'id': 'test', 'created': 0, 'model': 'demo', 'object': 'chat.completion',
        'choices': [{'index': 0, 'finish_reason': 'stop',
                     'message': {'role': 'assistant', 'content': '0.4'}}],
    })
    async def create_async(**kwargs):
        return response
    monkeypatch.setattr(agent.client.chat.completions, 'create', lambda **kwargs: response)
    monkeypatch.setattr(agent.async_client.chat.completions, 'create', create_async)
    def no_pricing(*args):
        pytest.fail('Missing token usage must not be priced')
    monkeypatch.setattr(agent, '_calculate_cost', no_pricing)
    messages = [{'role': 'user', 'content': 'Estimate the probability.'}]
    result = asyncio.run(agent.async_completions(messages)) if asynchronous else agent.completions(messages)
    assert result.content == '0.4'
    assert result.token_usage is None
    assert result.raw['choices'][0]['message']['content'] == '0.4'
    assert result.raw['usage'] is None
    assert agent.all_token_usage.total_tokens == 0
