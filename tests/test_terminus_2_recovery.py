"""Transport recovery preserves payload content and retries only known faults."""
import ast
import asyncio
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def runtime():
    # The extension is injected into the pinned Harbor interpreter at runtime.
    # Exercise its wire boundary with a scripted provider, without model calls.
    source = Path(__file__).resolve().parents[1] / 'core/agents/terminus_2_images.py'
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and n.name in {'response_content', 'ImageLiteLLM'}]
    class ContextLengthExceededError(Exception):
        pass
    class Provider:
        async def call(self, prompt, history, response_format, logging_path, **kwargs):
            self.sent.append(deepcopy((prompt, history, kwargs)))
            result = self.replies.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        _call_responses = call
    ns = {'LiteLLM': Provider, 'ContextLengthExceededError': ContextLengthExceededError}
    exec(compile(tree, str(source), 'exec'), ns)
    client = ns['ImageLiteLLM']()
    client.sent = []
    return client, ContextLengthExceededError


def test_empty_reply_recovery_preserves_content_and_original_history(runtime):
    client, _ = runtime
    history = [{'role': 'user', 'content': 'Original task rules'},
               {'role': 'assistant', 'content': ''},
               {'role': 'assistant', 'content': '', 'reasoning_content': 'real reasoning'},
               {'role': 'assistant', 'content': '', 'tool_calls': [{'name': 'read'}]},
               {'role': 'assistant', 'content': 'valid reply'}]
    original = deepcopy(history)
    client.replies = [ValueError("the message with role 'assistant' must not be empty"), 'ok']
    assert asyncio.run(client.call('next', history)) == 'ok'
    assert history == original
    assert client.sent[1][1] == [history[0], *history[2:]]
    assert client.transport_recoveries == {'empty_assistant_message': 1}
    client.replies = ['next reply']
    assert asyncio.run(client.call('next turn', history)) == 'next reply'
    assert len(client.sent) == 3
    assert client.sent[-1][1] == client.sent[1][1]
    assert history == original


def test_expired_response_replays_full_image_history_without_chain(runtime):
    client, _ = runtime
    history = [{'role': 'user', 'content': 'Original task rules'},
               {'role': 'assistant', 'content': 'Previous answer'}]
    prompt = [{'type': 'text', 'text': 'Image'},
              {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA', 'detail': 'auto'}}]
    original = deepcopy((history, prompt))
    client.replies = [ValueError('referenced response not found or expired'), 'ok']
    assert asyncio.run(client._call_responses(prompt, history, previous_response_id='expired')) == 'ok'
    assert (history, prompt) == original
    sent_prompt, sent_history, kwargs = client.sent[1]
    assert kwargs['previous_response_id'] is None
    assert sent_history == history
    assert sent_prompt[1] == {'type': 'input_image', 'image_url': 'data:image/png;base64,AA', 'detail': 'auto'}
    assert client.transport_recoveries == {'expired_response_chain': 1}


def test_image_size_error_enters_existing_compaction_once(runtime):
    client, overflow = runtime
    client.replies = [ValueError('Downloaded image content cannot exceed 30MB')] * 2
    with pytest.raises(overflow):
        asyncio.run(client.call('image'))
    assert client._byte_overflow_pending
    with pytest.raises(ValueError):
        asyncio.run(client.call('image'))
    assert client.transport_recoveries == {'request_size_compaction': 1}


def test_other_errors_are_not_retried(runtime):
    client, _ = runtime
    client.replies = [ValueError('invalid API credential')]
    with pytest.raises(ValueError, match='credential'):
        asyncio.run(client.call('prompt'))
    assert len(client.sent) == 1
