import json
from pathlib import Path

import pytest

from core.agents.deepseek_harness import DeepSeekHarnessAgent
from core.agents import deepseek_harness_runner as runner
from core.agents.factory import make_agent


def adapter(**kwargs):
    return DeepSeekHarnessAgent(model='deepseek-v4-pro', version='0.1.5rc1', **kwargs)


def test_official_install_config_and_registration():
    agent = adapter()
    agent.apply_tool_policy({'web_search': False, 'web_fetch': False})
    assert agent.setup() == []
    payload = json.loads(agent.blob(row={'id': 'test'}, prompt='unchanged', files=[], skip_dirs=(),
                                   stage_src='', timeout_s=20, env={}))
    assert payload['task']['content'] == 'unchanged'
    assert payload['task']['deepseek_harness']['base_url'] == 'https://api.deepseek.com/v1'
    assert 'deepseek-harness-sdk==0.1.5rc1' in agent.install()['install']
    assert 'deepseek-harness-runtime-bin' in agent.install()['check']
    assert 'vegamo' not in agent.install()['install']
    assert make_agent('deepseek-harness', model='deepseek-v4-pro').name() == 'deepseek-harness'


@pytest.mark.parametrize('config', [{'reasoning_effort': 'bogus'}, {'max_tokens': 0},
                                    {'max_tokens': True}, {'thinking_enabled': True}])
def test_invalid_generation_rejected(config):
    agent = adapter(generation_config=config)
    agent.apply_tool_policy({})
    assert agent.setup()


@pytest.mark.parametrize('policy', [{'web_search': True}, {'web_fetch': True}, {'unknown': False}])
def test_unsupported_policy_rejected(policy):
    with pytest.raises(ValueError):
        adapter().apply_tool_policy(policy)


def test_native_capture_converts_user_reasoning_call_and_result():
    raw = json.loads((Path(__file__).parent / 'fixtures/deepseek_harness/bash.json').read_text())
    trajectory = adapter().to_trajectory(raw)
    assert trajectory.steps[0].message == 'Create proof.txt using bash.'
    step = next(s for s in trajectory.steps if s.tool_calls)
    assert step.reasoning_content == 'I will create the requested file using the shell.'
    assert step.tool_calls[0].function_name == 'bash'
    assert step.tool_calls[0].arguments['command'] == 'printf official-harness > proof.txt'
    assert step.observation.results[0].content == '(no output)'
    assert trajectory.steps[-1].message == 'Created proof.txt.'
    assert trajectory.final_metrics.total_prompt_tokens == 200
    assert trajectory.notes is None


def test_descendant_call_ids_do_not_collide_and_ptc_is_retained():
    raw = json.loads((Path(__file__).parent / 'fixtures/deepseek_harness/bash.json').read_text())
    child = json.loads(json.dumps(raw['messages']))
    for e in child:
        e['session_id'] = 'child'
    raw['messages'] += child + [{'session_id': 'child', 'type': 'tool/ptc-dispatch',
        'data': {'subCallId': 'nested', 'name': 'read', 'arguments': {'file_path': '/tmp/gps/a.gpx'},
                 'content': [{'type': 'text', 'text': 'reference answer'}]}}]
    traj = adapter().to_trajectory(raw)
    calls = [c.tool_call_id for s in traj.steps for c in s.tool_calls or []]
    assert len(calls) == len(set(calls)) == 3
    assert traj.steps[-1].observation.results[0].content == 'reference answer'
    assert traj.notes is None


def test_subagent_notice_preserves_quoted_reasoning_as_received_content():
    raw = {'messages': [{'session_id': 'parent', 'type': 'user/message', 'data': {
        'message': {'id': 'notice', 'source': {'kind': 'subagent-settled'}, 'content': [
            {'type': 'text', 'text': 'The child finished. Its closing message:'},
            {'type': 'reasoning', 'text': 'I inspected the reference document.'},
            {'type': 'text', 'text': 'Here are the findings.'},
        ]}}}]}
    step = adapter().to_trajectory(raw).steps[0]
    assert step.source == 'user'
    assert step.reasoning_content is None
    assert step.llm_call_count is None
    assert step.message == (
        'The child finished. Its closing message:'
        '\n[reasoning block in received message]\n'
        'I inspected the reference document.\nHere are the findings.')
    from core.render import render_trajectory
    rendered = render_trajectory(adapter().to_trajectory(raw))
    assert 'I inspected the reference document.' in rendered
    assert 'THINKING:' not in rendered


def test_policy_patch_and_gateway_fail_closed():
    patch = runner.profile_patch()
    assert {'tool-web', 'web-search-deepseek', 'web-fetch-http'} <= {x['id'] for x in patch if x.get('disabled')}
    original = {'messages': [{'role': 'user', 'content': 'unchanged'}],
                'tools': [{'function': {'name': 'bash'}}]}
    assert runner.filter_request(original) is original
    with pytest.raises(ValueError, match='Disabled native web'):
        runner.filter_request({'tools': [{'function': {'name': 'web_search'}}]})


def test_observer_receives_native_actions():
    raw = json.loads((Path(__file__).parent / 'fixtures/deepseek_harness/bash.json').read_text())
    records = [x for e in raw['messages'] for x in runner.observer_messages(e)]
    assert any(b.get('name') == 'bash' for x in records for b in x['content'])
    assert any(b.get('_type') == 'ToolResultBlock' for x in records for b in x['content'])


@pytest.mark.parametrize('termination', ['timeout', 'max_turns'])
def test_watchdog_failure_is_a_judgeable_limit(tmp_path, monkeypatch, termination):
    import sys
    import threading
    from types import SimpleNamespace
    from core.agents import errors
    from core.trial import classify_failure

    closed = threading.Event()

    class Harness:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def run(self, *args, **kwargs):
            assert closed.wait(2), 'watchdog did not close the harness'
            raise RuntimeError('runtime closed by watchdog')

        def close(self):
            closed.set()

    class Gateway:
        def __init__(self, *args):
            self.limit_hit = threading.Event()
            if termination == 'max_turns':
                self.limit_hit.set()
            self.url, self.token = 'http://127.0.0.1', 'local-test'
            self.error, self.calls, self.tools = None, [], []

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, 'deepseek_harness', SimpleNamespace(DeepSeekHarness=Harness))
    monkeypatch.setattr(runner, 'Gateway', Gateway)
    monkeypatch.setattr(runner, 'WORKDIR', str(tmp_path))
    monkeypatch.setenv('DSH_TEST_KEY', 'local-test')
    task = {'id': 'watchdog', 'model': 'deepseek-v4-pro', 'content': 'test',
            'timeout_s': 0 if termination == 'timeout' else 30,
            'deepseek_harness': {'api_key_env': 'DSH_TEST_KEY', 'base_url': 'http://127.0.0.1',
                                'generation': {'reasoning_effort': 'high'}}}
    record = runner.run(task, {'task_install_env': lambda: {},
                              'collect_deliverables': lambda *args: []},
                        {'Upstream': lambda *args, **kwargs: None})
    assert record['terminal_reason'] == termination
    expected = errors.EpisodeTimeoutError if termination == 'timeout' else errors.MaxTurnsError
    assert classify_failure(adapter(), record['error'], None, record, tmp_path/'no-stderr') is expected
