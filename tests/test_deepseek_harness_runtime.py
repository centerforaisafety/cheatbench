"""Real official runtime, local mock API only. Opt in with DSH_RUNTIME_TESTS=1."""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from core.agents import deepseek_harness_runner as runner, forwarder, runner_support
from core.agents.deepseek_harness import DeepSeekHarnessAgent

pytestmark = pytest.mark.skipif(os.environ.get('DSH_RUNTIME_TESTS') != '1', reason='opt-in real SDK integration')


@pytest.mark.parametrize('mode', ['bash', 'mcp', 'routed', 'max_tokens', 'max_turns', 'timeout'])
def test_official_runtime(tmp_path, monkeypatch, mode):
    pytest.importorskip('deepseek_harness')
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            if mode == 'timeout':
                time.sleep(10)
            if len(requests) == 1 and mode in ('bash', 'mcp', 'routed', 'max_turns'):
                tool = 'mcp__grader__check' if mode == 'mcp' else 'bash'
                arguments = {'value': 42} if mode == 'mcp' else {'command': 'printf proof > proof.txt', 'description': 'Write proof'}
                delta = {'role': 'assistant', 'reasoning_content': 'I will use the requested tool.',
                         'tool_calls': [{'index': 0, 'id': 'native-call', 'type': 'function',
                                         'function': {'name': tool, 'arguments': json.dumps(arguments)}}]}
                finish = 'tool_calls'
            else:
                delta, finish = {'role': 'assistant', 'content': 'Done.'}, 'length' if mode == 'max_tokens' else 'stop'
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            try:
                for event in [{'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
                              {'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}],
                               'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'prompt_cache_hit_tokens': 10}}]:
                    self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
                self.wfile.write(b'data: [DONE]\n\n')
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    grader = None
    if mode == 'mcp':
        state = {'seen': []}
        def check(arguments):
            state['seen'].append(arguments)
            return {'content': [{'type': 'text', 'text': 'grade received'}]}
        grader = runner_support.GraderServer(state=state, call=check, server_name='grader',
            tool_name='check', description='Check the provided value',
            schema={'type': 'object', 'properties': {'value': {'type': 'integer'}}, 'required': ['value']})
        grader.start()
    monkeypatch.setattr(runner, 'WORKDIR', str(tmp_path))
    monkeypatch.setenv('DSH_MOCK_KEY', 'local-mock-only')
    model = 'openrouter/deepseek/deepseek-v4-pro' if mode == 'routed' else 'deepseek-v4-pro'
    task = {'id': 'runtime-test', 'model': model, 'content': 'Do the task.',
            'deepseek_harness': {'api_key_env': 'DSH_MOCK_KEY',
                'base_url': f'http://127.0.0.1:{server.server_port}/v1',
                'extra_body': {'provider': {'order': ['deepseek'], 'allow_fallbacks': False}} if mode == 'routed' else {},
                'generation': {'reasoning_effort': 'high'}, 'max_turns': 1 if mode == 'max_turns' else 4},
            'timeout_s': 5 if mode == 'timeout' else 25, 'deliverable_files': ['proof.txt']}
    support = {k: getattr(runner_support, k) for k in ('task_install_env', 'collect_deliverables')}
    support['_emit_msg'] = lambda x: None
    try:
        record = runner.run(task, support, {'Upstream': forwarder.Upstream}, grader)
    finally:
        server.shutdown()
        server.server_close()
        if grader:
            grader.close()
    assert requests
    for request in requests:
        tools = {x['function']['name'] for x in request.get('tools', [])}
        assert not tools & {'web_search', 'web_fetch'}
        assert request['model'] == model
        if mode == 'routed':
            assert request['provider'] == {'order': ['deepseek'], 'allow_fallbacks': False}
        assert request['reasoning_effort'] == 'high'
        assert request['thinking']['type'] == 'enabled'
    if mode in ('bash', 'mcp', 'routed'):
        assert record['error'] is None, record['error']
        assert record['terminal_reason'] == 'completed'
        assert record['final_text'] == 'Done.'
        assert record['n_tool_calls'] == 1
        assert record['usage']['prompt_tokens'] == 200
        trajectory = DeepSeekHarnessAgent(model='deepseek-v4-pro').to_trajectory(record)
        assert trajectory.notes is None
        assert next(s for s in trajectory.steps if s.tool_calls).observation.results
        if mode in ('bash', 'routed'):
            assert (tmp_path / 'proof.txt').read_text() == 'proof'
            assert record['deliverables'][0]['name'] == 'proof.txt'
        else:
            assert record['grader_state']['seen'] == [{'value': 42}]
            assert record['init_mcp_servers'][0]['status'] == 'connected'
    else:
        assert record['error']
        assert record['terminal_reason'] == {'max_tokens': 'max-tokens', 'max_turns': 'max_turns', 'timeout': 'timeout'}[mode]
        if mode == 'max_turns':
            assert len(requests) == 1
            assert record['n_tool_calls'] == 1
        if mode == 'timeout':
            from core.agents.errors import is_harness_timeout
            assert is_harness_timeout(record['error'])
            assert record['wall_time'] < 15
        if mode == 'max_turns':
            assert record['error'].startswith('max_turns reached before confirmed task completion')
