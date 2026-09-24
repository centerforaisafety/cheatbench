"""Native Google calls must preserve payloads and pin the requested main model."""
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from core.agents import gemini_cli_runner as runner


@pytest.mark.parametrize('status', [200, 429])
@pytest.mark.parametrize('method,cli_model,expected', [
    ('generateContent', 'gemini-3.5-flash', 'gemini-3.8-flash'),
    ('streamGenerateContent', 'gemini-3.5-flash', 'gemini-3.8-flash'),
    ('countTokens', 'gemini-3.5-flash', 'gemini-3.8-flash'),
    ('generateContent', 'gemini-3.1-pro', 'gemini-3.8-flash'),
    ('generateContent', 'gemini-3.5-flash-lite', 'gemini-3.5-flash-lite'),
])
def test_native_wire(method, cli_model, expected, status):
    seen = []
    output = b'data: {"text":"first"}\n\ndata: {"text":"last"}\n\n' if method.startswith('stream') else b'{"totalTokens":17}'
    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            seen.append((self.path, self.headers['x-goog-api-key'], self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(status)
            self.send_header('Content-Type', 'text/event-stream' if method.startswith('stream') else 'application/json')
            self.send_header('Content-Length', str(len(output)))
            self.end_headers(); self.wfile.write(output)
    host = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    thread = threading.Thread(target=host.serve_forever, daemon=True); thread.start()
    relay = runner.NativeRelay(base_url=f'http://127.0.0.1:{host.server_port}', api_key='upstream-secret',
                               model_id='gemini/gemini-3.8-flash', model_slug='gemini-3.8-flash', exclude_tools=[])
    relay.start()
    body = json.dumps({'contents':[{'parts':[{'text':'test','thoughtSignature':'untouched'}]}],
                       'generationConfig':{'thinkingConfig':{'thinkingLevel':'HIGH'}}}).encode()
    try:
        url = relay.url + f'/v1beta/models/{cli_model}:{method}?alt=sse&key={relay.token}'
        try:
            response = urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=5)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            assert response.code == status
            assert response.read() == output
        assert seen == [(f'/v1beta/models/{expected}:{method}?alt=sse', 'upstream-secret', body)]
        assert relay.calls[0]['model'] == cli_model
        assert relay.calls[0]['upstream_model'] == expected
        assert relay.calls[0]['status'] == status
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(relay.url + f'/v1beta/models/{cli_model}:{method}', data=body), timeout=5)
        assert error.value.code == 403
        assert len(seen) == 1
    finally:
        relay.close(); host.shutdown();host.server_close();thread.join()


def test_native_run_uses_relay_and_keeps_key_out_of_cli(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv('GOOGLE_TEST_KEY', 'upstream-secret')
    relay = SimpleNamespace(url='http://127.0.0.1:1234', token='local-token',
                            upstream_url='https://generativelanguage.googleapis.com',
                            model_id='gemini-3.8-flash', start=lambda: None)
    monkeypatch.setattr(runner, 'NativeRelay', lambda **kw: relay)
    monkeypatch.setattr(runner, 'write_settings', lambda *a: None)
    monkeypatch.setattr(runner, 'compose_command', lambda *a: 'cli')
    class Captured(Exception): pass
    def capture(command, env, timeout):
        assert env['GOOGLE_GEMINI_BASE_URL'] == relay.url
        assert env['GEMINI_API_KEY'] == relay.token
        assert 'upstream-secret' not in env.values()
        raise Captured
    monkeypatch.setattr(runner, 'run_cli', capture)
    with pytest.raises(Captured):
        runner.run({'content':'test','gemini':{'route':'native','api_key_env':'GOOGLE_TEST_KEY',
                    'model_id':'gemini/gemini-3.8-flash','model_slug':'gemini-3.8-flash'}}, install=None)
