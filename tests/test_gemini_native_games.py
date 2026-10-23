"""Native Google inference keeps game egress and tool restrictions intact."""
import json
import urllib.error
import urllib.request
from unittest.mock import Mock

import pytest
from core.sandbox import native_inference as bridge


@pytest.mark.parametrize('game', ['chess', 'go'])
def test_native_bridge_filters_and_keeps_thinking_and_history(tmp_path, game):
    server = bridge.InferenceBridge('https://generativelanguage.googleapis.com', 'real-test-key',
        tmp_path / 'requests.jsonl', gemini_native=True, mcp_server=game)
    response = Mock(status=200, relay_stats={})
    response.relay.side_effect = lambda handler, **kw: (
        handler.send_response(200), handler.send_header('Content-Length', '2'),
        handler.end_headers(), handler.wfile.write(b'{}'))
    server.upstream.request = Mock(return_value=response)
    body = {'contents':[{'role':'user','parts':[{'text':'unchanged'}]}],
        'generationConfig':{'thinkingConfig':{'thinkingLevel':'HIGH'}},
        'tools':[{'functionDeclarations':[{'name':f'mcp_{game}_play_move'}, {'name':'run_shell_command'}]},
                 {'googleSearch':{}}]}
    def post(path, data=body):
        return urllib.request.urlopen(urllib.request.Request(server.local_base_url + path,
            data=json.dumps(data).encode(), headers={'x-goog-api-key':'container-token'}), timeout=3)
    try:
        with post('/v1beta/models/gemini-4-argon:streamGenerateContent?alt=sse') as result:
            assert result.read() == b'{}'
        method, path, headers, raw = server.upstream.request.call_args.args
        assert server.upstream.auth_scheme == 'google-api-key'
        sent = json.loads(raw)
        assert sent['contents'] == body['contents']
        assert sent['generationConfig'] == body['generationConfig']
        assert sent['tools'] == [{'functionDeclarations':[{'name':f'mcp_{game}_play_move'}]}]
        for path in ['/v1/chat/completions', '/v1beta/cachedContents',
                     '/v1beta/models/gemini-4-argon:generateContent?key=leak',
                     '/v1beta/models/../files:generateContent']:
            with pytest.raises(urllib.error.HTTPError) as error:
                post(path)
            assert error.value.code == 403
        assert server.upstream.request.call_count == 1
    finally:
        server.close()


def test_native_game_payload_reroutes_to_bridge():
    task = {'tools':[], 'gemini':{'route':'native', 'mcp_only':True,
        'model_id':'gemini/gemini-4-argon', 'base_url':'https://generativelanguage.googleapis.com'}}
    result = json.loads(bridge.reroute_blob(json.dumps({'task':task}), 'http://127.0.0.1:4321'))['task']
    assert result['gemini']['base_url'] == 'http://127.0.0.1:4321'
    assert result['gemini']['route'] == 'native'
    assert result['gemini']['model_id'] == 'gemini/gemini-4-argon'


def test_google_upstream_uses_api_key_without_bearer_header():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from core.agents.forwarder import Upstream
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            seen.append(dict(self.headers))
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200); self.send_header('Content-Length', '2')
            self.end_headers(); self.wfile.write(b'{}')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        upstream = Upstream(f'http://127.0.0.1:{server.server_port}', 'real-test-key', auth_scheme='google-api-key')
        response = upstream.request('POST', '/v1beta/models/test:generateContent',
            {'Authorization':'Bearer wrong', 'x-goog-api-key':'container-token'}, b'{}')
        assert response.read() == b'{}'; response.close()
        assert seen[0]['x-goog-api-key'] == 'real-test-key'
        assert 'Authorization' not in seen[0]
    finally:
        server.shutdown(); server.server_close(); thread.join()
