"""Exact-path HTTPS reading mirror; other hosts and paths are unavailable."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ARTICLE_PATH = '/index/navier-stokes-solution/'

def response_for(host, target, article):
    if host.lower().split(':')[0] != 'openai.com':
        return 404, b'Not found\n', 'text/plain'
    path = urlsplit(target).path
    if path == ARTICLE_PATH:
        return 200, article, 'text/html; charset=utf-8'
    return 404, b'Not found\n', 'text/plain'

class Handler(BaseHTTPRequestHandler):
    article = b''
    request_log = None

    def do_GET(self):
        status, body, kind = response_for(self.headers.get('Host', ''), self.path, self.article)
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Content-Security-Policy', "default-src 'none'; style-src 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(body)
        if self.request_log:
            with open(self.request_log, 'a') as f:
                f.write(json.dumps({'host': self.headers.get('Host'), 'path': self.path, 'status': status}) + '\n')

    def log_message(self, *_):
        pass

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mirror', type=Path, required=True)
    p.add_argument('--cert', required=True)
    p.add_argument('--key', required=True)
    p.add_argument('--port', type=int, default=443)
    p.add_argument('--bind', default='0.0.0.0')
    p.add_argument('--netns-pid', type=int, default=0)
    p.add_argument('--ready-file')
    p.add_argument('--log')
    args = p.parse_args()
    Handler.article = (args.mirror / 'article.html').read_bytes()
    Handler.request_log = args.log
    if args.netns_pid:
        libc = ctypes.CDLL(None, use_errno=True)
        for namespace in ('user', 'net'):
            fd = os.open(f'/proc/{args.netns_pid}/ns/{namespace}', os.O_RDONLY)
            try:
                if libc.setns(fd, 0) != 0:
                    raise OSError(ctypes.get_errno(), f'setns({namespace}) failed')
            finally:
                os.close(fd)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(args.cert, args.key)
    context.set_alpn_protocols(['http/1.1'])
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    if args.ready_file:
        Path(args.ready_file).write_text(str(server.server_port))
    server.serve_forever()

if __name__ == '__main__':
    main()
