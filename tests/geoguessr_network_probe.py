"""Compute-node probe: local task works, public egress fails, API relay works.

Pass a built GeoGuessr environment directory containing data.jsonl.
No paid model calls or credentials are used.
"""
import http.server
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import sandbox, stage, trial

env = Path(sys.argv[1]).resolve()
row = json.loads((env / 'data.jsonl').read_text().splitlines()[0])
task = trial.load_task('geoguessr')
assert task.network.denies and task.private_net
class API(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'bridge-ok')
    def log_message(self, *args): pass
server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), API)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f'http://127.0.0.1:{server.server_port}/'
bootstrap = 'import sys,json;d=json.load(sys.stdin);exec(d["code"])'
probe = '''
import json,os,socket,subprocess
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError,HTTPError
ns={};exec(d['stage'],ns);ns['write'](d['files']);os.chdir('/workspace')
assert os.getuid()==1000
assert Path('image/img_01.jpg').is_file()
assert Path('/tmp/gps/img_01_gps.gpx').is_file()
assert urlopen(d['bridge'],timeout=10).read()==b'bridge-ok'
for url in ['https://example.org/','https://www.google.com/']:
    try: urlopen(url,timeout=5)
    except HTTPError as e: raise AssertionError('Public server responded: '+url) from e
    except (URLError,OSError): pass
    else: raise AssertionError('Public egress allowed: '+url)
try: socket.create_connection(('1.1.1.1',443),timeout=5)
except OSError: pass
else: raise AssertionError('Direct IP egress allowed')
try: urlopen('https://api.openai.com/v1/models',timeout=15)
except HTTPError as e: assert e.code==401,e.code
else: raise AssertionError('Expected unauthenticated API response')
Path('submission.txt').write_text('-10.41566, 105.67558\\n')
output=subprocess.check_output(['python','submit.pyc','submission.txt'],text=True)
assert '1/1' in output,output
print(json.dumps({'public_hosts':'blocked','direct_ip':'blocked','model_api':401,'local_bridge':'reachable','local_checker':'passed'}))
'''
sandbox.ensure_path()
with tempfile.TemporaryDirectory(prefix='geo-net-probe-') as tmp:
    proc=subprocess.Popen(sandbox.container_argv(sandbox.resolve_image('python311'),bootstrap,private_net=True,pass_key=False),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=sandbox.spawn_env(''))
    sb=sandbox.EpisodeSandbox(proc.pid,('python','-c',bootstrap),Path(tmp))
    try:
        sb.start_network()
        sb.lock_egress(sandbox.hosts_for(task.network,'https://api.openai.com'),sandbox.api_loopback_ports(url))
        out,err=proc.communicate(json.dumps({'code':probe,'bridge':url,'stage':(ROOT/'core/stage.py').read_text(),'files':stage.collect(row,env)}).encode(),timeout=100)
        if proc.returncode: raise RuntimeError(err.decode()[-4000:])
        print(out.decode())
    finally:
        sb.close()
        if proc.poll() is None: proc.kill()
        proc.wait(timeout=10)
        server.shutdown()
