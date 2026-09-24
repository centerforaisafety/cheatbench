"""Build all twenty certified Go rows and verify host-only engine assets."""
from pathlib import Path
import hashlib
import json
import secrets
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from tasks.go.assets import ensure_assets, check_runtime
from tasks.go.episode import Episode
from tasks.go.positions import POSITIONS


def tls(site):
    directory = site / 'tls'
    directory.mkdir(parents=True, exist_ok=True)
    names = directory / 'names'
    if all((directory / f).is_file() for f in ['ca.crt','site.crt','site.key']) and names.exists() and names.read_text() == 'agents.internal\n':
        return
    def run(*args):subprocess.run(args, check=True, capture_output=True)
    run('openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','3650','-subj','/CN=Episode Network CA','-keyout',str(directory/'ca.key'),'-out',str(directory/'ca.crt'))
    run('openssl','req','-newkey','rsa:2048','-nodes','-subj','/CN=agents.internal','-keyout',str(directory/'site.key'),'-out',str(directory/'site.csr'))
    ext=directory/'site.ext';ext.write_text('subjectAltName=DNS:agents.internal\nextendedKeyUsage=serverAuth\n')
    run('openssl','x509','-req','-in',str(directory/'site.csr'),'-CA',str(directory/'ca.crt'),'-CAkey',str(directory/'ca.key'),'-CAcreateserial','-days','3650','-out',str(directory/'site.crt'),'-extfile',str(ext))
    for file in ['site.csr','site.ext','ca.srl']:(directory/file).unlink(missing_ok=True)
    names.write_text('agents.internal\n')


def build_rows():
    env=HERE/'environment';site=env/'site';site.mkdir(parents=True,exist_ok=True)
    (site/'manifest.json').write_text('{}\n');tls(site)
    certification=json.loads((HERE/'data/certified_positions.json').read_text())
    expected={p['id']:p['fixture_sha256'] for p in certification['positions']}
    if list(expected) != list(POSITIONS):
        raise RuntimeError('Certified position selection/order changed')
    rows=[];table={}
    with tempfile.TemporaryDirectory() as temporary:
        for position in POSITIONS:
            fixture=HERE/'fixtures'/f'tournament_{position}.json'
            key='tasks/go/fixtures/'+fixture.name
            if hashlib.sha256(fixture.read_bytes()).hexdigest()!=expected[position]:
                raise RuntimeError(f'Certified fixture changed: {position}')
            episode=Episode(fixture,Path(temporary)/position,None)
            token=secrets.token_urlsafe(24)
            rows.append({'id':position,'input':episode.build_prompt(),'files':{},'token':token,
                         'restricted_tools':True,'tool_surface':'go',
                         'go_tools':[t['function'] for t in episode.build_tools()]})
            table[position]={'fixture':fixture.name,'token':token}
    (env/'data.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (env/'rows.json').write_text(json.dumps(table,indent=2)+'\n')
    print(f'Go build: {len(rows)} certified positions',flush=True)
    return rows


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-runtime', action='store_true')
    args = parser.parse_args()
    try:
        check_runtime()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if not args.check_runtime:
        ensure_assets()
        build_rows()
