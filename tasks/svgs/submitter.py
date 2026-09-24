"""Compiled client: sends only the submitted SVG to the host-owned leaderboard."""
import json,sys,urllib.request
from pathlib import Path
here=Path(sys.argv[0]).resolve().parent
config=json.loads((here/'leaderboard_config.json').read_text())
try:
 if len(sys.argv)!=2:raise ValueError('Usage: python leaderboard/leaderboard.pyc drawing.svg')
 raw=Path(sys.argv[1]).read_text()
 req=urllib.request.Request(config['endpoint'],data=json.dumps({'submission':raw}).encode(),headers={'Authorization':'Bearer '+config['token'],'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=240) as response:result=json.load(response)
 log=Path(result['log']);log.parent.mkdir(exist_ok=True);log.write_text(json.dumps(result['log_data'],indent=2)+'\n')
 print(result['standings']);print('Submissions:',result['submissions']);print('Log:',result['log'])
except Exception as e:
 print('Submission failed:',str(e),file=sys.stderr);raise SystemExit(1)
