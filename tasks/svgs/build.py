import json,py_compile,sys
from pathlib import Path
TASK=Path(__file__).resolve().parent
ENV=TASK/'environment/workshop'
def build():
 if sys.version_info[:2] != (3, 11):
  raise SystemExit("SVG checker compilation requires Python 3.11 to match the python311 container image")
 shared=ENV/'shared/leaderboard';shared.mkdir(parents=True,exist_ok=True)
 py_compile.compile(str(TASK/'submitter.py'),cfile=str(shared/'leaderboard.pyc'),dfile='leaderboard.py',doraise=True)
 rows=[]
 for r in json.loads((TASK/'catalog.json').read_text())['rows']:
  folder=ENV/'rows'/r['id'];folder.mkdir(parents=True,exist_ok=True)
  rubric=json.loads((TASK/'rubrics'/f"{r['id']}.json").read_text());assert sum(c['max_points'] for c in rubric['criteria'])==100
  (folder/'rubric.json').write_text(json.dumps(rubric,ensure_ascii=False,indent=2)+'\n')
  location=f"/codabench/competitions/{r['competition_id']}/rubric.json"
  (folder/'leaderboard_config.json').write_text(json.dumps({'round':'final','rubric':location,'endpoint':'RUNTIME_ENDPOINT','token':'RUNTIME_TOKEN'})+'\n')
  rel=lambda p:str(p.relative_to(ENV))
  rows.append({'id':r['id'],'input':r['instruction'],'vars':{'submitter':'python leaderboard/leaderboard.pyc drawing.svg'},'files':{'leaderboard/leaderboard.pyc':'shared/leaderboard/leaderboard.pyc','leaderboard/leaderboard_config.json':rel(folder/'leaderboard_config.json')},'files_abs':{location:rel(folder/'rubric.json')},'meta':{'competition_id':r['competition_id'],'competitor_count':5,'deliverable':'drawing.svg'}})
 (ENV/'data.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));return rows
if __name__=='__main__':print('Built',len(build()),'rubric-only rows')
