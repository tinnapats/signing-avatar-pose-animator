"""Finish the already-running extraction and score without an additional user turn."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json,subprocess,time,sys,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent;O=D/'oracle200-20261011'
P=Path('C:/Users/newtu/signing-avatar-github-newton-upload/.venv/Scripts/python.exe')
read=lambda p:_local_json_loads(p.read_text(encoding='utf-8'))
start=time.monotonic();last=-1;changed=start
def status(state,**kw):
 (O/'STATUS.json').write_text(json.dumps({'state':state,**kw},indent=2),encoding='utf-8')
try:
 while not (O/'extraction_complete.json').exists():
  count=len(list((O/'rtm').glob('*.json')))
  if count!=last:last=count;changed=time.monotonic();print('Extraction',count,'/200',flush=True)
  status('extracting',completed_clips=count,total_clips=200)
  if time.monotonic()-changed>900:raise RuntimeError('No completed clip for 15 minutes; inspect extraction process. Partial results preserved.')
  if time.monotonic()-start>14400:raise RuntimeError('Four-hour watchdog reached. Partial results preserved.')
  time.sleep(20)
 status('scoring',completed_clips=200,total_clips=200)
 with (O/'scoring.log').open('w',encoding='utf-8') as log:
  subprocess.run([str(P),'-u',str(D/'oracle200.py'),'score'],stdout=log,stderr=subprocess.STDOUT,check=True)
 result=read(O/'results.json');assert result['clips']==200 and result['first30_exact_reproduction']
 root=Path('C:/Users/newtu/signing-avatar-github-newton-upload')
 for f,h in read(O/'protocol.json')['production'].items():assert hashlib.sha256((root/f).read_bytes()).hexdigest()==h
 status('completed',completed_clips=200,total_clips=200,production_unchanged=True,summary=result['summary'])
 print(json.dumps(result['summary']),flush=True)
except Exception as e:
 status('failed',error=str(e),partial_data_preserved=True);raise
