
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import sys,json,hashlib,time,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
D=Path(__file__).resolve().parent;O=D/'improve-exp12-20261013';A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit');sys.path.insert(0,str(A))
import run_audit as audit
import compare_slgcn_best as pilot
read=lambda p:_local_json_loads(p.read_text(encoding='utf-8'))
def main():
 s=importlib.util.spec_from_file_location('metrics_only',D/'rtm-eval.py');ev=importlib.util.module_from_spec(s);s.loader.exec_module(ev)
 st,norm,vocab=audit.load_model();sl,cfg=pilot.load_slgcn();old=read(A/'slgcn200_20261002/results.json');old={(r['label'],r['model']):r for r in old['predictions'] if r['stage']=='pre'};pred={v:[] for v in ['ST-GCN','SL-GCN','ST50_SL50','ST75_SL25']};(O/'exp1-inference').mkdir(exist_ok=True)
 for i,path in enumerate(sorted((A/'slgcn200_20261002/cases').glob('*.json'))):
  c=read(path);a=audit.frame_array([c['payload']['frames'][j] for j in c['frameIndices']]);x=norm({'frames':torch.from_numpy(a[:,audit.PICK].transpose(2,0,1).copy())})['frames'];start=time.perf_counter()
  with torch.inference_mode():ls=st(x.unsqueeze(0))[0];ll=sl(x.unsqueeze(0))[0];ps=ls.softmax(0);pl=ll.softmax(0)
  probs={'ST-GCN':ps,'SL-GCN':pl,'ST50_SL50':.5*ps+.5*pl,'ST75_SL25':.75*ps+.25*pl}
  for v,p in probs.items():
   top=p.topk(5);r=dict(id=path.stem,label=c['label'],prediction=vocab[int(top.indices[0])],top5=[vocab[int(k)] for k in top.indices],target_probability=float(p[vocab.index(c['label'])]));pred[v].append(r)
   if v in ['ST-GCN','SL-GCN']:
    before=old[c['label'],v];assert r['top5']==before['top5'] and abs(r['target_probability']-before['target_probability'])<1e-5
  np.savez_compressed(O/'exp1-inference'/(path.stem+'.npz'),input=x.numpy(),st_logits=ls.numpy(),sl_logits=ll.numpy(),inference_ms=(time.perf_counter()-start)*1000)
  if (i+1)%20==0:print('E1',i+1,flush=True)
 pairs={v:dict(gains=[b['label'] for a,b in zip(pred['ST-GCN'],pred[v]) if a['prediction']!=a['label'] and b['prediction']==b['label']],losses=[b['label'] for a,b in zip(pred['ST-GCN'],pred[v]) if a['prediction']==a['label'] and b['prediction']!=b['label']]) for v in pred if v!='ST-GCN'}
 both=sum(a['prediction']==a['label'] and b['prediction']==b['label'] for a,b in zip(pred['ST-GCN'],pred['SL-GCN']));union=sum(a['prediction']==a['label'] or b['prediction']==b['label'] for a,b in zip(pred['ST-GCN'],pred['SL-GCN']))
 out=dict(summary={v:ev.metrics(rows) for v,rows in pred.items()},pairs=pairs,both_correct=both,oracle_union_not_achieved=union,baseline_reproduced=True,protocol_sha256=hashlib.sha256((O/'PROTOCOL.md').read_bytes()).hexdigest(),predictions=pred)
 (O/'exp1-results.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out['summary']),flush=True)
if __name__=='__main__':main()
