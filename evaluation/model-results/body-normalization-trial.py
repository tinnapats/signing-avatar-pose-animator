
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import sys,json,hashlib,importlib.util,itertools
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
BASE=Path(__file__).parent;O=BASE/'body-normalization-trial-20261009';O.mkdir(exist_ok=True)
A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit');sys.path.insert(0,str(A));import run_audit as audit
read=lambda p:_local_json_loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
Q=BASE/'hand-quality-trial-20261009';B=A/'pixels30_20261007'
spec=importlib.util.spec_from_file_location('e',BASE/'rtm-eval.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
COMBOS=list(itertools.product(['R','M'],repeat=3));NAMES=['B'+b+'_C'+c+'_S'+s for b,c,s in COMBOS]
protocol=dict(rules='Development30 only. Factorial diagnostic: body B, center C, scale S each R=RTMW/M=MediaPipe. Identical baseline hybrid hands in raw pixels for all8. Clip-level center mean shoulder midpoint, scale reciprocal mean shoulder width using unchanged upstream method. No imputation/masking changes, timing or model changes. Normalization source can differ from body for diagnosis; not a validated production recipe.',variants=NAMES,production=read(Q/'protocol.json')['production'],input_hashes={p.stem:sha(p) for p in sorted((Q/'inputs').glob('*.npz'))},normalizer_sha256=sha(A/'upstream_pose_transforms.py'),checkpoint_archive_sha256=sha(A/'wlasl_stgcn.zip'))
if (O/'protocol.json').exists():assert read(O/'protocol.json')==protocol
else:save(O/'protocol.json',protocol)
assert not (O/'results.json').exists()
model,norm,vocab=audit.load_model();pred={k:[] for k in NAMES};diag=[];normalized=O/'normalized_inputs';normalized.mkdir(exist_ok=True)
for clip in read(B/'capture_manifest.json'):
 z=np.load(Q/'inputs'/(clip['id']+'.npz'));raw={'R':z['baseline'],'M':z['mp_body_same_hands']};assert np.array_equal(raw['R'][:,33:],raw['M'][:,33:]);tensors={k:torch.from_numpy(v[:,audit.PICK].copy()) for k,v in raw.items()};params={k:norm.calc_center_and_scale(v) for k,v in tensors.items()};saved={}
 for name,(body,center,scale) in zip(NAMES,COMBOS):
  cn=params[center][0];sc=params[scale][1];x=((tensors[body]-cn)*sc).permute(2,0,1).contiguous();assert torch.isfinite(x).all()
  if body==center==scale:
   expected=norm({'frames':tensors[body].permute(2,0,1).clone()})['frames'];assert torch.equal(x,expected)
  with torch.inference_mode():p=model(x.unsqueeze(0))[0].softmax(0)
  top=p.topk(5);pred[name].append(dict(label=clip['label'],prediction=vocab[int(top.indices[0])],top5=[vocab[int(j)] for j in top.indices],target_probability=float(p[vocab.index(clip['label'])])));saved[name]=x.numpy()
 np.savez_compressed(normalized/(clip['id']+'.npz'),**saved,times=z['times'])
 cR,sR=params['R'];cM,sM=params['M'];diag.append(dict(id=clip['id'],label=clip['label'],center_R=np.asarray(cR).tolist(),center_M=np.asarray(cM).tolist(),center_distance_px=float(torch.linalg.norm(cM-cR)),scale_R=float(sR),scale_M=float(sM),scale_M_over_R=float(sM/sR),MP_missing_shoulder_frames=int((~np.all(np.any(raw['M'][:,[11,12]]!=0,axis=2),axis=1)).sum()),RTMW_missing_shoulder_frames=int((~np.all(np.any(raw['R'][:,[11,12]]!=0,axis=2),axis=1)).sum())))
 print('SCORED',clip['label'],flush=True)
old=read(Q/'results.json')
for name,prev in [('BR_CR_SR','baseline'),('BM_CM_SM','mp_body_same_hands')]:
 for x,y in zip(pred[name],old['predictions'][prev]):assert x['prediction']==y['prediction'] and x['top5']==y['top5'] and np.isclose(x['target_probability'],y['target_probability'],atol=1e-6)
for f,h in protocol['production'].items():assert sha(A.parent/f)==h
result=dict(summary={k:e.metrics(v) for k,v in pred.items()},predictions=pred,parameters=diag,production_unchanged=True,baseline_reproduced=True,hands_identical_in_raw_pixels=True,frames=1964,scope='Development30; no confirmation set opened')
save(O/'results.json',result);print(json.dumps(result['summary']),flush=True)
