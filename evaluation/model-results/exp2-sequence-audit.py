
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import sys,json,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
D=Path(__file__).resolve().parent;O=D/'exp2-sequence-audit-20261014';B=D/'oracle200-20261011';P=D/'pixel-only-plan-20261012';A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit')
read=lambda p:_local_json_loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def runs(mask):
 d=np.diff(np.r_[False,mask,False].astype(int));starts=np.flatnonzero(d==1);ends=np.flatnonzero(d==-1);return [dict(start=int(s),frames=int(e-s)) for s,e in zip(starts,ends)]
def geometry():
 ev=mod('seqconvert',D/'rtm-eval.py');rows=[];totals={};examples={};edges=[(0,j) for j in [1,5,9,13,17]]+[(j,j+1) for start in [1,5,9,13,17] for j in range(start,start+3)]
 for c in read(B/'capture_manifest.json'):
  z=np.load(P/'arrays'/(c['id']+'.npz'));rt=np.load(B/'rtm'/(c['id']+'.npz'));raw=z['raw'];xy=rt['rtmw_det'];sc=rt['rtmw_det_scores'];T=len(raw);assert np.array_equal(ev.convert(xy,sc),raw);assert T==len(c['frames']);assert np.allclose(rt['times'],np.arange(T)/30);assert np.allclose([f['time'] for f in c['frames']],np.arange(T)/30);assert np.isfinite(xy).all() and np.isfinite(sc).all()
  shoulders=(sc[:,5]>=.3)&(sc[:,6]>=.3);width=np.linalg.norm(xy[:,5]-xy[:,6],axis=1);good=shoulders&(width>=10);scale=float(np.median(width[good])) if good.any() else 1.;signals={};body=sc[:,[5,6,7,8,9,10]]<.3
  signals['body_any_low_confidence']=body.any(1);signals['shoulder_missing']=~shoulders;signals['shoulder_width_small']=shoulders&(width<10);bodyjump=np.zeros(T,bool)
  for j in [5,6,7,8,9,10]:bodyjump[1:]|=(sc[1:,j]>=.3)&(sc[:-1,j]>=.3)&(np.linalg.norm(np.diff(xy[:,j],axis=0),axis=1)>.5*scale)
  signals['body_jump']=bodyjump;hands=[]
  for s,b in enumerate([33,54]):
   q=raw[:,b:b+21];present=np.any(q!=0,axis=(1,2));ds=z['distances'][:,s];off=present&(ds[:,0]>6);majority=present&((ds>6).mean(1)>.5);assert np.array_equal(off&majority,z['flags'][:,s]);missing=~present|z['flags'][:,s];lengths=np.stack([np.linalg.norm(q[:,j]-q[:,k],axis=1) for j,k in edges],axis=1);collapse=present&((lengths<.5).sum(1)>=5);extreme=np.zeros_like(lengths,bool)
   for j in range(20):
    valid=present&(lengths[:,j]>.5)
    if valid.any():median=float(np.median(lengths[valid,j]));extreme[:,j]=(lengths[:,j]<median/3)|(lengths[:,j]>median*3)
   shape=present&(extreme.sum(1)>=3);jump=np.zeros(T,bool);jump[1:]=present[1:]&present[:-1]&(np.linalg.norm(np.diff(q[:,0],axis=0),axis=1)>.5*scale);w=9+s;e=7+s;forearm=np.linalg.norm(xy[:,w]-xy[:,e],axis=1);joint=present&(sc[:,w]>=.3)&(sc[:,e]>=.3)&(forearm>=10);disagree=joint&(np.linalg.norm(q[:,0]-xy[:,w],axis=1)>.65*forearm)
   masks=dict(raw_missing=~present,post_gate_missing=missing,wrist_off_skin=off,majority_off_skin=majority,old_gate=z['flags'][:,s],collapsed=collapse,projected_length_outlier=shape,wrist_jump=jump,wrist_body_disagreement=disagree)
   h=dict(side=s,counts={k:int(v.sum()) for k,v in masks.items()},raw_missing_runs=runs(~present),post_gate_missing_runs=runs(missing));hands.append(h)
   for k,v in masks.items():
    totals[k]=totals.get(k,0)+int(v.sum())
    if v.any() and (k not in examples or int(v.sum())>examples[k]['count']):examples[k]=dict(id=c['id'],frame=int(np.flatnonzero(v)[0]),side=s,count=int(v.sum()))
  for k,v in signals.items():totals[k]=totals.get(k,0)+int(v.sum())
  totals['frames']=totals.get('frames',0)+T;rows.append(dict(id=c['id'],frames=T,shoulder_scale=scale,body_counts={k:int(v.sum()) for k,v in signals.items()},hands=hands))
 longest=max((dict(id=r['id'],side=h['side'],**run) for r in rows for h in r['hands'] for run in h['post_gate_missing_runs']),key=lambda r:r['frames'],default=None);save(O/'coverage.json',dict(totals=totals,clips=rows,examples=examples,longest_missing_run=longest));print(json.dumps(dict(totals=totals,longest=longest)))
def tests():
 import torch
 trans=mod('seq_trans',A/'upstream_pose_transforms.py');norm=trans.CenterAndScaleNormalize(reference_points_preset='shoulder_mediapipe_holistic_minimal_27');ev=mod('seq_ev',D/'rtm-eval.py');pick=[0,2,5,11,12,13,14,33,37,38,41,42,45,46,49,50,53,54,58,59,62,63,66,67,70,71,74];expected=[0,1,2,5,6,7,8]+[91+j for j in [0,4,5,8,9,12,13,16,17,20]]+[112+j for j in [0,4,5,8,9,12,13,16,17,20]];checks=[]
 probe=np.arange(266,dtype=np.float32).reshape(1,133,2)+1;assert np.array_equal(ev.convert(probe,np.ones((1,133)))[:,pick],probe[:,expected]);checks.append('133->75->27 anatomical mapping exact')
 assert not ev.convert(probe,np.zeros((1,133))).any();assert ev.convert(probe,np.full((1,133),.3,dtype=np.float32))[:,pick].all();assert not ev.convert(probe,np.full((1,133),.299,dtype=np.float32)).any();boundary64=bool(ev.convert(probe,np.full((1,133),.3,dtype=np.float64))[:,pick].all());checks.append('float32 production confidence threshold passes at0.3 and rejects0.299; float64 exact0.3 mean can round below threshold')
 x=torch.arange(2*7*27,dtype=torch.float32).reshape(2,7,27)+20;x[0,:,3]=100;x[1,:,3]=200;x[0,:,4]=200;x[1,:,4]=200
 fn=lambda t:norm({'frames':t.clone()})['frames'];y=fn(x);assert torch.allclose(fn(x+torch.tensor([17.,-23.])[:,None,None]),y,atol=1e-5);assert torch.allclose(fn(x*2.5),y,atol=1e-5);checks.append('fully observed translation and isotropic-scale invariance')
 missing=x.clone();missing[:,:,7:]=0;ym=fn(missing);assert torch.allclose(ym[:,:,7],torch.tensor([-1.5,-2.])[:,None].expand(2,7));checks.append('missing zero maps to negative shoulder-relative coordinates, documented behavior')
 allzero=torch.zeros_like(x);assert torch.equal(fn(allzero),allzero);checks.append('all-zero degenerate shoulders remain finite')
 exact=0;missing_slots=0;nonzero_slots=0;maxmissing=0.;shoulderbad=0
 for c in read(B/'capture_manifest.json'):
  z=np.load(P/'arrays'/(c['id']+'.npz'));a=z['raw'].copy()
  for s,b in enumerate([33,54]):a[z['flags'][:,s],b:b+21]=0
  inp=torch.from_numpy(a[:,pick].transpose(2,0,1).copy());out=fn(inp).numpy();saved=np.load(P/'inference'/(c['id']+'.npz'))['support_gate_input'];assert np.array_equal(out,saved);assert np.isfinite(out).all();exact+=1
  for s,sl in enumerate([slice(7,17),slice(17,27)]):
   mask=~np.any(a[:,[33,54][s]:[33,54][s]+21],axis=(1,2));missing_slots+=int(mask.sum());vals=out[:,mask,sl];nonzero_slots+=int(np.any(vals!=0,axis=(0,2)).sum());maxmissing=max(maxmissing,float(np.abs(vals).max()) if vals.size else 0.)
  shoulderbad+=int((~np.any(a[:,11],axis=1)|~np.any(a[:,12],axis=1)).sum())
 checks.append('all200 cached normalized baseline inputs reproduced bit-for-bit')
 # Independent affine arithmetic tests over rotations/scales (without any renderer points).
 for theta in [0,np.pi/2,-np.pi/2,np.pi]:
  for scale in [.8,1.,1.2]:
   rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])*scale;pts=np.array([[0.,0.],[120.,320.],[808.,808.]]);shift=np.array([42.,-75.]);assert np.allclose(((pts@rot.T+shift)-shift)@np.linalg.inv(rot).T,pts)
 checks.append('synthetic affine inverse round trips at all tested rotations/scales')
 save(O/'tests.json',dict(checks=checks,clips_exact=exact,synthetic_float64_exact_threshold_all_accepted=boundary64,missing_hand_frames=missing_slots,missing_nonzero_after_normalization=nonzero_slots,max_absolute_normalized_missing_coordinate=maxmissing,frames_with_missing_shoulder=shoulderbad,warning='Missing-value behavior matches frozen upstream normalization; not automatically a bug and not changed.'))
 print(read(O/'tests.json'))
if __name__=='__main__':{'geometry':geometry,'tests':tests}[sys.argv[1]]()
