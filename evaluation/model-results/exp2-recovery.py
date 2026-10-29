
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import sys,json,time,hashlib,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
D=Path(__file__).resolve().parent; B=D/'oracle200-20261011'; P=D/'pixel-only-plan-20261012'; O=D/'exp2-recovery-20261013'
A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit')
def read(p):return _local_json_loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def extract():
 base=module('recovery_base',D/'rtm-avatar.py');cv2,RTMPose,_=base.setup()
 mp=A/'pixels30_20261007/rtm_trial_20261008/model_cache/hub/checkpoints/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.onnx'
 assert hashlib.sha256(mp.read_bytes()).hexdigest()=='39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29'
 hand=RTMPose(str(mp),model_input_size=(256,256),backend='onnxruntime',device='cpu')
 (O/'arrays').mkdir(exist_ok=True);events=[];started=time.perf_counter()
 for c in read(B/'capture_manifest.json'):
  z=np.load(P/'arrays'/(c['id']+'.npz'));rt=np.load(B/'rtm'/(c['id']+'.npz'));raw=z['raw'];flags=z['flags'];a=raw.copy()
  for side,b in enumerate([33,54]):a[flags[:,side],b:b+21]=0
  wrist=a.copy();flow=a.copy();cache={}
  def frame(i):
   if i not in cache:
    f=c['frames'][i];p=B/'frames'/c['id']/f['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
    im=base.image(cv2,p);blue,green,red=[im[:,:,k].astype(np.int16) for k in range(3)]
    skin=(red>=100)&(red-green>=15)&(red-blue>=20)&(green>=45)&(green-blue>=-5)
    cache[i]=(im,cv2.distanceTransform((~skin).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE))
   return cache[i]
  def context(i,s):
   w,e=9+s,7+s;q=rt['rtmw_det'][i];sc=rt['rtmw_det_scores'][i];length=float(np.linalg.norm(q[w]-q[e]))
   return q[w],q[e],length,bool(min(sc[w],sc[e])>=.3 and length>=10)
  def accept(q,sc,i,s):
   w,e,l,valid=context(i,s)
   if not valid or not np.isfinite(q).all() or not np.isfinite(sc).all():return False
   if min(float(sc.mean()),float(sc[0]))<.3 or np.linalg.norm(q[0]-w)>.65*l:return False
   im,dist=frame(i);xy=np.rint(q).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<im.shape[1])&(xy[:,1]>=0)&(xy[:,1]<im.shape[0]);ds=np.full(21,999.);ds[inside]=dist[xy[inside,1],xy[inside,0]]
   return bool(ds[0]<=6 and (ds>6).mean()<=.5)
  def detect(i,s,bb,kind,seed=None):
   t=time.perf_counter();q,sc=hand(frame(i)[0],bboxes=[np.asarray(bb,dtype=np.float32)]);q,sc=q[0],sc[0];ok=accept(q,sc,i,s)
   events.append(dict(id=c['id'],frame=int(i),side=int(s),kind=kind,seed=seed,box=np.asarray(bb).tolist(),points=q.tolist(),scores=sc.tolist(),accepted=ok,ms=(time.perf_counter()-t)*1000))
   return q,ok
  for i,s in np.argwhere(flags):
   i,s=int(i),int(s);b=[33,54][s];w,e,l,valid=context(i,s)
   if not valid:
    events.append(dict(id=c['id'],frame=i,side=s,kind='invalid_body',accepted=False));continue
   center=w+.25*(w-e);size=float(np.clip(2.2*l,96,256));bb=np.r_[center-size/2,center+size/2];q,ok=detect(i,s,bb,'wrist')
   if ok:wrist[i,b:b+21]=q;flow[i,b:b+21]=q;continue
   for j in range(i-1,max(-1,i-4),-1):
    src=91+21*s;seed=raw[j,b:b+21]
    if flags[j,s] or not np.any(seed) or not accept(seed,rt['rtmw_det_scores'][j,src:src+21],j,s):continue
    im0=frame(j)[0];im1=frame(i)[0];lo=seed.min(0);hi=seed.max(0);ctr=(lo+hi)/2;sz=float(np.clip(1.5*max(hi-lo),64,256));bb0=np.r_[ctr-sz/2,ctr+sz/2]
    mask=np.zeros(im0.shape[:2],np.uint8);x0,y0,x1,y1=np.rint(bb0).astype(int);mask[max(0,y0):min(mask.shape[0],y1),max(0,x0):min(mask.shape[1],x1)]=255
    g0=cv2.cvtColor(im0,cv2.COLOR_BGR2GRAY);g1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY);corners=cv2.goodFeaturesToTrack(g0,maxCorners=60,qualityLevel=.01,minDistance=3,mask=mask)
    if corners is None or len(corners)<6:break
    nxt,st,_=cv2.calcOpticalFlowPyrLK(g0,g1,corners,None,winSize=(21,21),maxLevel=3);back,stb,_=cv2.calcOpticalFlowPyrLK(g1,g0,nxt,None,winSize=(21,21),maxLevel=3)
    good=(st[:,0]>0)&(stb[:,0]>0)&(np.linalg.norm(back[:,0]-corners[:,0],axis=1)<=1.5)
    if good.sum()<6 or good.mean()<.5:break
    delta=(nxt-corners)[:,0][good];med=np.median(delta,axis=0);inlier=np.linalg.norm(delta-med,axis=1)<=3
    if inlier.sum()<6 or inlier.mean()<.5:break
    shift=np.median(delta[inlier],axis=0);q,ok=detect(i,s,bb0+np.tile(shift,2),'flow',j)
    if ok:flow[i,b:b+21]=q
    break
  assert np.array_equal(a[:,:33],wrist[:,:33]) and np.array_equal(a[:,:33],flow[:,:33])
  np.savez_compressed(O/'arrays'/(c['id']+'.npz'),baseline=a,wrist=wrist,flow=flow)
  print('EXTRACT',c['id'],'flagged',int(flags.sum()),flush=True)
 save(O/'extraction.json',dict(events=events,seconds=time.perf_counter()-started,protocol_sha256=hashlib.sha256((O/'PROTOCOL.md').read_bytes()).hexdigest(),model_sha256=hashlib.sha256(mp.read_bytes()).hexdigest()))
def score():
 import torch
 sys.path.insert(0,str(A));import run_audit as audit
 ev=module('rec_ev',D/'rtm-eval.py');model,norm,vocab=audit.load_model();pred={v:[] for v in ['baseline','wrist','flow']};old=read(P/'results.json')['predictions']['support_gate'];(O/'inference').mkdir(exist_ok=True)
 for c,before in zip(read(B/'capture_manifest.json'),old):
  z=np.load(O/'arrays'/(c['id']+'.npz'));archive={}
  for v in pred:
   x=norm({'frames':torch.from_numpy(z[v][:,audit.PICK].transpose(2,0,1).copy())})['frames']
   with torch.inference_mode():log=model(x.unsqueeze(0))[0];p=log.softmax(0)
   top=p.topk(5);row=dict(id=c['id'],label=c['label'],prediction=vocab[int(top.indices[0])],top5=[vocab[int(k)] for k in top.indices],target_probability=float(p[vocab.index(c['label'])]));pred[v].append(row);archive[v+'_input']=x.numpy();archive[v+'_logits']=log.numpy()
   if v=='baseline':assert row['top5']==before['top5'] and abs(row['target_probability']-before['target_probability'])<1e-5
  np.savez_compressed(O/'inference'/(c['id']+'.npz'),**archive);print('SCORE',c['id'],flush=True)
 pairs={}
 for v in ['wrist','flow']:
  pairs[v]={'gains':[b['label'] for a,b in zip(pred['baseline'],pred[v]) if a['prediction']!=a['label'] and b['prediction']==b['label']], 'losses':[b['label'] for a,b in zip(pred['baseline'],pred[v]) if a['prediction']==a['label'] and b['prediction']!=b['label']]}
 result=dict(summary={v:ev.metrics(rows) for v,rows in pred.items()},pairs=pairs,baseline_reproduced=True,predictions=pred);save(O/'results.json',result);print(json.dumps(result['summary']),flush=True)
if __name__=='__main__':{'extract':extract,'score':score}[sys.argv[1]]()
