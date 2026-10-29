
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
D=Path(__file__).resolve().parent;O=D/'exp2-bidirectional-20261013';OLD=D/'improve-exp12-20261013';B=D/'oracle200-20261011';P=D/'pixel-only-plan-20261012';A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit')
read=lambda p:_local_json_loads(p.read_text(encoding='utf-8'))
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def extract():
 import cv2
 cv2.setNumThreads(1);base=mod('bi_image',D/'rtm-avatar.py');source={}
 for r in read(OLD/'exp2-extraction.json')['events']:source.setdefault(r['id'],[]).append(r)
 # Whitelist clip identity and pixel metadata; no label enters extraction decisions.
 catalog=[dict(id=c['id'],frames=c['frames']) for c in read(B/'capture_manifest.json')]
 (O/'arrays').mkdir(exist_ok=True);(O/'events').mkdir(exist_ok=True);started=time.perf_counter();accepted=0;attempts=0
 for c in catalog:
  z=np.load(P/'arrays'/(c['id']+'.npz'));raw=z['raw'];flags=z['flags'];rt=np.load(B/'rtm'/(c['id']+'.npz'));previous=np.load(OLD/'exp2-arrays'/(c['id']+'.npz'));a=previous['baseline'];out=a.copy();cache={};events=[]
  def frame(i):
   if i not in cache:
    f=c['frames'][i];p=B/'frames'/c['id']/f['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];im=base.image(cv2,p);blue,green,red=[im[:,:,k].astype(np.int16) for k in range(3)];skin=(red>=100)&(red-green>=15)&(red-blue>=20)&(green>=45)&(green-blue>=-5);cache[i]=(im,cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),cv2.distanceTransform((~skin).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE))
   return cache[i]
  def seed(i,s,direction):
   b=[33,54][s]
   for step in [1,2,3]:
    j=i+direction*step
    if j<0 or j>=len(raw):break
    q=raw[j,b:b+21];sc=rt['rtmw_det_scores'][j,91+21*s:112+21*s]
    if flags[j,s] or not np.any(q) or min(sc.mean(),sc[0])<.3:continue
    span=float(max(q.max(0)-q.min(0)))
    if span<10:continue
    im,_,dist=frame(j);xy=np.rint(q).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<im.shape[1])&(xy[:,1]>=0)&(xy[:,1]<im.shape[0]);ds=np.full(21,999.);ds[inside]=dist[xy[inside,1],xy[inside,0]]
    if ds[0]<=6 and (ds>6).mean()<=.5:return j,q,span
   return None
  def track(i,ss):
   if ss is None:return dict(ok=False,reason='no_reliable_seed')
   j,q,span=ss;_,g0,_=frame(j);_,g1,_=frame(i);ctr=(q.min(0)+q.max(0))/2;size=float(np.clip(1.5*span,64,256));bb=np.rint(np.r_[ctr-size/2,ctr+size/2]).astype(int);x0,y0,x1,y1=bb;mask=np.zeros(g0.shape,np.uint8);x0,x1=np.clip([x0,x1],0,g0.shape[1]);y0,y1=np.clip([y0,y1],0,g0.shape[0]);mask[y0:y1,x0:x1]=255
   corners=cv2.goodFeaturesToTrack(g0,maxCorners=60,qualityLevel=.01,minDistance=3,mask=mask);diag=dict(seed=j,span=span,ok=False)
   if corners is None or len(corners)<6:return dict(diag,reason='too_few_corners')
   nxt,st,_=cv2.calcOpticalFlowPyrLK(g0,g1,corners,None,winSize=(21,21),maxLevel=3);back,stb,_=cv2.calcOpticalFlowPyrLK(g1,g0,nxt,None,winSize=(21,21),maxLevel=3);good=(st[:,0]>0)&(stb[:,0]>0)&(np.linalg.norm(back[:,0]-corners[:,0],axis=1)<=1.5);diag.update(corners=len(corners),survived=int(good.sum()))
   if good.sum()<6 or good.mean()<.5:return dict(diag,reason='forward_backward_failed')
   delta=(nxt-corners)[:,0][good];median=np.median(delta,axis=0);keep=np.linalg.norm(delta-median,axis=1)<=3;diag['inliers']=int(keep.sum())
   if keep.sum()<6 or keep.mean()<.5:return dict(diag,reason='inconsistent_flow')
   shift=np.median(delta[keep],axis=0);return dict(diag,ok=True,reason='ok',predicted_wrist=(q[0]+shift).tolist(),shift=shift.tolist())
  for row in source.get(c['id'],[]):
   i=row['frame'];record=dict(id=c['id'],frame=i,tracks={},candidates=row['candidates'],accepted=[],rejections=[]);tracks={};proposals=[]
   for s in [0,1]:
    before=track(i,seed(i,s,-1));after=track(i,seed(i,s,1));attempts+=2;pair=dict(before=before,after=after,agrees=False);record['tracks'][s]=pair
    if before['ok'] and after['ok']:
     span=(before['span']+after['span'])/2;diff=float(np.linalg.norm(np.array(before['predicted_wrist'])-after['predicted_wrist']));pair.update(distance=diff,agrees=bool(diff<=max(12,.25*span)))
     if pair['agrees']:tracks[s]=(np.array(before['predicted_wrist']),np.array(after['predicted_wrist']),span)
   for k,ca in enumerate(row['candidates']):
    if not ca['valid']:continue
    q=np.array(ca['points']);costs={s:max(float(np.linalg.norm(q[0]-w0)),float(np.linalg.norm(q[0]-w1)))/max(24,.75*span) for s,(w0,w1,span) in tracks.items()};ok={s:v for s,v in costs.items() if v<=1}
    if not ok:record['rejections'].append(dict(candidate=k,reason='no_bidirectional_match'));continue
    if len(ok)==2 and abs(ok[0]-ok[1])<.2:record['rejections'].append(dict(candidate=k,reason='ambiguous_side'));continue
    s=min(ok,key=ok.get)
    if not flags[i,s]:record['rejections'].append(dict(candidate=k,reason='retained_side'));continue
    opp=a[i,[54,33][s]:[54,33][s]+21]
    if np.any(opp) and np.linalg.norm(q[0]-opp[0])<24:record['rejections'].append(dict(candidate=k,reason='duplicate'));continue
    proposals.append((ok[s],s,k,q))
   used=set()
   for cost,s,k,q in sorted(proposals,key=lambda x:x[0]):
    if s in used:continue
    used.add(s);b=[33,54][s];out[i,b:b+21]=q;record['accepted'].append(dict(side=s,candidate=k,cost=cost));accepted+=1
   events.append(record)
  assert np.isfinite(out).all() and np.array_equal(out[:,:33],a[:,:33])
  for s,b in enumerate([33,54]):assert np.array_equal(out[~flags[:,s],b:b+21],a[~flags[:,s],b:b+21])
  np.savez_compressed(O/'arrays'/(c['id']+'.npz'),baseline=a,forward=previous['independent'],bidirectional=out);(O/'events'/(c['id']+'.json')).write_text(json.dumps(events),encoding='utf-8')
  if (int(c['id'])+1)%20==0:print('EXTRACT',c['id'],'accepted',accepted,flush=True)
 ex=dict(seconds=time.perf_counter()-started,tracking_attempts=attempts,accepted_hand_frames=accepted,protocol_sha256=hashlib.sha256((O/'PROTOCOL.md').read_bytes()).hexdigest());(O/'extraction.json').write_text(json.dumps(ex,indent=2),encoding='utf-8');print(ex)
def score():
 import torch
 sys.path.insert(0,str(A));import run_audit as audit
 ev=mod('bi_metrics',D/'rtm-eval.py');model,norm,vocab=audit.load_model();pred={v:[] for v in ['baseline','forward','bidirectional']};old=read(OLD/'exp2-results.json')['predictions'];(O/'inference').mkdir(exist_ok=True)
 for idx,c in enumerate(read(B/'capture_manifest.json')):
  z=np.load(O/'arrays'/(c['id']+'.npz'));archive={}
  for v in pred:
   x=norm({'frames':torch.from_numpy(z[v][:,audit.PICK].transpose(2,0,1).copy())})['frames']
   with torch.inference_mode():log=model(x.unsqueeze(0))[0];p=log.softmax(0)
   t=p.topk(5);r=dict(id=c['id'],label=c['label'],prediction=vocab[int(t.indices[0])],top5=[vocab[int(k)] for k in t.indices],target_probability=float(p[vocab.index(c['label'])]));pred[v].append(r);archive[v+'_input']=x.numpy();archive[v+'_logits']=log.numpy()
   if v!='bidirectional':
    before=old['baseline' if v=='baseline' else 'independent'][idx];assert r['top5']==before['top5'] and abs(r['target_probability']-before['target_probability'])<1e-5
  np.savez_compressed(O/'inference'/(c['id']+'.npz'),**archive)
  if (idx+1)%20==0:print('SCORE',idx+1,flush=True)
 gains=[b['label'] for a,b in zip(pred['baseline'],pred['bidirectional']) if a['prediction']!=a['label'] and b['prediction']==b['label']];losses=[b['label'] for a,b in zip(pred['baseline'],pred['bidirectional']) if a['prediction']==a['label'] and b['prediction']!=b['label']];out=dict(summary={v:ev.metrics(rr) for v,rr in pred.items()},gains=gains,losses=losses,predictions=pred,baselines_reproduced=True);(O/'results.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(out['summary'])
if __name__=='__main__':{'extract':extract,'score':score}[sys.argv[1]]()
