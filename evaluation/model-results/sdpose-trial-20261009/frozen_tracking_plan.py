"""Fixed 3-frame image tracker on development30 and disjoint historical confirmation30."""

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
ROOT=Path(__file__).resolve().parent;V=ROOT.parent
A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit');B=A/'pixels30_20261007';R=B/'rtm_trial_20261008';T=ROOT/'wrist_crop_tight20_20261011'
C=V/'confirm30-crop-20261009';O=ROOT/'frozen_tracking_plan_20261011';O.mkdir(exist_ok=True)
read=lambda p:_local_json_loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
SETS={'development':(B,R),'confirmation':(C,C/'rtm')}
def register():
 labels=[{c['label'] for c in read(p/'capture_manifest.json')} for p,_ in SETS.values()];assert not labels[0]&labels[1]
 p={'status':'registered','sets':'30 development plus30 disjoint historical confirmation; confirmation was used in earlier unrelated experiments, not a pristine test set',
 'seed':'Accepted tight20 hand, current RTMW wrist/elbow >=.30 and >=6 trackable image corners. Acceptance includes wrist+mean pose .30 and wrist distance <=.65forearm; duplicate20px rule.',
 'tracking':'Only when current tight20 hand rejected. Max3frames since last detector seed, never renewed by tracking. LK corners60 quality.01 distance3, window21 levels3, forward-back error<=1.5px, median translation residual<=3px, min6 inliers and fraction>=.5. Translate bbox then rerun hand model on CURRENT image. Reject failed confidence/body match/duplicate. Existing accepted crop always wins; otherwise RTMW fallback.',
 'unchanged':'No smoothing, bone constraints, wrist translation or reference-fed correction. Fixed rules before both sets. No tuning after scores.',
 'quality':'Separate flow consistency, pose confidence, wrist/body and duplicate rejection reasons; not a single composite score',
 'model':read(R/'hand_model.json'),'hashes':{str(p):sha(p) for p in [Path(__file__),V/'optical-hand-track.py',T/'protocol.json']}}
 save(O/'protocol.json',p)
def extract():
 if not (O/'protocol.json').exists():register()
 base=module('base',V/'rtm-avatar.py');cv2,rt,_=base.setup();info=read(R/'hand_model.json');assert sha(Path(info['path']))==info['sha256']
 hand=rt(info['path'],model_input_size=(256,256),backend='onnxruntime',device='cpu')
 def features(gray,box):
  mask=np.zeros_like(gray);x1,y1,x2,y2=np.rint(box).astype(int);x1=max(0,x1);y1=max(0,y1);x2=min(gray.shape[1]-1,x2);y2=min(gray.shape[0]-1,y2)
  if x2<=x1 or y2<=y1:return None
  mask[y1:y2+1,x1:x2+1]=255;return cv2.goodFeaturesToTrack(gray,maxCorners=60,qualityLevel=.01,minDistance=3,mask=mask)
 for setname,(folder,bodyfolder) in SETS.items():
  outdir=O/setname;outdir.mkdir(exist_ok=True)
  for c in read(folder/'capture_manifest.json'):
   cid=c['id'];target=outdir/f'{cid}.npz'
   if target.exists():assert sha(target)==read(target.with_suffix('.json'))['sha256'];continue
   assert sha(bodyfolder/f'{cid}.npz')==read(bodyfolder/f'{cid}.json')['sha256']
   body=np.load(bodyfolder/f'{cid}.npz');cached=np.load(T/f'{cid}.npz') if setname=='development' else None
   states={};previous=None;out={k:[] for k in ['points','mask','boxes','tracked','track_mask','base_ms','track_ms']};logs=[]
   for i,f in enumerate(c['frames']):
    path=folder/'frames'/cid/f['file'];assert sha(path)==f['sha256'];im=base.image(cv2,path);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);bp=body['rtmw_det'][i];bs=body['rtmw_det_scores'][i]
    start=time.perf_counter()
    if cached is not None:p=cached['points'][i].copy();accepted=cached['accepted'][i].copy();boxes=cached['boxes'][i].copy();basems=float(cached['ms'][i])
    else:
     p=np.zeros((2,21,2),np.float32);accepted=np.zeros(2,bool);boxes=np.zeros((2,4),np.float32);dist=np.full(2,np.inf)
     for side,(w,e) in enumerate([(9,7),(10,8)]):
      if min(bs[w],bs[e])<.3:continue
      delta=bp[w]-bp[e];length=float(np.linalg.norm(delta))
      if length<10:continue
      center=bp[w]+.25*delta;size=np.clip(2.2*length,96,256);original=np.r_[center-size/2,center+size/2]
      mid=(original[:2]+original[2:])/2;half=(original[2:]-original[:2])*.8/2;box=np.r_[mid-half,mid+half];boxes[side]=box
      pp,ss=hand(im,bboxes=[box]);p[side]=pp[0];dist[side]=np.linalg.norm(p[side,0]-bp[w]);accepted[side]=ss[0].mean()>=.3 and ss[0,0]>=.3 and dist[side]<=.65*length
     if accepted.all() and np.linalg.norm(p[0,0]-p[1,0])<20:accepted[int(np.argmax(dist))]=False
     basems=1000*(time.perf_counter()-start)
    start=time.perf_counter();tp=np.zeros_like(p);tm=np.zeros(2,bool);frame_logs=[];proposals=[]
    for side,(w,e) in enumerate([(9,7),(10,8)]):
     log={'side':side,'status':'no_seed'};frame_logs.append(log)
     if accepted[side]:
      q=features(gray,boxes[side])
      if min(bs[w],bs[e])>=.3 and q is not None and len(q)>=6:states[side]={'seed':i,'box':boxes[side].copy(),'features':q};log['status']='seed'
      else:states.pop(side,None);log['status']='seed_quality_failed'
      continue
     if side not in states:continue
     st=states[side];q=st['features'];log['age']=i-st['seed']
     if i-st['seed']>3 or q is None or len(q)<6:states.pop(side,None);log['status']='expired_or_few_features';continue
     nxt,ok,_=cv2.calcOpticalFlowPyrLK(previous,gray,q,None,winSize=(21,21),maxLevel=3)
     back,ok2,_=cv2.calcOpticalFlowPyrLK(gray,previous,nxt,None,winSize=(21,21),maxLevel=3) if nxt is not None else (None,None,None)
     if nxt is None or back is None:states.pop(side,None);log['status']='flow_failed';continue
     good=(ok.ravel()>0)&(ok2.ravel()>0)&(np.linalg.norm(back-q,axis=2).ravel()<=1.5)
     diff=(nxt-q).reshape(-1,2);shift=np.median(diff[good],axis=0) if good.any() else np.zeros(2);inlier=good&(np.linalg.norm(diff-shift,axis=1)<=3)
     log['inliers']=int(inlier.sum());log['corners']=len(q)
     if inlier.sum()<6 or inlier.mean()<.5:states.pop(side,None);log['status']='flow_inconsistent';continue
     box=st['box']+np.tile(np.median(diff[inlier],axis=0),2);states[side]={'seed':st['seed'],'box':box,'features':features(gray,box)}
     pp,ss=hand(im,bboxes=[box]);pp=pp[0];ss=ss[0];ratio=float(np.linalg.norm(pp[0]-bp[w])/max(np.linalg.norm(bp[w]-bp[e]),10))
     log.update(wrist_ratio=ratio,mean_confidence=float(ss.mean()),wrist_confidence=float(ss[0]))
     if ss.mean()<.3 or ss[0]<.3:log['status']='pose_confidence_failed';continue
     if min(bs[w],bs[e])<.3 or ratio>.65:log['status']='body_match_failed';continue
     proposals.append((ratio,side,pp,log));log['status']='proposal'
    used=[p[s,0] for s in [0,1] if accepted[s]]
    for _,side,pp,log in sorted(proposals,key=lambda x:x[0]):
     if any(np.linalg.norm(pp[0]-u)<20 for u in used):log['status']='duplicate_failed';continue
     tp[side]=pp;tm[side]=True;used.append(pp[0]);log['status']='accepted_track'
    assert not np.any(tm&accepted)
    for k,v in [('points',p),('mask',accepted),('boxes',boxes),('tracked',tp),('track_mask',tm),('base_ms',basems),('track_ms',1000*(time.perf_counter()-start))]:out[k].append(v)
    logs.append(frame_logs);previous=gray
   np.savez_compressed(target,**{k:np.array(v) for k,v in out.items()},times=body['times']);save(target.with_suffix('.json'),{'label':c['label'],'sha256':sha(target),'added':int(np.sum(out['track_mask']))});save(outdir/f'{cid}_logs.json',logs)
   print(setname,cid,c['label'],'added',int(np.sum(out['track_mask'])),flush=True)
 p=read(O/'protocol.json');p['status']='extracted';save(O/'protocol.json',p)
if __name__=='__main__':extract()
