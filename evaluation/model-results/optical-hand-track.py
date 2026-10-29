import sys,time,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('ah',Path(__file__).with_name('anime-hands.py'));ah=importlib.util.module_from_spec(s);s.loader.exec_module(ah)
np=ah.np;B=ah.B;R=ah.R;A=ah.O;O=Path(__file__).parent/'optical-hand-track-20261009';O.mkdir(exist_ok=True);read=ah.read;save=ah.save;sha=ah.sha
def extract():
 info=read(R/'hand_model.json');assert sha(Path(info['path']))==info['sha256']
 protocol=dict(manifest_sha256=sha(B/'capture_manifest.json'),model=info,production=read(A/'protocol.json')['production'],rules='Development30 only. Baseline anime boxes/accepted hands cached unchanged. Seed tracking only from accepted detector+pose hand; sparse pyramidal Lucas-Kanade image feature translation, re-read RTMPose-hand current RGB when baseline hand absent. Max3frames since last accepted detector seed, never extend TTL with tracking success. No holding/averaging landmarks, no renderer/payload input. Original handconfidence .3/body .3/wristdistance .65forearm. Existing baseline accepted hand always wins; tracked wrists within20px of another used wrist rejected. On failure use previous_best MP/RTMW fallback, not frozen points.',tracker=dict(corners=60,quality=.01,min_distance=3,lk_window=21,levels=3,fb_error_max=1.5,residual_max_px=3,min_inliers=6,min_inlier_fraction=.5,box_scale='fixed; median optical-flow translation only',max_gap=3),comparison=['baseline','flow_fallback'])
 if (O/'protocol.json').exists():assert read(O/'protocol.json')==protocol
 else:save(O/'protocol.json',protocol)
 cv2,rt,det=ah.base.setup();hand=rt(info['path'],model_input_size=(256,256),backend='onnxruntime',device='cpu')
 def features(gray,box):
  mask=np.zeros_like(gray);x1,y1,x2,y2=np.rint(box).astype(int);x1=max(x1,0);y1=max(y1,0);x2=min(x2,gray.shape[1]-1);y2=min(y2,gray.shape[0]-1)
  if x2<=x1 or y2<=y1:return None
  mask[y1:y2+1,x1:x2+1]=255;return cv2.goodFeaturesToTrack(gray,maxCorners=60,qualityLevel=.01,minDistance=3,mask=mask)
 for c in read(B/'capture_manifest.json'):
  target=O/(c['id']+'.npz')
  if target.exists():assert sha(target)==read(target.with_suffix('.json'))['sha256'];continue
  raw=read(A/(c['id']+'_raw.json'));old=np.load(A/(c['id']+'.npz'));body=np.load(R/(c['id']+'.npz'));states={};prevgray=None;points=[];masks=[];times=[];records=[]
  for i,f in enumerate(c['frames']):
   path=B/'frames'/c['id']/f['file'];assert sha(path)==f['sha256'];im=ah.base.image(cv2,path);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);start=time.perf_counter();p=np.zeros((2,21,2),np.float32);mask=np.zeros(2,bool);proposals=[];logs=[]
   for side in [0,1]:
    if old['plain_mask'][i,side]:
     j=raw[i]['assignment']['plain'][side];box=np.array(raw[i]['boxes'][j],np.float32);states[side]=dict(box=box,seed=i,features=features(gray,box));logs.append(dict(side=side,status='detector_seed'));continue
    log=dict(side=side,status='no_seed');logs.append(log)
    if side not in states:continue
    st=states[side];log['age']=i-st['seed'];q=st['features']
    if i-st['seed']>3 or q is None or len(q)<6:log['status']='expired_or_few_features';states.pop(side,None);continue
    nxt,ok,_=cv2.calcOpticalFlowPyrLK(prevgray,gray,q,None,winSize=(21,21),maxLevel=3);back,ok2,_=cv2.calcOpticalFlowPyrLK(gray,prevgray,nxt,None,winSize=(21,21),maxLevel=3) if nxt is not None else (None,None,None)
    if nxt is None or back is None:log['status']='flow_failed';states.pop(side,None);continue
    good=(ok.ravel()>0)&(ok2.ravel()>0)&(np.linalg.norm(back-q,axis=2).ravel()<=1.5);diff=(nxt-q).reshape(-1,2);shift=np.median(diff[good],axis=0) if good.any() else np.zeros(2);inlier=good&(np.linalg.norm(diff-shift,axis=1)<=3);log['inliers']=int(inlier.sum());log['features']=len(q)
    if inlier.sum()<6 or inlier.mean()<.5:log['status']='flow_inconsistent';states.pop(side,None);continue
    shift=np.median(diff[inlier],axis=0);box=st['box']+np.tile(shift,2);log['box']=box.tolist();states[side]=dict(box=box,seed=st['seed'],features=features(gray,box))
    pp,ss=hand(im,bboxes=[box]);pp=pp[0];ss=ss[0];log.update(points=pp.tolist(),scores=ss.tolist());w,e=[(9,7),(10,8)][side];bp=body['rtmw_det'][i];bs=body['rtmw_det_scores'][i];d=float(np.linalg.norm(pp[0]-bp[w])/max(np.linalg.norm(bp[w]-bp[e]),10));log['wrist_ratio']=d
    if ss.mean()<.3 or ss[0]<.3:log['status']='pose_confidence_rejected';continue
    if bs[w]<.3 or bs[e]<.3 or d>.65:log['status']='body_wrist_rejected';continue
    log['status']='proposal';proposals.append((d,side,pp,log))
   used=[old['plain'][i,side,0] for side in [0,1] if old['plain_mask'][i,side]]
   for d,side,pp,log in sorted(proposals,key=lambda x:x[0]):
    if any(np.linalg.norm(pp[0]-u)<20 for u in used):log['status']='duplicate_wrist_rejected';continue
    log['status']='accepted_flow';p[side]=pp;mask[side]=True;used.append(pp[0])
   points.append(p);masks.append(mask);times.append((time.perf_counter()-start)*1000);records.append(logs);prevgray=gray
  np.savez_compressed(target,points=np.array(points),mask=np.array(masks),times=old['times'],ms=np.array(times));save(O/(c['id']+'_raw.json'),records);save(target.with_suffix('.json'),dict(label=c['label'],sha256=sha(target),raw_sha256=sha(O/(c['id']+'_raw.json')),added=int(np.sum(masks))));print('TRACKED',c['id'],c['label'],'added',np.sum(masks),flush=True)
 print('EXTRACTION COMPLETE',flush=True)
def score():
 s=importlib.util.spec_from_file_location('ae',Path(__file__).with_name('anime-eval.py'));ae=importlib.util.module_from_spec(s);s.loader.exec_module(ae);e=ae.e
 def data(c):
  a=ae.data(c);z=np.load(O/(c['id']+'.npz'));assert sha(O/(c['id']+'.npz'))==read(O/(c['id']+'.json'))['sha256'];old=np.load(A/(c['id']+'.npz'));assert not np.any(z['mask']&old['plain_mask']);assert np.array_equal(z['times'],old['times']);result=a['anime_replace_best'].copy()
  for side,off in enumerate([33,54]):
   mask=z['mask'][:,side];result[mask,off:off+21]=z['points'][mask,side]
  out=dict(mediapipe=a['mediapipe'],baseline=a['anime_replace_best'],flow_fallback=result);folder=O/'inputs';folder.mkdir(exist_ok=True);np.savez_compressed(folder/(c['id']+'.npz'),**out,times=z['times']);return out
 e.data=data;e.VARIANTS=['mediapipe','baseline','flow_fallback'];e.RESULT_NAME=str(O/'results.json');e.evaluate();new=read(O/'results.json');old=read(A/'results.json')
 for x,y in zip(new['predictions']['baseline'],old['predictions']['anime_replace_best']):assert x['prediction']==y['prediction'] and x['top5']==y['top5'] and np.isclose(x['target_probability'],y['target_probability'],atol=1e-6)
 print('BASELINE VERIFIED')
if __name__=='__main__':score() if len(sys.argv)>1 and sys.argv[1]=='score' else extract()
