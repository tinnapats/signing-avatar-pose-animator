import sys,json,time,hashlib,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
D=Path(__file__).resolve().parent;O=D/'improve-exp12-20261013';B=D/'oracle200-20261011';P=D/'pixel-only-plan-20261012';A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def extract():
 base=module('ind_base',D/'rtm-avatar.py');anime=module('ind_anime',D/'anime-hands.py');cv2,RTMPose,_=base.setup();import onnxruntime as ort
 detpath=A/'pixels30_20261007/anime_hand_trial_20261009/model.onnx';hp=A/'pixels30_20261007/rtm_trial_20261008/model_cache/hub/checkpoints/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.onnx'
 assert hashlib.sha256(detpath.read_bytes()).hexdigest()=='408750ad39645fcdc0c5e774aa45a73941b2e785fc5611fb7d3d9790a41899c0'
 assert hashlib.sha256(hp.read_bytes()).hexdigest()=='39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29'
 detector=ort.InferenceSession(str(detpath),providers=['CPUExecutionProvider']);hand=RTMPose(str(hp),model_input_size=(256,256),backend='onnxruntime',device='cpu');(O/'exp2-arrays').mkdir(exist_ok=True);events=[];start=time.perf_counter();framecount=0;flags_total=0
 for c in read(B/'capture_manifest.json'):
  z=np.load(P/'arrays'/(c['id']+'.npz'));rt=np.load(B/'rtm'/(c['id']+'.npz'));raw=z['raw'];flags=z['flags'];a=raw.copy();flags_total+=int(flags.sum())
  for s,b in enumerate([33,54]):a[flags[:,s],b:b+21]=0
  candidate=a.copy();cache={}
  def frame(i):
   if i not in cache:
    f=c['frames'][i];p=B/'frames'/c['id']/f['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];im=base.image(cv2,p);blue,green,red=[im[:,:,k].astype(np.int16) for k in range(3)];skin=(red>=100)&(red-green>=15)&(red-blue>=20)&(green>=45)&(green-blue>=-5);cache[i]=(im,cv2.distanceTransform((~skin).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE))
   return cache[i]
  def quality(q,sc,i):
   span=float(max(q.max(0)-q.min(0)));conf=bool(min(sc.mean(),sc[0])>=.3)
   im,dist=frame(i);xy=np.rint(q).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<im.shape[1])&(xy[:,1]>=0)&(xy[:,1]<im.shape[0]);ds=np.full(21,999.);ds[inside]=dist[xy[inside,1],xy[inside,0]];support=bool(ds[0]<=6 and (ds>6).mean()<=.5)
   return conf and support and span>=10,dict(confidence_ok=conf,skin_ok=support,span=span)
  for i in np.flatnonzero(flags.any(1)):
   i=int(i);framecount+=1;im=frame(i)[0];boxes,boxscores=anime.detect(detector,im);seeds={};proposals=[]
   # Side identity comes only from a previous image-derived, reliable hand.
   for s,b in enumerate([33,54]):
    for j in range(i-1,max(-1,i-4),-1):
     if flags[j,s] or not np.any(raw[j,b:b+21]):continue
     q=raw[j,b:b+21];valid,diag=quality(q,rt['rtmw_det_scores'][j,91+21*s:112+21*s],j)
     if valid:seeds[s]=(j,q,diag['span']);break
   row=dict(id=c['id'],frame=i,flagged=flags[i].tolist(),seeds={str(s):j for s,(j,_,_) in seeds.items()},candidates=[],accepted=[])
   for k,bb in enumerate(boxes):
    q,sc=hand(im,bboxes=[bb]);q,sc=q[0],sc[0];valid,diag=quality(q,sc,i);entry=dict(box=bb.tolist(),boxscore=float(boxscores[k]),points=q.tolist(),scores=sc.tolist(),quality=diag,valid=valid);row['candidates'].append(entry)
    if not valid:continue
    costs={s:float(np.linalg.norm(q[0]-seed[0]))/max(24,.75*span) for s,(j,seed,span) in seeds.items()};compatible={s:v for s,v in costs.items() if v<=1};entry['costs']=costs
    if len(compatible)==2 and abs(compatible[0]-compatible[1])<.2:entry['rejection']='ambiguous_side';continue
    if not compatible:entry['rejection']='no_temporal_match';continue
    s=min(compatible,key=compatible.get)
    if not flags[i,s]:entry['rejection']='matches_retained_side';continue
    opposite=a[i,[54,33][s]:[54,33][s]+21]
    if np.any(opposite) and np.linalg.norm(q[0]-opposite[0])<24:entry['rejection']='duplicate_retained_hand';continue
    proposals.append((compatible[s],s,k,q))
   used=set()
   for cost,s,k,q in sorted(proposals,key=lambda x:x[0]):
    if s in used:continue
    used.add(s);b=[33,54][s];candidate[i,b:b+21]=q;row['accepted'].append(dict(side=s,candidate=k,cost=cost))
   events.append(row)
  assert np.array_equal(a[:,:33],candidate[:,:33])
  for s,b in enumerate([33,54]):assert np.array_equal(a[~flags[:,s],b:b+21],candidate[~flags[:,s],b:b+21])
  np.savez_compressed(O/'exp2-arrays'/(c['id']+'.npz'),baseline=a,independent=candidate)
  if (int(c['id'])+1)%20==0:print('E2 EXTRACT',c['id'],flush=True)
 out=dict(events=events,seconds=time.perf_counter()-start,flagged_hand_frames=flags_total,processed_frames=framecount,accepted_hand_frames=sum(len(r['accepted']) for r in events),boxes=sum(len(r['candidates']) for r in events),protocol_sha256=hashlib.sha256((O/'PROTOCOL.md').read_bytes()).hexdigest())
 (O/'exp2-extraction.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='events'}),flush=True)
def score():
 import torch
 sys.path.insert(0,str(A));import run_audit as audit
 ev=module('ind_metrics',D/'rtm-eval.py');model,norm,vocab=audit.load_model();pred={v:[] for v in ['baseline','independent']};old=read(P/'results.json')['predictions']['support_gate'];(O/'exp2-inference').mkdir(exist_ok=True)
 for c,before in zip(read(B/'capture_manifest.json'),old):
  z=np.load(O/'exp2-arrays'/(c['id']+'.npz'));archive={}
  for v in pred:
   x=norm({'frames':torch.from_numpy(z[v][:,audit.PICK].transpose(2,0,1).copy())})['frames']
   with torch.inference_mode():log=model(x.unsqueeze(0))[0];p=log.softmax(0)
   top=p.topk(5);row=dict(id=c['id'],label=c['label'],prediction=vocab[int(top.indices[0])],top5=[vocab[int(k)] for k in top.indices],target_probability=float(p[vocab.index(c['label'])]));pred[v].append(row);archive[v+'_input']=x.numpy();archive[v+'_logits']=log.numpy()
   if v=='baseline':assert row['top5']==before['top5'] and abs(row['target_probability']-before['target_probability'])<1e-5
  np.savez_compressed(O/'exp2-inference'/(c['id']+'.npz'),**archive)
  if (int(c['id'])+1)%20==0:print('E2 SCORE',c['id'],flush=True)
 pairs=dict(gains=[b['label'] for a,b in zip(pred['baseline'],pred['independent']) if a['prediction']!=a['label'] and b['prediction']==b['label']],losses=[b['label'] for a,b in zip(pred['baseline'],pred['independent']) if a['prediction']==a['label'] and b['prediction']!=b['label']])
 out=dict(summary={v:ev.metrics(rows) for v,rows in pred.items()},pairs=pairs,predictions=pred,baseline_reproduced=True);(O/'exp2-results.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out['summary']),flush=True)
if __name__=='__main__':{'extract':extract,'score':score}[sys.argv[1]]()
