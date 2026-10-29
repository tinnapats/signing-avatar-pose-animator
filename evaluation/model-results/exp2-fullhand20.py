import sys,json,time,hashlib,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
D=Path(__file__).resolve().parent;O=D/'exp2-fullhand20-20261014';B=D/'oracle200-20261011';P=D/'pixel-only-plan-20261012';A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def prepare():
 mf=read(B/'capture_manifest.json');ids=sorted([c['id'] for c in mf],key=lambda cid:hashlib.sha256(('exp2-fullhand20-20261014:'+cid).encode()).hexdigest())[:20];out=dict(ids=ids,total_frames=sum(len(c['frames']) for c in mf if c['id'] in ids),protocol_sha256=hashlib.sha256((O/'PROTOCOL.md').read_bytes()).hexdigest());save(O/'selection.json',out);print(out)
def extract():
 base=mod('fh_base',D/'rtm-avatar.py');anime=mod('fh_anime',D/'anime-hands.py');cv2,RTMPose,_=base.setup();import onnxruntime as ort
 dp=A/'pixels30_20261007/anime_hand_trial_20261009/model.onnx';hp=A/'pixels30_20261007/rtm_trial_20261008/model_cache/hub/checkpoints/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.onnx';assert hashlib.sha256(dp.read_bytes()).hexdigest()=='408750ad39645fcdc0c5e774aa45a73941b2e785fc5611fb7d3d9790a41899c0';assert hashlib.sha256(hp.read_bytes()).hexdigest()=='39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29';det=ort.InferenceSession(str(dp),providers=['CPUExecutionProvider']);hand=RTMPose(str(hp),model_input_size=(256,256),backend='onnxruntime',device='cpu');mf={c['id']:dict(id=c['id'],frames=c['frames']) for c in read(B/'capture_manifest.json')};(O/'arrays').mkdir(exist_ok=True);(O/'events').mkdir(exist_ok=True);start=time.perf_counter();counts=dict(frames=0,boxes=0,valid_candidates=0,accepted=0)
 for cid in read(O/'selection.json')['ids']:
  c=mf[cid];z=np.load(P/'arrays'/(cid+'.npz'));raw=z['raw'];rt=np.load(B/'rtm'/(cid+'.npz'));gate=raw.copy()
  for s,b in enumerate([33,54]):gate[z['flags'][:,s],b:b+21]=0
  out=raw.copy();out[:,33:]=0;cache={};events=[]
  def frame(i):
   if i not in cache:
    f=c['frames'][i];p=B/'frames'/cid/f['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];im=base.image(cv2,p);b,g,r=[im[:,:,k].astype(np.int16) for k in range(3)];skin=(r>=100)&(r-g>=15)&(r-b>=20)&(g>=45)&(g-b>=-5);cache[i]=(im,cv2.distanceTransform((~skin).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE))
   return cache[i]
  def quality(q,sc,i):
   im,dist=frame(i);xy=np.rint(q).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<im.shape[1])&(xy[:,1]>=0)&(xy[:,1]<im.shape[0]);ds=np.full(21,999.);ds[inside]=dist[xy[inside,1],xy[inside,0]];span=float(max(q.max(0)-q.min(0)));frac=float((ds>6).mean());return bool(np.isfinite(q).all() and min(sc.mean(),sc[0])>=.3 and span>=10 and ds[0]<=6 and frac<=.5),span,frac
  for i in range(len(raw)):
   im=frame(i)[0];boxes,bs=anime.detect(det,im);counts['frames']+=1;counts['boxes']+=len(boxes);anchors={};proposals=[];rec=dict(frame=i,anchors={},candidates=[],selected=[])
   for s,b in enumerate([33,54]):
    for j in range(i,max(-1,i-4),-1):
     if z['flags'][j,s] or not np.any(raw[j,b:b+21]):continue
     q=raw[j,b:b+21];ok,span,_=quality(q,rt['rtmw_det_scores'][j,91+21*s:112+21*s],j)
     if ok:anchors[s]=(q[0],span);rec['anchors'][s]=j;break
   for k,bb in enumerate(boxes):
    q,sc=hand(im,bboxes=[bb]);q,sc=q[0],sc[0];ok,span,frac=quality(q,sc,i);ca=dict(box=bb.tolist(),points=q.tolist(),scores=sc.tolist(),valid=ok);rec['candidates'].append(ca)
    if not ok:continue
    counts['valid_candidates']+=1;costs={s:float(np.linalg.norm(q[0]-w))/max(24,.75*span0) for s,(w,span0) in anchors.items()};match={s:v for s,v in costs.items() if v<=1};ca['costs']=costs
    if not match:ca['rejection']='no_anchor_match';continue
    if len(match)==2 and abs(match[0]-match[1])<.2:ca['rejection']='ambiguous';continue
    s=min(match,key=match.get);proposals.append((match[s]+.25*frac,s,k,q))
   used=set();wrists=[]
   for cost,s,k,q in sorted(proposals,key=lambda t:(t[0],t[2])):
    if s in used or any(np.linalg.norm(q[0]-w)<24 for w in wrists):continue
    used.add(s);wrists.append(q[0]);b=[33,54][s];out[i,b:b+21]=q;rec['selected'].append(dict(side=s,candidate=k,cost=cost));counts['accepted']+=1
   events.append(rec)
   # Keep only the4 frames needed for current/past-anchor pixel checks.
   for old in list(cache):
    if old<i-3:del cache[old]
  assert np.array_equal(out[:,:33],raw[:,:33]) and np.isfinite(out).all();np.savez_compressed(O/'arrays'/(cid+'.npz'),raw=raw,support_gate=gate,fullhand=out);save(O/'events'/(cid+'.json'),events);print('CLIP',cid,'frames',len(raw),'accepted',sum(len(r['selected']) for r in events),flush=True)
 save(O/'extraction.json',dict(**counts,seconds=time.perf_counter()-start));print(counts)
def score():
 import torch
 sys.path.insert(0,str(A));import run_audit as audit
 ev=mod('fh_metrics',D/'rtm-eval.py');model,norm,vocab=audit.load_model();mf={c['id']:c for c in read(B/'capture_manifest.json')};old=read(P/'results.json')['predictions'];prior={v:{r['id']:r for r in old[k]} for v,k in [('raw','baseline'),('support_gate','support_gate')]};pred={v:[] for v in ['raw','support_gate','fullhand']};(O/'inference').mkdir(exist_ok=True)
 for cid in read(O/'selection.json')['ids']:
  c=mf[cid];z=np.load(O/'arrays'/(cid+'.npz'));archive={}
  for v in pred:
   x=norm({'frames':torch.from_numpy(z[v][:,audit.PICK].transpose(2,0,1).copy())})['frames']
   with torch.inference_mode():log=model(x.unsqueeze(0))[0];p=log.softmax(0)
   t=p.topk(5);r=dict(id=cid,label=c['label'],prediction=vocab[int(t.indices[0])],top5=[vocab[int(j)] for j in t.indices],target_probability=float(p[vocab.index(c['label'])]));pred[v].append(r);archive[v+'_input']=x.numpy();archive[v+'_logits']=log.numpy()
   if v in prior:assert r['top5']==prior[v][cid]['top5'] and abs(r['target_probability']-prior[v][cid]['target_probability'])<1e-5
  np.savez_compressed(O/'inference'/(cid+'.npz'),**archive);print('SCORE',cid,flush=True)
 pairs={v:dict(gains=[b['label'] for a,b in zip(pred[v],pred['fullhand']) if a['prediction']!=a['label'] and b['prediction']==b['label']],losses=[b['label'] for a,b in zip(pred[v],pred['fullhand']) if a['prediction']==a['label'] and b['prediction']!=b['label']]) for v in ['raw','support_gate']};out=dict(summary={v:ev.metrics(rr) for v,rr in pred.items()},pairs=pairs,predictions=pred,baselines_reproduced=True);save(O/'results.json',out);print(out['summary'])
if __name__=='__main__':{'prepare':prepare,'extract':extract,'score':score}[sys.argv[1]]()
