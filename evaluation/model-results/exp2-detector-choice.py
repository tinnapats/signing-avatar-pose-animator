
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
D=Path(__file__).resolve().parent;O=D/'exp2-detector-choice-20261013';OLD=D/'improve-exp12-20261013';B=D/'oracle200-20261011';P=D/'pixel-only-plan-20261012';A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit')
read=lambda p:_local_json_loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def rt():
 base=mod('choicebase',D/'rtm-avatar.py');cv2,RTMPose,_=base.setup();hp=A/'pixels30_20261007/rtm_trial_20261008/model_cache/hub/checkpoints/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.onnx';assert hashlib.sha256(hp.read_bytes()).hexdigest()=='39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29';hand=RTMPose(str(hp),model_input_size=(256,256),backend='onnxruntime',device='cpu');mf={c['id']:c['frames'] for c in read(B/'capture_manifest.json')};(O/'crops').mkdir(exist_ok=True);out=[];start=time.perf_counter()
 for row in read(OLD/'exp2-extraction.json')['events']:
  f=mf[row['id']][row['frame']];p=B/'frames'/row['id']/f['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];im=base.image(cv2,p)
  for k,ca in enumerate(row['candidates']):
   bb=np.array(ca['box']);center=(bb[:2]+bb[2:])/2;side=1.25*float(max(bb[2:]-bb[:2]));mat=np.array([[256/side,0,128-center[0]*256/side],[0,256/side,128-center[1]*256/side]]);inv=cv2.invertAffineTransform(mat);roi=cv2.warpAffine(im,mat,(256,256),borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255));name=f"{row['id']}-{row['frame']:04d}-{k}.png";cv2.imwrite(str(O/'crops'/name),roi);inner=np.array([25.6,25.6,230.4,230.4]);pre,_,_=hand.preprocess(roi,inner);restored=pre*np.array(hand.std)+np.array(hand.mean);assert np.max(np.abs(restored-roi))<1e-3
   q,sc=hand(roi,bboxes=[inner]);q=np.c_[q[0],np.ones(21)]@inv.T;out.append(dict(id=row['id'],frame=row['frame'],box_index=k,crop=name,crop_sha256=hashlib.sha256((O/'crops'/name).read_bytes()).hexdigest(),inverse=inv.tolist(),points=q.tolist(),scores=sc[0].tolist()))
  if len(out) and len(out)%50<3:print('RT',len(out),flush=True)
 save(O/'rt.json',dict(rows=out,seconds=time.perf_counter()-start));print('RT COMPLETE',len(out))
def mp():
 import cv2,mediapipe as mp
 cv2.setNumThreads(1);rows=[];start=time.perf_counter()
 with mp.solutions.hands.Hands(static_image_mode=True,max_num_hands=2,model_complexity=1,min_detection_confidence=.5) as hand:
  for i,r in enumerate(read(O/'rt.json')['rows']):
   p=O/'crops'/r['crop'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['crop_sha256'];im=cv2.imread(str(p));res=hand.process(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));points=[]
   for h in res.multi_hand_landmarks or []:points.append((np.array([[j.x*256,j.y*256,1] for j in h.landmark])@np.array(r['inverse']).T).tolist())
   rows.append(dict(crop=r['crop'],hands=points))
   if (i+1)%100==0:print('MP',i+1,flush=True)
 save(O/'mp.json',dict(rows=rows,seconds=time.perf_counter()-start));print('MP COMPLETE',len(rows))
def select():
 import cv2
 cv2.setNumThreads(1);base=mod('selbase',D/'rtm-avatar.py');rt=read(O/'rt.json')['rows'];mp=read(O/'mp.json')['rows'];assert len(rt)==len(mp)==497;group={}
 for r,m in zip(rt,mp):
  assert r['crop']==m['crop'];group.setdefault((r['id'],r['frame']),[]).append((r,m))
 old={(r['id'],r['frame']):r for r in read(OLD/'exp2-extraction.json')['events']};events=[];(O/'arrays').mkdir(exist_ok=True);start=time.perf_counter()
 for c in read(B/'capture_manifest.json'):
  z=np.load(P/'arrays'/(c['id']+'.npz'));a=np.load(OLD/'exp2-arrays'/(c['id']+'.npz'))['baseline'];outs={v:a.copy() for v in ['RT','MP','mixed']}
  for i in np.flatnonzero(z['flags'].any(1)):
   i=int(i);proposals=[];f=c['frames'][i];im=base.image(cv2,B/'frames'/c['id']/f['file']);b,g,r=[im[:,:,k].astype(np.int16) for k in range(3)];skin=(r>=100)&(r-g>=15)&(r-b>=20)&(g>=45)&(g-b>=-5);dist=cv2.distanceTransform((~skin).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);ev=dict(id=c['id'],frame=i,proposals=[],selected={})
   for rr,mm in group.get((c['id'],i),[]):
    candidates=[('RT',rr['points'],bool(min(np.mean(rr['scores']),rr['scores'][0])>=.3))]+[('MP',q,True) for q in mm['hands']]
    for model,points,confok in candidates:
     q=np.array(points);span=float(max(q.max(0)-q.min(0)));xy=np.rint(q).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<im.shape[1])&(xy[:,1]>=0)&(xy[:,1]<im.shape[0]);ds=np.full(21,999.);ds[inside]=dist[xy[inside,1],xy[inside,0]];valid=bool(confok and np.isfinite(q).all() and span>=10 and ds[0]<=6 and (ds>6).mean()<=.5);entry=dict(model=model,box_index=rr['box_index'],points=points,valid=valid);ev['proposals'].append(entry)
     if not valid:continue
     costs={}
     for ss,j in old[c['id'],i]['seeds'].items():
      s=int(ss);seed=z['raw'][j,[33,54][s]:[33,54][s]+21];costs[s]=float(np.linalg.norm(q[0]-seed[0]))/max(24,.75*float(max(seed.max(0)-seed.min(0))))
     ok={s:v for s,v in costs.items() if v<=1}
     if not ok or (len(ok)==2 and abs(ok[0]-ok[1])<.2):continue
     s=min(ok,key=ok.get)
     if not z['flags'][i,s]:continue
     opp=a[i,[54,33][s]:[54,33][s]+21]
     if np.any(opp) and np.linalg.norm(q[0]-opp[0])<24:continue
     score=ok[s]+.25*float((ds>6).mean());entry.update(side=s,quality_score=score);proposals.append((score,0 if model=='RT' else 1,rr['box_index'],len(ev['proposals'])-1,s,q,model))
   for v in outs:
    used=set();boxes=set();wrists=[];sel=[]
    for score,_,box,k,s,q,model in sorted(proposals,key=lambda t:t[:4]):
     if v!='mixed' and model!=v:continue
     if s in used or box in boxes or any(np.linalg.norm(q[0]-w)<24 for w in wrists):continue
     used.add(s);boxes.add(box);wrists.append(q[0]);off=[33,54][s];outs[v][i,off:off+21]=q;sel.append(dict(proposal=k,side=s,model=model))
    ev['selected'][v]=sel
   events.append(ev)
  for out in outs.values():
   assert np.array_equal(a[:,:33],out[:,:33]) and np.isfinite(out).all()
   for s,b in enumerate([33,54]):assert np.array_equal(a[~z['flags'][:,s],b:b+21],out[~z['flags'][:,s],b:b+21])
  np.savez_compressed(O/'arrays'/(c['id']+'.npz'),baseline=a,**outs)
 save(O/'selection.json',dict(events=events,seconds=time.perf_counter()-start,protocol_sha256=hashlib.sha256((O/'PROTOCOL.md').read_bytes()).hexdigest()));print('SELECT COMPLETE')
def score():
 import torch
 sys.path.insert(0,str(A));import run_audit as audit
 ev=mod('metrics',D/'rtm-eval.py');model,norm,vocab=audit.load_model();pred={v:[] for v in ['baseline','RT','MP','mixed']};old=read(P/'results.json')['predictions']['support_gate'];(O/'inference').mkdir(exist_ok=True)
 for idx,c in enumerate(read(B/'capture_manifest.json')):
  z=np.load(O/'arrays'/(c['id']+'.npz'));archive={}
  for v in pred:
   x=norm({'frames':torch.from_numpy(z[v][:,audit.PICK].transpose(2,0,1).copy())})['frames']
   with torch.inference_mode():log=model(x.unsqueeze(0))[0];p=log.softmax(0)
   top=p.topk(5);r=dict(id=c['id'],label=c['label'],prediction=vocab[int(top.indices[0])],top5=[vocab[int(j)] for j in top.indices],target_probability=float(p[vocab.index(c['label'])]));pred[v].append(r);archive[v+'_input']=x.numpy();archive[v+'_logits']=log.numpy()
   if v=='baseline':assert r['top5']==old[idx]['top5'] and abs(r['target_probability']-old[idx]['target_probability'])<1e-5
  np.savez_compressed(O/'inference'/(c['id']+'.npz'),**archive)
  if (idx+1)%20==0:print('SCORE',idx+1,flush=True)
 pairs={v:dict(gains=[b['label'] for a,b in zip(pred['baseline'],pred[v]) if a['prediction']!=a['label'] and b['prediction']==b['label']],losses=[b['label'] for a,b in zip(pred['baseline'],pred[v]) if a['prediction']==a['label'] and b['prediction']!=b['label']]) for v in ['RT','MP','mixed']};out=dict(summary={v:ev.metrics(rr) for v,rr in pred.items()},pairs=pairs,predictions=pred,baseline_reproduced=True);save(O/'results.json',out);print(out['summary'])
if __name__=='__main__':{'rt':rt,'mp':mp,'select':select,'score':score}[sys.argv[1]]()
