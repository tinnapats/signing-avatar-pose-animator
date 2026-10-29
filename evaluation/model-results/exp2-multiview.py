
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
D=Path(__file__).resolve().parent;O=D/'exp2-multiview-20261013';OLD=D/'improve-exp12-20261013';B=D/'oracle200-20261011';P=D/'pixel-only-plan-20261012';A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit')
read=lambda p:_local_json_loads(p.read_text(encoding='utf-8'))
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def extract():
 base=mod('mv_base',D/'rtm-avatar.py');cv2,RTMPose,_=base.setup();hp=A/'pixels30_20261007/rtm_trial_20261008/model_cache/hub/checkpoints/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.onnx';assert hashlib.sha256(hp.read_bytes()).hexdigest()=='39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29';hand=RTMPose(str(hp),model_input_size=(256,256),backend='onnxruntime',device='cpu')
 source={}
 for r in read(OLD/'exp2-extraction.json')['events']:source.setdefault(r['id'],[]).append(r)
 (O/'arrays').mkdir(exist_ok=True);(O/'events').mkdir(exist_ok=True);started=time.perf_counter();totalviews=0;accepted=0;consensuses=0
 for c in read(B/'capture_manifest.json'):
  z=np.load(P/'arrays'/(c['id']+'.npz'));a=np.load(OLD/'exp2-arrays'/(c['id']+'.npz'))['baseline'];out=a.copy();raw=z['raw'];rt=np.load(B/'rtm'/(c['id']+'.npz'));events=[]
  for row in source.get(c['id'],[]):
   i=row['frame'];f=c['frames'][i];p=B/'frames'/c['id']/f['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];im=base.image(cv2,p);blue,green,red=[im[:,:,k].astype(np.int16) for k in range(3)];skin=(red>=100)&(red-green>=15)&(red-blue>=20)&(green>=45)&(green-blue>=-5);dist=cv2.distanceTransform((~skin).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);record=dict(id=c['id'],frame=i,candidates=[],accepted=[]);proposals=[]
   for k,oldcand in enumerate(row['candidates']):
    bb=np.asarray(oldcand['box']);center=(bb[:2]+bb[2:])/2;side=float(max(bb[2:]-bb[:2]));views=[];valid=[]
    for scale in [1.,1.2]:
     for angle in [0,90,-90,180]:
      mat=cv2.getRotationMatrix2D(tuple(center),angle,320/(1.25*side*scale));mat[:,2]+=np.array([160,160])-center;inv=cv2.invertAffineTransform(mat);test=np.c_[np.array([[0,0],center,[808,808]]),np.ones(3)];mapped=test@mat.T;assert np.max(np.abs(np.c_[mapped,np.ones(3)]@inv.T-test[:,:2]))<1e-4
      roi=cv2.warpAffine(im,mat,(320,320),borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255));q,sc=hand(roi,bboxes=[np.array([32,32,288,288])]);q=np.c_[q[0],np.ones(21)]@inv.T;sc=sc[0];totalviews+=1;span=float(max(q.max(0)-q.min(0)));xy=np.rint(q).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<im.shape[1])&(xy[:,1]>=0)&(xy[:,1]<im.shape[0]);ds=np.full(21,999.);ds[inside]=dist[xy[inside,1],xy[inside,0]];ok=bool(np.isfinite(q).all() and min(sc.mean(),sc[0])>=.3 and span>=10 and ds[0]<=6 and (ds>6).mean()<=.5)
      views.append(dict(angle=angle,scale=scale,points=q.tolist(),scores=sc.tolist(),valid=ok,span=span))
      if ok:valid.append(len(views)-1)
    entry=dict(box=bb.tolist(),views=views,selected=None);record['candidates'].append(entry)
    if len(valid)<3:continue
    qs=np.asarray([views[j]['points'] for j in valid]);sp=np.asarray([views[j]['span'] for j in valid]);distance=np.sqrt(np.mean(np.sum((qs[:,None]-qs[None,:])**2,axis=-1),axis=-1))/np.maximum((sp[:,None]+sp[None,:])/2,10);neighbors=distance<=.2;best=min(range(len(valid)),key=lambda j:(-int(neighbors[j].sum()),float(distance[j,neighbors[j]].mean()),j))
    if neighbors[best].sum()<3:continue
    selected=valid[best];q=qs[best];entry['selected']=selected;entry['agreement_views']=int(neighbors[best].sum());consensuses+=1;costs={}
    for ss,j in row['seeds'].items():
     s=int(ss);seed=raw[j,[33,54][s]:[33,54][s]+21];span=float(max(seed.max(0)-seed.min(0)));costs[s]=float(np.linalg.norm(q[0]-seed[0]))/max(24,.75*span)
    compatible={s:v for s,v in costs.items() if v<=1};entry['costs']=costs
    if not compatible:entry['rejection']='no_temporal_match';continue
    if len(compatible)==2 and abs(compatible[0]-compatible[1])<.2:entry['rejection']='ambiguous_side';continue
    s=min(compatible,key=compatible.get)
    if not z['flags'][i,s]:entry['rejection']='retained_side';continue
    opposite=a[i,[54,33][s]:[54,33][s]+21]
    if np.any(opposite) and np.linalg.norm(q[0]-opposite[0])<24:entry['rejection']='duplicate';continue
    proposals.append((compatible[s],s,k,q))
   used=set()
   for cost,s,k,q in sorted(proposals,key=lambda x:x[0]):
    if s in used:continue
    used.add(s);b=[33,54][s];out[i,b:b+21]=q;record['accepted'].append(dict(side=s,candidate=k,cost=cost));accepted+=1
   events.append(record)
  assert np.array_equal(a[:,:33],out[:,:33]) and np.isfinite(out).all()
  for s,b in enumerate([33,54]):assert np.array_equal(a[~z['flags'][:,s],b:b+21],out[~z['flags'][:,s],b:b+21])
  np.savez_compressed(O/'arrays'/(c['id']+'.npz'),baseline=a,multiview=out,independent=np.load(OLD/'exp2-arrays'/(c['id']+'.npz'))['independent']);(O/'events'/(c['id']+'.json')).write_text(json.dumps(events),encoding='utf-8')
  if (int(c['id'])+1)%10==0:print('EXTRACT',c['id'],'views',totalviews,'accepted',accepted,flush=True)
 result=dict(seconds=time.perf_counter()-started,views=totalviews,consensus_boxes=consensuses,accepted_hand_frames=accepted,protocol_sha256=hashlib.sha256((O/'PROTOCOL.md').read_bytes()).hexdigest());(O/'extraction.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result,flush=True)
def score():
 import torch
 sys.path.insert(0,str(A));import run_audit as audit
 ev=mod('mv_metrics',D/'rtm-eval.py');model,norm,vocab=audit.load_model();pred={v:[] for v in ['baseline','independent','multiview']};old=read(OLD/'exp2-results.json')['predictions'];(O/'inference').mkdir(exist_ok=True)
 for idx,c in enumerate(read(B/'capture_manifest.json')):
  z=np.load(O/'arrays'/(c['id']+'.npz'));archive={}
  for v in pred:
   x=norm({'frames':torch.from_numpy(z[v][:,audit.PICK].transpose(2,0,1).copy())})['frames']
   with torch.inference_mode():log=model(x.unsqueeze(0))[0];p=log.softmax(0)
   t=p.topk(5);r=dict(id=c['id'],label=c['label'],prediction=vocab[int(t.indices[0])],top5=[vocab[int(k)] for k in t.indices],target_probability=float(p[vocab.index(c['label'])]));pred[v].append(r);archive[v+'_input']=x.numpy();archive[v+'_logits']=log.numpy()
   if v in old:assert r['top5']==old[v][idx]['top5'] and abs(r['target_probability']-old[v][idx]['target_probability'])<1e-5
  np.savez_compressed(O/'inference'/(c['id']+'.npz'),**archive)
  if (idx+1)%20==0:print('SCORE',idx+1,flush=True)
 gains=[b['label'] for a,b in zip(pred['baseline'],pred['multiview']) if a['prediction']!=a['label'] and b['prediction']==b['label']];losses=[b['label'] for a,b in zip(pred['baseline'],pred['multiview']) if a['prediction']==a['label'] and b['prediction']!=b['label']];result=dict(summary={v:ev.metrics(rr) for v,rr in pred.items()},gains=gains,losses=losses,predictions=pred,baselines_reproduced=True);(O/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result['summary'],flush=True)
if __name__=='__main__':{'extract':extract,'score':score}[sys.argv[1]]()
