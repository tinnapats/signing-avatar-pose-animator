import sys,json,hashlib,importlib.util,time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent;F=D/'exp2-fullhand20-20261014';R=D/'exp2-rejection-audit-20261014';B=D/'oracle200-20261011';O=D/'exp2-wrist-temporal-20261014';A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def save(n,x):(O/n).write_text(json.dumps(x,indent=2),encoding='utf-8')
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def extract():
 import cv2
 cv2.setNumThreads(1);base=mod('wt_image',D/'rtm-avatar.py');start=time.perf_counter()
 catalog={c['id']:c['frames'] for c in read(B/'capture_manifest.json')}
 candidates={}
 for r in read(R/'candidates.json'):candidates[(r['id'],r['frame'],r['candidate'])]=r
 totals=dict(eligible=0,accepted=0);all_events=[]
 (O/'arrays').mkdir(exist_ok=True)
 for cid in read(F/'selection.json')['ids']:
  old=np.load(F/'arrays'/(cid+'.npz'));baseline=old['fullhand'];out=baseline.copy();ev=read(F/'events'/(cid+'.json'));cache={}
  def gray(i):
   if i not in cache:
    f=catalog[cid][i];p=B/'frames'/cid/f['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];cache[i]=cv2.cvtColor(base.image(cv2,p),cv2.COLOR_BGR2GRAY)
   return cache[i]
  def seed(i,s,sgn):
   for step in [1,2,3]:
    j=i+sgn*step
    if not 0<=j<len(ev):break
    for sel in ev[j]['selected']:
     if sel['side']==s:
      q=np.array(ev[j]['candidates'][sel['candidate']]['points'],np.float32)
      return j,q
   return None
  def track(i,seedval):
   if seedval is None:return dict(ok=False,reason='no_strict_seed')
   j,q=seedval;g0,g1=gray(j),gray(i);span=float(np.ptp(q,axis=0).max());ctr=(q.min(0)+q.max(0))/2;size=float(np.clip(1.5*span,64,256));bb=np.rint(np.r_[ctr-size/2,ctr+size/2]).astype(int);mask=np.zeros(g0.shape,np.uint8);x0,x1=np.clip(bb[[0,2]],0,g0.shape[1]);y0,y1=np.clip(bb[[1,3]],0,g0.shape[0]);mask[y0:y1,x0:x1]=255
   p=cv2.goodFeaturesToTrack(g0,maxCorners=60,qualityLevel=.01,minDistance=3,mask=mask)
   diag=dict(ok=False,seed=j,span=span)
   if p is None or len(p)<6:return dict(diag,reason='few_corners')
   nxt,st,_=cv2.calcOpticalFlowPyrLK(g0,g1,p,None,winSize=(21,21),maxLevel=3);back,sb,_=cv2.calcOpticalFlowPyrLK(g1,g0,nxt,None,winSize=(21,21),maxLevel=3)
   good=(st[:,0]>0)&(sb[:,0]>0)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<=1.5)
   if good.sum()<6 or good.mean()<.5:return dict(diag,reason='fb_failed')
   delta=(nxt-p)[:,0][good];keep=np.linalg.norm(delta-np.median(delta,axis=0),axis=1)<=3
   if keep.sum()<6 or keep.mean()<.5:return dict(diag,reason='inconsistent_flow')
   return dict(diag,ok=True,wrist=(q[0]+np.median(delta[keep],axis=0)).tolist())
  for i,event in enumerate(ev):
   eligible=[(k,candidates[(cid,i,k)]) for k in range(len(event['candidates'])) if candidates[(cid,i,k)]['group']=='wrist_conf']
   if not eligible:continue
   tracks={};rec=dict(id=cid,frame=i,tracks={},decisions=[])
   for s,b in enumerate([33,54]):
    if np.any(baseline[i,b:b+21]):continue
    a,z=track(i,seed(i,s,-1)),track(i,seed(i,s,1));rec['tracks'][s]=dict(before=a,after=z)
    if a['ok'] and z['ok']:
     span=(a['span']+z['span'])/2;w0,w1=np.array(a['wrist']),np.array(z['wrist']);limit=max(12,.25*span)
     if np.linalg.norm(w0-w1)<=limit:tracks[s]=(w0,w1,limit)
   proposals=[]
   for k,r in eligible:
    totals['eligible']+=1;q=np.array(r['points'],np.float32)
    costs={s:max(np.linalg.norm(q[0]-w0),np.linalg.norm(q[0]-w1))/lim for s,(w0,w1,lim) in tracks.items()};match={s:float(v) for s,v in costs.items() if v<=1}
    dec=dict(candidate=k,matches=match,accepted=False);rec['decisions'].append(dec)
    if len(match)!=1:dec['reason']='no_unique_temporal_match';continue
    s=next(iter(match));proposals.append((match[s],k,s,q,dec))
   used=set()
   for cost,k,s,q,dec in sorted(proposals,key=lambda x:(x[0],x[1])):
    b=[33,54][s];opp=out[i,[54,33][s]:[54,33][s]+21]
    if s in used or (np.any(opp) and np.linalg.norm(q[0]-opp[0])<24):dec['reason']='occupied_or_duplicate';continue
    out[i,b:b+21]=q;used.add(s);dec.update(accepted=True,side=s);totals['accepted']+=1
   all_events.append(rec)
  assert np.array_equal(out[:,:33],baseline[:,:33]) and np.isfinite(out).all()
  for s,b in enumerate([33,54]):
   present=np.any(baseline[:,b:b+21]!=0,axis=(1,2));assert np.array_equal(out[present,b:b+21],baseline[present,b:b+21])
  np.savez_compressed(O/'arrays'/(cid+'.npz'),fullhand=baseline,temporal=out)
  print(cid,totals,flush=True)
 save('events.json',all_events);save('extraction.json',dict(**totals,seconds=time.perf_counter()-start,protocol_sha256=hashlib.sha256((O/'PROTOCOL.md').read_bytes()).hexdigest()))
def score():
 import torch
 sys.path.insert(0,str(A));import run_audit as audit
 ev=mod('wt_metrics',D/'rtm-eval.py');model,norm,vocab=audit.load_model();mf={c['id']:c for c in read(B/'capture_manifest.json')};prior={r['id']:r for r in read(F/'results.json')['predictions']['fullhand']};pred={v:[] for v in ['fullhand','temporal']};(O/'inference').mkdir(exist_ok=True)
 for cid in read(F/'selection.json')['ids']:
  z=np.load(O/'arrays'/(cid+'.npz'));archive={}
  for v in pred:
   x=norm({'frames':torch.from_numpy(z[v][:,audit.PICK].transpose(2,0,1).copy())})['frames']
   with torch.inference_mode():log=model(x.unsqueeze(0))[0];p=log.softmax(0)
   t=p.topk(5);label=mf[cid]['label'];r=dict(id=cid,label=label,prediction=vocab[int(t.indices[0])],top5=[vocab[int(j)] for j in t.indices],target_probability=float(p[vocab.index(label)]));pred[v].append(r);archive[v+'_input']=x.numpy();archive[v+'_logits']=log.numpy()
   if v=='fullhand':assert r['top5']==prior[cid]['top5'] and abs(r['target_probability']-prior[cid]['target_probability'])<1e-5
  np.savez_compressed(O/'inference'/(cid+'.npz'),**archive);print('SCORE',cid,flush=True)
 save('results.json',dict(summary={v:ev.metrics(rr) for v,rr in pred.items()},predictions=pred,baseline_reproduced=True));print(read(O/'results.json')['summary'])
if __name__=='__main__':{'extract':extract,'score':score}[sys.argv[1]]()
