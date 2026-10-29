import sys,json,time,hashlib,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
D=Path(__file__).resolve().parent;B=D/'oracle200-20261011';O=D/'pixel-only-plan-20261012';A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def extract():
 import cv2
 cv2.setNumThreads(1);ev=module('support_ev',D/'rtm-eval.py');folder=O/'arrays';folder.mkdir(exist_ok=True);manifest=read(B/'capture_manifest.json')
 for c in manifest:
  target=folder/(c['id']+'.npz')
  if target.exists():continue
  z=np.load(B/'rtm'/(c['id']+'.npz'));raw=ev.convert(z['rtmw_det'],z['rtmw_det_scores']);flags=np.zeros((len(raw),2),bool);distances=np.zeros((len(raw),2,21),np.float32);ms=[]
  for i,f in enumerate(c['frames']):
   start=time.perf_counter();p=B/'frames'/c['id']/f['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];im=cv2.imread(str(p),-1)
   if im.shape[2]==4:
    alpha=im[:,:,3:4].astype(np.float32)/255;im=np.rint(im[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8)
   blue,green,red=[im[:,:,k].astype(np.int16) for k in range(3)];skin=(red>=100)&(red-green>=15)&(red-blue>=20)&(green>=45)&(green-blue>=-5)
   distance=cv2.distanceTransform((~skin).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
   for side,b in enumerate([33,54]):
    q=raw[i,b:b+21]
    if not np.any(q):continue
    xy=np.rint(q).astype(int);ok=(xy[:,0]>=0)&(xy[:,0]<im.shape[1])&(xy[:,1]>=0)&(xy[:,1]<im.shape[0]);ds=np.full(21,999.,np.float32);ds[ok]=distance[xy[ok,1],xy[ok,0]];distances[i,side]=ds;flags[i,side]=ds[0]>6 and (ds>6).mean()>.5
   ms.append((time.perf_counter()-start)*1000)
  flags3=np.zeros_like(flags);run=np.zeros(2,int)
  for i in range(len(flags)):run=np.where(flags[i],run+1,0);flags3[i]=run>=3
  np.savez_compressed(target,raw=raw,flags=flags,flags3=flags3,distances=distances,ms=ms)
  print('SUPPORT',c['id'],int(flags.sum()),int(flags3.sum()),flush=True)
 save(O/'extraction.json',{'clips':len(manifest),'input':'PNG + RTMW only; no renderer references','protocol_sha256':hashlib.sha256((O/'PROTOCOL.md').read_bytes()).hexdigest()})
def score():
 import torch
 sys.path.insert(0,str(A));import run_audit as audit
 ev=module('support_ev_score',D/'rtm-eval.py');model,norm,vocab=audit.load_model();manifest=read(B/'capture_manifest.json');old=read(B/'results.json')['predictions']['pixels'];pred={v:[] for v in ['baseline','support_gate','support_gate3']};changed={v:0 for v in pred};times=[];out=O/'inference';out.mkdir(exist_ok=True)
 for c,before in zip(manifest,old):
  target=O/'arrays'/(c['id']+'.npz');deadline=time.monotonic()+1800
  while not target.exists():
   if time.monotonic()>deadline:raise RuntimeError('Extraction did not provide next clip within 30 minutes')
   time.sleep(2)
  # Extraction writes the small archive in place; allow completion before opening.
  time.sleep(.2)
  z=np.load(target);archive={};times.extend(z['ms'].tolist())
  for v in pred:
   a=z['raw'].copy()
   if v!='baseline':
    mask=z['flags' if v=='support_gate' else 'flags3'];changed[v]+=int(mask.sum())
    for side,b in enumerate([33,54]):a[mask[:,side],b:b+21]=0
   x=norm({'frames':torch.from_numpy(a[:,audit.PICK].transpose(2,0,1).copy())})['frames']
   with torch.inference_mode():log=model(x.unsqueeze(0))[0];p=log.softmax(0)
   top=p.topk(5);row={'id':c['id'],'label':c['label'],'prediction':vocab[int(top.indices[0])],'top5':[vocab[int(j)] for j in top.indices],'target_probability':float(p[vocab.index(c['label'])])};pred[v].append(row);archive[v+'_input']=x.numpy();archive[v+'_logits']=log.numpy()
   if v=='baseline':assert row['top5']==before['top5'] and abs(row['target_probability']-before['target_probability'])<1e-5
  np.savez_compressed(out/(c['id']+'.npz'),**archive);print('SCORE',c['id'],flush=True)
 dev={'149','125','195','045','122','147','178','057'};summary={g:{v:ev.metrics([r for r in rows if g=='all200' or ((r['id'] in dev)==(g=='review8'))]) for v,rows in pred.items()} for g in ['all200','review8','remaining192']}
 pairs={}
 for v in ['support_gate','support_gate3']:
  gains=[];losses=[];different=[]
  for a,b in zip(pred['baseline'],pred[v]):
   if a['prediction']!=b['prediction']:different.append({'id':a['id'],'label':a['label'],'before':a['prediction'],'after':b['prediction']})
   if a['prediction']!=a['label'] and b['prediction']==b['label']:gains.append(a['label'])
   if a['prediction']==a['label'] and b['prediction']!=b['label']:losses.append(a['label'])
  pairs[v]={'gains':gains,'losses':losses,'changed_predictions':different}
 result={'summary':summary,'pairs':pairs,'changed_hand_frames':changed,'mean_preparation_ms_per_frame':float(np.mean(times)),'frames':len(times),'baseline_reproduced':True,'predictions':pred};save(O/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k!='predictions'}),flush=True)
if __name__=='__main__':{'extract':extract,'score':score}[sys.argv[1]]()
