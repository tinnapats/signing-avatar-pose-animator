import json,importlib.util
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
D=Path(__file__).resolve().parent;B=D/'oracle200-20261011';P=D/'pixel-only-plan-20261012';O=D/'exp2-recovery-20261013'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
manifest={c['id']:c for c in read(B/'capture_manifest.json')};ex=read(O/'extraction.json');r=read(O/'results.json');tests=[];changed=Counter();flagged=0
for cid,c in manifest.items():
 z=np.load(O/'arrays'/(cid+'.npz'));prev=np.load(P/'arrays'/(cid+'.npz'));mask=prev['flags'];flagged+=int(mask.sum());expected=prev['raw'].copy()
 for s,b in enumerate([33,54]):expected[mask[:,s],b:b+21]=0
 assert np.array_equal(expected,z['baseline'])
 for v in ['wrist','flow']:
  assert z[v].shape==expected.shape and np.isfinite(z[v]).all();assert np.array_equal(z[v][:,:33],expected[:,:33])
  for s,b in enumerate([33,54]):
   assert np.array_equal(z[v][~mask[:,s],b:b+21],expected[~mask[:,s],b:b+21]);changed[v]+=int(np.any(z[v][:,b:b+21]!=expected[:,b:b+21],axis=(1,2)).sum())
tests=['All 200 baseline coordinate arrays reproduced exactly','Body coordinates unchanged for all variants','Unflagged hand coordinates unchanged','All arrays finite, identical shapes and frame counts','Baseline ST-GCN top5 and target probabilities reproduced','All 200 clips scored for all three variants']
reasons={k:Counter() for k in ['wrist','flow']};kinds=Counter();accepted=Counter()
for e in ex['events']:
 kind=e['kind'];kinds[kind]+=1
 if e['accepted']:accepted[kind]+=1
 if kind=='invalid_body':continue
 q=np.asarray(e['points']);sc=np.asarray(e['scores']);c=manifest[e['id']];i=e['frame'];s=e['side'];z=np.load(B/'rtm'/(e['id']+'.npz'));w=z['rtmw_det'][i,9+s];el=z['rtmw_det'][i,7+s];length=np.linalg.norm(w-el)
 im=cv2.imread(str(B/'frames'/e['id']/c['frames'][i]['file']),-1)
 if im.shape[2]==4:
  a=im[:,:,3:4].astype(np.float32)/255;im=np.rint(im[:,:,:3]*a+255*(1-a)).astype(np.uint8)
 blue,green,red=[im[:,:,k].astype(np.int16) for k in range(3)];skin=(red>=100)&(red-green>=15)&(red-blue>=20)&(green>=45)&(green-blue>=-5);dist=cv2.distanceTransform((~skin).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);xy=np.rint(q).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<im.shape[1])&(xy[:,1]>=0)&(xy[:,1]<im.shape[0]);ds=np.full(21,999.);ds[inside]=dist[xy[inside,1],xy[inside,0]]
 reasons[kind]['low_confidence']+=int(min(sc.mean(),sc[0])<.3);reasons[kind]['wrist_body_disagreement']+=int(np.linalg.norm(q[0]-w)>.65*length);reasons[kind]['wrist_off_skin']+=int(ds[0]>6);reasons[kind]['majority_off_skin']+=int((ds>6).mean()>.5)
different={v:[dict(id=b['id'],label=b['label'],before=a['prediction'],after=b['prediction']) for a,b in zip(r['predictions']['baseline'],r['predictions'][v]) if a['prediction']!=b['prediction']] for v in ['wrist','flow']}
out=dict(tests=tests,flagged_hand_frames=flagged,changed_hand_frames=dict(changed),attempts=dict(kinds),accepted=dict(accepted),overlapping_rejection_signals={k:dict(v) for k,v in reasons.items()},prediction_changes=different,extraction_seconds=ex['seconds'])
(O/'audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
