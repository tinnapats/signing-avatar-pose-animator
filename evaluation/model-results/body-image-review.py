
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json,hashlib
from pathlib import Path
import cv2,numpy as np
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');R=B/'rtm_trial_20261008';O=Path(__file__).parent/'body-image-review-20261009';O.mkdir(exist_ok=True)
read=lambda p:_local_json_loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();manifest=read(B/'capture_manifest.json');refs={c['id']:c for c in read(B/'coordinate_reference.json')}
chosen=[0,6,12,17,23,29];errors={'RTMW':[],'MediaPipe':[]};rows=[];worst=[]
for c in manifest:
 z=np.load(R/(c['id']+'.npz'));assert sha(R/(c['id']+'.npz'))==read(R/(c['id']+'.json'))['sha256'];raw=read(B/'detected'/(c['id']+'.json'))['rows'];samples=refs[c['id']]['samples']
 for i,ref in enumerate(samples):
  for side,ri,mi in [('left',9,15),('right',10,16)]:
   h=ref['hands'][side]
   if len(h)!=21:continue
   gt=np.array(h['0']);entry=dict(id=c['id'],label=c['label'],frame=i,side=side)
   if z['rtmw_det_scores'][i,ri]>=.3:entry['RTMW']=float(np.linalg.norm(z['rtmw_det'][i,ri]-gt));errors['RTMW'].append(entry['RTMW'])
   if raw[i]['pose'] and raw[i]['pose'][mi][3]>=.3:
    p=np.array(raw[i]['pose'][mi][:2])*[c['width'],c['height']];entry['MediaPipe']=float(np.linalg.norm(p-gt));errors['MediaPipe'].append(entry['MediaPipe'])
   rows.append(entry)
 def render(indices,name):
  panels=[]
  for i in indices:
   f=c['frames'][i];path=B/'frames'/c['id']/f['file'];assert sha(path)==f['sha256'];im=cv2.imread(str(path),-1);alpha=im[:,:,3:4].astype(np.float32)/255;im=np.rint(im[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8);row=[]
   for model in ['RTMW','MediaPipe']:
    img=im.copy();pts=z['rtmw_det'][i] if model=='RTMW' else np.array(raw[i]['pose'])[:,:2]*[c['width'],c['height']] if raw[i]['pose'] else np.zeros((33,2));sc=z['rtmw_det_scores'][i] if model=='RTMW' else np.array(raw[i]['pose'])[:,3] if raw[i]['pose'] else np.zeros(33)
    for side,ids,color in [('L',[5,7,9] if model=='RTMW' else [11,13,15],(220,70,0)),('R',[6,8,10] if model=='RTMW' else [12,14,16],(0,60,230))]:
     for a,b in zip(ids,ids[1:]):
      if sc[a]>=.3 and sc[b]>=.3:cv2.line(img,tuple(np.rint(pts[a]).astype(int)),tuple(np.rint(pts[b]).astype(int)),color,2)
     for j,label in zip(ids,['S','E','W']):
      if sc[j]>=.3:
       p=tuple(np.rint(pts[j]).astype(int));cv2.circle(img,p,5,color,-1);cv2.putText(img,side+label,(p[0]+7,p[1]-7),0,.5,color,1)
     h=samples[i]['hands'][{'L':'left','R':'right'}[side]]
     if len(h)==21:cv2.circle(img,tuple(np.rint(h['0']).astype(int)),8,(0,150,0),2)
    # Fixed identical upper-body viewport; no pose-dependent crop used for scoring.
    img=img[:570,60:680];img=cv2.resize(img,(496,456));band=np.full((55,496,3),255,np.uint8);cv2.putText(band,f"{c['label']} f{i} | {model}",(8,20),0,.6,(0,0,0),1);cv2.putText(band,'S=shoulder E=elbow W=wrist | green=rendered hand wrist',(8,43),0,.41,(0,0,0),1);row.append(np.vstack([band,img]))
   panels.append(np.hstack(row))
  cv2.imwrite(str(O/name),np.vstack(panels))
 if int(c['id']) in chosen:render(np.rint(np.linspace(0,len(samples)-1,3)).astype(int),c['id']+'.png')
 # One largest RTMW mismatch per clip, for explicitly error-selected diagnostic review.
 rr=[x for x in rows if x['id']==c['id'] and 'RTMW' in x]
 if rr:worst.append(max(rr,key=lambda x:x['RTMW']))
worst=sorted(worst,key=lambda x:x['RTMW'],reverse=True)
common=[x for x in rows if 'RTMW'in x and 'MediaPipe'in x]
def stats(v):return dict(n=len(v),mean_px=float(np.mean(v)),rmse_px=float(np.sqrt(np.mean(np.square(v)))),median_px=float(np.median(v)),p95_px=float(np.percentile(v,95)),within10px_pct=float(np.mean(np.array(v)<=10)*100))
result=dict(individual={k:stats(v) for k,v in errors.items()},common={k:stats([x[k] for x in common]) for k in errors},reference_hand_frames=len(rows),worst_per_clip=worst,rows=rows,notes='Body wrist vs rendered detailed-hand wrist; may differ anatomically by definition. No shoulder/elbow ground truth. MP visibility and RTMW confidence >=.3 for this diagnostic only; not used to edit model inputs. Uniform image review indices0,6,12,17,23,29 first/middle/last frames, selected before inspecting. Missing reference excludes native SVG hand frames, not proof of rest.')
(O/'wrist_diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
for f,h in read(B/'protocol.json')['production'].items():assert sha(B.parent.parent/f)==h
print(json.dumps({k:result[k] for k in ['individual','common','reference_hand_frames']}));print('worst',json.dumps(worst[:3]))
