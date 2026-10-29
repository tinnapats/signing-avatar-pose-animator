
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import cv2

B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007')
R=B/'rtm_trial_20261008'; A=B/'anime_hand_trial_20261009'
O=Path(__file__).parent/'hand-failure-audit-20261009'
O.mkdir(exist_ok=True)
read=lambda p:_local_json_loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
manifest={c['id']:c for c in read(B/'capture_manifest.json')}
rows=[]; examples={}; inputs={}; native=Counter()
def coverage(g,box):
 box=np.asarray(box); center=(box[:2]+box[2:])/2; size=max(box[2:]-box[:2])*1.25
 return float(np.mean(np.all((g>=center-size/2)&(g<=center+size/2),axis=1)))
for c in read(B/'coordinate_reference.json'):
 cid=c['id']; cap=manifest[cid]; raw=read(A/(cid+'_raw.json')); z=np.load(R/(cid+'.npz'))
 assert sha(R/(cid+'.npz'))==read(R/(cid+'.json'))['sha256']
 assert sha(A/(cid+'.npz'))==read(A/(cid+'.json'))['sha256']
 inputs[cid]=dict(raw_sha256=sha(A/(cid+'_raw.json')),body_sha256=sha(R/(cid+'.npz')))
 for i,ref in enumerate(c['samples']):
  assert ref['index']==cap['frames'][i]['sourceIndex']
  rr=raw[i]; pts=np.asarray(rr['points']); sc=np.asarray(rr['scores']); body=z['rtmw_det'][i]; bs=z['rtmw_det_scores'][i]
  for side,name in enumerate(['left','right']):
   h=ref['hands'][name]; assigned=rr['assignment']['plain'][side]
   if len(h)!=21:
    native['without_detailed_reference']+=1; native['accepted_without_detailed_reference']+=int(assigned>=0)
    continue
   g=np.array([h[str(j)] for j in range(21)])
   cov=[coverage(g,b) for b in rr['boxes']]; covered=[j for j,v in enumerate(cov) if v>=.8]
   conf=[j for j in covered if sc[j].mean()>=.3 and sc[j,0]>=.3]
   w,e=[(9,7),(10,8)][side]; length=max(float(np.linalg.norm(body[w]-body[e])),10)
   valid=bool(bs[w]>=.3 and bs[e]>=.3)
   dist=[float(np.linalg.norm(p[0]-body[w])/length) for p in pts]
   gated=[j for j in conf if valid and dist[j]<=.65]
   if not covered: reason='01_no_covering_box'
   elif not conf: reason='02_hand_confidence_rejected'
   elif not valid: reason='03_body_anchor_rejected'
   elif not gated: reason='04_wrist_distance_rejected'
   elif assigned not in gated: reason='05_assignment_other_or_missing'
   else: reason='06_accepted_covering_candidate'
   rmse=float(np.sqrt(np.mean(np.sum((pts[assigned]-g)**2,axis=1)))) if assigned>=0 else None
   row=dict(id=cid,label=cap['label'],frame=i,source_index=ref['index'],side=name,reason=reason,assigned=assigned,coverage=cov,confidence_candidates=conf,gated_candidates=gated,wrist_distance_ratios=dist,body_valid=valid,assigned_rmse_px=rmse)
   rows.append(row)
   if reason not in examples:examples[reason]=(row,rr,g,body[w])
counts=Counter(r['reason'] for r in rows)
assert len(rows)==1872
save(O/'diagnostics.json',dict(counts=dict(counts),reference_hand_frames=len(rows),without_reference=dict(native),rows=rows))
save(O/'protocol.json',dict(rules='Read-only cached diagnostic. Mutually exclusive sequential failure categories. Covering box: >=80% of renderer reference21 inside square bbox with padding1.25. All thresholds unchanged. Reference is used ONLY in this diagnostic, never in inference or recognition. A covering candidate can be wrong hand/oversized; coverage is optimistic, not correctness. No detailed reference does NOT prove rest or native SVG.',inputs=inputs,reference_sha256=sha(B/'coordinate_reference.json'),manifest_sha256=sha(B/'capture_manifest.json')))
edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
for reason,(row,rr,g,wrist) in examples.items():
 c=manifest[row['id']]; f=c['frames'][row['frame']]; p=B/'frames'/row['id']/f['file']; assert sha(p)==f['sha256']
 im=cv2.imread(str(p),-1)
 if im.shape[2]==4:
  alpha=im[:,:,3:4].astype(np.float32)/255; im=np.rint(im[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8)
 for j,box in enumerate(rr['boxes']):
  x1,y1,x2,y2=map(int,box); cv2.rectangle(im,(x1,y1),(x2,y2),(180,100,0),2);cv2.putText(im,str(j),(x1,y1),0,.6,(180,100,0),2)
 for a,b in edges:cv2.line(im,tuple(g[a].astype(int)),tuple(g[b].astype(int)),(0,180,0),2)
 for j,points in enumerate(rr['points']):
  pp=np.array(points).astype(int)
  for a,b in edges:cv2.line(im,tuple(pp[a]),tuple(pp[b]),(0,0,220) if j==row['assigned'] else (150,150,150),1)
 cv2.circle(im,tuple(wrist.astype(int)),8,(255,0,255),2)
 band=np.full((100,808,3),255,np.uint8)
 for k,t in enumerate([reason,f"{row['label']} frame {row['frame']} {row['side']} assigned={row['assigned']}",'Green: renderer reference | Red: assigned | Gray: candidates | Magenta: body wrist']):cv2.putText(band,t,(10,25+30*k),0,.48,(0,0,0),1)
 cv2.imwrite(str(O/(reason+'.png')),np.vstack([band,im]))
for path,h in read(A/'protocol.json')['production'].items():assert sha(B.parent.parent/path)==h
print(json.dumps(dict(counts=dict(counts),no_reference=dict(native),output=str(O)),ensure_ascii=True))
