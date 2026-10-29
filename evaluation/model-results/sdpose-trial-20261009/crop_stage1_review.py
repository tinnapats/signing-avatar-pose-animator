
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json
from pathlib import Path
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parent;O=ROOT/'crop_stage1_20261011'
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007')
manifest=_local_json_loads((B/'capture_manifest.json').read_text())
panels=[]
for cid in ['001','002','005']:
 c=next(c for c in manifest if c['id']==cid);i=len(c['frames'])//2
 h=np.load(B/'rtm_trial_20261008/hand_crop'/f'{cid}.npz')
 im=cv2.imread(str(B/'frames'/cid/c['frames'][i]['file']),-1)
 if im.shape[2]==4:
  alpha=im[:,:,3:4].astype(np.float32)/255;im=np.rint(im[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8)
 for side,color in [(0,(220,80,0)),(1,(0,60,220))]:
  box=h['boxes'][i,side]
  if box[2]<=box[0]:continue
  center=(box[:2]+box[2:])/2;half=(box[2:]-box[:2])*1.25/2
  cv2.rectangle(im,tuple(np.rint(center-half).astype(int)),tuple(np.rint(center+half).astype(int)),color,2)
  points=h['points'][i,side]
  for chain in [(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)]:
   for a,b in zip(chain,chain[1:]):cv2.line(im,tuple(np.rint(points[a]).astype(int)),tuple(np.rint(points[b]).astype(int)),color,1)
  cv2.putText(im,f"{'L' if side==0 else 'R'} accepted={h['accepted'][i,side]}",(10,52+side*23),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
 cv2.putText(im,f"{c['label']} frame {i} / {len(c['frames'])}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),2)
 panels.append(cv2.resize(im,(505,505)))
cv2.imwrite(str(O/'fixed_midframe_review.png'),np.concatenate(panels,axis=1))
print('Saved fixed midframe review')
