
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
from pathlib import Path
import json,numpy as np,cv2
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');O=B/'rotated_roi_20261008'
read=lambda p:_local_json_loads(p.read_text(encoding='utf8'))
refs={c['id']:c for c in read(B/'coordinate_reference.json')};chosen={};panels=[]
for cap in read(B/'capture_manifest.json'):
 for d in read(O/(cap['id']+'.json'))['added']:
  if d['angle'] in chosen:continue
  g=refs[cap['id']]['samples'][d['frame']]['hands'][d['side']]
  if len(g)!=21:continue
  chosen[d['angle']]=(cap,d,g)
for angle,(cap,d,g) in sorted(chosen.items()):
 i=d['frame'];off=33 if d['side']=='left' else 54;q=np.load(O/(cap['id']+'.npz'))['rest_roi3'][i,off:off+21];gt=np.array([g[str(j)] for j in range(21)]);im=cv2.imread(str(B/'frames'/cap['id']/cap['frames'][i]['file']),cv2.IMREAD_UNCHANGED);a=im[:,:,3:4]/255;im=np.uint8(np.rint(im[:,:,:3]*a+255*(1-a)))
 for p in gt:cv2.circle(im,tuple(np.rint(p).astype(int)),4,(0,170,0),-1)
 for p in q:cv2.circle(im,tuple(np.rint(p).astype(int)),3,(200,0,200),-1)
 im=cv2.resize(im,(404,404));im=cv2.copyMakeBorder(im,50,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255));cv2.putText(im,f"{cap['label']} f{i} angle{angle}",(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1);cv2.putText(im,'Green=reference Purple=detected',(5,40),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1);panels.append(im)
cv2.imwrite(str(O/'review.png'),np.concatenate(panels,axis=1));print('review angles',list(chosen))
