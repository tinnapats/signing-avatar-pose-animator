
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json
from pathlib import Path
import numpy as np,cv2
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');O=B/'wrist_tracking'
read=lambda p:_local_json_loads(p.read_text(encoding='utf8'))
refs={c['id']:c for c in read(B/'coordinate_reference.json')};panels=[];examples=[]
for source in ['crop','flow']:
 count=0
 for f in sorted(O.glob('[0-9][0-9][0-9].json')):
  c=read(f)
  for r in c['rows']:
   candidates=[s for s in ['left','right'] if r['sources'][s]==source and len(refs[c['id']]['samples'][r['frame']]['hands'][s])==21]
   if not candidates:continue
   side=candidates[0];im=cv2.imread(str(B/'frames'/c['id']/f"{r['frame']:04d}.png"),cv2.IMREAD_UNCHANGED);a=im[:,:,3:4]/255;im=np.uint8(im[:,:,:3]*a+255*(1-a))
   g=refs[c['id']]['samples'][r['frame']]['hands'][side];q=np.array(r['track'][side]);gt=np.array([g[str(j)] for j in range(21)])
   for p in gt:cv2.circle(im,tuple(np.rint(p).astype(int)),4,(0,170,0),-1)
   for p in q:cv2.circle(im,tuple(np.rint(p).astype(int)),3,(200,0,200),-1)
   if side in r['rois']:
    x,y,xx,yy=r['rois'][side];cv2.rectangle(im,(x,y),(xx,yy),(200,100,0),2)
   im=cv2.resize(im,(404,404));im=cv2.copyMakeBorder(im,40,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255));cv2.putText(im,f"{source}: {c['label']} f{r['frame']} {side}",(5,23),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1);panels.append(im);examples.append(dict(source=source,label=c['label'],frame=r['frame'],side=side,rmse=float(np.sqrt(np.mean(np.sum((q-gt)**2,axis=1))))));count+=1;break
  if count==3:break
while len(panels)%3:panels.append(np.full_like(panels[0],255))
cv2.imwrite(str(O/'review_added.png'),np.concatenate([np.concatenate(panels[i:i+3],axis=1) for i in range(0,len(panels),3)],axis=0));(O/'review_examples.json').write_text(json.dumps(examples,indent=2),encoding='utf8');print(examples)
