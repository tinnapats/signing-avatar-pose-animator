
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
from PIL import Image, ImageDraw

D=Path(__file__).resolve().parent
F=D/'exp2-fullhand20-20261014'
B=D/'oracle200-20261011'
O=D/'exp2-rejection-audit-20261014'
O.mkdir(exist_ok=True)
read=lambda p:_local_json_loads(p.read_text(encoding='utf-8'))
def save(name,obj):
 (O/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
# Frozen before inspection: rank candidate IDs by hash, select three per exact
# failure combination plus three valid controls. No label or recognition input.
save('protocol.json',dict(selection='first 3 sha256(audit-v1:clip:frame:candidate) per exact gate combination',thresholds=dict(confidence=.30,span=10,skin_distance=6,max_offskin_fraction=.5),source='saved pixel-only candidates and original PNGs',changes_to_pipeline=False))
mf={c['id']:c['frames'] for c in read(B/'capture_manifest.json')}
rows=[]; mismatches=[]
for cid in read(F/'selection.json')['ids']:
 for event in read(F/'events'/(cid+'.json')):
  i=event['frame']; p=B/'frames'/cid/mf[cid][i]['file']
  assert hashlib.sha256(p.read_bytes()).hexdigest()==mf[cid][i]['sha256']
  im=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
  if im.shape[2]==4:
   alpha=im[:,:,3:4].astype(np.float32)/255
   im=np.rint(im[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8)
  b,g,r=[im[:,:,k].astype(np.int16) for k in range(3)]
  skin=(r>=100)&(r-g>=15)&(r-b>=20)&(g>=45)&(g-b>=-5)
  dist=cv2.distanceTransform((~skin).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
  for k,c in enumerate(event['candidates']):
   q=np.array(c['points'],dtype=np.float32);sc=np.array(c['scores'],dtype=np.float32)
   finite=bool(np.isfinite(q).all());ds=np.full(21,999.)
   if finite:
    xy=np.rint(q).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<im.shape[1])&(xy[:,1]>=0)&(xy[:,1]<im.shape[0])
    ds[inside]=dist[xy[inside,1],xy[inside,0]]
   span=float(max(q.max(0)-q.min(0)));frac=float((ds>6).mean())
   flags=dict(nonfinite=not finite,mean_conf=bool(sc.mean()<.3),wrist_conf=bool(sc[0]<.3),small_span=span<10,wrist_offskin=bool(ds[0]>6),majority_offskin=frac>.5)
   failed=[n for n,v in flags.items() if v];group='+'.join(failed) or 'valid'
   if (not failed)!=c['valid']:mismatches.append([cid,i,k])
   rows.append(dict(id=cid,frame=i,candidate=k,group=group,failed=failed,mean_conf=float(sc.mean()),wrist_conf=float(sc[0]),span=span,wrist_distance=float(ds[0]),offskin_fraction=frac,box=c['box'],points=c['points'],path=str(p),rank=hashlib.sha256(f'audit-v1:{cid}:{i}:{k}'.encode()).hexdigest()))
print('processed',len(rows),flush=True)
counts=Counter(r['group'] for r in rows); gates=Counter(g for r in rows for g in r['failed'])
selected=[r for group in sorted(counts) for r in sorted((x for x in rows if x['group']==group),key=lambda x:x['rank'])[:3]]
save('audit.json',dict(total=len(rows),rejected=sum(bool(r['failed']) for r in rows),gate_counts=dict(gates),exclusive_combinations=dict(counts),reproduction_mismatches=mismatches))
save('candidates.json',rows);save('review-selection.json',selected)
edges=[(0,1),(1,2),(2,3),(3,4)]+[(0,j) for j in [5,9,13,17]]+[(j,j+1) for j in [5,6,7,9,10,11,13,14,15,17,18,19]]
for page in range((len(selected)+5)//6):
 canvas=Image.new('RGB',(1200,6*230),'white');draw=ImageDraw.Draw(canvas)
 for n,r in enumerate(selected[page*6:page*6+6]):
  src=Image.open(r['path']).convert('RGBA');white=Image.new('RGBA',src.size,'white');white.alpha_composite(src);src=white.convert('RGB')
  q=np.array(r['points']);x0,y0,x1,y1=r['box'];pad=max(x1-x0,y1-y0)*.3
  box=(int(max(0,x0-pad)),int(max(0,y0-pad)),int(min(src.width,x1+pad)),int(min(src.height,y1+pad)))
  y=n*230
  for col,overlay in enumerate([False,True]):
   tile=src.crop(box);scale=min(210/tile.width,210/tile.height);tile=tile.resize((int(tile.width*scale),int(tile.height*scale)))
   if overlay:
    td=ImageDraw.Draw(tile);pts=(q-np.array(box[:2]))*scale
    for a,b in edges:td.line([tuple(pts[a]),tuple(pts[b])],fill='blue',width=2)
    for j,(xx,yy) in enumerate(pts):td.ellipse((xx-2,yy-2,xx+2,yy+2),fill='red');td.text((xx+2,yy),str(j),fill='black')
   canvas.paste(tile,(col*220,y+15))
  draw.text((450,y+15),f"{r['id']} frame {r['frame']} candidate {r['candidate']}\n{r['group']}\nmean conf {r['mean_conf']:.3f}; wrist conf {r['wrist_conf']:.3f}\nspan {r['span']:.1f}px; wrist offskin {r['wrist_distance']:.1f}px\noffskin points {r['offskin_fraction']:.1%}",fill='black')
 canvas.save(O/f'review-{page+1:02d}.png')
print(json.dumps(read(O/'audit.json'),indent=2))
