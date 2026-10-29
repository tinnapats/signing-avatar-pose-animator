
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
D=Path(__file__).resolve().parent;O=D/'hand-reference60-20261014';B=D/'oracle200-20261011'
rows=_local_json_loads((D/'exp2-finger-image-audit-20261014/motion-diagnostics.json').read_text(encoding='utf-8'))['candidates']
ids=_local_json_loads((D/'exp2-fullhand20-20261014/selection.json').read_text())['ids']
mf={c['id']:c['frames'] for c in _local_json_loads((B/'capture_manifest.json').read_text())}
(O/'images').mkdir(parents=True,exist_ok=True)
samples=[]
for cid in ids:
 frames=len(mf[cid]);pool=[r for r in rows if r['id']==cid]
 for third in range(3):
  eligible=[r for r in pool if min(2,r['frame']*3//frames)==third]
  if not eligible:raise ValueError((cid,third,'no candidate'))
  r=min(eligible,key=lambda r:hashlib.sha256(f"reference60-v1:{cid}:{r['frame']}:{r['candidate']}".encode()).hexdigest())
  f=mf[cid][r['frame']];p=B/'frames'/cid/f['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
  im=Image.open(p).convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im);im=bg.convert('RGB')
  x0,y0,x1,y1=r['box'];pad=max(x1-x0,y1-y0)*.5
  crop=[max(0,int(x0-pad)),max(0,int(y0-pad)),min(im.width,int(x1+pad+1)),min(im.height,int(y1+pad+1))]
  sid=f"{cid}-{r['frame']:04d}-{r['candidate']}";im.crop(crop).save(O/'images'/f'{sid}.png');im.save(O/'images'/f'{sid}-full.png')
  samples.append(dict(id=sid,clip=cid,frame=r['frame'],candidate=r['candidate'],image=f'images/{sid}.png',full=f'images/{sid}-full.png',crop=crop,width=im.width,height=im.height,source_sha256=f['sha256'],split='development_seen',sampling_third=third))
assert len(samples)==60 and len({r['id'] for r in samples})==60
data=dict(schema='avatar-hand-reference-v1',dataset_id='hand-reference60-20261014',status='UNANNOTATED_NOT_GROUND_TRUTH',samples=samples)
(O/'manifest.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
(O/'data.js').write_text('window.REFERENCE_DATA='+json.dumps(data)+';',encoding='utf-8')
sheet=Image.new('RGB',(1200,1500),'white');dr=ImageDraw.Draw(sheet)
for i,s in enumerate(samples):
 im=Image.open(O/s['image']);im.thumbnail((180,120));x=(i%6)*200;y=(i//6)*150;sheet.paste(im,(x,y+20));dr.text((x,y),s['id'],fill='black')
sheet.save(O/'contact-sheet.png')
print('Created 60 image pairs; 20 clips, 3 temporal thirds per clip; no predicted points or word labels exported.')
