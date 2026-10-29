
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
D=Path(__file__).resolve().parent;B=D/'oracle200-20261011';O=D/'exp2-recovery-20261013'
read=lambda p:_local_json_loads(p.read_text(encoding='utf-8'))
manifest={c['id']:c for c in read(B/'capture_manifest.json')};events=read(O/'extraction.json')['events'];chosen=[];seen=set()
for kind in ['flow','wrist']:
 for e in events:
  if e['kind']==kind and e['accepted'] and e['id'] not in seen:
   chosen.append(e);seen.add(e['id'])
   if len(chosen)>=6:break
 if len(chosen)>=6:break
board=Image.new('RGB',(1200,440*max(1,len(chosen))),'white');font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);draw=ImageDraw.Draw(board)
for n,e in enumerate(chosen):
 c=manifest[e['id']];im=Image.open(B/'frames'/e['id']/c['frames'][e['frame']]['file']).convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im);im=bg.convert('RGB');q=np.asarray(e['points']);rt=np.load(B/'rtm'/(e['id']+'.npz'));old=rt['rtmw_det'][e['frame'],91+21*e['side']:112+21*e['side']];w=rt['rtmw_det'][e['frame'],9+e['side']]
 ctr=(q.min(0)+q.max(0))/2;size=max(120,float(max(q.max(0)-q.min(0))+50));box=np.r_[ctr-size/2,ctr+size/2].astype(int);size=box[2]-box[0]
 for k,pts in enumerate([None,old,q]):
  crop=im.crop(tuple(box)).resize((380,380));cd=ImageDraw.Draw(crop)
  if pts is not None:
   for j,p in enumerate(pts):
    x,y=(p-box[:2])*380/size;cd.ellipse((x-3,y-3,x+3,y+3),fill='red');cd.text((x+3,y),str(j),font=font,fill='blue',stroke_width=1,stroke_fill='white')
  board.paste(crop,(400*k,440*n+45));draw.text((400*k+5,440*n+5),[f"{e['id']} f{e['frame']} side{e['side']} {e['kind']}",'Original RTMW','Accepted new detection'][k],font=font,fill='black')
board.save(O/'recovery-review.png');(O/'review-selection.json').write_text(json.dumps(chosen,indent=2),encoding='utf-8');print('selected',[(e['id'],e['frame'],e['kind']) for e in chosen])
