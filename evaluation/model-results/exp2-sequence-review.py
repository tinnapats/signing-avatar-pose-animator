
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
D=Path(__file__).resolve().parent;O=D/'exp2-sequence-audit-20261014';B=D/'oracle200-20261011'
read=lambda p:_local_json_loads(p.read_text(encoding='utf-8'))
coverage=read(O/'coverage.json');mf={c['id']:c for c in read(B/'capture_manifest.json')};keys=['raw_missing','wrist_off_skin','majority_off_skin','collapsed','projected_length_outlier','wrist_jump'];board=Image.new('RGB',(1200,440*len(keys)),'white');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);selected=[]
for n,key in enumerate(keys):
 e=coverage['examples'][key];selected.append(dict(signal=key,**e));c=mf[e['id']];z=np.load(B/'rtm'/(e['id']+'.npz'));i=e['frame'];side=e['side'];hand=z['rtmw_det'][i,91+21*side:112+21*side];ctr=np.median(hand,axis=0);sz=max(140,min(300,float(max(hand.max(0)-hand.min(0))+40)));bb=np.r_[ctr-sz/2,ctr+sz/2].astype(int);sz=bb[2]-bb[0]
 for col,j in enumerate([max(i-1,0),i,i]):
  p=B/'frames'/e['id']/c['frames'][j]['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==c['frames'][j]['sha256'];im=Image.open(p).convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im);im=bg.convert('RGB');q=z['rtmw_det'][j];scores=z['rtmw_det_scores'][j];ids=list(range(5,11))+list(range(91+21*side,112+21*side));dd=ImageDraw.Draw(im)
  for u,v in [(5,7),(7,9),(6,8),(8,10),(5,6)]:dd.line([tuple(q[u]),tuple(q[v])],fill='green',width=2)
  for k in ids:
   x,y=q[k];dd.ellipse((x-2,y-2,x+2,y+2),fill='red' if k>=91 else 'green')
  panel=(im.crop(tuple(bb)) if col==2 else im).resize((390,390));board.paste(panel,(col*400,n*440+45));draw.text((col*400+3,n*440+3),[f"{e['id']} {key}",f"frame {i}, hand {side}",'Raw detector points (not truth)'][col],font=font,fill='black')
board.save(O/'diagnostic-examples.png');(O/'review-selection.json').write_text(json.dumps(selected,indent=2),encoding='utf-8');print(selected)
