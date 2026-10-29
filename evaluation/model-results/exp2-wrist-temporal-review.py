import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
D=Path(__file__).resolve().parent;O=D/'exp2-wrist-temporal-20261014';R=D/'exp2-rejection-audit-20261014';F=D/'exp2-fullhand20-20261014'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
rows={(r['id'],r['frame'],r['candidate']):r for r in read(R/'candidates.json')};accepted=[]
for e in read(O/'events.json'):
 for d in e['decisions']:
  if d['accepted']:
   key=(e['id'],e['frame'],d['candidate']);accepted.append(dict(**rows[key],side=d['side'],tracks=e['tracks'][str(d['side'])],review_rank=hashlib.sha256((':'.join(map(str,key))).encode()).hexdigest()))
selected=sorted(accepted,key=lambda r:r['review_rank'])[:3]
(O/'review-selection.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
canvas=Image.new('RGB',(1200,900),'white');draw=ImageDraw.Draw(canvas)
edges=[(0,j) for j in [1,5,9,13,17]]+[(j,j+1) for j in [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]]
for n,r in enumerate(selected):
 src=Image.open(r['path']).convert('RGBA');bg=Image.new('RGBA',src.size,'white');bg.alpha_composite(src);src=bg.convert('RGB');q=np.array(r['points']);bb=np.array(r['box']);lo=np.minimum(bb[:2],q.min(0))-25;hi=np.maximum(bb[2:],q.max(0))+25;box=(max(0,int(lo[0])),max(0,int(lo[1])),min(src.width,int(hi[0])),min(src.height,int(hi[1])))
 for col in [0,1]:
  tile=src.crop(box);scale=min(280/tile.width,280/tile.height);tile=tile.resize((round(tile.width*scale),round(tile.height*scale)))
  if col:
   td=ImageDraw.Draw(tile);pts=(q-box[:2])*scale
   for a,b in edges:td.line([tuple(pts[a]),tuple(pts[b])],fill='blue',width=2)
   for j,(x,y) in enumerate(pts):td.ellipse((x-2,y-2,x+2,y+2),fill='red');td.text((x+2,y),str(j),fill='black')
   for direction,color in [('before','green'),('after','purple')]:
    w=(np.array(r['tracks'][direction]['wrist'])-box[:2])*scale;td.ellipse((w[0]-5,w[1]-5,w[0]+5,w[1]+5),outline=color,width=2)
  canvas.paste(tile,(col*290,n*300+10))
 draw.text((590,n*300+20),f"Clip {r['id']} frame {r['frame']} candidate {r['candidate']}\nmean confidence {r['mean_conf']:.3f}\nwrist confidence {r['wrist_conf']:.3f}\nGreen circle: projected wrist from earlier image\nPurple circle: projected wrist from later image\nBlue/red: current image candidate, NOT copied tracking points",fill='black')
canvas.save(O/'review.png')
print('selected',[(r['id'],r['frame'],r['candidate']) for r in selected])
