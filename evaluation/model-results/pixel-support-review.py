import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
D=Path(__file__).resolve().parent;O=D/'pixel-only-plan-20261012';B=D/'oracle200-20261011';manifest={c['id']:c for c in json.loads((B/'capture_manifest.json').read_text())};cases=[]
for p in sorted((O/'arrays').glob('*.npz')):
 z=np.load(p);where=np.argwhere(z['flags'])
 if len(where):
  i,side=map(int,where[0]);cases.append((p.stem,i,side,z['raw'][i,33+side*21:54+side*21],z['distances'][i,side]))
 if len(cases)==6:break
board=Image.new('RGB',(1000,400*len(cases)),'white');d=ImageDraw.Draw(board)
for row,(cid,i,side,q,dist) in enumerate(cases):
 f=manifest[cid]['frames'][i];im=Image.open(B/'frames'/cid/f['file']).convert('RGBA');base=Image.new('RGBA',im.size,'white');base.alpha_composite(im);im=base.convert('RGB');board.paste(im.resize((400,400)),(0,row*400))
 center=q[0];box=(int(center[0]-100),int(center[1]-100),int(center[0]+100),int(center[1]+100));crop=im.crop(box).resize((400,400));cd=ImageDraw.Draw(crop)
 for j,point in enumerate(q):
  x=(point[0]-box[0])*2;y=(point[1]-box[1])*2;color='red' if dist[j]>6 else 'blue';cd.ellipse((x-3,y-3,x+3,y+3),fill=color)
 board.paste(crop,(400,row*400));d.text((805,row*400+15),f'{cid} frame{i}\nside{side}\nwrist gap {dist[0]:.1f}px\nunsupported\n{int((dist>6).sum())}/21',fill='black')
board.save(O/'support_flag_review.jpg',quality=94);print([(c,i,s) for c,i,s,q,ds in cases])
