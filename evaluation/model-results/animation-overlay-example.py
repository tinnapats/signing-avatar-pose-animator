
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
D=Path(__file__).resolve().parent;B=D/'oracle200-20261011';O=D/'animation-overlay-20261013';O.mkdir(exist_ok=True)
read=lambda p:_local_json_loads(p.read_text(encoding='utf-8'))
cid='001';c=next(x for x in read(B/'capture_manifest.json') if x['id']==cid);assert c['label']=='hello'
z=np.load(B/'rtm'/(cid+'.npz'));p=z['rtmw_det'];s=z['rtmw_det_scores'];indices=np.rint(np.linspace(0,len(p)-1,6)).astype(int).tolist()
fontpath='C:/Windows/Fonts/arial.ttf'
font=lambda size:ImageFont.truetype(fontpath,size)
colors={'body':'#728C9C','left':'#2989D5','right':'#ED7D31'}
edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
pick=[0,4,5,8,9,12,13,16,17,20]
board=Image.new('RGB',(2400,1220),'white');draw=ImageDraw.Draw(board)
def center(x,y,text,size=25):draw.text((x,y),text,fill='#223c4c',font=font(size),anchor='mt')
center(1200,18,'Experiment 2 — RTMW keypoints detected from actual Animation images: hello',37)
center(1200,70,'Points overlaid on the original frames. No Renderer reference coordinates.',26)
audit=[]
for col,i in enumerate(indices):
 f=c['frames'][i];file=B/'frames'/cid/f['file'];assert hashlib.sha256(file.read_bytes()).hexdigest()==f['sha256']
 src=Image.open(file).convert('RGBA');im=Image.new('RGBA',src.size,'white');im.alpha_composite(src);im=im.convert('RGB');overlay=im.copy();d=ImageDraw.Draw(overlay)
 def pt(j):return tuple(map(float,p[i,j]))
 for j,k in [(0,1),(0,2),(5,6),(5,7),(7,9),(6,8),(8,10)]:
  if s[i,j]>=.3 and s[i,k]>=.3:d.line([pt(j),pt(k)],fill=colors['body'],width=3)
 for j in [0,1,2,5,6,7,8,9,10]:
  if s[i,j]>=.3:
   x,y=pt(j);d.ellipse((x-4,y-4,x+4,y+4),fill=colors['body'],outline='white',width=1)
 hands={}
 for side,off in [('left',91),('right',112)]:
  valid=bool(s[i,off]>=.3 and s[i,off:off+21].mean()>=.3);hands[side]=valid
  if not valid:continue
  for j,k in edges:d.line([pt(off+j),pt(off+k)],fill=colors[side],width=3)
  for j in range(21):
   x,y=pt(off+j);radius=4 if j in pick else 3;d.ellipse((x-radius,y-radius,x+radius,y+radius),fill=colors[side],outline='white',width=1)
 overlay.save(O/f'hello-frame-{i:03d}-overlay.png')
 x=col*400;center(x+200,123,f'Frame {i} | {f["time"]:.2f} s',25);board.paste(overlay.resize((400,400),Image.Resampling.LANCZOS),(x,165))
 center(x+200,586,'Detected right-hand crop',23)
 cx,cy=p[i,112];box=(int(cx-88),int(cy-88),int(cx+88),int(cy+88));crop=overlay.crop(box).resize((360,360),Image.Resampling.LANCZOS);board.paste(crop,(x+20,630))
 center(x+200,1010,f'Hands accepted: {sum(hands.values())}/2',21)
 audit.append({'frame':i,'time':f['time'],'source':str(file),'source_sha256':f['sha256'],'hand_accepted':hands,'crop_box':box,'raw_body9':p[i,[0,1,2,5,6,7,8,9,10]].tolist(),'left21':p[i,91:112].tolist(),'right21':p[i,112:133].tolist()})
for j,(label,color) in enumerate([('Body/face + body wrists',colors['body']),('Left hand: 21 detected points',colors['left']),('Right hand: 21 detected points',colors['right'])]):
 x=80+j*790;draw.line((x,1093,x+70,1093),fill=color,width=7);draw.text((x+88,1075),label,font=font(25),fill='#223c4c')
center(1200,1135,'Display: 9 body/face points + 21 per accepted hand. Recognizer input: selected 27 points, not all displayed points.',23)
center(1200,1176,'Detection can be misplaced even when confidence passes. Enlarged crops follow detector wrists, not ground truth.',23)
board.save(O/'hello-animation-keypoints.png')
(O/'provenance.json').write_text(json.dumps({'experiment':2,'scope':'RTMW detections from PNG only; no Renderer coordinates, no recognition rerun. Whole-hand acceptance same mean and wrist >=0.30 as original trial. Body shown with wrists (9), recognizer uses7body and10eachhand.','clip':c['label'],'samples':audit},indent=2),encoding='utf-8')
print(json.dumps({'frames':indices,'output':str(O/'hello-animation-keypoints.png')}))
