import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
D=Path(__file__).resolve().parent;B=D/'oracle200-20261011';O=D/'exp2-recovery-20261013';O.mkdir(exist_ok=True)
manifest={x['id']:x for x in json.loads((B/'capture_manifest.json').read_text())};samples=[('001',0,1),('001',36,1),('002',46,0),('006',37,1),('014',37,0),('015',44,0)]
board=Image.new('RGB',(1800,1280),'white');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
for n,(cid,i,side) in enumerate(samples):
 z=np.load(B/'rtm'/(cid+'.npz'));q=z['rtmw_det'][i,91+21*side:112+21*side];w=z['rtmw_det'][i,9+side];e=z['rtmw_det'][i,7+side];im=Image.open(B/'frames'/cid/manifest[cid]['frames'][i]['file']).convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im);im=bg.convert('RGB');x=(n%3)*600;y=(n//3)*640;board.paste(im.resize((200,200)),(x,y+35))
 # Fixed window around body wrist, not Renderer or recognition information.
 mid=w;box=(int(mid[0]-100),int(mid[1]-100),int(mid[0]+100),int(mid[1]+100));crop=im.crop(box).resize((390,390));cd=ImageDraw.Draw(crop)
 for j,p in enumerate(q):
  xx=(p[0]-box[0])*390/200;yy=(p[1]-box[1])*390/200;cd.ellipse((xx-3,yy-3,xx+3,yy+3),fill='red');cd.text((xx+4,yy-12),str(j),font=font,fill='#173e9a',stroke_width=1,stroke_fill='white')
 for p,label in [(w,'BODY WRIST'),(e,'ELBOW')]:
  xx=(p[0]-box[0])*390/200;yy=(p[1]-box[1])*390/200;cd.rectangle((xx-5,yy-5,xx+5,yy+5),outline='#008000',width=3);cd.text((xx+6,yy),label,font=font,fill='#006000',stroke_width=1,stroke_fill='white')
 board.paste(crop,(x,y+240));d.text((x+5,y+5),f'{cid} frame{i}, anatomical '+('left' if side==0 else 'right'),font=font,fill='black')
board.save(O/'numbered-review.png');print(str(O/'numbered-review.png'))
