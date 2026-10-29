import json
from pathlib import Path
from PIL import Image,ImageDraw
O=Path(__file__).resolve().parent
samples=json.loads((O/'manifest.json').read_text())['samples'];trans=[]
for page in range(5):
 sheet=Image.new('RGB',(1080,1440),'#eef1f4');d=ImageDraw.Draw(sheet)
 for local,s in enumerate(samples[page*12:page*12+12]):
  idx=page*12+local;im=Image.open(O/s['image']).convert('RGB');scale=min(300/im.width,300/im.height);w,h=round(im.width*scale),round(im.height*scale);im=im.resize((w,h));ox,oy=(300-w)//2,(300-h)//2
  tile=Image.new('RGB',(300,300),'white');tile.paste(im,(ox,oy));x,y=(local%3)*360+35,(local//3)*360+40;sheet.paste(tile,(x,y));d.text((x-25,y-35),f"{idx+1:02d} {s['id']}",fill='black')
  for tick in range(0,301,50):d.text((x+tick-6,y-17),str(tick),fill='#555555');d.text((x-27,y+tick-5),str(tick),fill='#555555')
  trans.append(dict(id=s['id'],offset=[ox,oy],scale=[w/(s['crop'][2]-s['crop'][0]),h/(s['crop'][3]-s['crop'][1])]))
 sheet.save(O/f'blank-review-{page+1}.png')
(O/'draft-transforms.json').write_text(json.dumps(trans,indent=2))
