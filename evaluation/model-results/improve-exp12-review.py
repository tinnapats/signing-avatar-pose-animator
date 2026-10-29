import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
D=Path(__file__).resolve().parent;O=D/'improve-exp12-20261013';B=D/'oracle200-20261011'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
ex=read(O/'exp2-extraction.json');mf={c['id']:c for c in read(B/'capture_manifest.json')};chosen=[];seen=set()
for r in ex['events']:
 if r['accepted'] and r['id'] not in seen:chosen.append(r);seen.add(r['id'])
 if len(chosen)==6:break
if not chosen:chosen=ex['events'][:6]
board=Image.new('RGB',(1200,440*len(chosen)),'white');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
for n,r in enumerate(chosen):
 c=mf[r['id']];im=Image.open(B/'frames'/r['id']/c['frames'][r['frame']]['file']).convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im);im=bg.convert('RGB');overview=im.resize((380,380));ov=ImageDraw.Draw(overview)
 for cand in r['candidates']:
  bb=np.asarray(cand['box'])*380/im.width;ov.rectangle(tuple(bb),outline='green',width=2)
 board.paste(overview,(0,n*440+45));draw.text((5,n*440+5),f"Clip {r['id']} frame {r['frame']} boxes",font=font,fill='black')
 if not r['accepted']:continue
 accepted=r['accepted'][0];cand=r['candidates'][accepted['candidate']];q=np.asarray(cand['points']);ctr=(q.max(0)+q.min(0))/2;sz=max(120,float(max(q.max(0)-q.min(0))+50));bb=np.r_[ctr-sz/2,ctr+sz/2].astype(int);sz=bb[2]-bb[0]
 for k in [1,2]:
  crop=im.crop(tuple(bb)).resize((380,380));cd=ImageDraw.Draw(crop)
  if k==2:
   for j,p in enumerate(q):
    x,y=(p-bb[:2])*380/sz;cd.ellipse((x-3,y-3,x+3,y+3),fill='red');cd.text((x+3,y),str(j),font=font,fill='blue',stroke_width=1,stroke_fill='white')
  board.paste(crop,(400*k,n*440+45));draw.text((400*k+5,n*440+5),'Image crop' if k==1 else 'Accepted hand '+str(accepted['side']),font=font,fill='black')
board.save(O/'exp2-review.png');(O/'review-selection.json').write_text(json.dumps(chosen,indent=2),encoding='utf-8')
print('review count',len(chosen))
