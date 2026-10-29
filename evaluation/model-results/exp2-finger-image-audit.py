import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
D=Path(__file__).resolve().parent;O=D/'exp2-finger-image-audit-20261014';B=D/'oracle200-20261011';R=D/'exp2-rejection-audit-20261014'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def save(n,o):(O/n).write_text(json.dumps(o,indent=2),encoding='utf-8')
def rgb(p):
 im=Image.open(p).convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im);return bg.convert('RGB')
def main():
 mf={c['id']:c['frames'] for c in read(B/'capture_manifest.json')};rows=[];cache={}
 for r in read(R/'candidates.json'):
  cid,i=r['id'],r['frame']
  if i==0:continue
  if cache.get('key')!=(cid,i):
   imgs=[]
   for j in [i-1,i]:
    f=mf[cid][j];p=B/'frames'/cid/f['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];imgs.append(np.array(rgb(p)))
   cache=dict(key=(cid,i),imgs=imgs)
  prev,cur=cache['imgs'];x0,y0,x1,y1=np.rint(r['box']).astype(int);x0,x1=np.clip([x0,x1],0,cur.shape[1]);y0,y1=np.clip([y0,y1],0,cur.shape[0])
  if x1<=x0 or y1<=y0:continue
  delta=np.abs(cur[y0:y1,x0:x1].astype(float)-prev[y0:y1,x0:x1]).max(2);motion=float((delta>20).mean())
  group=('high' if motion>=.1 else 'low')+'_'+('pass' if not r['failed'] else 'reject')
  rows.append(dict(**r,motion_fraction=motion,audit_group=group,audit_rank=hashlib.sha256(f"finger-audit-v1:{cid}:{i}:{r['candidate']}".encode()).hexdigest(),previous=str(B/'frames'/cid/mf[cid][i-1]['file'])))
 chosen=[];used=set()
 for g in ['high_pass','high_reject','low_pass','low_reject']:
  n=0
  for r in sorted((x for x in rows if x['audit_group']==g),key=lambda x:x['audit_rank']):
   if r['id'] in used:continue
   chosen.append(r);used.add(r['id']);n+=1
   if n==3:break
 save('selection.json',chosen);save('motion-diagnostics.json',dict(counts=dict(Counter(r['audit_group'] for r in rows)),candidates=rows))
 edges=[(0,j) for j in [1,5,9,13,17]]+[(j,j+1) for j in [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]]
 for page in range(3):
  canvas=Image.new('RGB',(1250,1320),'white');draw=ImageDraw.Draw(canvas)
  for row,r in enumerate(chosen[page*4:page*4+4]):
   q=np.array(r['points']);bb=np.array(r['box']);lo=np.minimum(bb[:2],q.min(0))-20;hi=np.maximum(bb[2:],q.max(0))+20;src=rgb(r['path']);box=(max(0,int(lo[0])),max(0,int(lo[1])),min(src.width,int(hi[0])),min(src.height,int(hi[1])))
   for col in range(3):
    tile=(rgb(r['previous']) if col==0 else src).crop(box);scale=min(290/tile.width,290/tile.height);tile=tile.resize((round(tile.width*scale),round(tile.height*scale)))
    if col==2:
     td=ImageDraw.Draw(tile);pts=(q-box[:2])*scale
     for a,b in edges:td.line([tuple(pts[a]),tuple(pts[b])],fill='#0066cc',width=2)
     for j,(x,y) in enumerate(pts):td.ellipse((x-2,y-2,x+2,y+2),fill='red');td.text((x+3,y),str(j),fill='black')
    canvas.paste(tile,(col*300,row*330+30))
    draw.text((col*300+5,row*330+5),['Previous image','Current image','Current detected 21 points'][col],fill='black')
   draw.text((905,row*330+35),f"{r['id']} / frame {r['frame']} / c{r['candidate']}\n{r['audit_group']}\nImage change {r['motion_fraction']:.1%}\nMean conf {r['mean_conf']:.3f}\nWrist conf {r['wrist_conf']:.3f}\nGate: {r['group']}",fill='black')
  canvas.save(O/f'review-{page+1:02d}.png')
 print([(r['id'],r['frame'],r['candidate'],r['audit_group']) for r in chosen])
if __name__=='__main__':main()
