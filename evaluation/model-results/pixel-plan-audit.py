import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
D=Path(__file__).resolve().parent;B=D/'oracle200-20261011';O=D/'pixel-only-plan-20261012';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
r=read(B/'results.json');manifest={c['id']:c for c in read(B/'capture_manifest.json')};refs={c['id']:c for c in read(B/'coordinate_reference.json')}
groups={k:[] for k in ['improved','worsened','both_correct','both_wrong']};rows=[]
for a,b in zip(r['predictions']['pixels'],r['predictions']['mask_only']):
 assert a['id']==b['id'] and a['label']==b['label'];cid=a['id'];ac=a['prediction']==a['label'];bc=b['prediction']==b['label'];g=('both_correct' if ac else 'both_wrong') if ac==bc else ('improved' if bc else 'worsened')
 z=np.load(B/'rtm'/(cid+'.npz'));p=z['rtmw_det'];s=z['rtmw_det_scores'];valid=np.stack([(s[:,j:j+21].mean(1)>=.3)&(s[:,j]>=.3) for j in [91,112]],axis=1)
 present=np.array([[bool(t['hands'][side]) for side in ['left','right']] for t in refs[cid]['samples']]);removed=valid&~present
 row={'id':cid,'label':a['label'],'group':g,'pixels':a['prediction'],'masked':b['prediction'],'frames':len(p),'removed_hand_frames':int(removed.sum()),'detected_hand_frames':int(valid.sum()),'reference_present_hand_frames':int(present.sum())}
 groups[g].append(row);rows.append(row)
summary={g:{'clips':len(v),'removed_hand_frames':sum(c['removed_hand_frames'] for c in v),'detected_hand_frames':sum(c['detected_hand_frames'] for c in v)} for g,v in groups.items()}
save(O/'diagnosis.json',{'summary':summary,'clips':rows,'selection':'Two clips/group by sha256(id), four evenly spaced changed frames each. Outcome stratification for diagnosis ONLY. Not representative accuracy sample.'})
selected=[]
for group,items in groups.items():
 for c in sorted(items,key=lambda x:hashlib.sha256(x['id'].encode()).hexdigest())[:2]:
  cid=c['id'];z=np.load(B/'rtm'/(cid+'.npz'));s=z['rtmw_det_scores'];p=z['rtmw_det'];rr=refs[cid]['samples'];indices=[i for i,t in enumerate(rr) if any(not t['hands'][side] and s[i,j]>=.3 and s[i,j:j+21].mean()>=.3 for side,j in [('left',91),('right',112)])]
  if not indices:indices=list(range(len(rr)))
  ii=sorted(set(indices[int(k)] for k in np.linspace(0,len(indices)-1,4).round().astype(int)))
  board=Image.new('RGB',(1212,450*len(ii)),'white');draw=ImageDraw.Draw(board)
  for row,i in enumerate(ii):
   f=manifest[cid]['frames'][i];im=Image.open(B/'frames'/cid/f['file']).convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im);im=bg.convert('RGB');y=row*450
   board.paste(im.resize((404,404)),(0,y+25));draw.text((5,y+5),f'{cid} frame {i} | IMAGE ONLY',(0,0,0))
   for side,j,col in [('left',91,1),('right',112,2)]:
    q=p[i,j:j+21];cx,cy=q[0];box=(int(cx-90),int(cy-90),int(cx+90),int(cy+90));crop=im.crop(box).resize((404,404));board.paste(crop,(col*404,y+25));draw.text((col*404+5,y+5),f'{side} detector wrist crop',(0,0,0))
   selected.append({'id':cid,'frame':i,'image':str(B/'frames'/cid/f['file']),'board':f'{cid}.jpg','row':row,'left':'unreviewed','right':'unreviewed'})
  board.save(O/f'{cid}.jpg',quality=94)
save(O/'review_samples.json',selected);print(json.dumps({'summary':summary,'selected':[(c['id'],c['label'],c['group']) for g,v in groups.items() for c in sorted(v,key=lambda x:hashlib.sha256(x['id'].encode()).hexdigest())[:2]]}),flush=True)
