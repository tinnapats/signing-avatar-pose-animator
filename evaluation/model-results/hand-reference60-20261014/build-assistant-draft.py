import json
from pathlib import Path
from collections import Counter
from PIL import Image,ImageDraw
O=Path(__file__).resolve().parent
read=lambda n:json.loads((O/n).read_text(encoding='utf-8'))
data=read('manifest.json');tr=read('draft-transforms.json');obs=read('assistant-draft-observations.json');records=[]
assert len(obs['categories'])==len(data['samples'])==60
for i,s in enumerate(data['samples']):
 points=[dict(state='ambiguous',x=None,y=None) for _ in range(21)]
 for j,xy in zip(obs['landmark_order'],obs['tips'].get(str(i+1),[])):
  x=s['crop'][0]+(xy[0]-tr[i]['offset'][0])/tr[i]['scale'][0];y=s['crop'][1]+(xy[1]-tr[i]['offset'][1])/tr[i]['scale'][1]
  assert 0<=x<s['width'] and 0<=y<s['height'];points[j]=dict(state='visible',x=round(x,2),y=round(y,2))
 n=sum(p['state']=='visible' for p in points)
 notes=('AI visual draft only. '+('Five visible fingertip estimates; wrist center and internal joints are not unambiguously visible.' if n else 'Finger identity/joint centers are not sufficiently clear in this stylized crop; no coordinates guessed.')+' Category is visual appearance, not a claim about sign meaning or rest phase. Anatomical side not assigned. Requires independent review; NOT ground truth.')
 records.append(dict(id=s['id'],category=obs['categories'][i],side='unknown',crop_quality='multiple_hands' if i+1 in obs['multiple_hands'] else 'complete',notes=notes,reviewer='AI-assisted visual draft (NOT human verified)',reviewed=False,points=points))
payload=dict(schema='avatar-hand-labels-v1',dataset_id=data['dataset_id'],coordinate_system='original_full_image_pixels',status='AI_VISUAL_DRAFT_NOT_GROUND_TRUTH',provenance='Manual visual estimates from blank pixels, without predicted/Renderer keypoints. No training or scoring.',records=records)
(O/'assistant-draft.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
(O/'assistant-draft-data.js').write_text('window.ASSISTANT_DRAFT='+json.dumps(payload,ensure_ascii=False)+';',encoding='utf-8')
stats=dict(images_visually_screened=60,images_with_tip_estimates=sum(any(p['state']=='visible' for p in r['points']) for r in records),visible_point_estimates=sum(p['state']=='visible' for r in records for p in r['points']),ambiguous_points=sum(p['state']=='ambiguous' for r in records for p in r['points']),independently_verified_images=0,reviewed_flags_true=0,categories=dict(Counter(r['category'] for r in records)))
(O/'draft-summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
for page in range(5):
 sheet=Image.new('RGB',(1080,1440),'#eef1f4');draw=ImageDraw.Draw(sheet)
 for local,(s,r) in enumerate(zip(data['samples'][page*12:page*12+12],records[page*12:page*12+12])):
  i=page*12+local;im=Image.open(O/s['image']).convert('RGB');w=round(im.width*tr[i]['scale'][0]);h=round(im.height*tr[i]['scale'][1]);tile=Image.new('RGB',(300,300),'white');tile.paste(im.resize((w,h)),tuple(tr[i]['offset']));td=ImageDraw.Draw(tile)
  for j,p in enumerate(r['points']):
   if p['state']!='visible':continue
   x=(p['x']-s['crop'][0])*tr[i]['scale'][0]+tr[i]['offset'][0];y=(p['y']-s['crop'][1])*tr[i]['scale'][1]+tr[i]['offset'][1];td.ellipse((x-3,y-3,x+3,y+3),outline='red',width=2);td.text((x+4,y-10),str(j),fill='blue')
  x,y=(local%3)*360+30,(local//3)*360+40;sheet.paste(tile,(x,y));draw.text((x,y-30),f"{i+1:02d} {s['id']} / {r['category']}",fill='black');draw.text((x,y+305),'AI DRAFT / '+('5 tips' if str(i+1) in obs['tips'] else 'ambiguous - no point guessed'),fill='black')
 sheet.save(O/f'assistant-draft-review-{page+1}.png')
print(stats)
