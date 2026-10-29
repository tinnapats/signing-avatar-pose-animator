import json
from pathlib import Path
O=Path(__file__).resolve().parent/'pixel-only-plan-20261012'
samples=json.loads((O/'review_samples.json').read_text(encoding='utf-8'));labels=[]
for r in samples:
 cid=r['id'];i=r['frame']
 styles={'149':['flat','flat'],'125':['flat','outlined' if i!=69 else 'flat'],'195':['flat','flat'],'045':['flat','outlined'],'122':['flat','outlined'],'147':['outlined','outlined'],'178':['flat','flat'],'057':['outlined','outlined']}[cid]
 if cid=='149' and i==83:styles=['ambiguous_overlap','ambiguous_overlap']
 for side,style in zip(['left','right'],styles):
  labels.append({'id':cid,'frame':i,'side':side,'visible_hand':True,'drawing_style':style,'semantic_rest_or_active':'not_established','hand_overlap':(cid=='149' and i==83) or (cid=='057' and i in [16,24]),'detector_crop_misses_fingers':cid=='045' and side=='left','joint_accuracy':'not_annotated','reviewer':'assistant_visual_review','independent_human_verified':False})
assert len(labels)==64
(O/'visual_annotations.json').write_text(json.dumps({'scope':'Pilot labels from inspected images, not independent ground truth; no semantic ASL/rest labels inferred from drawing style.','annotations':labels},indent=2),encoding='utf-8')
print({s:sum(x['drawing_style']==s for x in labels) for s in ['flat','outlined','ambiguous_overlap']})
