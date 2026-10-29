import json,hashlib,shutil
from pathlib import Path
import cv2,numpy as np
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');BASE=Path(__file__).parent;O=BASE/'body-reference-v1-20261009';O.mkdir(exist_ok=True);(O/'images').mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();manifest=read(B/'capture_manifest.json');refs={c['id']:c for c in read(B/'coordinate_reference.json')};diagnostic=read(BASE/'body-image-review-20261009'/'wrist_diagnostics.json')
selection={}
for idx in [0,6,12,17,23,29]:
 c=manifest[idx]
 for f in np.rint(np.linspace(0,len(c['frames'])-1,3)).astype(int):selection[(c['id'],int(f))]=['uniform_review']
for r in diagnostic['worst_per_clip'][:6]:selection.setdefault((r['id'],r['frame']),[]).append('error_selected_RTMW_wrist')
items=[];panels=[]
for (cid,i),strata in selection.items():
 c=next(c for c in manifest if c['id']==cid);f=c['frames'][i];src=B/'frames'/cid/f['file'];assert sha(src)==f['sha256'];name=f'{cid}_{i:04d}.png';dst=O/'images'/name
 if dst.exists():assert sha(dst)==sha(src)
 else:shutil.copyfile(src,dst)
 candidate={}
 for side in ['left','right']:
  hand=refs[cid]['samples'][i]['hands'][side]
  candidate[side+'_wrist']=dict(x=hand['0'][0],y=hand['0'][1],source='rendered_detailed_hand_origin',status='candidate_not_human_verified') if len(hand)==21 else None
 items.append(dict(id=f'{cid}_{i:04d}',label=c['label'],clip_id=cid,frame=i,source_index=f['sourceIndex'],time=f['time'],image='images/'+name,sha256=sha(dst),width=c['width'],height=c['height'],strata=strata,annotation_status='pending',annotator=None,reviewer=None,joints={side+'_'+joint:dict(x=None,y=None,visibility='unreviewed',uncertainty_px=None,note='',review_status='pending') for side in ['left','right'] for joint in ['shoulder','elbow','wrist']},reference_candidates=candidate))
 im=cv2.imread(str(src),-1);alpha=im[:,:,3:4].astype(np.float32)/255;im=np.rint(im[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8);im=cv2.resize(im,(260,260));band=np.full((45,260,3),255,np.uint8);cv2.putText(band,f"{cid} {c['label']} f{i}",(5,17),0,.43,(0,0,0),1);cv2.putText(band,'uniform' if strata==['uniform_review'] else 'error-selected',(5,36),0,.42,(0,0,0),1);panels.append(np.vstack([band,im]))
dataset=dict(version=1,status='DRAFT_NOT_VERIFIED',coordinate_system='Original image pixels; top-left origin; anatomical left/right, not screen left/right; no camera inversion.',selection='18 fixed first/middle/last frames from six predetermined development clips plus top6 RTMW body-wrist-error frames, deduplicated. Strata must be reported separately; neither is an unbiased random sample.',independent_from_predictions='No detector coordinates prefilled. Renderer wrist candidates are separate and not accepted labels.',items=items)
target=O/'annotations.json'
if target.exists():raise RuntimeError('Refuse overwrite annotations that may have been edited')
target.write_text(json.dumps(dataset,ensure_ascii=False,indent=2),encoding='utf8')
while len(panels)%4:panels.append(np.full_like(panels[0],255))
cv2.imwrite(str(O/'contact-sheet.png'),np.vstack([np.hstack(panels[i:i+4]) for i in range(0,len(panels),4)]))
print(json.dumps(dict(images=len(items),joints=len(items)*6,status=dataset['status'],output=str(O))))
