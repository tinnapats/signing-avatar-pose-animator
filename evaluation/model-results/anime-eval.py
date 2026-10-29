import sys,importlib.util,json
from pathlib import Path
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('ev',Path(__file__).with_name('rtm-eval.py'));e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
np=e.np;R=e.O;B=e.B;O=B/'anime_hand_trial_20261009';read=e.read;save=e.save;old_data=e.data
NAMES=['mediapipe','rtmw_det','previous_best','wrist_crop_only','anime_only','anime_tracked','anime_replace_best','anime_tracked_replace_best','mp_anime_rtmw']
def data(c):
 a=old_data(c);z=np.load(O/(c['id']+'.npz'));assert e.sha(O/(c['id']+'.npz'))==read(O/(c['id']+'.json'))['sha256'];assert np.allclose(z['times'],[f['time'] for f in c['frames']]);mp=a['mediapipe'];rt=a['rtmw_det'];best=rt.copy()
 for off in [33,54]:
  valid=np.any(mp[:,off:off+21]!=0,axis=(1,2));best[valid,off:off+21]=mp[valid,off:off+21]
 out=dict(mediapipe=mp,rtmw_det=rt,previous_best=best,wrist_crop_only=a['crop_only'])
 for src,only,fallback in [('plain','anime_only','anime_replace_best'),('tracked','anime_tracked','anime_tracked_replace_best')]:
  aa=rt.copy();aa[:,33:]=0;bb=best.copy()
  for side,off in enumerate([33,54]):
   valid=z[src+'_mask'][:,side];aa[valid,off:off+21]=z[src][valid,side];bb[valid,off:off+21]=z[src][valid,side]
  out[only]=aa;out[fallback]=bb
 out['mp_anime_rtmw']=best.copy()
 for side,off in enumerate([33,54]):
  valid=(~np.any(mp[:,off:off+21]!=0,axis=(1,2)))&z['tracked_mask'][:,side];out['mp_anime_rtmw'][valid,off:off+21]=z['tracked'][valid,side]
 return {v:out[v] for v in NAMES}
def score():
 save(O/'evaluation_protocol.json',dict(variants=NAMES,rules='All body RTMW except mediapipe baseline. previous_best keeps MP hands when present else RTMW. anime_only missing stays0; replace_best uses anime when accepted else previous_best. mp_anime_rtmw preserves MP first then accepted tracked anime then RTMW. Freeze before scoring. Same STGCN, clip normalization, no temporal resampling.'))
 e.data=data;e.VARIANTS=NAMES;e.RESULT_NAME=str(O/'results.json');e.evaluate()
 result=read(O/'results.json');old=read(R/'hybrid_results.json')
 for new,previous in zip(result['predictions']['previous_best'],old['predictions']['rtmw_body_mp_priority']):
  assert new['label']==previous['label'] and new['prediction']==previous['prediction'] and new['top5']==previous['top5']
  assert np.isclose(new['target_probability'],previous['target_probability'],atol=1e-6)
def review():
 import cv2
 (O/'review').mkdir(exist_ok=True)
 for c in read(B/'capture_manifest.json'):
  if c['id'] not in ['001','002','005','011','016','025'] or not (O/(c['id']+'.npz')).exists():continue
  a=data(c);raw=read(O/(c['id']+'_raw.json'));panels=[]
  for i in np.rint(np.linspace(0,len(c['frames'])-1,3)).astype(int):
   im=cv2.imread(str(B/'frames'/c['id']/c['frames'][i]['file']),-1);alpha=im[:,:,3:4].astype(np.float32)/255;im=np.rint(im[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8);row=[]
   for v in ['previous_best','wrist_crop_only','anime_only','anime_tracked','mp_anime_rtmw']:
    img=im.copy()
    if v.startswith('anime'):
     for box in raw[i]['boxes']:cv2.rectangle(img,tuple(np.array(box[:2],int)),tuple(np.array(box[2:],int)),(0,150,0),1)
    for off,color in [(33,(255,100,0)),(54,(0,50,255))]:
     q=a[v][i,off:off+21]
     if not np.any(q):continue
     for chain in [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]:
      for j,k in zip(chain,chain[1:]):cv2.line(img,tuple(np.rint(q[j]).astype(int)),tuple(np.rint(q[k]).astype(int)),color,2)
     for p in q:cv2.circle(img,tuple(np.rint(p).astype(int)),3,color,-1)
    img=cv2.resize(img,(320,320));img=cv2.copyMakeBorder(img,30,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255));cv2.putText(img,v+' f'+str(i),(3,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1);row.append(img)
   panels.append(np.concatenate(row,axis=1))
  cv2.imwrite(str(O/'review'/(c['id']+'.png')),np.concatenate(panels,axis=0))
 print('review saved')
if __name__=='__main__':review() if len(sys.argv)>1 else score()
