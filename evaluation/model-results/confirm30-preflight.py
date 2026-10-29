import sys,json
from pathlib import Path
import cv2,mediapipe as mp,numpy as np
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');O=Path(__file__).parent/'confirm30-crop-20261009'
c=json.loads((B/'capture_manifest.json').read_text(encoding='utf8'))[0];old=np.load(B/'arrays'/'000.npz')['primary'];a=np.zeros_like(old)
with mp.solutions.holistic.Holistic(static_image_mode=False,smooth_landmarks=True,enable_segmentation=False,refine_face_landmarks=False,model_complexity=1,min_detection_confidence=.5,min_tracking_confidence=.5) as h:
 for i,f in enumerate(c['frames']):
  im=cv2.imread(str(B/'frames'/'000'/f['file']),-1);alpha=im[:,:,3:4].astype(np.float32)/255;im=np.rint(im[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8);r=h.process(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
  if r.pose_landmarks:
   for j,p in enumerate(r.pose_landmarks.landmark):
    if p.visibility>0:a[i,j]=[p.x*c['width'],p.y*c['height']]
  for lm,off in [(r.left_hand_landmarks,33),(r.right_hand_landmarks,54)]:
   if lm:a[i,off:off+21]=np.array([[p.x,p.y] for p in lm.landmark],np.float32)*[c['width'],c['height']]
same=np.array_equal(a,old);maxerr=float(np.max(abs(a-old)));assert np.allclose(a,old,atol=.001),maxerr
(O/'mp_reproduction.json').write_text(json.dumps(dict(clip=c['label'],frames=len(a),exact=same,max_abs_error_px=maxerr,rules='Known development clip control only; no selection or tuning on confirmation scores. Holistic primary output reproduced without unused secondary Hands detector.')),encoding='utf8');print('REPRODUCED',same,maxerr)
