import sys,json,time,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit')
B=A/'pixels30_20261007'; O=B/'complexity2_20261008'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def extract():
 import cv2,mediapipe as mp
 O.mkdir(exist_ok=True);manifest=read(B/'capture_manifest.json')
 if not (O/'protocol.json').exists():save(O/'protocol.json',dict(capture_sha256=sha(B/'capture_manifest.json'),fps=30,complexity=2,rules='Exact original PNG order and timestamps; white alpha composite; no resampling, mirroring, cropping, hand repair or gates. Fresh Holistic per clip, confidence .5/.5.',production=read(B/'protocol.json')['production']))
 assert sha(B/'capture_manifest.json')==read(O/'protocol.json')['capture_sha256']
 for n,c in enumerate(manifest):
  target=O/(c['id']+'.npz'); side=O/(c['id']+'.json')
  if target.exists() and side.exists():assert sha(target)==read(side)['sha256'];continue
  assert c['fps']==30;raw=[];start=time.perf_counter()
  with mp.solutions.holistic.Holistic(static_image_mode=False,model_complexity=2,min_detection_confidence=.5,min_tracking_confidence=.5) as model:
   for i,f in enumerate(c['frames']):
    p=B/'frames'/c['id']/f['file'];assert sha(p)==f['sha256'];assert abs(f['time']-i/30)<1e-6
    im=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
    if im.shape[2]==4:
     alpha=im[:,:,3:4]/255.;im=np.rint(im[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8)
    d=model.process(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));a=np.zeros((75,4),np.float32)
    for lm,off in [(d.pose_landmarks,0),(d.left_hand_landmarks,33),(d.right_hand_landmarks,54)]:
     if lm:a[off:off+len(lm.landmark)]=[[v.x,v.y,v.z,v.visibility] for v in lm.landmark]
    raw.append(a)
  raw=np.array(raw);points=raw[:,:,:2]*np.array([c['width'],c['height']],np.float32)
  np.savez_compressed(target,points=points,raw=raw,times=[f['time'] for f in c['frames']])
  save(side,dict(id=c['id'],label=c['label'],frames=len(raw),fps=30,seconds=time.perf_counter()-start,sha256=sha(target),mediapipe=mp.__version__))
  print(n+1,'/30',c['label'],len(raw),flush=True)
 print('Extraction complete',flush=True)
def verify():
 import cv2
 n=changed=0
 for c in read(B/'capture_manifest.json'):
  for f in c['frames']:
   im=cv2.imread(str(B/'frames'/c['id']/f['file']),-1)
   if im.shape[2]==4:
    a=im[:,:,3:4].astype(np.float32)/255;d=im[:,:,3:4]/255.
    changed+=int(np.any(np.rint(im[:,:,:3]*a+255*(1-a)).astype(np.uint8)!=np.rint(im[:,:,:3]*d+255*(1-d)).astype(np.uint8)))
   n+=1
 print('RGB input check',n,'changed',changed,flush=True);assert changed==0
if __name__=='__main__':verify() if len(sys.argv)>1 else extract()
