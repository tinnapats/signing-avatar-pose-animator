
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import os,sys,json,time,hashlib,importlib.metadata
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit');B=A/'pixels30_20261007';O=B/'rtm_trial_20261008'
os.environ['TORCH_HOME']=str(O/'model_cache')
read=lambda p:_local_json_loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
URL='https://download.openmmlab.com/mmpose/v1/projects/'
MODELS={
 'det':(URL+'rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',(640,640)),
 'dw':(URL+'rtmposev1/onnx_sdk/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip',(192,256)),
 'rtmw':(URL+'rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip',(192,256)),
 'hand':(URL+'rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip',(256,256))}
def setup():
 import onnxruntime as ort
 original=ort.InferenceSession
 def session(*a,**kw):
  s=ort.SessionOptions();s.intra_op_num_threads=4;s.inter_op_num_threads=1;kw['sess_options']=s;return original(*a,**kw)
 ort.InferenceSession=session
 import cv2
 cv2.setNumThreads(1)
 from rtmlib import RTMPose,YOLOX
 return cv2,RTMPose,YOLOX
def load(k,rt,det):
 url,size=MODELS[k];m=(det if k=='det' else rt)(url,model_input_size=size,backend='onnxruntime',device='cpu')
 save(O/(k+'_model.json'),dict(url=url,path=m.onnx_model,sha256=sha(Path(m.onnx_model)),input_size=size,providers=m.session.get_providers()))
 return m
def image(cv2,p):
 im=cv2.imread(str(p),-1)
 if im.shape[2]==4:
  a=im[:,:,3:4].astype(np.float32)/255;im=np.rint(im[:,:,:3]*a+255*(1-a)).astype(np.uint8)
 return im
def extract():
 O.mkdir(exist_ok=True);cv2,rt,det=setup();manifest=read(B/'capture_manifest.json')
 protocol=dict(manifest_sha256=sha(B/'capture_manifest.json'),models={k:list(v) for k,v in MODELS.items()},fps=30,frames=sum(len(c['frames']) for c in manifest),production=read(B/'protocol.json')['production'],rules='No smoothing/tracking/imputation/resampling. COCO133 (not OpenPose134). Two bbox strata: largest YOLOX box, no detection => missing; fixed full image bbox independently. Confidence hand mean>=.3 and wrist>=.3; selected body>=.3. All raw points/conf saved. Never use renderer reference at inference. Single-clip STGCN unchanged. CPU4threads.',versions={k:importlib.metadata.version(k) for k in ['rtmlib','onnxruntime','numpy']})
 if not (O/'protocol.json').exists():save(O/'protocol.json',protocol)
 else:assert read(O/'protocol.json')==protocol
 detector=load('det',rt,det);poses={k:load(k,rt,det) for k in ['dw','rtmw']}
 for n,c in enumerate(manifest):
  target=O/(c['id']+'.npz');side=O/(c['id']+'.json')
  if target.exists() and side.exists():assert sha(target)==read(side)['sha256'];continue
  arrays={k:[] for k in ['boxes','det_ms','pose_ms']}
  for k in poses:
   for crop in ['det','full']:
    arrays[k+'_'+crop]=[];arrays[k+'_'+crop+'_scores']=[]
  for i,f in enumerate(c['frames']):
   p=B/'frames'/c['id']/f['file'];assert sha(p)==f['sha256'];assert abs(f['time']-i/30)<1e-6
   im=image(cv2,p);start=time.perf_counter();boxes=detector(im);arrays['det_ms'].append((time.perf_counter()-start)*1000)
   box=None
   if len(boxes):box=np.array(max(boxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1])))[:4]
   arrays['boxes'].append(np.full(4,np.nan) if box is None else box);timings=[]
   for k,pose in poses.items():
    for crop in ['det','full']:
     start=time.perf_counter()
     if crop=='det' and box is None:pts=np.zeros((133,2),np.float32);scores=np.zeros(133,np.float32)
     else:
      bb=box if crop=='det' else np.array([0,0,im.shape[1],im.shape[0]])
      pts,scores=pose(im,bboxes=[bb]);pts=pts[0];scores=scores[0];assert pts.shape==(133,2)
     arrays[k+'_'+crop].append(pts);arrays[k+'_'+crop+'_scores'].append(scores);timings.append((time.perf_counter()-start)*1000)
   arrays['pose_ms'].append(timings)
  np.savez_compressed(target,**{k:np.array(v) for k,v in arrays.items()},times=[f['time'] for f in c['frames']]);save(side,dict(id=c['id'],label=c['label'],frames=len(c['frames']),sha256=sha(target),detected=int(np.isfinite(np.array(arrays['boxes'])[:,0]).sum()),pose_ms_mean=np.mean(arrays['pose_ms'],axis=0).tolist()))
  print(n+1,'/30',c['label'],'frames',len(c['frames']),'ms',np.round(np.mean(arrays['pose_ms'],axis=0),1),flush=True)
 print('DONE',flush=True)
if __name__=='__main__':extract()
