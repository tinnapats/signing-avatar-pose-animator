import sys,importlib.util,time,json
from pathlib import Path
sys.dont_write_bytecode=True
def module(name,file):
 s=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ah=module('ah','anime-hands.py');np=ah.np;read=ah.read;save=ah.save;sha=ah.sha
O=Path(__file__).parent/'confirm30-crop-20261009';OLD=ah.B;ROOT=OLD.parent.parent
def check():
 for f,h in read(O/'protocol.json')['production'].items():assert sha(ROOT/f)==h
def mp_extract():
 import cv2,mediapipe as mp
 (O/'arrays').mkdir(exist_ok=True);(O/'detected').mkdir(exist_ok=True)
 for c in read(O/'capture_manifest.json'):
  target=O/'arrays'/(c['id']+'.npz');meta=target.with_suffix('.json')
  if target.exists():assert sha(target)==read(meta)['sha256'];continue
  rows=[];points=np.zeros((len(c['frames']),75,2),np.float32)
  def pack(lm):return [[p.x,p.y,p.z,p.visibility] for p in lm.landmark] if lm else []
  with mp.solutions.holistic.Holistic(static_image_mode=False,smooth_landmarks=True,enable_segmentation=False,refine_face_landmarks=False,model_complexity=1,min_detection_confidence=.5,min_tracking_confidence=.5) as model:
   for i,f in enumerate(c['frames']):
    path=O/'frames'/c['id']/f['file'];assert sha(path)==f['sha256'];im=ah.base.image(cv2,path);a=model.process(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));r=dict(pose=pack(a.pose_landmarks),left=pack(a.left_hand_landmarks),right=pack(a.right_hand_landmarks));rows.append(r)
    if r['pose']:
     for j,p in enumerate(r['pose']):
      if p[3]>0:points[i,j]=[p[0]*c['width'],p[1]*c['height']]
    for side,off in [('left',33),('right',54)]:
     if r[side]:points[i,off:off+21]=np.array(r[side],np.float32)[:,:2]*[c['width'],c['height']]
  np.savez_compressed(target,primary=points,times=[f['time'] for f in c['frames']]);save(O/'detected'/(c['id']+'.json'),rows);save(meta,dict(sha256=sha(target),label=c['label'],version=mp.__version__));print('MEDIAPIPE',c['id'],c['label'],flush=True)
 check();print('MEDIAPIPE COMPLETE',flush=True)
def rtm_extract():
 cv2,rt,det=ah.base.setup();import onnxruntime as ort
 folder=O/'rtm';folder.mkdir(exist_ok=True);models={}
 for k in ['det','rtmw','hand']:
  info=read(ah.R/(k+'_model.json'));assert sha(Path(info['path']))==info['sha256'];models[k]=(det if k=='det' else rt)(info['path'],model_input_size=tuple(info['input_size']),backend='onnxruntime',device='cpu')
 assert sha(ah.O/'model.onnx')=='408750ad39645fcdc0c5e774aa45a73941b2e785fc5611fb7d3d9790a41899c0';anime=ort.InferenceSession(str(ah.O/'model.onnx'),providers=['CPUExecutionProvider'])
 save(folder/'protocol.json',read(O/'protocol.json'))
 for c in read(O/'capture_manifest.json'):
  target=folder/(c['id']+'.npz');meta=target.with_suffix('.json')
  if target.exists():assert sha(target)==read(meta)['sha256'];continue
  out={k:[] for k in ['rtmw_det','rtmw_det_scores','baseline','baseline_mask','wide_120','wide_120_mask','ms']};details=[];detected=0
  for i,f in enumerate(c['frames']):
   path=O/'frames'/c['id']/f['file'];assert sha(path)==f['sha256'];im=ah.base.image(cv2,path);start=time.perf_counter();bodyboxes=models['det'](im)
   if len(bodyboxes):
    box=max(bodyboxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1]))[:4];pp,ss=models['rtmw'](im,bboxes=[box]);body,bs=pp[0],ss[0];detected+=1
   else:body=np.zeros((133,2),np.float32);bs=np.zeros(133,np.float32)
   out['rtmw_det'].append(body);out['rtmw_det_scores'].append(bs);boxes,conf=ah.detect(anime,im);record=dict(boxes=boxes.tolist(),confidence=conf.tolist(),variants={})
   for name,factor in [('baseline',1.),('wide_120',1.2)]:
    candidates=[]
    for b in boxes:
     mid=(b[:2]+b[2:])/2;half=(b[2:]-b[:2])*factor/2;bb=np.r_[mid-half,mid+half];pp,ss=models['hand'](im,bboxes=[bb]);candidates.append((pp[0],ss[0]))
    choice=ah.assign(candidates,body,bs,{},i,False);p=np.zeros((2,21,2),np.float32);mask=np.zeros(2,bool)
    for side,j in enumerate(choice):
     if j>=0:p[side]=candidates[j][0];mask[side]=True
    out[name].append(p);out[name+'_mask'].append(mask);record['variants'][name]=dict(points=[p.tolist() for p,s in candidates],scores=[s.tolist() for p,s in candidates],assignment=choice)
   out['ms'].append((time.perf_counter()-start)*1000);details.append(record)
  np.savez_compressed(target,**{k:np.array(v) for k,v in out.items()},times=[f['time'] for f in c['frames']]);save(folder/(c['id']+'_raw.json'),details);save(meta,dict(sha256=sha(target),raw_sha256=sha(folder/(c['id']+'_raw.json')),detected=detected,label=c['label']));print('RTM',c['id'],c['label'],'frames',len(details),flush=True)
 check();print('RTM COMPLETE',flush=True)
def score():
 e=module('ev','rtm-eval.py');e.B=O;e.O=O/'rtm'
 def data(c):
  z=np.load(e.O/(c['id']+'.npz'));mpfile=O/'arrays'/(c['id']+'.npz');assert sha(mpfile)==read(mpfile.with_suffix('.json'))['sha256'];mp=np.load(mpfile)['primary'];rt=e.convert(z['rtmw_det'],z['rtmw_det_scores']);best=rt.copy()
  for off in [33,54]:
   mask=np.any(mp[:,off:off+21]!=0,axis=(1,2));best[mask,off:off+21]=mp[mask,off:off+21]
  out={'mediapipe':mp,'previous_best':best}
  assert sha(e.O/(c['id']+'.npz'))==read(e.O/(c['id']+'.json'))['sha256'];assert np.allclose(z['times'],[f['time'] for f in c['frames']])
  for name in ['baseline','wide_120']:
   a=best.copy()
   for side,off in enumerate([33,54]):
    mask=z[name+'_mask'][:,side];a[mask,off:off+21]=z[name][mask,side]
   out[name]=a
  folder=O/'model_inputs';folder.mkdir(exist_ok=True);np.savez_compressed(folder/(c['id']+'.npz'),**out,times=z['times']);return out
 e.data=data;e.VARIANTS=['mediapipe','previous_best','baseline','wide_120'];e.RESULT_NAME=str(O/'results.json');e.evaluate();r=read(O/'results.json');r['frames']=sum(len(c['frames']) for c in read(O/'capture_manifest.json'));save(O/'results.json',r);check()
if __name__=='__main__':{'mp':mp_extract,'rtm':rtm_extract,'score':score}[sys.argv[1]]()
