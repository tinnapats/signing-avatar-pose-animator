import sys,json,time,hashlib,urllib.request,importlib.util,itertools
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
spec=importlib.util.spec_from_file_location('base',Path(__file__).with_name('rtm-avatar.py'));base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
B=base.B;R=base.O;O=B/'anime_hand_trial_20261009';read=base.read;save=base.save;sha=base.sha
URL='https://huggingface.co/deepghs/anime_hand_detection/resolve/main/hand_detect_v1.0_s/'
def prepare():
 O.mkdir(exist_ok=True)
 for name in ['model.onnx','labels.json','threshold.json','model_artifacts.json']:
  p=O/name
  if not p.exists():urllib.request.urlretrieve(URL+name,p)
 assert sha(O/'model.onnx')=='408750ad39645fcdc0c5e774aa45a73941b2e785fc5611fb7d3d9790a41899c0'
 import onnxruntime as ort
 opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.log_severity_level=3
 s=ort.InferenceSession(str(O/'model.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
 info=dict(input=[dict(name=x.name,shape=x.shape,type=x.type) for x in s.get_inputs()],output=[dict(name=x.name,shape=x.shape) for x in s.get_outputs()],metadata=s.get_modelmeta().custom_metadata_map,threshold=read(O/'threshold.json'),labels=read(O/'labels.json'))
 save(O/'model_info.json',info);print(json.dumps(info),flush=True)
def nms(boxes,conf):
 order=np.argsort(conf)[::-1];keep=[]
 while len(order):
  j=order[0];keep.append(j);rest=order[1:]
  lo=np.maximum(boxes[j,:2],boxes[rest,:2]);hi=np.minimum(boxes[j,2:],boxes[rest,2:]);inter=np.prod(np.maximum(hi-lo+1,0),axis=1);area=np.prod(boxes[:,2:]-boxes[:,:2]+1,axis=1);iou=inter/(area[j]+area[rest]-inter);order=rest[iou<=.7]
 return keep
def detect(session,im):
 from PIL import Image
 rgb=Image.fromarray(im[:,:,::-1]);arr=np.asarray(rgb.resize((640,640)),np.float32).transpose(2,0,1)/255
 raw=session.run(['output0'],{'images':arr[None]})[0][0];assert raw.shape[0]==5,raw.shape
 raw=raw[:,raw[4]>.395].T
 if not len(raw):return np.empty((0,4)),np.empty(0)
 boxes=np.c_[raw[:,:2]-raw[:,2:4]/2,raw[:,:2]+raw[:,2:4]/2];conf=raw[:,4];keep=nms(boxes,conf);boxes=boxes[keep];boxes=boxes/640*np.array([im.shape[1],im.shape[0],im.shape[1],im.shape[0]]);boxes=np.rint(np.clip(boxes,0,im.shape[0]))
 return boxes,conf[keep]
def assign(candidates,body,scores,previous,frame,tracked):
 n=len(candidates);cost=np.full((2,n),np.inf)
 for side,(w,e) in enumerate([(9,7),(10,8)]):
  if scores[w]<.3 or scores[e]<.3:continue
  length=max(float(np.linalg.norm(body[w]-body[e])),10)
  for j,(p,s) in enumerate(candidates):
   d=float(np.linalg.norm(p[0]-body[w]))/length
   if s.mean()<.3 or s[0]<.3 or d>.65:continue
   cost[side,j]=d
   if tracked and side in previous and frame-previous[side][0]<=3:cost[side,j]+=.25*min(float(np.linalg.norm(p[0]-previous[side][1]))/length,1.)
 opts=[-1]+list(range(n));best=(-1,-1);bestcost=2.
 for l,r in itertools.product(opts,repeat=2):
  if l>=0 and l==r:continue
  c=(1 if l<0 else cost[0,l])+(1 if r<0 else cost[1,r])
  if c<bestcost:bestcost=c;best=(l,r)
 return best
def extract():
 import onnxruntime as ort
 cv2,rt,det=base.setup();opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1;opts.log_severity_level=3
 # base.setup configures the session factory; detector and hand each use CPU4threads.
 session=ort.InferenceSession(str(O/'model.onnx'),providers=['CPUExecutionProvider']);hand=base.load('hand',rt,det)
 protocol=dict(model_sha256=sha(O/'model.onnx'),manifest_sha256=sha(B/'capture_manifest.json'),body_source='RTMW-det cached',detector=dict(resize=640,threshold=.395,nms_iou=.7,preprocess='Pillow RGB resize, float32 CHW /255; source imgutils default640 when imgsz absent'),rules='All original frames/timestamps. RTMPose-hand bbox default padding1.25. Global one-to-one assignment to body wrists, mean hand score>=.3,wrist>=.3,distance<=.65forearm; unmatched missing. Temporal variant adds .25*capped distance to previous accepted wrist within3frames, no invented frames. No renderer/payload input. No smoothing.',production=read(R/'protocol.json')['production'])
 if not (O/'protocol.json').exists():save(O/'protocol.json',protocol)
 else:assert read(O/'protocol.json')==protocol
 for c in read(B/'capture_manifest.json'):
  target=O/(c['id']+'.npz');sidecar=target.with_suffix('.json')
  if target.exists():assert sha(target)==read(sidecar)['sha256'];continue
  z=np.load(R/(c['id']+'.npz'));body=z['rtmw_det'];bs=z['rtmw_det_scores'];out={k:[] for k in ['plain','tracked','plain_scores','tracked_scores','plain_mask','tracked_mask','ms']};rawrows=[];prev={}
  for i,f in enumerate(c['frames']):
   p=B/'frames'/c['id']/f['file'];assert sha(p)==f['sha256'];im=base.image(cv2,p);start=time.perf_counter();boxes,conf=detect(session,im);cand=[]
   for box in boxes:
    pp,ss=hand(im,bboxes=[box]);cand.append((pp[0],ss[0]))
   choices={k:assign(cand,body[i],bs[i],prev,i,k=='tracked') for k in ['plain','tracked']}
   for k,choice in choices.items():
    points=np.zeros((2,21,2),np.float32);sc=np.zeros((2,21),np.float32);mask=np.zeros(2,bool)
    for s,j in enumerate(choice):
     if j<0:continue
     points[s],sc[s]=cand[j];mask[s]=True
     if k=='tracked':prev[s]=(i,points[s,0].copy())
    out[k].append(points);out[k+'_scores'].append(sc);out[k+'_mask'].append(mask)
   out['ms'].append((time.perf_counter()-start)*1000);rawrows.append(dict(index=i,boxes=boxes.tolist(),box_scores=conf.tolist(),points=[p.tolist() for p,s in cand],scores=[s.tolist() for p,s in cand],assignment=choices))
  np.savez_compressed(target,**{k:np.array(v) for k,v in out.items()},times=z['times']);save(O/(c['id']+'_raw.json'),rawrows);save(sidecar,dict(label=c['label'],frames=len(rawrows),sha256=sha(target),boxes=sum(len(r['boxes']) for r in rawrows),plain=int(np.sum(out['plain_mask'])),tracked=int(np.sum(out['tracked_mask'])),mean_ms=float(np.mean(out['ms']))))
  print(c['id'],c['label'],'boxes',sum(len(r['boxes']) for r in rawrows),'accepted',np.sum(out['plain_mask']),np.sum(out['tracked_mask']),flush=True)
 print('COMPLETE',flush=True)
if __name__=='__main__':extract() if len(sys.argv)>1 else prepare()
