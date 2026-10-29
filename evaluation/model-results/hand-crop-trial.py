import sys,json,time,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('ah',Path(__file__).with_name('anime-hands.py')); ah=importlib.util.module_from_spec(spec);spec.loader.exec_module(ah)
np=ah.np;B=ah.B;R=ah.R;A=ah.O;O=Path(__file__).parent/'hand-crop-trial-20261009';O.mkdir(exist_ok=True)
read=ah.read;save=ah.save;sha=ah.sha
FACTORS={'tight_080':.8,'wide_120':1.2}
def extract():
 info=read(R/'hand_model.json');assert sha(Path(info['path']))==info['sha256']
 protocol=dict(factors=FACTORS,effective_padding={'tight_080':1.,'baseline':1.25,'wide_120':1.5},model=info,manifest_sha256=sha(B/'capture_manifest.json'),rules='Freeze before inference/scoring. Same cached anime detector boxes, centered scale only; no bbox clipping. Hand model256x256 defaultpadding1.25. Confidence hand/wrist .30 body .30 distance .65forearm and stateless assignment unchanged. Same frames/times, previous_best fallback. Reference never used for crop/input/assignment. Reused30 development clips.',production=read(A/'protocol.json')['production'])
 if (O/'protocol.json').exists():assert read(O/'protocol.json')==protocol
 else:save(O/'protocol.json',protocol)
 cv2,rt,det=ah.base.setup();hand=rt(info['path'],model_input_size=(256,256),backend='onnxruntime',device='cpu')
 for c in read(B/'capture_manifest.json'):
  target=O/(c['id']+'.npz')
  if target.exists():assert sha(target)==read(target.with_suffix('.json'))['sha256'];continue
  raw=read(A/(c['id']+'_raw.json'));z=np.load(R/(c['id']+'.npz'));out={};details={}
  for name in FACTORS:out[name]=[];out[name+'_mask']=[];out[name+'_ms']=[];details[name]=[]
  for i,f in enumerate(c['frames']):
   path=B/'frames'/c['id']/f['file'];assert sha(path)==f['sha256'];im=ah.base.image(cv2,path)
   # Preflight confirms cached baseline is reproducible with this local ONNX runtime.
   if i==0 and c['id']=='000':
    for j,b in enumerate(raw[i]['boxes']):
     pp,ss=hand(im,bboxes=[b]);assert np.allclose(pp[0],raw[i]['points'][j],atol=.01);assert np.allclose(ss[0],raw[i]['scores'][j],atol=1e-5)
   for name,factor in FACTORS.items():
    start=time.perf_counter();candidates=[];boxes=[]
    for b in raw[i]['boxes']:
     b=np.asarray(b,dtype=float);mid=(b[:2]+b[2:])/2;half=(b[2:]-b[:2])*factor/2;box=np.r_[mid-half,mid+half];pp,ss=hand(im,bboxes=[box]);boxes.append(box.tolist());candidates.append((pp[0],ss[0]))
    choice=ah.assign(candidates,z['rtmw_det'][i],z['rtmw_det_scores'][i],{},i,False);p=np.zeros((2,21,2),np.float32);mask=np.zeros(2,bool)
    for side,j in enumerate(choice):
     if j>=0:p[side]=candidates[j][0];mask[side]=True
    out[name].append(p);out[name+'_mask'].append(mask);out[name+'_ms'].append((time.perf_counter()-start)*1000)
    details[name].append(dict(boxes=boxes,points=[p.tolist() for p,s in candidates],scores=[s.tolist() for p,s in candidates],assignment=choice))
  np.savez_compressed(target,**{k:np.array(v) for k,v in out.items()},times=z['times']);save(O/(c['id']+'_raw.json'),details);save(target.with_suffix('.json'),dict(sha256=sha(target),raw_sha256=sha(O/(c['id']+'_raw.json')),source_raw_sha256=sha(A/(c['id']+'_raw.json')),label=c['label']))
  print('EXTRACTED',c['id'],c['label'],flush=True)
 print('EXTRACTION COMPLETE',flush=True)
def score():
 spec=importlib.util.spec_from_file_location('ae',Path(__file__).with_name('anime-eval.py'));ae=importlib.util.module_from_spec(spec);spec.loader.exec_module(ae);e=ae.e
 def data(c):
  a=ae.data(c);z=np.load(O/(c['id']+'.npz'));assert sha(O/(c['id']+'.npz'))==read(O/(c['id']+'.json'))['sha256'];assert np.allclose(z['times'],[f['time'] for f in c['frames']]);out=dict(mediapipe=a['mediapipe'],previous_best=a['previous_best'],baseline=a['anime_replace_best'])
  for name in FACTORS:
   p=a['previous_best'].copy()
   for side,off in enumerate([33,54]):
    valid=z[name+'_mask'][:,side];p[valid,off:off+21]=z[name][valid,side]
   out[name]=p
  return out
 e.data=data;e.VARIANTS=['mediapipe','previous_best','baseline']+list(FACTORS);e.RESULT_NAME=str(O/'results.json');e.evaluate()
 prior=read(A/'results.json');new=read(O/'results.json')
 for a,b in zip(prior['predictions']['anime_replace_best'],new['predictions']['baseline']):assert a['prediction']==b['prediction'] and a['top5']==b['top5'] and np.isclose(a['target_probability'],b['target_probability'],atol=1e-6)
 print('BASELINE VERIFIED',flush=True)
if __name__=='__main__':score() if len(sys.argv)>1 and sys.argv[1]=='score' else extract()
