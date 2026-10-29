import sys, importlib.util, itertools
from pathlib import Path
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('ae',Path(__file__).with_name('anime-eval.py')); ae=importlib.util.module_from_spec(spec);spec.loader.exec_module(ae)
e=ae.e; np=e.np; O=Path(__file__).parent/'hand-failure-audit-20261009'
NAMES=['mediapipe','previous_best','confidence_030','confidence_020','confidence_040']
accepted={}; changes={}
def assign(points,scores,body,bs,threshold):
 n=len(points); cost=np.full((2,n),np.inf)
 for side,(w,elbow) in enumerate([(9,7),(10,8)]):
  if bs[w]<.3 or bs[elbow]<.3:continue
  length=max(float(np.linalg.norm(body[w]-body[elbow])),10)
  for j,(p,s) in enumerate(zip(points,scores)):
   distance=float(np.linalg.norm(p[0]-body[w]))/length
   if s.mean()>=threshold and s[0]>=threshold and distance<=.65:cost[side,j]=distance
 best=(-1,-1); bestcost=2.
 for l,r in itertools.product([-1]+list(range(n)),repeat=2):
  if l>=0 and l==r:continue
  value=(1 if l<0 else cost[0,l])+(1 if r<0 else cost[1,r])
  if value<bestcost:bestcost=value;best=(l,r)
 return best
def data(c):
 old=ae.data(c); out={'mediapipe':old['mediapipe'],'previous_best':old['previous_best']}
 raw=e.read(ae.O/(c['id']+'_raw.json')); z=np.load(e.O/(c['id']+'.npz'))
 for name,t in [('confidence_030',.3),('confidence_020',.2),('confidence_040',.4)]:
  a=old['previous_best'].copy(); choices=[]
  for i,rr in enumerate(raw):
   p=np.asarray(rr['points']); s=np.asarray(rr['scores']); choice=assign(p,s,z['rtmw_det'][i],z['rtmw_det_scores'][i],t);choices.append(choice)
   if t==.3:assert list(choice)==rr['assignment']['plain']
   for side,off in enumerate([33,54]):
    if choice[side]>=0:a[i,off:off+21]=p[choice[side]]
  choices=np.array(choices);accepted.setdefault(name,0);accepted[name]+=int((choices>=0).sum());out[name]=a
  if t==.3:assert np.array_equal(a,old['anime_replace_best'])
 outpath=O/'model_inputs';outpath.mkdir(exist_ok=True)
 np.savez_compressed(outpath/(c['id']+'.npz'),**out,times=z['times'])
 return out
e.save(O/'confidence_protocol.json',dict(variants=NAMES,thresholds=[.3,.2,.4],rules='Frozen before model scoring. Change hand mean AND wrist confidence together only. Detector .395, body confidence .3, wrist distance .65, global one-to-one stateless assignment, fallback previous_best unchanged. Cached image-only points; no renderer reference in input. Original30FPS/time and STGCN fixed. Thirty development clips reused; not held-out.'))
e.data=data;e.VARIANTS=NAMES;e.RESULT_NAME=str(O/'confidence_results.json');e.evaluate()
e.save(O/'accepted_counts.json',accepted)
r=e.read(O/'confidence_results.json'); prior=e.read(ae.O/'results.json')
for a,b in zip(r['predictions']['confidence_030'],prior['predictions']['anime_replace_best']):
 assert a['prediction']==b['prediction'] and a['top5']==b['top5'] and np.isclose(a['target_probability'],b['target_probability'],atol=1e-6)
print('BASELINE REPRODUCED; accepted counts',accepted,flush=True)
