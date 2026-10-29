import sys,importlib.util,json
from pathlib import Path
sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('ae',Path(__file__).with_name('anime-eval.py'));ae=importlib.util.module_from_spec(s);s.loader.exec_module(ae)
e=ae.e;np=e.np;O=Path(__file__).parent/'hand-quality-trial-20261009';O.mkdir(exist_ok=True)
NAMES=['mediapipe','previous_best','baseline','anchor_guard','shape_guard','combined_guard','mp_body_same_hands']
EDGES=[(0,1),(1,2),(2,3),(3,4)]+[(0,k) for k in [5,9,13,17]]+[(k+j,k+j+1) for k in [5,9,13,17] for j in range(3)]
protocol=dict(variants=NAMES,scope='Development30 only; do not open confirmation results or select thresholds from them.',rules={'base':'Original bbox, confidence .30, anime accepted else MP/RTMW fallback. No source renderer points supplied. Same time/recognizer.', 'anchor_guard':'Reject anime only when fallback exists and valid RTMW wrist/elbow; candidate wrist distance to body exceeds fallback distance by >.15 forearm lengths.', 'shape_guard':'Compare current candidate/fallback to previous selected hand only on adjacent frame. Score=max(median absolute log bone-length ratio /log2, median wrist-relative displacement/(.75*previous palm scale)). Bone length floor2px, palm scale median wrist-MCP distance floor10px. Reject only if candidate score>1 AND >fallback score+.25. No coordinate editing/hold/interpolation. 2D foreshortening can trigger false rejection.', 'combined_guard':'Union of anchor and shape rejections, with its own selected history.', 'mp_body_same_hands':'Diagnostic: baseline hands unchanged; body block uses MP instead of RTMW. Not a hand repair.'},selection='Frozen before scores; no sweep; no per-word rules.',production=e.read(ae.O/'protocol.json')['production'])
if (O/'protocol.json').exists():assert e.read(O/'protocol.json')==protocol
else:e.save(O/'protocol.json',protocol)
def quality(q,prev):
 lengths=lambda p:np.array([max(float(np.linalg.norm(p[b]-p[a])),2.) for a,b in EDGES])
 bone=float(np.median(abs(np.log(lengths(q)/lengths(prev))))/np.log(2))
 scale=max(float(np.median(np.linalg.norm(prev[[5,9,13,17]]-prev[0],axis=1))),10.)
 displacement=float(np.median(np.linalg.norm((q-q[0])-(prev-prev[0]),axis=1))/(.75*scale))
 return max(bone,displacement)
def data(c):
 a=ae.data(c);base=a['anime_replace_best'];fallback=a['previous_best'];z=np.load(ae.O/(c['id']+'.npz'));body=np.load(ae.R/(c['id']+'.npz'));out=dict(mediapipe=a['mediapipe'],previous_best=fallback,baseline=base);logs=[]
 for name in ['anchor_guard','shape_guard','combined_guard']:
  result=base.copy()
  for i in range(len(base)):
   for side,off in enumerate([33,54]):
    if not z['plain_mask'][i,side] or not np.any(fallback[i,off:off+21]):continue
    candidate=base[i,off:off+21];backup=fallback[i,off:off+21];w,elbow=[(9,7),(10,8)][side];bp=body['rtmw_det'][i];bs=body['rtmw_det_scores'][i];anchor=False;shape=False;ratio_c=ratio_f=None;qc=qf=None
    if bs[w]>=.3 and bs[elbow]>=.3:
     length=max(float(np.linalg.norm(bp[w]-bp[elbow])),10);ratio_c=float(np.linalg.norm(candidate[0]-bp[w])/length);ratio_f=float(np.linalg.norm(backup[0]-bp[w])/length);anchor=ratio_c>ratio_f+.15
    if i>0 and np.any(result[i-1,off:off+21]):
     qc=quality(candidate,result[i-1,off:off+21]);qf=quality(backup,result[i-1,off:off+21]);shape=qc>1 and qc>qf+.25
    reject=(anchor if name=='anchor_guard' else shape if name=='shape_guard' else anchor or shape)
    if reject:result[i,off:off+21]=backup
    logs.append(dict(variant=name,frame=i,side=side,reject=bool(reject),anchor=bool(anchor),shape=bool(shape),candidate_wrist_ratio=ratio_c,fallback_wrist_ratio=ratio_f,candidate_shape=qc,fallback_shape=qf))
  out[name]=result
 out['mp_body_same_hands']=base.copy();out['mp_body_same_hands'][:,:33]=a['mediapipe'][:,:33]
 folder=O/'inputs';folder.mkdir(exist_ok=True);np.savez_compressed(folder/(c['id']+'.npz'),**out,times=z['times']);e.save(folder/(c['id']+'.json'),dict(sha256=e.sha(folder/(c['id']+'.npz')),rules=logs))
 return out
if __name__=='__main__':
 e.data=data;e.VARIANTS=NAMES;e.RESULT_NAME=str(O/'results.json');e.evaluate()
 old=e.read(ae.O/'results.json');new=e.read(O/'results.json')
 for x,y in zip(new['predictions']['baseline'],old['predictions']['anime_replace_best']):assert x['prediction']==y['prediction'] and x['top5']==y['top5'] and np.isclose(x['target_probability'],y['target_probability'],atol=1e-6)
 print('BASELINE VERIFIED')
