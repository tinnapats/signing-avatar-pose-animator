"""Source-to-rendered 2D hand fidelity; not linguistic accuracy."""
import json, argparse, copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import export_pose_animator_sequence as p
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'fidelity_results'
def xy(k): return np.array([k['position']['x'],k['position']['y']])
def prepare():
 OUT.mkdir(exist_ok=True); cases=[]; db=ROOT/'SLclean.sqlite3'
 for phrase in ['abdomen','hello','love','test yes','hello please','thank you yes no']:
  paths=p.resolve_selected_paths(p.discover_clips(db),db,[],phrase)
  raw=p.build_frames(p.concat_clips(paths,513,513,0,target_fps=30),513,513,0)
  affine={}; original=p.fit_frames_to_canvas
  def capture(frames,width,height):
   before=np.array([xy(k) for f in frames for k in f['pose']['keypoints']]); result=original(frames,width,height)
   after=np.array([xy(k) for f in result for k in f['pose']['keypoints']])
   scale,dx=np.linalg.lstsq(np.column_stack([before[:,0],np.ones(len(before))]),after[:,0],rcond=None)[0]
   dy=np.mean(after[:,1]-before[:,1]*scale)
   assert np.max(np.abs(after-(before*scale+[dx,dy])))<1e-6
   affine.update(scale=float(scale),dx=float(dx),dy=float(dy)); return result
  with patch.object(p,'fit_frames_to_canvas',capture): payload=p.build_payload(db,[],phrase,30,513,513,0,0,2,1.2,2)
  assert len(payload['frames'])==(len(raw)-1)*2+1
  for f in raw:
   points=list(f['pose']['keypoints'])
   for h in f.get('hands',{}).values(): points.extend(h.get('keypoints',[]))
   for k in points:
    for axis in ['x','y']: k['position'][axis]=k['position'][axis]*affine['scale']+affine['d'+axis]
   f.pop('face',None)
  cases.append(dict(phrase=phrase,payload=payload,reference=raw,frameIndices=list(range(0,len(payload['frames']),2))))
  print(phrase,len(raw),flush=True)
 (OUT/'inputs.json').write_text(json.dumps(cases),encoding='utf-8')
def summarize():
 cases=json.loads((OUT/'inputs.json').read_text()); rendered=json.loads((OUT/'rendered.json').read_text()); rows=[]; pooled=[[],[],[]]
 def metrics(name,a,b,c):
  return dict(phrase=name,points=len(a),coverage=100*len(c)/len(a),position_pck5=100*float(np.mean(np.array(a)<=.05)),shape_pck10=100*float(np.mean(np.array(b)<=.1)),shape_pck20=100*float(np.mean(np.array(b)<=.2)),shape_pck30=100*float(np.mean(np.array(b)<=.3)),rmse_canvas_px=float(np.sqrt(np.mean(np.square(c)))))
 for case,result in zip(cases,rendered,strict=True):
  assert case['phrase']==result['phrase']; a,b,c=[],[],[]
  for ref,sample in zip(case['reference'],result['samples'],strict=True):
   pose={k['part']:k for k in ref['pose']['keypoints']}
   if min(pose[s]['score'] for s in ['leftShoulder','rightShoulder'])<=0: continue
   shoulder=np.linalg.norm(xy(pose['leftShoulder'])-xy(pose['rightShoulder']))
   if shoulder<=1e-6: continue
   for side,h in ref.get('hands',{}).items():
    if not h.get('observed'): continue
    keys=h['keypoints']; palm=np.linalg.norm(xy(keys[0])-xy(keys[9]))
    if palm<=1e-6: continue
    pred=sample['hands'][side]
    for i,k in enumerate(keys):
     if k.get('score',0)<=0: continue
     q=pred.get(str(i)); err=np.linalg.norm(np.array(q)-xy(k)) if q is not None else float('inf')
     a.append(err/shoulder)
     if q is not None: c.append(float(err*sample['zoom']))
     if i:
      err=np.linalg.norm((np.array(q)-np.array(pred['0']))-(xy(k)-xy(keys[0]))) if q is not None and '0' in pred else float('inf')
      b.append(err/palm)
  rows.append(metrics(case['phrase'],a,b,c))
  for pool,values in zip(pooled,[a,b,c]): pool.extend(values)
 rows.append(metrics('Pooled',*pooled))
 (OUT/'metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
 report=['# Rendered hand fidelity (six-sequence pilot)','', 'Measured from actual Paper.js hand-stroke endpoints. Camera transform removed; source uses the SAME global fit as output. No fitted per-hand alignment.','', '|Sequence|Points|Coverage %|Position PCK@5%|Shape @10%|Shape @20%|Shape @30%|RMSE px|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows: report.append('| '+r['phrase']+' | '+str(r['points'])+' | '+' | '.join(f'{r[k]:.2f}' for k in ['coverage','position_pck5','shape_pck10','shape_pck20','shape_pck30','rmse_canvas_px'])+' |')
 report+=['','PCK: percentage within a distance threshold. Position threshold = 5% of source shoulder width. Shape = wrist-relative coordinates, thresholds = 10/20/30% of source wrist-to-middle-MCP length. Missing rendered points fail PCK. RMSE is over available points, in 513x513 canvas pixels including zoom.','','Limitations: observed source hands only, resampled to 30 FPS BEFORE motion processing. Output uses upsample x2, Gaussian sigma 1.2/radius 2 and default repair/stabilization/articulation. Compare original timestamps only; inserted frames have no independent truth. Pooled result is point-weighted. This is 2D hand centreline fidelity, NOT whole-avatar/pixel/depth/linguistic accuracy. Six selected sequences, not the entire dataset. Source detection itself can contain errors. Renderer hand sizing and wrist anchoring count as deviations; shape removes translation only.']
 (OUT/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8'); print(json.dumps(rows,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','summarize']); args=parser.parse_args()
 prepare() if args.stage=='prepare' else summarize()