"""Paired geometry audit. Reference alignments are diagnostics, never model inputs."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007')
O=ROOT/'wrist_shape_20261011';O.mkdir(exist_ok=True)
assert not (O/'protocol.json').exists()
chains=[(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)]
triples=[c[i:i+3] for c in chains for i in range(3)]
def measure(q,g):
 x=(q-q[0])[1:];y=(g-g[0])[1:]
 u,sv,vt=np.linalg.svd(x.T@y)
 d=np.eye(2);d[-1,-1]=1 if np.linalg.det(u@vt)>=0 else -1
 rot=u@d@vt;den=np.sum(x*x)
 scale=max(0.,float(np.sum((x@rot)*y)/den)) if den>1e-12 else 0.
 return {'wrist_sq':float(np.sum((q[0]-g[0])**2)),
  'total_sq':float(np.mean(np.sum((q-g)**2,axis=1))),
  'wrist_aligned_sq':float(np.mean(np.sum((x-y)**2,axis=1))),
  'similarity_residual_sq':float(np.mean(np.sum((scale*x@rot-y)**2,axis=1))),
  'fitted_scale':scale}
def angles(q):
 out=[]
 for a,b,c in triples:
  u=q[a]-q[b];v=q[c]-q[b];den=np.linalg.norm(u)*np.linalg.norm(v)
  out.append(float(np.degrees(np.arccos(np.clip(np.dot(u,v)/den,-1,1)))) if den>1e-8 else None)
 return out
# Geometry invariance checks; no data-dependent calibration.
rng=np.random.default_rng(5);g=rng.normal(size=(21,2));g[0]=0
m=measure(g+[8,6],g);assert abs(m['wrist_sq']-100)<1e-8 and m['wrist_aligned_sq']<1e-20
rot=np.array([[0.,-1.],[1.,0.]])
m=measure(2*g@rot+[8,6],g);assert m['similarity_residual_sq']<1e-20 and m['wrist_aligned_sq']>0
q=g.copy();q[8]+=[3,4];assert measure(q,g)['similarity_residual_sq']>0
sources=[ROOT/'tracking3/results.json',ROOT/'confirmation3/results.json']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol={'status':'registered','selection':'All six existing SDPose clips, no exclusions by word or score',
 'primary_subset':'Per-hand frames with full 21-point renderer reference AND both detectors pass original wrist+mean confidence .30',
 'secondary_subset':'All full-reference hand-frames including low confidence, reported separately',
 'metrics':'Wrist RMSE; 21-point total RMSE; 20 non-wrist points after wrist translation; optional optimal 2D rotation+positive uniform scale residual, no reflection; 15 projected joint-angle MAE on identical valid triples',
 'no_changes':'No threshold search, no recognition rerun, no production edits, no reference coordinates fed to model',
 'hashes':{str(p):sha(p) for p in [Path(__file__),*sources,B/'coordinate_reference.json']}}
(O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
rows=sum([_local_json_loads(p.read_text()) for p in sources],[])
refs={c['id']:{f['index']:f for f in c['samples']} for c in _local_json_loads((B/'coordinate_reference.json').read_text())}
cache={cid:np.load(B/'rtm_trial_20261008'/f'{cid}.npz') for cid in {r['id'] for r in rows}}
samples=[]
for r in rows:
 c=cache[r['id']];i=r['frame'];assert abs(c['times'][i]-r['time'])<1e-6
 ref=refs[r['id']][r['sourceIndex']];assert ref['pngMatches'] and ref['pixelRatio']==1
 for side,b in [('left',91),('right',112)]:
  h=ref['hands'][side]
  if not all(str(j) in h for j in range(21)):continue
  g=np.array([h[str(j)] for j in range(21)])
  qq={'SDPose':np.array(r['keypoints'])[b:b+21],'RTMW':c['rtmw_det'][i,b:b+21]}
  ss={'SDPose':np.array(r['scores'])[b:b+21],'RTMW':c['rtmw_det_scores'][i,b:b+21]}
  valid={m:bool(s[0]>=.3 and s.mean()>=.3) for m,s in ss.items()}
  aa={m:angles(q) for m,q in qq.items()};ga=angles(g)
  common_angles=[j for j in range(15) if ga[j] is not None and all(aa[m][j] is not None for m in qq)]
  for model,q in qq.items():
   samples.append({'label':r['label'],'id':r['id'],'frame':i,'side':side,'model':model,
    'common_accepted':all(valid.values()),'accepted':valid[model],**measure(q,g),
    'angle_errors':[abs(aa[model][j]-ga[j]) for j in common_angles]})
def aggregate(ss):
 out={'hand_frames':len(ss)}
 for key in ['wrist','total','wrist_aligned','similarity_residual']:
  out[key+'_rmse_px']=float(np.sqrt(np.mean([s[key+'_sq'] for s in ss]))) if ss else None
 ang=sum([s['angle_errors'] for s in ss],[])
 out['joint_angle_mae_deg']=float(np.mean(ang)) if ang else None;out['angle_samples']=len(ang)
 return out
labels=list(dict.fromkeys(r['label'] for r in rows));summary={}
for label in labels:
 summary[label]={subset:{m:aggregate([s for s in samples if s['label']==label and s['model']==m and (subset=='all_reference' or s['common_accepted'])]) for m in ['SDPose','RTMW']} for subset in ['common_accepted','all_reference']}
result={'summary':summary,'samples':samples,'frame_count':len(rows),'synthetic_tests_passed':True}
(O/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
lines=['# แยกตำแหน่งข้อมือกับรูปมือ','',
 'ตรวจ 6 คลิป 422 เฟรม ใช้ Renderer reference สำหรับตรวจความตรงของพิกัดเท่านั้น ไม่ใช่ ground truth ความถูกต้องทาง ASL',
 'ตารางหลักใช้ hand-frame ที่ทั้งสองโมเดลผ่าน confidence 0.30 และมี reference ครบ 21 จุดชุดเดียวกัน ไม่กรองตาม confidence ของ RTMW body และไม่จำเป็นต้องเห็นมืออีกข้าง จึงมีตัวหารต่างจากรอบตรวจมือซ้ำ',
 '1. Wrist RMSE: ระยะผิดของข้อมือ',
 '2. Wrist-aligned: ย้ายข้อมือให้ตรงกันแล้ววัดอีก 20 จุด ยังรวมการหันมือ ขนาด และรูปร่างนิ้ว ไม่ใช่ข้อผิดพลาดของนิ้วล้วน',
 '3. Similarity residual: ยอมให้หมุนในระนาบและปรับขนาดรวมอย่างเหมาะสม โดยไม่สะท้อนภาพ แล้ววัด 20 จุดที่เหลือ ใช้เพื่อวิเคราะห์เท่านั้น ไม่แก้ input จริง',
 '4. Angle MAE: ความต่างมุมข้อ 15 มุมในภาพ 2D ไม่ใช่มุมข้อ 3D; มือหันขอบ/การบัง/สลับตัวตนทำให้ค่าเปลี่ยนได้','',
 '| คำ | โมเดล | hand-frames | Wrist px | Wrist-aligned px | Similarity residual px | Angle MAE deg |',
 '|---|---|---:|---:|---:|---:|---:|']
for l in labels:
 for m,g in summary[l]['common_accepted'].items():
  lines.append(f"| {l} | {m} | {g['hand_frames']} | {g['wrist_rmse_px']:.2f} | {g['wrist_aligned_rmse_px']:.2f} | {g['similarity_residual_rmse_px']:.2f} | {g['joint_angle_mae_deg']:.2f} |")
lines+=['','ค่า Wrist และ Wrist-aligned ไม่ใช่องค์ประกอบอิสระที่นำมาบวกเป็น Total RMSE ได้ เพราะมี cross term และใช้จำนวนจุดต่างกัน',
 'ข้อมูลอ้างอิงไม่มีมือบางข้าง/บางเฟรม จึงไม่อนุมานว่าโมเดลผิดหรือมือพักจากการไม่มี reference เก็บผลทุก reference รวม low confidence แยกใน results.json',
 'การจัดแนวใช้ reference แบบ oracle เพื่อวินิจฉัยเท่านั้น ไม่ใช่วิธีปรับปรุงคะแนนใช้งานจริง ไม่สร้างเกณฑ์ผ่าน/ตกจากตัวเลขในชุดนี้']
(O/'REPORT_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
protocol['status']='completed';(O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
print(json.dumps({l:s['common_accepted'] for l,s in summary.items()}),flush=True)
