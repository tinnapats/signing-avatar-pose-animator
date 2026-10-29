"""Isolated 2D adaptation of production functions; frozen 2x4 experiment."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import sys,json,hashlib,importlib.util,copy,time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit');B=A/'pixels30_20261007'
O=ROOT/'production_hand_ablation_20261011';O.mkdir(exist_ok=True)
assert not (O/'protocol.json').exists()
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
sys.path.insert(0,str(A.parent));sys.path.insert(0,str(A))
import run_audit as audit
prod=load('hand_production_audit',A.parent/'export_pose_animator_sequence.py')
ev=load('rtm_eval',ROOT.parent/'rtm-eval.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol={'status':'registered','clips':'All six SDPose clips, no per-word selection','sources':['RTMW','SDPose'],
 'variants':['raw','bone','smooth','smooth_bone'],'body':'RTMW fixed, normalized body asserted equal',
 'bone':'Actual production stabilize_hand_bone_lengths defaults radius=3, min=.72,max=1.28,iterations=3; x/y only, no invented depth',
 'smoothing':'Actual production apply_hand_gaussian_smoothing sigma=1.2,radius=2,preserve_flip_steps=False',
 'adaptations':'Each hand separately, only contiguous runs passing original mean+wrist confidence .30. Original individual scores supplied; original scores restored after smoothing. Rejected frames unchanged, no gap fill. No new constraints on 2D joint angles.',
 'combined_order':'Gaussian then bone, as production order','time':'Original timestamps, no resampling',
 'references':'Only postprocessing evaluation; never fed into filters or model',
 'limitations':'2D projected bone lengths legitimately vary. Pilot on developmental clips, not independent validation. Offline symmetric filters use future frames.',
 'hashes':{str(p):sha(p) for p in [Path(__file__),A.parent/'export_pose_animator_sequence.py',ROOT.parent/'rtm-eval.py',A/'wlasl_stgcn.zip']}}
(O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
rows=sum([_local_json_loads((ROOT/f/'results.json').read_text()) for f in ['tracking3','confirmation3']],[])
def process(k,s,mode):
 out=k.copy()
 if mode=='raw':return out
 for side,b in [('left',91),('right',112)]:
  valid=(s[:,b]>=.3)&(s[:,b:b+21].mean(1)>=.3)
  cuts=np.flatnonzero(np.diff(np.r_[False,valid,False])).reshape(-1,2)
  for start,end in cuts:
   frames=[]
   for i in range(start,end):
    hands={}
    for name in ['left','right']:
     hands[name]={'keypoints':[{'position':{'x':float(k[i,b+j,0]) if name==side else 0.,'y':float(k[i,b+j,1]) if name==side else 0.},'score':float(s[i,b+j]) if name==side else 0.} for j in range(21)]}
    frames.append({'hands':hands})
   if mode in ['smooth','smooth_bone']:prod.apply_hand_gaussian_smoothing(frames,sigma=1.2,radius=2,preserve_flip_steps=False)
   # Preserve original confidence for bone candidate selection too.
   for n,f in enumerate(frames):
    for j,kp in enumerate(f['hands'][side]['keypoints']):kp['score']=float(s[start+n,b+j])
   if mode in ['bone','smooth_bone']:prod.stabilize_hand_bone_lengths(frames)
   for n,f in enumerate(frames):out[start+n,b:b+21]=[[p['position']['x'],p['position']['y']] for p in f['hands'][side]['keypoints']]
  assert np.array_equal(out[~valid,b:b+21],k[~valid,b:b+21])
 assert np.array_equal(out[:,:91],k[:,:91]) and np.isfinite(out).all()
 if mode=='bone':assert np.array_equal(out[:,[91,112]],k[:,[91,112]])
 return out
model,norm,vocab=audit.load_model();pred={};inputs={};logits={};processed={};metadata={};timing={}
for cid in sorted({r['id'] for r in rows}):
 rr=[r for r in rows if r['id']==cid];label=rr[0]['label'];c=np.load(B/'rtm_trial_20261008'/f'{cid}.npz')
 assert np.allclose(c['times'],[r['time'] for r in rr])
 fixed=None
 for source in ['RTMW','SDPose']:
  k=c['rtmw_det'].copy();s=c['rtmw_det_scores'].copy()
  if source=='SDPose':k[:,91:]=np.array([r['keypoints'] for r in rr])[:,91:];s[:,91:]=np.array([r['scores'] for r in rr])[:,91:]
  for mode in ['raw','bone','smooth','smooth_bone']:
   v=source+'_'+mode;start=time.perf_counter();q=process(k,s,mode);timing[cid+'_'+v]=1000*(time.perf_counter()-start)
   a=ev.convert(q,s);rawa=ev.convert(k,s)
   for b in [33,54]:assert np.array_equal(np.any(a[:,b:b+21]!=0,axis=(1,2)),np.any(rawa[:,b:b+21]!=0,axis=(1,2)))
   t=norm({'frames':torch.from_numpy(a[:,audit.PICK].transpose(2,0,1).copy())})['frames']
   if fixed is None:fixed=t[:,:,:7].clone()
   assert torch.equal(fixed,t[:,:,:7]) and torch.isfinite(t).all()
   with torch.inference_mode():o=model(t.unsqueeze(0))[0];p=o.softmax(0)
   top=p.topk(5);target=vocab.index(label)
   res={'label':label,'prediction':vocab[int(top.indices[0])],'top5':[vocab[int(j)] for j in top.indices],
    'target_probability':float(p[target]),'target_rank':torch.argsort(p,descending=True).tolist().index(target)+1}
   if mode=='raw':
    if cid in ['000','001','002']:
     old=np.load(ROOT/'tracking3/rtmw_anchor_trial/stgcn_body_hybrid/logits.npz');key='rtmw_all_control' if source=='RTMW' else 'rtmw_body_raw'
    else:old=np.load(ROOT/'confirmation3/stgcn/logits.npz');key='rtmw' if source=='RTMW' else 'hybrid_raw'
    assert np.allclose(o.numpy(),old[cid+'_'+key],atol=1e-5,rtol=1e-5)
   pred.setdefault(v,[]).append(res);inputs[cid+'_'+v]=t.numpy();logits[cid+'_'+v]=o.numpy();processed[cid+'_'+v]=q
   metadata[cid+'_'+v]={'scores':s,'rows':rr}
   print(label,v,res['prediction'],res['target_rank'],flush=True)
np.savez_compressed(O/'processed.npz',**processed);np.savez_compressed(O/'inputs.npz',**inputs);np.savez_compressed(O/'logits.npz',**logits)
# Only now read references, after every model prediction and processed sequence is fixed.
refs={c['id']:{f['index']:f for f in c['samples']} for c in _local_json_loads((B/'coordinate_reference.json').read_text())}
geometry={};details=[]
for key,q in processed.items():
 cid,v=key.split('_',1);s=metadata[key]['scores'];rr=metadata[key]['rows'];source=v.split('_')[0]
 for i,r in enumerate(rr):
  ref=refs[cid][r['sourceIndex']];assert ref['pngMatches'] and ref['pixelRatio']==1
  for side,b in [('left',91),('right',112)]:
   h=ref['hands'][side]
   if not all(str(j) in h for j in range(21)):continue
   # Same points for every variant AND both detectors.
   scores=[metadata[cid+'_'+src+'_raw']['scores'][i,b:b+21] for src in ['RTMW','SDPose']]
   if not all(ss[0]>=.3 and ss.mean()>=.3 for ss in scores):continue
   g=np.array([h[str(j)] for j in range(21)]);hand=q[i,b:b+21]
   chain=[(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)]
   angles=[]
   for cc in chain:
    for j in range(3):
     a,bb,c=cc[j:j+3];vals=[]
     # Valid triples fixed by reference and both RAW detectors, not output.
     originals=[g]+[processed[cid+'_'+src+'_raw'][i,b:b+21] for src in ['RTMW','SDPose']]
     if any(np.linalg.norm(x[a]-x[bb])*np.linalg.norm(x[c]-x[bb])<=1e-8 for x in originals):continue
     for x in [hand,g]:
      u=x[a]-x[bb];w=x[c]-x[bb];den=np.linalg.norm(u)*np.linalg.norm(w)
      vals.append(float(np.degrees(np.arccos(np.clip(np.dot(u,w)/den,-1,1)))) if den>1e-8 else None)
     if None not in vals:angles.append(abs(vals[0]-vals[1]))
   details.append({'variant':v,'label':r['label'],'frame':i,'side':side,
    'total_sq':float(np.mean(np.sum((hand-g)**2,axis=1))),
    'wrist_sq':float(np.sum((hand[0]-g[0])**2)),
    'shape_sq':float(np.mean(np.sum(((hand-hand[0])[1:]-(g-g[0])[1:])**2,axis=1))), 'angles':angles})
for v in pred:
 dd=[d for d in details if d['variant']==v];angs=sum([d['angles'] for d in dd],[])
 geometry[v]={'hand_frames':len(dd),**{n+'_rmse_px':float(np.sqrt(np.mean([d[n+'_sq'] for d in dd]))) for n in ['total','wrist','shape']},'angle_mae_deg':float(np.mean(angs)),'angle_samples':len(angs)}
result={'summary':{v:ev.metrics(r) for v,r in pred.items()},'predictions':pred,'geometry':geometry,'geometry_details':details,'preprocess_ms':timing,'baseline_reproduced':True}
(O/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
assert sha(A.parent/'export_pose_animator_sequence.py')==protocol['hashes'][str(A.parent/'export_pose_animator_sequence.py')]
protocol['status']='completed';(O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
lines=['# ทดลองฟังก์ชันรักษามือจากงานหลักกับจุดจากภาพ','',
 'ใช้ฟังก์ชัน production จริง แต่ดัดแปลงการเรียกใช้กับ 2D โดยแบ่งช่วงตรวจพบมือต่อเนื่อง รักษา confidence เดิม ไม่มีเติมข้อมูล ไม่มีปรับลำตัวหรือเวลา ไม่ใส่ความลึกปลอม',
 'Gaussian sigma 1.2 radius 2; bone radius 3 ช่วง 0.72–1.28 เท่าของ median ท้องถิ่น 3 iterations; combined ทำ smoothing ก่อน bone',
 '6 คลิปพัฒนา: abdomen hello love test yes please ไม่ใช่ชุดทดสอบอิสระ ไม่เลือกวิธีรายคำด้วย label จริง ตรวจ logits baseline ซ้ำตรงกับรอบก่อน และ checksum production ไม่เปลี่ยน','',
 '| วิธี | ถูก/6 | Macro-F1 % | Top5 % | Total RMSE px | Wrist RMSE px | Wrist-aligned RMSE px | Angle MAE deg |','|---|---:|---:|---:|---:|---:|---:|---:|']
for v,m in result['summary'].items():
 g=geometry[v];lines.append(f"| {v} | {m['correct']}/6 | {m['f1']:.2f} | {m['top5']:.2f} | {g['total_rmse_px']:.2f} | {g['wrist_rmse_px']:.2f} | {g['shape_rmse_px']:.2f} | {g['angle_mae_deg']:.2f} |")
lines+=['','Macro เฉลี่ยเฉพาะ 6 true labels, zero division 0; geometry เฉพาะ hand-frame ที่ทั้งสอง detector ผ่านเกณฑ์และ reference ครบ 21 จุด ใช้ชุดเดียวกันทุกวิธี ไม่ใช่ ASL correctness',
 'Wrist-aligned ยังรวมทิศทาง/ขนาดมือ Angle เป็นมุมฉาย 2D ไม่ใช่ 3D ไม่ควรบังคับความยาวคงที่ในภาพ 2D และ smoothing ไม่แก้การจับมือผิดตำแหน่ง','',
 '| คำจริง | วิธี | คำทาย | อันดับคำจริง |','|---|---|---|---:|']
for v,rs in pred.items():
 for r in rs:lines.append(f"| {r['label']} | {v} | {r['prediction']} | {r['target_rank']} |")
(O/'REPORT_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(result['summary']),flush=True);print(json.dumps(geometry),flush=True)
