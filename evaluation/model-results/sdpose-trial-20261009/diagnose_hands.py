"""Post-hoc diagnosis, not validation or a label-dependent routing system."""
import sys,json,hashlib,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
O=ROOT/'hand_diagnosis_20261011';O.mkdir(exist_ok=True)
assert not (O/'protocol.json').exists()
A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit')
B=A/'pixels30_20261007'
sys.path.insert(0,str(A))
import run_audit as audit
spec=importlib.util.spec_from_file_location('rtm_eval',ROOT.parent/'rtm-eval.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol={'status':'registered','purpose':'Post-hoc diagnosis of hello improvement and please regression; not held-out validation',
 'variants':{'RR':'both RTMW','SR':'anatomical left SDPose, right RTMW','RS':'left RTMW, right SDPose','SS':'both SDPose'},
 'fixed':'RTMW body, original timestamps, converter .30 confidence, missing zeros, same ST-GCN and shoulder normalization',
 'hashes':{str(p):sha(p) for p in [Path(__file__),A/'wlasl_stgcn.zip',ROOT.parent/'rtm-eval.py']},
 'reference_usage':'Geometry evaluation only; never fed to recognition', 'production_edited':False}
(O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
model,norm,vocab=audit.load_model()
refs={c['id']:{s['index']:s for s in c['samples']} for c in json.loads((B/'coordinate_reference.json').read_text())}
results={};geometry={};arrays={};logits={}
def stats(a):
 a=np.asarray(a)
 return {'n':int(a.size),'rmse_px':float(np.sqrt(np.mean(a*a))) if a.size else None,'pck10_pct':float(np.mean(a<=10)*100) if a.size else None}
def longest_missing(valid):
 best=run=0
 for x in valid:
  run=0 if x else run+1;best=max(best,run)
 return best
for cid,label,folder in [('001','hello','tracking3'),('005','please','confirmation3')]:
 rows=[r for r in json.loads((ROOT/folder/'results.json').read_text()) if r['id']==cid]
 cache=np.load(B/'rtm_trial_20261008'/f'{cid}.npz')
 assert np.allclose(cache['times'],[r['time'] for r in rows])
 kp={'S':np.array([r['keypoints'] for r in rows]),'R':cache['rtmw_det']}
 sc={'S':np.array([r['scores'] for r in rows]),'R':cache['rtmw_det_scores']}
 results[label]={};geometry[label]={};baseline_body=None
 for v in ['RR','SR','RS','SS']:
  k=kp['R'].copy();s=sc['R'].copy()
  for source,b in zip(v,[91,112]):k[:,b:b+21]=kp[source][:,b:b+21];s[:,b:b+21]=sc[source][:,b:b+21]
  a=ev.convert(k,s)
  t=norm({'frames':torch.from_numpy(a[:,audit.PICK].transpose(2,0,1).copy())})['frames']
  assert torch.isfinite(t).all()
  if baseline_body is None:baseline_body=t[:,:,:7].clone()
  assert torch.equal(t[:,:,:7],baseline_body)
  with torch.inference_mode():out=model(t.unsqueeze(0))[0];p=out.softmax(0)
  top=p.topk(5);target=vocab.index(label)
  results[label][v]={'prediction':vocab[int(top.indices[0])],'top5':[vocab[int(j)] for j in top.indices],
   'target_rank':torch.argsort(p,descending=True).tolist().index(target)+1,'target_probability':float(p[target])}
  arrays[f'{cid}_{v}']=t.numpy();logits[f'{cid}_{v}']=out.numpy()
  if v in ('RR','SS'):
   oldpath=(ROOT/'tracking3/rtmw_anchor_trial/stgcn_body_hybrid/logits.npz') if cid=='001' else ROOT/'confirmation3/stgcn/logits.npz'
   oldkey=('rtmw_all_control' if v=='RR' else 'rtmw_body_raw') if cid=='001' else ('rtmw' if v=='RR' else 'hybrid_raw')
   assert np.allclose(out.numpy(),np.load(oldpath)[f'{cid}_{oldkey}'],atol=1e-5,rtol=1e-5)
 for side,b in [('left',91),('right',112)]:
  valid={m:(s[:,b]>=.3)&(s[:,b:b+21].mean(1)>=.3) for m,s in sc.items()}
  common=valid['S']&valid['R']
  geometry[label][side]={}
  for m in ['R','S']:
   allerr=[];accepted=[];commonerr=[];shape=[];wrist=[];detail=[]
   for i,r in enumerate(rows):
    ref=refs[cid][r['sourceIndex']];assert ref['pngMatches'] and ref['pixelRatio']==1
    h=ref['hands'][side]
    if not all(str(j) in h for j in range(21)):continue
    target=np.array([h[str(j)] for j in range(21)])
    q=kp[m][i,b:b+21];e=np.linalg.norm(q-target,axis=1)
    allerr.extend(e);wrist.append(e[0])
    if valid[m][i]:accepted.extend(e)
    if common[i]:commonerr.extend(e);shape.extend(np.linalg.norm((q-q[0])-(target-target[0]),axis=1))
    detail.append({'frame':i,'accepted':bool(valid[m][i]),'mean_error_px':float(e.mean())})
   geometry[label][side][m]={'accepted_frames':int(valid[m].sum()),'total_frames':len(rows),'longest_rejected_run':longest_missing(valid[m]),
    'all_reference_points':stats(allerr),'accepted_reference_points':stats(accepted),'common_accepted_points':stats(commonerr),
    'common_wrist_aligned_shape':stats(shape),'all_reference_wrists':stats(wrist),'frames':detail}
np.savez_compressed(O/'inputs.npz',**arrays);np.savez_compressed(O/'logits.npz',**logits)
result={'recognition':results,'geometry':geometry,'old_baselines_reproduced':True,'normalized_body_identical':True}
(O/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
protocol['status']='completed';(O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
lines=['# ตรวจมือ hello / please','',
 'วิเคราะห์ย้อนหลังสองคำที่ให้ผลต่างกัน ไม่ใช่ชุดทดสอบใหม่ ไม่ใช้พิกัด Renderer หรือคำตอบจริงเลือก input ให้โมเดล ลำตัว RTMW และ normalization เท่ากันทุกวิธี ตรวจ logits baseline ซ้ำตรงกับรอบก่อน',
 'R = RTMW, S = SDPose; ตัวแรกคือมือซ้ายตามกายวิภาค ตัวที่สองคือมือขวา ไม่ใช่ซ้ายขวาของภาพ','',
 '| คำจริง | มือ | คำทาย | อันดับคำจริง | Probability |','|---|---|---|---:|---:|']
for label,vs in results.items():
 for v,r in vs.items():lines.append(f"| {label} | {v} | {r['prediction']} | {r['target_rank']} | {r['target_probability']:.6e} |")
lines+=['','| คำ | มือ | ตัวตรวจจับ | ผ่าน confidence / เฟรม | ไม่ผ่านต่อเนื่องสูงสุด | RMSE จุดอ้างอิงทั้งหมด px | RMSE เฟรมผ่านร่วม px | Shape RMSE หลังย้ายข้อมือ px |','|---|---|---|---:|---:|---:|---:|---:|']
for label,sides in geometry.items():
 for side,ms in sides.items():
  for m,g in ms.items():
   def fmt(x):return 'N/A' if x is None else f'{x:.2f}'
   lines.append(f"| {label} | {side} | {m} | {g['accepted_frames']}/{g['total_frames']} | {g['longest_rejected_run']} | {fmt(g['all_reference_points']['rmse_px'])} | {fmt(g['common_accepted_points']['rmse_px'])} | {fmt(g['common_wrist_aligned_shape']['rmse_px'])} |")
lines+=['','RMSE ใช้ระยะ Euclidean ต่อจุด บนภาพ 808x808; common ใช้เฉพาะเฟรมที่ทั้งสองโมเดลผ่าน confidence และมี reference ครบ 21 จุด เพื่อเทียบบนจุดชุดเดียวกัน',
 'Shape ย้ายข้อมือมาที่ศูนย์อย่างเดียว ไม่หมุนหรือปรับขนาด จึงแยกความคลาดเคลื่อนตำแหน่งทั้งมือออกบางส่วน ไม่ใช่ความถูกต้อง ASL',
 'ช่วง reference ไม่ครบไม่ถูกนับว่าเป็นมือพักหรือท่าผิด การไม่ผ่าน confidence ทำให้ converter เดิมใส่ศูนย์ แต่รอบนี้ไม่เปลี่ยนวิธีเติมข้อมูล']
(O/'REPORT_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(results),flush=True)
