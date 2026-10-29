"""Audit actual selected hand points; reference never enters model input."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import sys,json,hashlib,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;T=ROOT/'wrist_crop_tight20_20261011';O=ROOT/'final_hand_geometry_20261011';O.mkdir(exist_ok=True)
assert not (O/'protocol.json').exists()
A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit');B=A/'pixels30_20261007';R=B/'rtm_trial_20261008'
read=lambda p:_local_json_loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf-8')
protocol={'status':'registered','purpose':'Actual tight_fallback hand geometry vs RTMW baseline, no new inference or correction',
 'reference':'Evaluation only. Missing reference not classified as rest or absence. Missing detector points excluded from geometry and counted.',
 'metrics':'Total21point RMS, wrist distance, wrist-aligned20point RMS, proper rotation+uniform positive scale residual20points (diagnostic oracle only).',
 'source_hashes':{str(p):sha(p) for p in [Path(__file__),T/'results.json',T/'model_inputs.npz',B/'coordinate_reference.json']}}
save(O/'protocol.json',protocol)
sys.path.insert(0,str(A));import run_audit as audit
spec=importlib.util.spec_from_file_location('ev',ROOT.parent/'rtm-eval.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
_,norm,_=audit.load_model() # Existing normalization, verify saved inputs exactly; no model prediction.
manifest=read(B/'capture_manifest.json');saved=np.load(T/'model_inputs.npz')
arrays={};cache={}
for c in manifest:
 cid=c['id'];z=np.load(R/f'{cid}.npz');h=np.load(T/f'{cid}.npz');cache[cid]=(z,h)
 for folder in [R,T]:assert sha(folder/f'{cid}.npz')==read(folder/f'{cid}.json')['sha256']
 a=ev.convert(z['rtmw_det'],z['rtmw_det_scores']);f=a.copy()
 for side,b in enumerate([33,54]):
  valid=h['accepted'][:,side];f[valid,b:b+21]=h['points'][valid,side]
 assert np.allclose(h['times'],[x['time'] for x in c['frames']])
 for name,q in [('rtmw',a),('tight_fallback',f)]:
  tensor=norm({'frames':torch.from_numpy(q[:,audit.PICK].transpose(2,0,1).copy())})['frames'].numpy()
  assert np.array_equal(tensor,saved[cid+'_'+name]),'Not actual model inputs'
 arrays[cid]=(a,f)
refs={c['id']:{f['index']:f for f in c['samples']} for c in read(B/'coordinate_reference.json')}
def measure(q,g):
 x=(q-q[0])[1:];y=(g-g[0])[1:];u,_,vt=np.linalg.svd(x.T@y);d=np.eye(2);d[-1,-1]=1 if np.linalg.det(u@vt)>=0 else -1
 rot=u@d@vt;scale=max(0,float(np.sum((x@rot)*y)/max(np.sum(x*x),1e-12)))
 return {'total_sq':float(np.mean(np.sum((q-g)**2,axis=1))),'wrist_sq':float(np.sum((q[0]-g[0])**2)),
 'shape_sq':float(np.mean(np.sum((x-y)**2,axis=1))),'aligned_sq':float(np.mean(np.sum((scale*x@rot-y)**2,axis=1)))}
rows=[];transitions=[]
for c in manifest:
 cid=c['id'];a,f=arrays[cid];z,h=cache[cid]
 for side,name,b in [(0,'left',33),(1,'right',54)]:
  previous=None
  for i,fr in enumerate(c['frames']):
   present=bool(np.any(f[i,b:b+21]));old_present=bool(np.any(a[i,b:b+21]));source='crop' if h['accepted'][i,side] else ('rtmw' if present else 'zero')
   if previous and present and previous[1]:transitions.append({'label':c['label'],'side':name,'frame':i,'switch':source!=previous[0],'wrist_step_px':float(np.linalg.norm(f[i,b]-f[i-1,b]))})
   previous=(source,present)
   ref=refs[cid][fr['sourceIndex']];hand=ref['hands'][name]
   if not all(str(j) in hand for j in range(21)):continue
   assert ref['pngMatches'] and ref['pixelRatio']==1
   g=np.array([hand[str(j)] for j in range(21)])
   rows.append({'id':cid,'label':c['label'],'side':name,'frame':i,'source':source,
    'final':measure(f[i,b:b+21],g) if present else None,'rtmw':measure(a[i,b:b+21],g) if old_present else None})
def stats(rr,field):
 good=[r[field] for r in rr if r[field] is not None]
 return {'hand_frames':len(good),**{k+'_rmse_px':float(np.sqrt(np.mean([r[k+'_sq'] for r in good]))) if good else None for k in ['total','wrist','shape','aligned']}}
paired=[r for r in rows if r['final'] is not None and r['rtmw'] is not None]
changed=[r for r in paired if r['source']=='crop'];better=sum(r['final']['total_sq']<r['rtmw']['total_sq']-1e-6 for r in changed);worse=sum(r['final']['total_sq']>r['rtmw']['total_sq']+1e-6 for r in changed)
summary={'reference_hand_frames':len(rows),'missing_final':sum(r['final'] is None for r in rows),
 'paired':{n:stats(paired,n) for n in ['rtmw','final']},'replaced_paired':{n:stats(changed,n) for n in ['rtmw','final']},'replaced_better':better,'replaced_worse':worse,
 'switches':{str(sw):{'n':len([t for t in transitions if t['switch']==sw]),'median_wrist_step_px':float(np.median([t['wrist_step_px'] for t in transitions if t['switch']==sw]))} for sw in [False,True]}}
perclip={c['label']:{n:stats([r for r in paired if r['id']==c['id']],n) for n in ['rtmw','final']} for c in manifest}
save(O/'results.json',{'summary':summary,'per_clip':perclip,'rows':rows,'transitions':transitions,'actual_normalized_inputs_verified':True})
lines=['# ตรวจพิกัดมือสุดท้ายที่เข้า ST-GCN จริง','',
 'ประกอบพิกัดจาก crop+RTMW ตามกฎเดิม ตรวจ normalized tensor ตรงกับ input ที่บันทึกจากรอบทำนายทุกคลิปและทุกจุด ก่อนอ่าน reference ไม่รันโมเดลทายใหม่และไม่แก้พิกัด',
 'Geometry วัดมือเต็ม21จุดก่อนเลือก10จุดต่อมือเข้า ST-GCN; model input27จุดตรวจซ้ำตรงทั้งหมด Reference เป็นจุด Renderer ไม่ใช่ความถูกต้อง ASL',
 'Wrist-aligned ตัดการเลื่อนข้อมือออกแต่ยังรวมทิศทางและขนาด; aligned ยอมหมุน/ปรับขนาดโดยไม่สะท้อน ใช้วินิจฉัย ไม่ใช่การแก้จริง','',
 '```json',json.dumps(summary,indent=2),'```','',
 '| คำ | RTMW total px | Final total px | Final wrist px | Final wrist-aligned px | Final similarity residual px |','|---|---:|---:|---:|---:|---:|']
for label,g in perclip.items():
 if g['final']['hand_frames']:lines.append(f"| {label} | {g['rtmw']['total_rmse_px']:.2f} | {g['final']['total_rmse_px']:.2f} | {g['final']['wrist_rmse_px']:.2f} | {g['final']['shape_rmse_px']:.2f} | {g['final']['aligned_rmse_px']:.2f} |")
lines+=['','ผลเปรียบเทียบใช้ reference hand-frame เดียวกันที่ทั้งสองวิธีมีข้อมูล ไม่ใช่ผลทุกเฟรม; การไม่มี reference ไม่ได้แปลว่าเป็นมือพัก',
 'การกระโดดข้อมือระหว่างเฟรมมี motion จริงปนอยู่ การต่างกันของกลุ่มเปลี่ยนแหล่งข้อมูลกับไม่เปลี่ยนยังไม่ใช่หลักฐานเชิงสาเหตุ',
 'ข้อมูล30คลิปพัฒนาเดิม เฟรมไม่เป็นอิสระ ไม่สร้าง threshold หรือเลือกโมเดลตามชื่อคำ']
(O/'REPORT_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');protocol['status']='completed';save(O/'protocol.json',protocol)
print(json.dumps(summary),flush=True)
print(json.dumps(sorted([(l,g['final']['total_rmse_px']) for l,g in perclip.items() if g['final']['hand_frames']],key=lambda x:-x[1])[:6]),flush=True)
