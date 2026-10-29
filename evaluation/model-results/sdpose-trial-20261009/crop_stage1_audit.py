"""Audit frozen cached wrist crops and rerun recognition; no new routing tuning."""
import sys,json,hashlib,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit');B=A/'pixels30_20261007';R=B/'rtm_trial_20261008';H=R/'hand_crop'
O=ROOT/'crop_stage1_20261011';O.mkdir(exist_ok=True)
assert not (O/'protocol.json').exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
manifest=read(B/'capture_manifest.json');assert len(manifest)==30
protocol={'status':'registered','phase':'Crop coverage / landmark diagnosis before further tracking',
 'data':'Existing 30 development clips. IDs 000-005 reported separately from 006-029. Neither subset is unseen across all historical experiments.',
 'crop_protocol':read(H/'protocol.json'),
 'variants':['rtmw','crop_only','crop_fallback'],
 'fallback':'Existing accepted mask, unchanged. Crop accepted -> crop points, otherwise RTMW points. No label/ref-based routing.',
 'quality':'Measure reference joints inside effective crop incl library padding 1.25, and errors inside fully covered reference hand-frames. Reference never used for selection.',
 'recognition':'Same STGCN2000, confidence/missing conventions, shoulder normalization, timing, no retraining',
 'source_hashes':{str(p):sha(p) for p in [Path(__file__),ROOT.parent/'rtm-hand.py',ROOT.parent/'rtm-eval.py',A/'wlasl_stgcn.zip',H/'protocol.json']}}
# Verify cache hashes and alignment before evaluation.
for c in manifest:
 for folder in [R,H]:assert sha(folder/(c['id']+'.npz'))==read(folder/(c['id']+'.json'))['sha256']
protocol['cache_hashes']={c['id']:sha(H/(c['id']+'.npz')) for c in manifest}
(O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
sys.path.insert(0,str(A))
import run_audit as audit
spec=importlib.util.spec_from_file_location('ev',ROOT.parent/'rtm-eval.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
model,norm,vocab=audit.load_model();pred={v:[] for v in protocol['variants']};inputs={};logits={}
old=read(R/'results.json')
for c in manifest:
 cid=c['id'];z=np.load(R/(cid+'.npz'));h=np.load(H/(cid+'.npz'))
 assert np.allclose(z['times'],h['times']) and np.allclose(z['times'],[f['time'] for f in c['frames']])
 raw=ev.convert(z['rtmw_det'],z['rtmw_det_scores']);variants={'rtmw':raw,'crop_only':raw.copy(),'crop_fallback':raw.copy()}
 variants['crop_only'][:,33:]=0
 for side,b in enumerate([33,54]):
  valid=h['accepted'][:,side]
  for v in ['crop_only','crop_fallback']:variants[v][valid,b:b+21]=h['points'][valid,side]
 fixed=None
 for v,a in variants.items():
  t=norm({'frames':torch.from_numpy(a[:,audit.PICK].transpose(2,0,1).copy())})['frames']
  if fixed is None:fixed=t[:,:,:7].clone()
  assert torch.equal(fixed,t[:,:,:7]) and torch.isfinite(t).all()
  with torch.inference_mode():out=model(t.unsqueeze(0))[0];p=out.softmax(0)
  top=p.topk(5);target=vocab.index(c['label'])
  rr={'id':cid,'label':c['label'],'prediction':vocab[int(top.indices[0])],'top5':[vocab[int(j)] for j in top.indices],
   'target_probability':float(p[target]),'target_rank':torch.argsort(p,descending=True).tolist().index(target)+1}
  before=next(r for r in old['predictions']['rtmw_det' if v=='rtmw' else v] if r['label']==c['label'])
  assert before['top5']==rr['top5'] and abs(before['target_probability']-rr['target_probability'])<1e-5
  pred[v].append(rr);inputs[cid+'_'+v]=t.numpy();logits[cid+'_'+v]=out.numpy()
 print(cid,c['label'],*[pred[v][-1]['prediction'] for v in variants],flush=True)
np.savez_compressed(O/'inputs.npz',**inputs);np.savez_compressed(O/'logits.npz',**logits)
# Post-hoc reference geometry only, never inference input.
refs={c['id']:{f['index']:f for f in c['samples']} for c in read(B/'coordinate_reference.json')}
details=[]
for c in manifest:
 cid=c['id'];z=np.load(R/(cid+'.npz'));h=np.load(H/(cid+'.npz'))
 for i,f in enumerate(c['frames']):
  ref=refs[cid][f['sourceIndex']];assert ref['pngMatches'] and ref['pixelRatio']==1
  for side,name,b in [(0,'left',91),(1,'right',112)]:
   hand=ref['hands'][name]
   if not all(str(j) in hand for j in range(21)):continue
   g=np.array([hand[str(j)] for j in range(21)]);box=h['boxes'][i,side];exists=bool(box[2]>box[0] and box[3]>box[1])
   mid=(box[:2]+box[2:])/2;half=(box[2:]-box[:2])*1.25/2
   inside=np.all((g>=mid-half)&(g<=mid+half),axis=1) if exists else np.zeros(21,bool)
   q=h['points'][i,side];rt=z['rtmw_det'][i,b:b+21];s=z['rtmw_det_scores'][i,b:b+21]
   details.append({'id':cid,'label':c['label'],'frame':i,'side':name,'has_crop':exists,'covered_joints':int(inside.sum()),'full_coverage':bool(inside.all()),
    'accepted':bool(h['accepted'][i,side]),'rtmw_accepted':bool(s[0]>=.3 and s.mean()>=.3),
    'crop_sq':float(np.mean(np.sum((q-g)**2,axis=1))) if exists else None,'rtmw_sq':float(np.mean(np.sum((rt-g)**2,axis=1))),
    'crop_wrist_sq':float(np.sum((q[0]-g[0])**2)) if exists else None,
    'crop_shape_sq':float(np.mean(np.sum(((q-q[0])[1:]-(g-g[0])[1:])**2,axis=1))) if exists else None})
def geom(ds):
 paired=[d for d in ds if d['full_coverage'] and d['accepted'] and d['rtmw_accepted']]
 return {'reference_hand_frames':len(ds),'has_crop':sum(d['has_crop'] for d in ds),'full_coverage':sum(d['full_coverage'] for d in ds),
 'full_coverage_rejected':sum(d['full_coverage'] and not d['accepted'] for d in ds),
 'partial_coverage_accepted':sum(not d['full_coverage'] and d['accepted'] for d in ds),
 'paired_full_accepted_hand_frames':len(paired),
 **{key+'_rmse_px':float(np.sqrt(np.mean([d[key+'_sq'] for d in paired]))) if paired else None for key in ['crop','rtmw','crop_wrist','crop_shape']}}
subsets={'six':lambda cid:int(cid)<6,'remaining24':lambda cid:int(cid)>=6,'all30':lambda cid:True}
summary={name:{v:ev.metrics([r for r in rr if choose(r['id'])]) for v,rr in pred.items()} for name,choose in subsets.items()}
geometry={name:geom([d for d in details if choose(d['id'])]) for name,choose in subsets.items()}
result={'summary':summary,'geometry':geometry,'predictions':pred,'details':details,'cached_results_reproduced':True,
 'per_clip_geometry':{c['label']:geom([d for d in details if d['id']==c['id']]) for c in manifest}}
(O/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
protocol['status']='completed';(O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
lines=['# ขั้นที่ 1: ตรวจ crop ข้อมือ RTMW และ RTMPose-hand','',
 'ตรวจ checksum ผล crop เดิมครบ 30 คลิป และรัน ST-GCN ใหม่จริง ผล top5 และ probability ซ้ำตรงกับรอบก่อนทุกวิธี ไม่ได้รันตัวตรวจมือใหม่ เพราะมีผลดิบที่ตรงเงื่อนไขอยู่แล้ว',
 'ตรึงเกณฑ์เดิม: confidence 0.30, กรอบจากข้อมือและศอก RTMW, padding 1.25; กฎยอมรับ crop และ fallback เหมือนเดิม ไม่มีปรับตามชื่อคำ',
 '6 คลิปล่าสุดและ 24 คลิปที่เหลือยังเป็นข้อมูลพัฒนาเดิมทั้งหมด ไม่ใช่ held-out test set','',
 '| กลุ่ม | วิธี | ถูก | Accuracy % | Macro-F1 % | Top5 % |','|---|---|---:|---:|---:|---:|']
for group,vs in summary.items():
 for v,m in vs.items():lines.append(f"| {group} | {v} | {m['correct']} | {m['accuracy']:.2f} | {m['f1']:.2f} | {m['top5']:.2f} |")
lines+=['','## คุณภาพ crop','',json.dumps(geometry,ensure_ascii=False,indent=2),
 'Coverage ตรวจว่าจุด Renderer ทั้ง 21 อยู่ในกรอบที่รวม padding จริง ไม่ได้ยืนยันพื้นที่ผิวมือทั้งหมดหรือการบัง; ไม่มี reference ไม่ถูกนับว่าเป็นมือพัก',
 'RMSE เปรียบเทียบบน hand-frame เดียวกันที่ crop ครอบ reference ครบและทั้งสองวิธีผ่านเกณฑ์เท่านั้น ผลนี้ไม่ครอบคลุมเฟรมที่ล้มเหลว จึงรายงานจำนวนที่ถูกตัดออกควบคู่',
 'Macro เฉลี่ยเฉพาะ true labels ของแต่ละกลุ่ม ไม่ใช่ macro ทั้ง 2000 classes; คะแนนไม่ใช่ความเข้าใจ ASL']
(O/'REPORT_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary),flush=True);print(json.dumps(geometry),flush=True)
