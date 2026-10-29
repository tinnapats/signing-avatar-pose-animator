import sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np,torch
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');O=B/'rest_roi3_confirm_20261008';O.mkdir(exist_ok=True)
sys.path.insert(0,str(B.parent));import run_audit as audit
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'results.json').exists(),'Keep completed results'
refs=read(B/'mixed_reference.json');prior=read(B/'rest_gate_flow/results.json')['predictions'];labels=read(B/'protocol.json')['labels'];rows=[];checks=[];model,norm,vocab=audit.load_model()
for c in refs:
 raw=read(B/'detected'/(c['id']+'.json'))['rows'];roi=read(B/'wrist_tracking'/(c['id']+'.json'))['rows'];z=np.load(B/'rest_gate_flow'/(c['id']+'.npz'));n=len(raw);points=np.zeros((n,75,2),np.float32);gate=np.zeros((n,2),bool)
 for i,(d,r) in enumerate(zip(raw,roi)):
  if d['pose']:points[i,:33]=np.array(d['pose'])[:,:2]*808
  for si,(s,wi,off) in enumerate([('left',15,33),('right',16,54)]):
   if r['track'][s]:points[i,off:off+21]=r['track'][s]
   pose=d['pose'];hand=r['track'][s]
   if pose and hand and all(pose[j][3]>=.5 for j in [11,12,23,24,wi]):
    p=np.array(pose)[:,:2]*808;h=np.array(hand);hip=(p[23]+p[24])/2;shoulder=(p[11]+p[12])/2;torso=np.linalg.norm(hip-shoulder);gate[i,si]=p[wi,1]>=hip[1]-.1*torso and sum(h[j,1]>h[0,1] for j in [4,8,12,16,20])>=4
 stable=np.zeros_like(gate)
 for si,off in enumerate([33,54]):
  for i in range(n-2):
   if gate[i:i+3,si].all():stable[i:i+3,si]=True
  points[stable[:,si],off:off+21]=0
 assert np.array_equal(stable,z['gate']);assert np.array_equal(points,z['rest_roi3']),c['label']
 checks.append(dict(label=c['label'],reconstructed_exact=True,detected_sha256=sha(B/'detected'/(c['id']+'.json')),tracking_sha256=sha(B/'wrist_tracking'/(c['id']+'.json')),coordinates_sha256=sha(B/'rest_gate_flow'/(c['id']+'.npz'))))
 for variant,a in [('image',z['image'].copy()),('rest_roi3',points.copy())]:
  for i,r in enumerate(c['samples']):
   assert r['pngMatches'];origin,ex,ey=np.array(r['affine'])*r['pixelRatio'];inv=np.linalg.inv(np.stack([ex-origin,ey-origin],axis=1));valid=np.any(a[i]!=0,axis=1);a[i,valid]=(a[i,valid]-origin)@inv.T
  x=norm({'frames':torch.from_numpy(a[:,audit.PICK,:].transpose(2,0,1).copy())})['frames'].unsqueeze(0);assert torch.isfinite(x).all()
  with torch.inference_mode():p=model(x)[0].softmax(0)
  top=p.topk(5);r=dict(label=c['label'],variant=variant,prediction=vocab[int(top.indices[0])],top5=[vocab[int(t)] for t in top.indices],target_probability=float(p[vocab.index(c['label'])]));old=next(q for q in prior if q['label']==c['label'] and q['variant']==variant);assert r['prediction']==old['prediction'] and r['top5']==old['top5'];rows.append(r)
 print('confirmed',c['label'],flush=True)
summary={}
for v in ['image','rest_roi3']:
 rr=[r for r in rows if r['variant']==v];ps=[];rs=[];fs=[]
 for l in labels:
  tp=sum(r['label']==l and r['prediction']==l for r in rr);fp=sum(r['label']!=l and r['prediction']==l for r in rr);p=tp/(tp+fp) if tp+fp else 0;ps.append(p);rs.append(tp);fs.append(2*p*tp/(p+tp) if p+tp else 0)
 summary[v]=dict(correct=sum(r['label']==r['prediction'] for r in rr),accuracy=100*np.mean(rs),precision=100*np.mean(ps),recall=100*np.mean(rs),f1=100*np.mean(fs),correct_words=[r['label'] for r in rr if r['label']==r['prediction']])
for f,h in read(B/'protocol.json')['production'].items():assert sha(B.parent.parent/f)==h
result=dict(summary=summary,predictions=rows,checks=checks,checkpoint_sha256=sha(B.parent/'wlasl_stgcn.zip'),scope='Reconstructed rest gate from cached image detections and3-frame optical flow; reran ST-GCN inference. No new image capture or MediaPipe inference. Uses camera metadata for coordinate alignment. No5-frame gap interpolation, rotated detection or bidirectional tracking.',production_unchanged=True)
(O/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
lines=['# ยืนยันวิธีคัดมือพัก + ตรวจข้อมือ + ติดตาม3เฟรม','', '8 ตุลาคม2026 — สร้างกฎคัดมือพักซ้ำจากข้อมูลตรวจจับภาพและtrackingที่บันทึกไว้ครบ30คำ แล้วรันST-GCNใหม่ ไม่ใช่การจับภาพหรือรันMediaPipeใหม่',
'','วิธีเฉพาะนี้: รักษาจุดHolisticเดิม → ตรวจROIข้อมือเมื่อมือหาย → optical flowเมื่อยังหาไม่พบ สูงสุด3เฟรมต่อเนื่องพร้อมforward/backwardตรวจความสอดคล้อง → คัดมือที่ใกล้สะโพกและ4/5ปลายนิ้วต่ำกว่าข้อมือต่อเนื่อง3เฟรม → จัดพิกัดด้วยcamera metadata → เลือก27จุดและnormalize → ST-GCN',
'','ไม่ได้ใช้การเติม5เฟรม การตรวจภาพหมุน หรือtrackingสองทิศทาง5เฟรม ผลพิกัดสร้างซ้ำตรงเดิมทุกค่า ผลTop1/Top5ตรงเดิมครบ30คำทั้งสองวิธี ไฟล์หลักไม่เปลี่ยน','', '| วิธี | ถูก/30 | Accuracy | Precision macro | Recall macro | F1 macro |','|---|---:|---:|---:|---:|---:|']
for v,s in summary.items():lines.append(f"| {v} | {s['correct']}/30 | {s['accuracy']:.2f}% | {s['precision']:.2f}% | {s['recall']:.2f}% | {s['f1']:.2f}% |")
lines+=['','คำที่ทายถูก: '+', '.join(summary['rest_roi3']['correct_words']),'', 'ยังมีmissing zeroในช่วงที่หาไม่ได้ และใช้metadataกล้องจากระบบ จึงยังไม่ใช่การอ่านภาพล้วนend-to-end ชุดพัฒนาเดิม30คำไม่ใช่held-out คะแนนนี้ไม่ใช่ความถูกต้องASLจากผู้ใช้ รายคำและhashแหล่งข้อมูลอยู่results.json']
(O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(summary,ensure_ascii=False,indent=2))
