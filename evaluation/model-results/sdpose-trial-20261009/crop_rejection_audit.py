import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;O=ROOT/'crop_stage1_20261011'
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');R=B/'rtm_trial_20261008'
res=json.loads((O/'results.json').read_text());cache={};records=[]
for d in res['details']:
 cid=d['id'];i=d['frame'];side=0 if d['side']=='left' else 1
 if cid not in cache:cache[cid]=(np.load(R/f'{cid}.npz'),np.load(R/'hand_crop'/f'{cid}.npz'))
 z,h=cache[cid];w,e=[(9,7),(10,8)][side];length=float(np.linalg.norm(z['rtmw_det'][i,w]-z['rtmw_det'][i,e]))
 s=h['scores'][i,side];dist=float(np.linalg.norm(h['points'][i,side,0]-z['rtmw_det'][i,w]))
 reasons=[]
 if not d['has_crop']:reasons.append('no_crop')
 else:
  if s.mean()<.3:reasons.append('mean_confidence')
  if s[0]<.3:reasons.append('wrist_confidence')
  if dist>.65*length:reasons.append('wrist_distance')
  if not reasons and not d['accepted']:reasons.append('duplicate_rejection')
 assert bool(reasons)==(not d['accepted'])
 records.append({**d,'reasons':reasons,'mean_confidence':float(s.mean()),'wrist_confidence':float(s[0]),'wrist_forearm_ratio':dist/max(length,1e-8)})
full=[d for d in records if d['full_coverage'] and not d['accepted']]
counts={r:sum(r in d['reasons'] for d in full) for r in ['no_crop','mean_confidence','wrist_confidence','wrist_distance','duplicate_rejection']}
conf_only=sum(set(d['reasons'])<=set(['mean_confidence','wrist_confidence']) for d in full)
out={'full_coverage_rejected':len(full),'overlapping_reasons':counts,'confidence_only':conf_only,'records':records}
(O/'rejections.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
lines=['','## ขั้นที่ 2: แยกสาเหตุไม่ยอมรับผล crop','',
 f"ใน {len(full)} hand-frames ที่กรอบครอบ reference ครบแต่ถูกปฏิเสธ (นับสาเหตุซ้ำกันได้):",'']
for r,n in counts.items():lines.append(f'- {r}: {n}')
lines += [f'- ถูกปฏิเสธด้วย confidence อย่างเดียว: {conf_only}', '',
 'ตรวจสูตรยอมรับย้อนหลังได้ตรง mask เดิมทุก reference hand-frame ไม่มีการลด threshold เพื่อเอาคะแนนเพิ่ม',
 'ผล crop-only 0/30 ประกอบด้วยผลของข้อมูลมือที่ถูกปฏิเสธและกลายเป็นศูนย์ด้วย จึงไม่ควรตีความว่าโมเดล RTMPose-hand ทายรูปมือผิดทุกเฟรม',
 'ข้อเสนอจากหลักฐาน: ยังไม่ต่อ tracking อัตโนมัติในรอบนี้ เพราะต้องตรวจความเชื่อถือได้ของจุดเริ่มก่อน ขั้นต่อไปควรตรวจ confidence กับคุณภาพจุดจริงบน crop ที่ครอบมือครบ และตรวจขนาด/สิ่งรบกวนในกรอบ โดยตรึงการออกแบบใหม่ก่อนดูคะแนน',
 'ยังไม่ทำ fine-tune และยังไม่ได้ยืนยันบนคลิปใหม่อิสระ ผล 24 คลิปแยกกลุ่มนี้เป็นข้อมูลพัฒนาเดิม ไม่ถือเป็น validation ใหม่']
with (O/'REPORT_TH.md').open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='records'}))
