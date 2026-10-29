
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont
D=Path(__file__).resolve().parent;O=D/'exp2-multiview-20261013';B=D/'oracle200-20261011';P=D/'pixel-only-plan-20261012'
read=lambda p:_local_json_loads(p.read_text(encoding='utf-8'))
def review():
 mf={c['id']:c for c in read(B/'capture_manifest.json')};chosen=[]
 for p in sorted((O/'events').glob('*.json')):
  for row in read(p):
   if row['accepted']:chosen.append(row);break
  if len(chosen)==6:break
 board=Image.new('RGB',(1200,max(1,len(chosen))*440),'white');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
 for n,r in enumerate(chosen):
  c=mf[r['id']];im=Image.open(B/'frames'/r['id']/c['frames'][r['frame']]['file']).convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im);im=bg.convert('RGB');accepted=r['accepted'][0];cand=r['candidates'][accepted['candidate']];v=cand['views'][cand['selected']];q=np.array(v['points']);rt=np.load(B/'rtm'/(r['id']+'.npz'));old=rt['rtmw_det'][r['frame'],91+21*accepted['side']:112+21*accepted['side']];ctr=(q.max(0)+q.min(0))/2;sz=max(120,float(max(q.max(0)-q.min(0))+50));bb=np.r_[ctr-sz/2,ctr+sz/2].astype(int);sz=bb[2]-bb[0]
  for k,pts in enumerate([None,old,q]):
   crop=im.crop(tuple(bb)).resize((380,380));cd=ImageDraw.Draw(crop)
   if pts is not None:
    for j,pt in enumerate(pts):
     x,y=(pt-bb[:2])*380/sz;cd.ellipse((x-3,y-3,x+3,y+3),fill='red');cd.text((x+3,y),str(j),font=font,fill='blue',stroke_width=1,stroke_fill='white')
   board.paste(crop,(k*400,n*440+45));labels=[f"{r['id']} f{r['frame']} hand{accepted['side']}",'Raw RTMW',f"Consensus {cand['agreement_views']}/8, {v['angle']}deg"]
   draw.text((400*k+4,440*n+4),labels[k],font=font,fill='black')
 board.save(O/'review.png');(O/'review-selection.json').write_text(json.dumps(chosen,indent=2),encoding='utf-8');print('review',len(chosen))
def report():
 result=read(O/'results.json');ex=read(O/'extraction.json');changed=0;counts=Counter();totalboxes=0
 for path in sorted((O/'arrays').glob('*.npz')):
  z=np.load(path);old=np.load(P/'arrays'/path.name);expected=old['raw'].copy()
  for s,b in enumerate([33,54]):expected[old['flags'][:,s],b:b+21]=0
  assert np.array_equal(z['baseline'],expected) and np.isfinite(z['multiview']).all() and expected.shape==z['multiview'].shape
  assert np.array_equal(expected[:,:33],z['multiview'][:,:33])
  for s,b in enumerate([33,54]):
   assert np.array_equal(expected[~old['flags'][:,s],b:b+21],z['multiview'][~old['flags'][:,s],b:b+21]);changed+=int(np.any(expected[:,b:b+21]!=z['multiview'][:,b:b+21],axis=(1,2)).sum())
  for row in read(O/'events'/(path.stem+'.json')):
   totalboxes+=len(row['candidates'])
   for a in row['accepted']:
    ca=row['candidates'][a['candidate']];v=ca['views'][ca['selected']];counts[str((v['angle'],v['scale']))]+=1
 assert changed==ex['accepted_hand_frames'] and ex['views']==8*totalboxes
 changes=[dict(label=b['label'],before=a['prediction'],after=b['prediction']) for a,b in zip(result['predictions']['baseline'],result['predictions']['multiview']) if a['prediction']!=b['prediction']]
 audit=dict(changed_hand_frames=changed,accepted_views=dict(counts),prediction_changes=changes,all_invariants_passed=True,clips=200);(O/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
 lines=['# Experiment 2: ตรวจมือหลายขนาด/หลายมุม — 13 ตุลาคม 2026','','รอบนี้ทำเฉพาะขั้นแรกของแผนใหม่ ไม่ทำ Experiment1 ไม่แก้ระบบหลัก ไม่ใช้ Renderer/CSV พิกัดก่อน render หรือเฉลยคำในการเลือกผล ใช้ภาพและพิกัดที่โมเดลอ่านจากภาพเท่านั้น ชุด200เป็น exploratory เดิม ไม่ใช่ held-out test','','## วิธี','','ใช้กรอบมืออิสระจากภาพที่ cache ไว้ 497กรอบ บน303ภาพของ341มือ-เฟรมที่ตัวกรองเดิมแจ้งปัญหา ตรวจ RTMPose-hand ที่มุม0,+90,-90,180องศาและขนาด1.0/1.2เท่า รวม8แบบ แปลงจุดกลับภาพต้นฉบับและตรวจ round-trip ของ affine transform','','คัด confidence/สีผิว/ขนาดตามเกณฑ์เดิม แล้วเลือกพิกัดที่เป็นตัวแทนกลุ่มผลสอดคล้องอย่างน้อย3แบบ ใช้จุดจากผลตรวจหนึ่งชุด ไม่เฉลี่ยรูปนิ้ว ใช้มือภาพก่อนหน้าไม่เกิน3เฟรมช่วยจับคู่ข้างเช่นรอบก่อน ไม่ค้างพิกัด ไม่ใช้ recovered hand เป็นseed ไม่ลดเกณฑ์หลังเห็นคะแนน','','ต่างจากงานเก่าที่หมุนภาพให้MediaPipe: รอบนี้เป็นRTMPose-handบนกรอบอิสระ และตรวจความสอดคล้องของจุด21จุดข้ามมุม/ขนาด','','## ผลโมเดลจริง','','| วิธี | ถูก/200 | Accuracy | Macro-P | Macro-R | Macro-F1 | Top5 |','|---|---:|---:|---:|---:|---:|---:|']
 for v,m in result['summary'].items():lines.append(f"| {v} | {m['correct']} | {m['accuracy']:.2f}% | {m['precision']:.2f}% | {m['recall']:.2f}% | {m['f1']:.2f}% | {m['top5']:.2f}% |")
 lines+=['',f"ตรวจรวม {ex['views']} ครั้ง; กรอบที่มีผลสอดคล้อง {ex['consensus_boxes']}; เติมได้จริง {changed} มือ-เฟรม",f"เวลา extraction {ex['seconds']:.2f} วินาที ไม่รวมRTMW/กรอบที่cacheไว้ ไม่รวมST-GCNและrender จึงไม่ใช่end-to-end latency",f"คำที่ถูกเพิ่ม: {result['gains']}",f"คำที่ถูกเดิมแต่ผิดใหม่: {result['losses']}",'','## ตรวจความถูกต้องและข้อจำกัด','','Baseline และวิธีกรอบอิสระเดิมถูกนำเข้าโมเดลซ้ำ top5ตรงเดิมและtarget probability ต่างไม่เกิน1e-5 พิกัดลำตัวและมือที่ไม่ถูกflagไม่เปลี่ยน ทุกarrayfinite,shapeและจำนวนเฟรมเดิม ตรวจจำนวนพิกัดที่เปลี่ยนตรงกับบันทึกยอมรับ เก็บinput/logitsครบ','','ความสอดคล้องหลายมุมไม่รับรองความถูกต้อง เพราะทุกมุมอาจผิดเหมือนกัน ผลนี้ทดสอบเฉพาะ341มือ-เฟรม ไม่ใช่การแก้ทุกมือที่หาย และไม่ใช่การวัดความเข้าใจASLของมนุษย์ ไม่มีPCK/joint errorเนื่องจากไม่มีground truthที่คนตรวจ','','ยังไม่ได้ทำการรวมตัวตรวจมือหลายโมเดลตามคุณภาพ หรือการติดตามจากทั้งก่อนและหลังในรอบนี้ ไม่ได้ใช้reserved30หรือfine-tune','','หลักฐาน: PROTOCOL.md, extraction.json, results.json, audit.json, events/, arrays/, inference/, review.png']
 (O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(audit,indent=2))
if __name__=='__main__':
 import sys
 {'review':review,'report':report}[sys.argv[1]]()
