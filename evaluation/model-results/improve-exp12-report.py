import json
from pathlib import Path
from collections import Counter
import numpy as np
D=Path(__file__).resolve().parent;O=D/'improve-exp12-20261013';P=D/'pixel-only-plan-20261012'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
r1=read(O/'exp1-results.json');r2=read(O/'exp2-results.json');ex=read(O/'exp2-extraction.json');changed=0;reasons=Counter();good=0
for p in sorted((O/'exp2-arrays').glob('*.npz')):
 z=np.load(p);old=np.load(P/'arrays'/p.name);expected=old['raw'].copy();flags=old['flags']
 for s,b in enumerate([33,54]):expected[flags[:,s],b:b+21]=0
 assert np.array_equal(z['baseline'],expected) and np.isfinite(z['independent']).all() and z['independent'].shape==expected.shape
 assert np.array_equal(expected[:,:33],z['independent'][:,:33])
 for s,b in enumerate([33,54]):
  assert np.array_equal(expected[~flags[:,s],b:b+21],z['independent'][~flags[:,s],b:b+21]);changed+=int(np.any(expected[:,b:b+21]!=z['independent'][:,b:b+21],axis=(1,2)).sum())
for r in ex['events']:
 for c in r['candidates']:
  good+=int(c['valid'])
  if 'rejection' in c:reasons[c['rejection']]+=1
assert changed==ex['accepted_hand_frames']
audit=dict(exp2_changed_hand_frames=changed,exp2_quality_passed_candidates=good,rejections=dict(reasons),baseline_reproduced_both_experiments=True,body_and_unflagged_hands_unchanged=True,exp2_prediction_changes=[dict(label=b['label'],before=a['prediction'],after=b['prediction']) for a,b in zip(r2['predictions']['baseline'],r2['predictions']['independent']) if a['prediction']!=b['prediction']])
(O/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
lines=['# ทดลองเพิ่มคะแนน Experiment 1 และ 2 — 13 ตุลาคม 2026','','ทดลองจริงบน 200 คลิปเดิม ไม่เปลี่ยน production ไม่ฝึกโมเดล ไม่ใช้ชุด reserved30 และไม่ใช้ข้อมูล Renderer ใน Experiment 2 ผลนี้เป็น exploratory ไม่ใช่ held-out validation','','## Experiment 1: รวมโมเดลแบบตรึงน้ำหนัก','','ใช้พิกัดก่อน Render และ frameIndices เดิม เลือก 27 จุด normalize เหมือนเดิม รัน ST-GCN และ SL-GCN ใหม่ครบ 200 คลิปแล้วเฉลี่ย softmax 2000 คลาส ไม่มีการเลือกคำตอบตามเฉลย ไม่มีการจำกัดคำตอบให้เหลือ 200 คำ และไม่ได้ไล่ค่าน้ำหนักเพื่อหาจุดสูงสุด','','| วิธี | ถูก/200 | Accuracy | Macro-P | Macro-R | Macro-F1 | Top5 |','|---|---:|---:|---:|---:|---:|---:|']
for v,m in r1['summary'].items():lines.append(f"| {v} | {m['correct']} | {m['accuracy']:.2f}% | {m['precision']:.2f}% | {m['recall']:.2f}% | {m['f1']:.2f}% | {m['top5']:.2f}% |")
lines+=['','ST50_SL50 เพิ่มคำที่ทายถูก 13 คำ แต่เสียคำเดิม 13 คำ ST75_SL25 ไม่มีคำใหม่ที่ถูกและเสีย 3 คำ จึงยังไม่ใช้ ensemble แทน ST-GCN เดิม Top5 ที่เพิ่มไม่ใช่การเพิ่ม Top1 Accuracy','','ทั้งสองโมเดลถูกพร้อมกัน 76 คำ ST ถูกฝ่ายเดียว 46 คำ SL ถูกฝ่ายเดียว 20 คำ รวมมีโมเดลใดโมเดลหนึ่งถูก 142/200 แต่ **142/200 ไม่ใช่คะแนนวิธีที่ทำได้จริง** เพราะการเลือกคำตอบที่ถูกทุกครั้งต้องรู้เฉลย','','## Experiment 2: หากรอบมือจากภาพโดยตรง','','ใช้ anime hand detector เดิมแบบทั้งภาพ threshold0.395 แล้วตรวจนิ้วด้วย RTMPose-hand คง threshold0.30 และตรวจ pixel support ไม่ใช้ body wrist ปัจจุบันบังคับ crop หรือคัด candidate ใช้มือ RTMW ที่ตรวจจากภาพก่อนหน้าภายใน 3 เฟรมและผ่านเกณฑ์คุณภาพเพื่อระบุข้างเท่านั้น ถ้าจับคู่ไม่ได้หรือกำกวมจะไม่เติม มือที่กู้คืนไม่ถูกใช้เป็น seed ต่อ','','รอบนี้เลือกเฉพาะมือที่ตัวกรองเดิมแจ้งปัญหา ไม่ใช่ทุกมือที่ตรวจไม่พบ ไม่ได้แก้ลำตัว ไม่ได้ใช้ชื่อคำใน extraction และไม่ได้ปรับเกณฑ์หลังเห็นคะแนน','','| วิธี | ถูก/200 | Accuracy | Macro-P | Macro-R | Macro-F1 |','|---|---:|---:|---:|---:|---:|']
for v,m in r2['summary'].items():lines.append(f"| {v} | {m['correct']} | {m['accuracy']:.2f}% | {m['precision']:.2f}% | {m['recall']:.2f}% | {m['f1']:.2f}% |")
lines+=['',f"- เลือก {ex['flagged_hand_frames']} hand-frames ใน {ex['processed_frames']} ภาพ",f"- ตรวจพบ {ex['boxes']} กรอบมือ (ยังไม่ใช่จำนวนมือที่ถูกต้อง)",f"- candidate ผ่าน confidence/skin/span: {good}",f"- จับคู่ข้างและเติมพิกัดได้จริง {changed} hand-frames",f"- เวลา extraction เพิ่ม {ex['seconds']:.2f} วินาที ไม่รวม RTMW cache, ST-GCN และการ render จึงไม่ใช่ end-to-end latency",f"- คำที่ถูกเพิ่ม: {r2['pairs']['gains']}",f"- คำที่ถูกเดิมแต่ผิดใหม่: {r2['pairs']['losses']}",'','## ข้อจำกัดและสิ่งที่ยังไม่ทำ','','ตัวกรองสีผิว/ความมั่นใจ/ขนาดไม่ใช่ ground truth การตรวจพบเพิ่มไม่ได้แปลว่านิ้วถูกเพิ่ม ต้องตรวจภาพประกอบ exp2-review.png แยกกัน จุดที่ซ้อน/ถูกบังไม่ควรเดาแล้วอ้างเป็น label จริง','','ยังไม่ได้ทดลองตัดช่วงพักหรือ fine-tune และยังไม่มี annotations ที่มนุษย์ตรวจรับรอง จึงไม่อ้างว่าทำแผนระยะยาวครบแล้ว การฝึกบนคลิปทดสอบหรือใช้ Renderer เป็นเฉลยใน inference ไม่อยู่ในขอบเขต','','## ตรวจซ้ำ','','Baseline ของทั้งสองโมเดลใน Experiment1 และ ST-GCN Experiment2 ให้ top5 ตรงเดิมและ target probability ต่างไม่เกิน 1e-5 พิกัดลำตัวและมือที่ไม่ถูก flag ใน Experiment2 ไม่เปลี่ยน ทุก array finite และ shape/frame count เดิม เก็บ input/logits ครบและตรวจจำนวนเฟรมที่แก้ตรงกับบันทึกการยอมรับ','','หลักฐาน: exp1-results.json, exp2-results.json, exp2-extraction.json, audit.json, PROTOCOL.md และโฟลเดอร์ inference/arrays ที่แยกตาม experiment']
(O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(audit,indent=2))
