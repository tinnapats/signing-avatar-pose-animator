import json, hashlib, math
from pathlib import Path
import numpy as np
import cv2
O=Path(__file__).parent/'hand-failure-audit-20261009'
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
r=read(O/'confidence_results.json'); d=read(O/'diagnostics.json'); counts=read(O/'accepted_counts.json')
base=r['predictions']['confidence_030']; paired={}
for name in ['confidence_020','confidence_040']:
 rr=r['predictions'][name]
 gained=[a['label'] for a,b in zip(rr,base) if a['prediction']==a['label'] and b['prediction']!=b['label']]
 lost=[a['label'] for a,b in zip(rr,base) if a['prediction']!=a['label'] and b['prediction']==b['label']]
 n=len(gained)+len(lost); p=min(1,2*sum(math.comb(n,j) for j in range(min(len(gained),len(lost))+1))/2**n) if n else 1
 paired[name]=dict(gained=gained,lost=lost,exact_mcnemar_p=p)
(O/'paired.json').write_text(json.dumps(paired,ensure_ascii=False,indent=2),encoding='utf8')
names={'01_no_covering_box':'ไม่พบกรอบที่ครอบจุดมืออย่างน้อย80%','02_hand_confidence_rejected':'มีกรอบครอบมือ แต่ความมั่นใจจุดมือไม่ผ่าน','03_body_anchor_rejected':'จุดข้อมือ/ศอกลำตัวไม่ผ่าน','04_wrist_distance_rejected':'ผ่านความมั่นใจ แต่ระยะข้อมือไม่ผ่าน','05_assignment_other_or_missing':'มีตัวเลือกผ่าน แต่การจับคู่เลือกอื่น/ไม่เลือก','06_accepted_covering_candidate':'ยอมรับตัวเลือกที่กรอบครอบมือ'}
lines=['# ตรวจสาเหตุการดึงจุดมือจากภาพ Avatar และทดลอง confidence','', '## ขอบเขต','9 ตุลาคม 2026 — ทดลองเฉพาะข้อมูล signing-avatar-github-tinnapat ไม่แก้ระบบหลัก ผลลัพธ์เก็บในโฟลเดอร์รายงานแยก','30 คลิปเดิม 1,964 เฟรม 30FPS เวลาเดิม ใช้ผลดึงจุดจากภาพที่บันทึกไว้แล้ว รัน ST-GCN ใหม่จริง ไม่สร้างคะแนนเอง','', '## 1. แยกสาเหตุ','หน่วยคือหนึ่งมือในหนึ่งเฟรม ไม่ใช่จำนวนเฟรมภาพไม่ซ้ำ มี detailed-hand reference 1,872 hand-frames','จัดหมวดตามลำดับ: กรอบ → ความมั่นใจมือ → ความมั่นใจลำตัว → ระยะข้อมือ → การจับคู่ หมวดไม่ซ้ำกัน แต่ไม่ได้พิสูจน์ว่าไม่มีสาเหตุร่วม','| หมวด | จำนวน | สัดส่วน |','|---|---:|---:|']
for k,v in names.items():
 n=d['counts'].get(k,0);lines.append(f'| {v} | {n} | {100*n/1872:.2f}% |')
lines+=['','กรอบครอบมือเป็นเกณฑ์เชิงเรขาคณิตแบบ optimistic: >=80% ของ21จุดอยู่ในกรอบสี่เหลี่ยม padding1.25 ไม่รับรองว่ากรอบนั้นเป็นมือข้างถูกหรือนิ้วถูก','ยอมรับกรอบครอบมือ471 ต่างจากจำนวนที่มีพิกัดให้เทียบ473 เพราะอีก2hand-framesยอมรับตัวเลือกซึ่งไม่ผ่านเกณฑ์ครอบ80%','อีก2,056hand-slotsไม่มี detailed reference และถูกยอมรับ1,836 ไม่สามารถสรุปจากการไม่มีreferenceเพียงอย่างเดียวว่าเป็นมือพักหรือมือผิด ต้องแยกชนิดRendererหรือดูภาพเพิ่มเติม','', '## 2. ทดลองเกณฑ์ความมั่นใจแบบควบคุม','ตรึง detector .395, body confidence .3, ระยะข้อมือ .65เท่าความยาวท่อนแขน, การจับคู่หนึ่งต่อหนึ่ง และ fallbackเดิม เปลี่ยนเฉพาะ mean hand confidence และ wrist confidence พร้อมกันเป็น .2/.3/.4','ทุกวิธีใช้ RTMWลำตัว; มือanimeที่ผ่านเกณฑ์แทนมือเดิม ถ้าไม่ผ่านใช้MediaPipeเมื่อพบ ไม่พบใช้RTMW ตัวเลขthresholdไม่ใช่เปอร์เซ็นต์ความถูกต้องที่calibrateแล้ว','ไม่มีใช้ Renderer/CSV/payload ในข้อมูลเข้าโมเดลหรือเลือกผล ใช้Rendererเฉพาะวัดgeometry','ตรึงทั้ง3ค่าก่อนรันคะแนน ทำซ้ำbaseline0.30ได้คำทายTop1/Top5และtarget probabilityตรงรอบก่อน','', '| วิธี | ถูก/30 | Macro-P | Macro-R | Macro-F1 | Top5 | PCK10รวมmissing | RMSE(px) |','|---|---:|---:|---:|---:|---:|---:|---:|']
for name in ['previous_best','confidence_020','confidence_030','confidence_040']:
 s=r['summary'][name];g=r['geometry'][name]
 lines.append(f"| {name} | {s['correct']}/30 | {s['precision']:.2f}% | {s['recall']:.2f}% | {s['f1']:.2f}% | {s['top5']:.2f}% | {g['pck10_all_reference_pct']:.2f}% | {g['rmse_px']:.2f} |")
lines+=['','Macroเฉลี่ย30คลาสคำจริง zero_division0 โมเดลยังทายได้2000คำ ไม่จำกัดคำตอบ; RMSEวัดระยะEuclideanในภาพ808×808 เฉพาะที่มีพิกัด ส่วนPCK10รวมmissingนับจุดที่หายว่าไม่ผ่าน','ตรวจmask geometryของhybridทั้ง3ค่าเหมือนกันทุกคลิป:1,870/1,872 hand-frames จึงเทียบRMSEกันบนมือเฟรมเดียวกันได้','', '## 3. ผลรายคำเทียบ0.30']
for name,p in paired.items():lines+=['',f"- {name}: กลับมาถูก {', '.join(p['gained']) or 'ไม่มี'}; จากถูกเป็นผิด {', '.join(p['lost']) or 'ไม่มี'}; exact McNemar p={p['exact_mcnemar_p']:.4f}"]
lines+=['','p-valueเป็นexploratory ไม่ปรับmultiple comparisons และไม่ใช่หลักฐานยืนยันเหนือกว่า','', '## 4. ข้อสรุปและงานต่อ','- คอขวดเชิงกระบวนการคือความมั่นใจมือไม่ผ่าน แม้มีกรอบครอบมือ ไม่ใช่ระยะข้อมือเป็นหลัก','- ลดconfidenceเป็น0.20รับมือเพิ่ม แต่คะแนนรู้จำลด5/30 และRMSEแย่ลง แม้PCKดีขึ้น แสดงว่ามีทั้งจุดที่ดีเพิ่มและoutlierมากขึ้น ไม่ควรลดเกณฑ์ทั้งหมดทันที','- เพิ่มconfidenceเป็น0.40ได้8/30 แต่PCKและRMSEด้อยกว่า0.30 จึงเป็นcandidateด้านrecognitionเท่านั้น ไม่ใช่หลักฐานว่าภาพหรือการดึงนิ้วถูกต้องกว่า','- เก็บ0.30เป็นgeometry baseline และ0.40เป็นrecognition candidate ยังไม่เปลี่ยนproduction','- ขั้นต่อไปเหมาะกับการเปรียบเทียบขนาดcropบนกรอบเดิม โดยตรึงconfidence0.30และตัวแปรอื่น ไม่ไล่thresholdเพิ่มบน30คำเดิม เป้าหมายคือเพิ่มมือที่อ่านนิ้วได้จริงโดยไม่เพิ่มoutlier','- หลังเลือกวิธีต้องตรึงค่าก่อนใช้คลิปใหม่30–60คลิป ผลชุดนี้เป็นdevelopment setทดลองซ้ำ ไม่ใช่held-out test และคะแนนไม่แทนความเข้าใจASLของคน','', '## จำนวนมือanimeที่ยอมรับทั้งหมด (รวมช่วงไม่มี detailed reference)']
for k,v in counts.items():lines.append(f'- {k}: {v}/3928 hand-slots')
lines+=['','0.40รับมือน้อยลงมากและกลับไปใช้fallbackบ่อยขึ้น ดังนั้นผลคะแนนที่เพิ่มอาจมาจากการลดการแทนที่มือเดิม ไม่ใช่โมเดลหานิ้วแม่นขึ้นเอง','ไม่รันdetector/poseใหม่ในรอบนี้ จึงไม่รายงานความเร็วextractionใหม่','', '## หลักฐาน','- diagnostics.json: เหตุผลทุกhand-frame','- confidence_protocol.json: ค่าทดลองที่ตรึงก่อนคะแนน','- confidence_results.json: คะแนนจริง, คำทายทุกคลิป, geometry, ซ้าย–ขวา','- model_inputs/: พิกัดภาพเข้าโมเดลทุกวิธีพร้อมtimestamp','- paired.json: คำได้/เสีย','- comparison.png: ตัวอย่างพิกัดของวิธีต่างๆบนคำที่คะแนนเปลี่ยน','- 01–06 PNG: ตัวอย่างสาเหตุ กรอบน้ำเงิน, referenceเขียว, จุดที่เลือกแดง, จุดตัวเลือกเทา, ข้อมือลำตัวม่วง','', '## คำทายรายคลิป','| คำ | 0.20 | 0.30 | 0.40 |','|---|---|---|---|']
for i,b in enumerate(base):lines.append('| '+b['label']+' | '+' | '.join(r['predictions'][v][i]['prediction'] for v in ['confidence_020','confidence_030','confidence_040'])+' |')
manifest=read(B/'capture_manifest.json'); refs={c['id']:c for c in read(B/'coordinate_reference.json')}; chosen=set(sum([p['gained']+p['lost'] for p in paired.values()],[]));panels=[]
edges=[(0,1),(1,2),(2,3),(3,4)]+[(0,k) for k in [5,9,13,17]]+[(k+j,k+j+1) for k in [5,9,13,17] for j in range(3)]
for c in manifest:
 z=np.load(O/'model_inputs'/(c['id']+'.npz')); masks=[np.any(z[k][:,33:].reshape(-1,2,21,2)!=0,axis=(2,3)) for k in ['confidence_020','confidence_030','confidence_040']]; assert np.array_equal(masks[0],masks[1]) and np.array_equal(masks[1],masks[2])
 if c['label'] not in chosen:continue
 delta=np.linalg.norm(z['confidence_020'][:,33:]-z['confidence_040'][:,33:],axis=2).mean(1); i=int(np.argmax(delta));f=c['frames'][i];p=B/'frames'/c['id']/f['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
 im=cv2.imread(str(p),-1);alpha=im[:,:,3:4].astype(np.float32)/255;im=np.rint(im[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8);row=[]
 for k in ['previous_best','confidence_020','confidence_030','confidence_040']:
  img=im.copy()
  for side,off,col in [('left',33,(255,60,0)),('right',54,(0,50,255))]:
   h=refs[c['id']]['samples'][i]['hands'][side]
   if len(h)==21:
    g=np.array([h[str(j)] for j in range(21)]).astype(int)
    for a,b in edges:cv2.line(img,tuple(g[a]),tuple(g[b]),(0,160,0),2)
   q=z[k][i,off:off+21].astype(int)
   if np.any(q):
    for a,b in edges:cv2.line(img,tuple(q[a]),tuple(q[b]),col,1)
  img=cv2.resize(img,(360,360));band=np.full((65,360,3),255,np.uint8);pred=r['predictions'][k][int(c['id'])]['prediction']
  cv2.putText(band,k,(6,20),0,.5,(0,0,0),1);cv2.putText(band,f"{c['label']} f{i}: {pred}",(6,45),0,.5,(0,0,0),1);row.append(np.vstack([band,img]))
 panels.append(np.hstack(row))
if panels:cv2.imwrite(str(O/'comparison.png'),np.vstack(panels))
(O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf8')
print(json.dumps(paired,ensure_ascii=True));print('Report and comparison saved. Identical masks verified.')
