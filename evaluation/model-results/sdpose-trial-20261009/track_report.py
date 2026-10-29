import json
from pathlib import Path
import numpy as np
import cv2
R=Path(__file__).resolve().parent/'tracking3'
O=R/'identity_trial'
m=json.loads((O/'metrics.json').read_text())
rows=json.loads((R/'results.json').read_text())
d=json.loads((O/'decisions.json').read_text())
x=np.load(O/'predictions.npz')
variants=['raw','body_anchor','body_anchor_temporal']
lines=['# SDPose: ทดลองจับคู่มือด้วยลำตัวและเวลา','',
f"รันภาพ Avatar ทั้ง 3 คลิป abdomen, hello, love รวม {len(rows)} เฟรม ที่เวลาเดิม 30 FPS ไม่ resample และไม่แก้ production",'',
'## วิธีที่เทียบ',
'1. raw: พิกัดดิบ SDPose',
'2. body_anchor: จับคู่ข้อมือของมือกับข้อมือลำตัวซ้าย/ขวา โดยไหล่–ศอก–ข้อมือและมือทั้งสองต้องผ่าน confidence 0.30 ตรวจอัตราความยาวท่อนแขนเพื่อคัดกรณีผิดรุนแรง',
'3. body_anchor_temporal: เพิ่มระยะจากข้อมือที่เลือกในเฟรมก่อน น้ำหนัก 0.35 ใช้ประวัติไม่เกิน 3 เฟรม',
'','ทุกวิธีใช้เฉพาะพิกัดจากภาพในการตัดสินใจ สลับเมื่อ cost ลดลงเกิน 0.15 เท่าความกว้างไหล่ ไม่ปรับ threshold หลังดูผล ไม่สร้างจุดใหม่ ไม่ค้างท่า และไม่เปลี่ยนจุดลำตัว',
'บันทึก decisions และ predictions ก่อนโหลดข้อมูลอ้างอิงเพื่อประเมินผล',
'','## ผลเทียบพิกัดมือกับ Renderer',
'ใช้จุดดิบครบทุกจุดของ hand-frames ที่มีข้อมูลอ้างอิงครบ 21 จุด โดยไม่กรอง confidence ใน metric วิธีนี้วัดทั้งความผิดของจุดและความผิดของฝั่งมือ ไม่ใช่ความถูกต้องของคำ ASL',
'','| วิธี | เฟรมที่สลับมือ | RMSE (px) ↓ | PCK10 ↑ | Wrist RMSE (px) ↓ |',
'|---|---:|---:|---:|---:|']
for v in variants:
    z=m['metrics'][v]
    lines.append(f"| {v} | {z['swapped_frames']} | {z['rmse_px']:.2f} | {z['pck10_pct']:.2f}% | {z['wrist_rmse_px']:.2f} |")
z=m['metrics']['raw']
lines += ['',f"จำนวนข้อมูลอ้างอิง: {z['reference_hand_frames']} hand-frames, {z['n_points']} จุดเหมือนกันทุกวิธี",'',
'## ผลรายคลิป','| คลิป | raw RMSE | body_anchor RMSE | +temporal RMSE |','|---|---:|---:|---:|']
for cid,label in [('000','abdomen'),('001','hello'),('002','love')]:
    vals=[m['per_clip'][v][cid]['rmse_px'] for v in variants]
    lines.append(f"| {label} | "+' | '.join(f'{v:.2f}' for v in vals)+' |')
for v in variants[1:]:
    c=[a for a in m['changes'] if a['variant']==v]
    better=sum(a['new_mean_error']<a['old_mean_error']-1e-8 for a in c)
    worse=sum(a['new_mean_error']>a['old_mean_error']+1e-8 for a in c)
    lines.append(f"\n{v}: ในมือที่ถูกสลับและมีอ้างอิง {len(c)} hand-frames ดีขึ้น {better}, แย่ลง {worse} (วัด mean joint error)")
lines+=['','## ข้อจำกัดและข้อสรุป',
'- นี่เป็นผลจาก SDPose FP16 UNet บนคลิปที่เคยดูผล pilot แล้ว เป็นการพัฒนาวิธี ไม่ใช่ชุดทดสอบอิสระ',
'- ยังไม่ได้รันโมเดลรู้จำคำ จึงไม่มี Precision/Recall/F1 ใหม่',
'- ข้อมือของลำตัวอาจระบุฝั่งผิดด้วย โดยเฉพาะไขว้แขน กฎอิงข้อมือจึงอาจรักษาความผิดเดิมแทนที่จะแก้ได้',
'- การติดตามตำแหน่งอย่างเดียวไม่สร้างข้อมูลฝั่งมือที่ถูกต้องขึ้นมาได้ หากเริ่มคลิปผิดและตำแหน่งยังต่อเนื่อง',
'- ยังไม่ควรแทนระบบหลัก ต้องดูทั้งผลรวม ผลรายคลิป และตัวอย่างผิด ไม่เลือกเฉพาะเฟรมที่ดีขึ้น',
'- ภาพตัวอย่างเลือกเฟรมแรกที่กฎเปลี่ยนผลในแต่ละคลิป หากไม่เปลี่ยนเลือกกลางคลิป ไม่เลือกจากค่า error']
valid=sum(v['body_anchor']['reason']=='cost_comparison' for v in d)
love=[i for i,r in enumerate(rows) if r['id']=='002']
failcounts={str(j):sum(rows[i]['scores'][j]<.3 for i in love) for j in (5,6,7,8,9,10)}
diag={'eligible_frames':valid,'abstained_frames':len(rows)-valid,
      'love_frames':len(love),'love_body_invalid_frames':sum(not d[i]['body_anchor']['body_valid'] for i in love),
      'love_body_confidence_failures':failcounts,
      'raw_equals_anchor':bool(np.array_equal(x['raw'],x['body_anchor'])),
      'raw_equals_temporal':bool(np.array_equal(x['raw'],x['body_anchor_temporal']))}
(O/'diagnostics.json').write_text(json.dumps(diag,indent=2),encoding='utf-8')
lines+=['','## สาเหตุที่ผลไม่เปลี่ยน',
f"- เข้าเปรียบเทียบ cost จริง {valid} เฟรม; งดตัดสินใจ {len(rows)-valid} เฟรม",
f"- love ไม่ผ่าน body gate {diag['love_body_invalid_frames']}/{len(love)} เฟรม; จำนวนเฟรม confidence <0.30 ตาม COCO joint ID: {failcounts}",
'- พิกัดผลลัพธ์ทั้งสองวิธีเท่ากับ raw ทุกจุดจริง ไม่ใช่แค่คะแนนที่ปัดเศษแล้วเท่ากัน',
'- จึงสรุปได้ว่ากฎที่ทดลองยังไม่มีประโยชน์เพิ่มเติมในชุดนี้ ไม่ใช่ข้อพิสูจน์ว่าการติดตามทุกวิธีใช้ไม่ได้ เพราะ temporal variant ถูก body gate เดียวกันงดตัดสินใจใน love',
'- ไม่แนะนำแทนระบบหลักด้วยกฎนี้ ขั้นต่อไปที่สมเหตุผลคือทดสอบ body anchor จาก RTMW ที่มีอยู่โดยตรึงมือ SDPose และเกณฑ์เดิม เพื่อแยกปัญหาจากแหล่งพิกัดลำตัว ไม่ลดเกณฑ์ตามคะแนน love']
edges=[(5,7),(7,9),(6,8),(8,10),(5,6)]
for b in (91,112):
    for chain in ((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)):
        edges.extend((b+a,b+c) for a,c in zip(chain,chain[1:]))
panels=[]
for cid in ('000','001','002'):
    indices=[i for i,r in enumerate(rows) if r['id']==cid]
    changed=[i for i in indices if any(d[i][v]['swapped'] for v in variants[1:])]
    i=changed[0] if changed else indices[len(indices)//2]
    tiles=[]
    for v in variants:
        im=cv2.imread(str(R/f"{rows[i]['tag']}_input.png"))
        kp=x[v][i]; s=x[v+'_scores'][i]
        for a,b in edges:
            if min(s[a],s[b])>=.3:
                color=(0,140,0) if a<91 else ((200,90,0) if a<112 else (0,60,240))
                cv2.line(im,tuple(np.rint(kp[a]).astype(int)),tuple(np.rint(kp[b]).astype(int)),color,2)
        for j in (9,10,91,112):
            cv2.putText(im,str(j),tuple(np.rint(kp[j]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
        cv2.putText(im,f"{v} {rows[i]['label']} f{rows[i]['frame']}",(5,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),2)
        tiles.append(cv2.resize(im,(404,404)))
    panels.append(np.hstack(tiles))
cv2.imwrite(str(O/'comparison.png'),np.vstack(panels))
(O/'REPORT_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Report and comparison image saved.')
