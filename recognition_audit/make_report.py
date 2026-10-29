"""Validate saved predictions and generate a transparent pilot report."""
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
result=json.loads((HERE/'results.json').read_text())
rows=result['predictions']
labels=result['labels_evaluated']
assert len(labels)==len(set(labels))==30
assert len(rows)==30*7
groups={name:[r for r in rows if r['variant']==name] for name in {r['variant'] for r in rows}}
native={r['label']:r for r in groups['source_native']}
for row in groups['corrupt_unselected_joints']:
    baseline=native[row['label']]
    assert row['top5']==baseline['top5']
    assert abs(row['target_probability']-baseline['target_probability'])<1e-8
for summary in result['summaries']:
    subset=groups[summary['variant']]
    assert len(subset)==30
    assert abs(summary['accuracy']-sum(r['label']==r['prediction'] for r in subset)/30*100)<1e-9
confusion={v:[dict(actual=a,predicted=b,count=n) for (a,b),n in sorted(Counter((r['label'],r['prediction']) for r in g).items())] for v,g in groups.items()}
(HERE/'confusion.json').write_text(json.dumps(confusion,indent=2),encoding='utf-8')
names={
 'source_native':'ต้นทางตามเฟรมเดิม',
 'source_30fps':'ต้นทางหลังปรับเป็น 30 FPS',
 'processed_30fps':'หลังประมวลผล ก่อน Renderer (เก็บทุก 2 เฟรมจาก 60 FPS)',
 'rendered_hands_payload_body':'มือจาก Renderer + ลำตัวจาก payload',
 'freeze_fingers':'ต้นทางแต่ค้างรูปนิ้วเทียบข้อมือ',
 'shuffle_frames':'ต้นทางแต่สลับลำดับเฟรม',
 'corrupt_unselected_joints':'ต้นทางแต่ทำจุดนิ้วที่โมเดลไม่ใช้ให้ผิด',
}
lines=['# ST-GCN / WLASL2000: ผลทดลองนำร่องบน tinnapat', '',
 '## ข้อสรุป', '',
 'รัน checkpoint จริงแล้ว ไม่ได้แต่งตัวเลข: ใช้ได้เป็นตัวประเมินเสริมว่าท่ายังถูกจำแนกเป็นคำเป้าหมายหรือไม่ แต่ไม่ผ่านเงื่อนไขการตรวจความถูกต้องของนิ้วทุกข้อหรือ Avatar ทั้งภาพด้วยตัวเดียว', '',
 '## ผล 30 คำ (หนึ่งคลิปต่อคำ)', '',
 '| เงื่อนไข | ถูก/30 | Accuracy | Macro-P | Macro-R | Macro-F1 | Top-5 | เวลา inference มัธยฐาน (ms) |',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
for name in names:
    s=next(s for s in result['summaries'] if s['variant']==name)
    correct=sum(r['label']==r['prediction'] for r in groups[name])
    lines.append('| '+names[name]+' | '+str(correct)+' | '+' | '.join(f'{s[k]:.2f}%' for k in ['accuracy','macro_precision','macro_recall','macro_f1','top5'])+f" | {s['median_runtime_ms']:.2f} |")
lines += ['', '## ความหมายและข้อจำกัดที่ต้องใช้คู่กับตาราง', '',
 '- 73.33% ในแถวมือจาก Renderer คือการจำแนกถูก 22/30 คลิปในชุดนี้ ไม่ใช่ภาพเหมือนต้นฉบับ 73.33% และไม่ใช่ผลทั้ง 2,000 คำ',
 '- โมเดลจำแนกจาก 2,000 classes เสมอ ไม่ได้จำกัดตัวเลือกให้เหลือ 30 คำ; 30 คำเป็นเพียงชุดทดสอบ',
 '- Macro-P/R/F1 เฉลี่ยเหนือ 30 คำที่มีตัวอย่างอ้างอิง ตั้ง zero_division=0 คำตอบนอกชุดทดสอบยังนับว่าผิด แต่ไม่เพิ่ม class ที่ไม่มี support เข้ามาใน macro average',
 '- คะแนน Macro-P/R/F1 เท่ากันในรอบนี้เพราะแต่ละคำมีเพียงหนึ่งตัวอย่างและรูปแบบคำตอบผิดที่เกิดขึ้น ไม่ใช่สูตรทั้งสามเหมือนกันทั่วไป; ดู confusion.json',
 '- เวลาเป็น forward pass + softmax + จัด top-5 บน CPU หลัง warmup ไม่รวมอ่าน SQLite เตรียมข้อมูล วาดภาพ หรือโหลดโมเดล และเป็นการวัดนำร่อง ไม่ใช่ benchmark ความเร็วที่ควบคุมทุกปัจจัย',
 '- แถว Renderer ใช้พิกัดปลายเส้นมือจริงที่ Paper.js วาดและชดเชยกล้องแล้ว แต่ 7 จุดร่างกายยังมาจาก payload จึงเป็นการตรวจมือหลัง Renderer ไม่ใช่ end-to-end visual recognition ของ Avatar ทั้งตัว',
 '- ไม่ส่งคำเป้าหมายหรือข้อความ Output เข้า network; label ใช้เทียบคะแนนภายหลังเท่านั้น',
 '- ยังไม่มีข้อมูล video_id/signer_id/split ต้นฉบับในฐานข้อมูลเพียงพอให้ตัดความเสี่ยงว่าคลิปซ้ำกับ training set ของ checkpoint ได้ ผลนี้ห้ามอ้างเป็นผล generalization บน unseen data',
 '- ทดสอบ 8 คำที่ใช้อยู่เดิม + 22 คำสุ่มโดย seed 20260926 เลือกก่อนทำนาย ไม่คัดทิ้งคำที่โมเดลทายผิด',
 '- ยังไม่ทดสอบการจำแนกประโยคต่อเนื่องหรือให้ผู้เชี่ยวชาญภาษามือประเมิน',
 '- การตรึงนิ้วหรือสลับเฟรมเป็น stress test ไม่ได้แปลว่าทุกท่าที่ถูกเปลี่ยนจะผิดความหมายเสมอ เช่นท่าค้างอาจไวต่อการสลับเฟรมน้อย', '',
 '## จุดบอดที่ยืนยันได้', '',
 'เมื่อจงใจย้ายข้อต่อนิ้วที่ไม่อยู่ใน 27 จุด input ให้ผิด ผล Top-5 และ probability ของคำเป้าหมายเหมือนเดิมครบ 30/30 คลิป เพราะจุดเหล่านั้นไม่เข้าตัวโมเดล เป็นข้อจำกัดเชิงโครงสร้าง ไม่สามารถแก้โดยการเปลี่ยน threshold ของ F1 ต้องใช้การวัดข้อต่อมือครบ 21 จุดต่อข้างร่วมด้วย', '',
 '## วิธีทดลอง', '',
 'ใช้ config และน้ำหนัก ST-GCN จาก release checkpoints_v1 ของ AI4Bharat/OpenHands ใช้ x/y 27 จุดและ CenterAndScaleNormalize ของผู้พัฒนา โหลด tensor strict=True ไม่มี fine-tune มีเพียงการแทน OmegaConf ด้วย dict และแก้ import ในตัว loader ชั่วคราว ไม่เปลี่ยนสมการโมเดล ใช้ weights_only=True และ inert callback stand-ins แทนการโหลด callback ของ training', '',
 'ใช้ Python/PyTorch CPU ที่มีอยู่ ไม่ติดตั้ง package เพิ่มและไม่แก้ environment ของระบบหลัก ไฟล์โมเดลและผลอยู่ใน recognition_audit ทั้งหมด ไม่แก้ฐานข้อมูล ไม่แก้ Renderer และไม่แตะ branch main หรือโฟลเดอร์ newton ในรอบนี้', '',
 '## รายคำ: ต้นทาง → ก่อน Renderer → มือหลัง Renderer', '',
 '| เป้าหมาย | ต้นทาง | หลังประมวลผล | มือหลัง Renderer |', '|---|---|---|---|']
for label in labels:
    found={r['variant']:r['prediction'] for r in rows if r['label']==label}
    lines.append('| '+label+' | '+' | '.join(found[k] for k in ['source_native','processed_30fps','rendered_hands_payload_body'])+' |')
lines += ['', '## ทำซ้ำ', '',
 'ใช้ Python environment ที่มี numpy, pandas, torch และ yaml; ใช้ Node/Vite/Playwright ตาม path ใน render_hands.mjs ซึ่งปรับผ่าน AUDIT_VITE, AUDIT_PLAYWRIGHT, AUDIT_BROWSER ได้', '',
 '1. python recognition_audit/run_audit.py prepare --count 30',
 '2. node recognition_audit/render_hands.mjs',
 '3. python recognition_audit/run_audit.py run',
 '4. python recognition_audit/make_report.py', '',
 'ผลตัวเลขรายคลิป: results.json; ตารางความสับสนแบบ sparse: confusion.json; ภาพกลางท่า: <คำ>.png; ข้อมูลทดสอบ: inputs.json; พิกัดมือที่วาด: rendered.json', '',
 'ต้องมี wlasl_stgcn.zip และ wlasl_metadata.zip จาก https://github.com/AI4Bharat/OpenHands/releases/tag/checkpoints_v1 ก่อนทำซ้ำ น้ำหนักและไฟล์ข้อมูลขนาดใหญ่ถูก ignore ใน Git แต่ไม่ได้ถูกลบ', '',
 'แหล่งโค้ด: https://github.com/AI4Bharat/OpenHands (license สำเนาอยู่ใน OpenHands_LICENSE.txt)', '',
 '## ขั้นต่อไปที่ยังไม่ได้ทำ', '',
 'หากต้องการประเมินภาพสุดท้ายทั้งตัว ต้องเพิ่มการสกัดข้อต่อจากวิดีโอ Avatar หรือโมเดลรับภาพ และตรวจความสำเร็จของการตรวจจับในภาพการ์ตูนแยกจากการจำแนกคำ รวมทั้งจัดชุดทดสอบที่พิสูจน์ได้ว่าไม่ซ้ำ training set ก่อนอ้างคะแนนวิจัยทั่วไป', '']
(HERE/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8')
tracked=['run_audit.py','render_hands.mjs','upstream_st_gcn.py','upstream_graph_utils.py','upstream_pose_transforms.py','wlasl_stgcn.zip','wlasl_metadata.zip','inputs.json','rendered.json']
(HERE/'manifest.json').write_text(json.dumps({name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in tracked},indent=2),encoding='utf-8')
print('Validated 210 predictions; blind-spot test identical on 30/30; report written.')
