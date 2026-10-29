"""Summarize the controlled experiment, including paired uncertainty."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'preprocessing_control'

def main():
    result=_local_json_loads((OUT/'results.json').read_text())
    manifest=_local_json_loads((HERE/'recognition_100/manifest.json').read_text())
    old,new=result['summaries']
    rows=result['predictions']
    changes=np.array([int(rows[i+1]['label']==rows[i+1]['prediction'])-int(rows[i]['label']==rows[i]['prediction']) for i in range(0,len(rows),2)])
    rng=np.random.default_rng(20260930)
    bootstrap=np.mean(rng.choice(changes,size=(10000,len(changes)),replace=True),axis=1)*100
    ci=np.quantile(bootstrap,[.025,.975]).tolist()
    g=len(result['gained']); l=len(result['lost']); discordant=g+l
    p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(g,l)+1))/2**discordant) if discordant else 1.
    coverage=[]
    for item in manifest:
        fresh=_local_json_loads((OUT/(item['metadata']['video_id']+'.json')).read_text())
        for side in ['pose','left_hand','right_hand']:
            coverage.append(dict(word=item['word'],side=side,frames=fresh['frames'],
                old=item['extraction']['detected_frames'][side],new=fresh['detected_frames'][side]))
    stats=dict(accuracy_difference_percentage_points=int(changes.sum()),paired_bootstrap95=ci,
               exact_mcnemar_p=p,gained=g,lost=l,coverage=coverage)
    (OUT/'statistics.json').write_text(json.dumps(stats,indent=2))
    lines=['# ทดลองควบคุม preprocessing: Holistic complexity 1 เทียบ 2','',
        '100 คลิปเดิม โมเดล ST-GCN เดิม FPS เดิม ไม่มีการปรับ Avatar หรือเลือกคำตามคะแนน เปลี่ยนเฉพาะ model_complexity จาก 1 เป็น 2', '',
        '| วิธีสกัด | ทายถูก/100 | Top-5 /100 | Macro Precision | Macro Recall | Macro-F1 |',
        '|---|---:|---:|---:|---:|---:|']
    for s in [old,new]:
        lines.append(f"| {s['variant']} | {s['correct']} | {s['top5']} | {s['macro_precision']:.2f}% | {s['macro_recall']:.2f}% | {s['macro_f1']:.2f}% |")
    lines+=['', f'คำที่เปลี่ยนจากผิดเป็นถูก {g} คำ; ถูกเป็นผิด {l} คำ',
        f'ผลต่าง accuracy {int(changes.sum()):+d} จุดเปอร์เซ็นต์; paired bootstrap 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]; exact McNemar p={p:.4f}',
        'ช่วงความไม่แน่นอนข้างต้น resample รายคลิป (หนึ่งคลิปต่อคำ) ไม่ครอบคลุมความไม่แน่นอนทุกแบบ เช่น ความสัมพันธ์ระหว่างผู้แสดง', '',
        '## ความถี่ที่ตรวจพบจุด (ไม่ใช่ความถูกต้องของจุด)',
        '| ส่วน | complexity 1 | complexity 2 |','|---|---:|---:|']
    for side in ['pose','left_hand','right_hand']:
        subset=[c for c in coverage if c['side']==side]; total=sum(c['frames'] for c in subset)
        lines.append(f"| {side} | {100*sum(c['old'] for c in subset)/total:.2f}% | {100*sum(c['new'] for c in subset)/total:.2f}% |")
    lines+=['', '## ข้อสรุปของรอบนี้',
        'complexity 2 ไม่ได้ช่วยเพิ่มคะแนนในชุดนี้: 9/100 เทียบ 14/100 เดิม จึงไม่เปลี่ยนระบบไปใช้ค่านี้จากหลักฐานรอบนี้',
        'ยังสรุปทั่วไปไม่ได้ว่า complexity 2 แย่กว่า: ช่วงความไม่แน่นอนครอบคลุม 0 และ McNemar p=0.125 ความต่างเพียงปัจจัยนี้ยังอธิบายคะแนนต่ำทั้งหมดไม่ได้',
        'ทั้งสองแบบยังรู้จำต้นทางต่ำ จึงพักการใช้ ST-GCN เป็นตัวชี้วัดหลักของ Avatar และไม่รัน downstream เพื่อเลือกคะแนนที่ดูดี', '',
        '## ตรวจความน่าเชื่อถือ',
        '- รัน complexity 1 ซ้ำ: top-5 และ target probability ตรงผลเดิมครบ 100 คลิป',
        '- SHA-256 วิดีโอตรงต้นฉบับ; จำนวนเฟรมและ FPS เท่ากัน; โหลด checkpoint เดิมแบบ strict',
        '- เลือก 27 จุด / normalize ด้วยหัวไหล่ / x-y / ลำดับคลาสตามวิธีเดิม ไม่เปิด z หรือ confidence เพิ่ม',
        '- Macro เฉลี่ย 100 target labels, zero division=0; ทายนอกกลุ่มยังนับผิด',
        '- ผลนี้วัดความสามารถรู้จำข้อมูลต้นทาง ไม่ใช่ความเข้าใจของผู้ใช้ และไม่ใช่คะแนนภาพ Avatar',
        '- ความต่างจาก reference ที่ยังเหลือ: วิดีโอ resized, เวอร์ชัน MediaPipe, batching/padding และขอบเขตคลิป',
        '- ชุด 100 คลิปนี้เป็นชุดพัฒนาวิธีประเมินแล้ว ห้ามอ้างเป็น untouched final test',
        '- พัก BiLSTM จนตรวจเวอร์ชันโค้ด/แกนข้อมูลที่ตรงกับ checkpoint ได้', '',
        '## ขั้นต่อไป',
        'อย่านำคะแนนต่ำของ recognizer ไปแปลเป็นเปอร์เซ็นต์ Avatar ผิด ควรตรวจ reference pose ที่ผู้พัฒนาโมเดลใช้หรือ pipeline ต้นฉบับให้ทำซ้ำได้ ก่อนรับรองโมเดลเป็นเครื่องมือวัด',
        'แยกวัดความตรงเชิงเรขาคณิต (จุด/มุม/ความยาวกระดูก โดยควบคุมเวลาและพิกัด) จากความเข้าใจภาษามือ (ให้ผู้รู้ ASL ดูภาพโดยไม่แสดงคำเฉลย)',
        'กำหนดและตรึงวิธีล่วงหน้า แล้วใช้ชุดใหม่ยืนยัน ไม่ปรับรายคำ ไม่เลือกเฉพาะโมเดลหรือเงื่อนไขที่คะแนนสูง', '']
    (OUT/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'evidence.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='evidence.json'},indent=2))
    print('\n'.join(lines))

if __name__=='__main__':main()
