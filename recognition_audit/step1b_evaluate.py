"""Graded hand scaling/anchoring; shares the verified step-1 evaluator."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json
import sys
import step1_evaluate as base

base.OUT=base.HERE/'step1b'
base.BASELINE_TOLERANCE=1e-6
base.NAMES={
 'S0A50':'ขนาด 0% / ยึดข้อมือ 50%',
 'S50A50':'ขนาด 50% / ยึดข้อมือ 50%',
 'S1A50':'ขนาด 100% / ยึดข้อมือ 50%',
 'S0A1':'ขนาด 0% / ยึดข้อมือ 100%',
 'S50A1':'ขนาด 50% / ยึดข้อมือ 100%',
 'S50A75':'ขนาด 50% / ยึดข้อมือ 75%',
 'S1A1':'ขนาด 100% / ยึดข้อมือ 100% (ปัจจุบัน)',
}
if sys.argv[1]=='prepare':
    base.OUT.mkdir(exist_ok=True)
    previous=_local_json_loads((base.HERE/'step1/production_before.json').read_text())
    assert previous==base.hashes(),'Production differs from step 1; regenerate references first'
    (base.OUT/'production_before.json').write_text(json.dumps(base.hashes()),encoding='utf-8')
    (base.OUT/'reference_fit.json').write_bytes((base.HERE/'step1/reference_fit.json').read_bytes())
    print('Snapshot saved; references validated against unchanged production.')
else:
    base.evaluate()
    report=base.OUT/'REPORT_TH.md'
    text=report.read_text(encoding='utf-8').replace('ขั้นที่ 1: แยกผลการปรับขนาดมือและการยึดข้อมือ','ขั้นที่ 1b: ปรับระดับขนาดมือและการยึดข้อมือ').replace('ด้วย 4 เงื่อนไข','ด้วย 7 เงื่อนไข').replace('step1_evaluate.py','step1b_evaluate.py').replace('step1_render.mjs','step1b_render.mjs')
    text=text.replace('ให้พิกัดตรงกับ audit ระบบปัจจุบันเดิมทุกเฟรม','ให้พิกัดตรงกับ audit ระบบปัจจุบันเดิมทุกเฟรมภายใน 1e-6 หน่วย (เผื่อความต่างจาก floating-point ในสูตรผสม)')
    text+='\n## ความหมายระดับการปรับ\n\nขนาด 50% หมายถึงใช้ครึ่งหนึ่งของการเปลี่ยนสัดส่วนจากระบบเดิม ไม่ใช่ย่อมือให้เหลือครึ่งหนึ่ง: scale = 1 + beta*(production_scale-1)\n\nยึดข้อมือ 50% หมายถึงตำแหน่งกึ่งกลางระหว่างข้อมือมือกับข้อมือแขน: wrist = hand_wrist + alpha*(body_wrist-hand_wrist)\n\nเป็นการเลือกค่าจากชุดทดลองเดิม ไม่ใช่ผลยืนยันบน held-out test set ห้ามสรุปว่าเป็นค่าที่ดีที่สุดทั่วไป; ยังไม่ทดสอบวิธีจำกัดช่วง scale แบบ clamp และไม่ได้ปรับ production\n'
    rows=_local_json_loads((base.OUT/'results.json').read_text())['summaries']
    text+='\n## ความต่อเนื่องบริเวณข้อมือ (ตัวชี้วัดพิกัด ไม่ใช่การตรวจขอบภาพ)\n\n| แบบ | Gap P95 (% ไหล่) | สัดส่วนที่ gap ไม่เกิน 2% ไหล่ |\n|---|---:|---:|\n'
    for r in rows:
        text+=f"| {base.NAMES[r['variant']]} | {r['wrist_gap_p95_percent']:.2f}% | {r['wrist_gap_within2_percent']:.2f}% |\n"
    text+='\n2% เป็นเกณฑ์ตรวจเชิงวิศวกรรมที่กำหนดสำหรับรอบนี้ ไม่ใช่เกณฑ์รับรองโดยผู้ใช้ภาษามือ\n'
    report.write_text(text,encoding='utf-8')
