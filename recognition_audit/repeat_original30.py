"""Repeat archived 30-clip protocol without replacing the original evidence."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import run_audit as original

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'repeat_original30_20260930'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def production():
    paths=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    return {p:sha(ROOT/p) for p in paths if (ROOT/p).is_file()}

def prepare():
    if OUT.exists():raise RuntimeError('Refuse to overwrite an existing repeat')
    manifest=_local_json_loads((HERE/'manifest.json').read_text())
    checks={p:dict(expected=h,actual=sha(HERE/p),matches=sha(HERE/p)==h) for p,h in manifest.items()}
    critical=['inputs.json','rendered.json','wlasl_stgcn.zip','wlasl_metadata.zip','upstream_st_gcn.py','upstream_graph_utils.py','upstream_pose_transforms.py']
    assert all(checks[p]['matches'] for p in critical),'Archived evidence differs from manifest'
    cases=_local_json_loads((HERE/'inputs.json').read_text())
    saved=_local_json_loads((HERE/'results.json').read_text())
    assert [c['phrase'] for c in cases]==saved['labels_evaluated']
    assert len(cases)==30 and len(saved['predictions'])==210
    assert sha(HERE/'wlasl_stgcn.zip')==saved['checkpoint_sha256']
    OUT.mkdir()
    shutil.copyfile(HERE/'inputs.json',OUT/'inputs.json')
    record=dict(hashes=checks,production_before=production(),original_results_sha256=sha(HERE/'results.json'),
        python=sys.version,torch=torch.__version__,numpy=np.__version__,labels=saved['labels_evaluated'],
        protocol='Frozen original inputs; fresh renderer capture and fresh inference; no new sample selection or hand modification')
    (OUT/'provenance.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({p:c['matches'] for p,c in checks.items()},indent=2))

def run():
    provenance=_local_json_loads((OUT/'provenance.json').read_text())
    old_capture=_local_json_loads((HERE/'rendered.json').read_text())
    fresh=_local_json_loads((OUT/'rendered.json').read_text())
    assert fresh==old_capture,'Fresh renderer capture differs: stop before claiming exact reproduction'
    assert sha(OUT/'inputs.json')==sha(HERE/'inputs.json')
    original.run(OUT)
    old=_local_json_loads((HERE/'results.json').read_text())
    new=_local_json_loads((OUT/'results.json').read_text())
    fields=['label','variant','prediction','top5']
    differences=[];max_probability_delta=0
    for a,b in zip(old['predictions'],new['predictions']):
        if any(a[k]!=b[k] for k in fields):differences.append(dict(old=a,new=b))
        max_probability_delta=max(max_probability_delta,abs(a['target_probability']-b['target_probability']))
    assert len(new['predictions'])==len(old['predictions'])==210
    summary_fields=['variant','n','accuracy','top5','macro_precision','macro_recall','macro_f1']
    summaries_match=all(all(a[k]==b[k] for k in summary_fields) for a,b in zip(old['summaries'],new['summaries']))
    assert production()==provenance['production_before'],'Production changed during repeat'
    assert sha(HERE/'results.json')==provenance['original_results_sha256'],'Original results overwritten'
    check=dict(fresh_renderer_exact=True,predictions_compared=210,prediction_differences=differences,
        maximum_target_probability_delta=max_probability_delta,summaries_match=summaries_match,
        production_unchanged=True,original_results_unchanged=True)
    (OUT/'comparison.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    lines=['# ทำซ้ำ ST-GCN 30 คำเดิม — 30 กันยายน 2026','',
        'ใช้ inputs เดิมที่ตรวจ SHA-256 กับ manifest แล้ว วาด Renderer ใหม่และ inference ใหม่ ไม่สุ่มคำใหม่ ไม่ใช้ S50A1 หรือ fallback รุ่นทดลอง ไม่สร้างข้อมูลจาก SQLite ใหม่ในรอบนี้','',
        '| เงื่อนไข | ถูก/30 | Macro-P | Macro-R | Macro-F1 | เวลา median ms |',
        '|---|---:|---:|---:|---:|---:|']
    for s in new['summaries']:
        lines.append(f"| {s['variant']} | {round(s['accuracy']*s['n']/100)}/30 | {s['macro_precision']:.2f}% | {s['macro_recall']:.2f}% | {s['macro_f1']:.2f}% | {s['median_runtime_ms']:.2f} |")
    lines += ['',f"พิกัด Renderer ใหม่ตรงข้อมูลเดิมทุกจุด: True; คำตอบ/Top-5 ต่าง {len(differences)}/210 รายการ; probability คำเป้าหมายต่างสูงสุด {max_probability_delta:.12g}; metrics ตรงเดิม: {summaries_match}",
        '', 'inputs, checkpoint, metadata และ upstream model/normalization ตรง manifest เดิม แต่ run_audit.py และ render_hands.mjs มี hash ต่างจาก manifest จึงไม่อ้างว่าไฟล์ช่วยรันหรือ environment ทุกส่วนเหมือนเดิมแบบ bit-for-bit ความตรงของผลต้องอ่านจากการเปรียบเทียบจริงข้างต้น',
        '', 'เวลาประมวลผลเป็นค่ารันใหม่ ไม่จำเป็นต้องเท่าครั้งก่อน ไม่ใช้เวลาเป็นเงื่อนไขความตรงของคะแนน',
        '', 'ข้อจำกัดเดิมยังคงอยู่: หนึ่งคลิปต่อคำ, ไม่ทราบ train overlap, macro เฉพาะ 30 true labels, โมเดลเลือกคำจาก 2,000 classes; มือหลัง Renderer + ลำตัว payload ไม่ใช่ภาพ Avatar ทั้งตัว และคะแนนไม่ใช่เปอร์เซ็นต์ความเหมือนหรือความเข้าใจ ASL',
        '', 'ผลซ้ำเป็นหลักฐาน reproducibility บนข้อมูลเดิม ไม่ใช่หลักฐาน generalization หรือความแม่นยำเพิ่มขึ้น',
        '', 'รายละเอียดเวอร์ชันและ hash อยู่ใน provenance.json; ผลเทียบรายรายการอยู่ comparison.json และ predictions อยู่ results.json; ผลรอบเก่าและไฟล์ระบบหลักไม่ถูกแก้ไข']
    (OUT/'REPORT_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(check,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else run()
