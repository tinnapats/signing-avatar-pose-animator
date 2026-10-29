"""Geometry-only fallback. Does not import or bypass blocked PyTorch DLLs."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'step1b'
NAMES={'S0A50':'ขนาด 0 / ข้อมือ 50','S50A50':'ขนาด 50 / ข้อมือ 50','S1A50':'ขนาด 100 / ข้อมือ 50','S0A1':'ขนาด 0 / ข้อมือ 100','S50A1':'ขนาด 50 / ข้อมือ 100','S50A75':'ขนาด 50 / ข้อมือ 75','S1A1':'ระบบปัจจุบัน 100 / 100'}
def dist(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])
def sub(a,b): return [a[0]-b[0],a[1]-b[1]]
def percentile(values,p):
    v=sorted(values); index=(len(v)-1)*p; i=int(index); j=min(i+1,len(v)-1)
    return v[i]+(v[j]-v[i])*(index-i)

cases=_local_json_loads((HERE/'inputs.json').read_text())
fits=_local_json_loads((OUT/'reference_fit.json').read_text())
capture=_local_json_loads((OUT/'rendered.json').read_text())
lookup={(r['variant'],r['phrase']):r['samples'] for r in capture}
assert len(lookup)==7*len(cases)==210
prior=_local_json_loads((HERE/'step1/rendered.json').read_text())
old={(r['variant'],r['phrase']):r['samples'] for r in prior}
for variant in ['S0A1','S1A1']:
    for case in cases:
        current=lookup[variant,case['phrase']]; previous=old[variant,case['phrase']]
        assert len(current)==len(previous)
        for a,b in zip(current,previous):
            for side in ['left','right']:
                assert a['hands'][side].keys()==b['hands'][side].keys()
                for k,v in a['hands'][side].items(): assert dist(v,b['hands'][side][k])<1e-6
rows=[]
for variant in NAMES:
    position=[];shape=[];gaps=[]
    for case in cases:
        fit=fits[case['phrase']]; samples=lookup[variant,case['phrase']]
        assert len(samples)==len(case['source30'])==len(case['frameIndices'])
        for t,(source,sample) in enumerate(zip(case['source30'],samples)):
            ref=[[p[0]*fit['scale']+fit['offset'][0],p[1]*fit['scale']+fit['offset'][1]] for p in source]
            shoulder=dist(ref[11],ref[12])
            if shoulder<=1e-6 or not all(any(source[i]) for i in [11,12]): continue
            frame=case['payload']['frames'][case['frameIndices'][t]]
            pose={k['part']:k for k in frame['pose']['keypoints']}
            for side,offset in [('left',33),('right',54)]:
                if not all(any(p) for p in source[offset:offset+21]): continue
                palm=dist(ref[offset+9],ref[offset])
                if palm<=1e-6: continue
                out=sample['hands'][side]
                for i in range(21):
                    p=out.get(str(i))
                    position.append(dist(p,ref[offset+i])/shoulder if p is not None else math.inf)
                    if i: shape.append(dist(sub(p,out['0']),sub(ref[offset+i],ref[offset]))/palm if p is not None and '0' in out else math.inf)
                wrist=pose[side+'Wrist']
                if '0' in out and wrist['score']>=.2:
                    gaps.append(dist(out['0'],[wrist['position']['x'],wrist['position']['y']])/shoulder)
    rows.append(dict(variant=variant,points=len(position),pck5=100*sum(e<=.05 for e in position)/len(position),shape_pck20=100*sum(e<=.2 for e in shape)/len(shape),gap_mean=100*statistics.mean(gaps),gap_p95=100*percentile(gaps,.95),gap_within2=100*sum(e<=.02 for e in gaps)/len(gaps),new_f1=None))
before=_local_json_loads((OUT/'production_before.json').read_text())
after={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines() if (ROOT/f).is_file()}
assert before==after,'Production files changed; do not attribute changes to experiment.'
(OUT/'geometry_results.json').write_text(json.dumps(dict(results=rows,production_unchanged=True,baseline_coordinates_verified=True,f1_status='Blocked: Windows Application Control WinError 4551 loading torch_global_deps.dll'),indent=2),encoding='utf-8')
lines=['# ขั้น 1b: ผลด้านพิกัดครบ — F1 แบบใหม่ยังติดข้อจำกัดระบบ','',
 'ตรวจ 30 คำเดิม 7 เงื่อนไขจาก browser capture จริง ใช้ตัวคำนวณ Python standard library เพราะ Windows Application Control บล็อก PyTorch DLL (WinError 4551) ไม่ได้แก้นโยบายหรือใช้วิธีข้ามการบล็อก','',
 '| ระดับการปรับ (%) | PCK@5% ไหล่ | รูปนิ้ว PCK@20% ฝ่ามือ | Gap เฉลี่ย (% ไหล่) | Gap P95 (% ไหล่) | Gap ผ่านเกณฑ์ 2% |','|---|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append('| '+NAMES[r['variant']]+' | '+' | '.join(f'{r[k]:.2f}' for k in ['pck5','shape_pck20','gap_mean','gap_p95','gap_within2'])+' |')
lines+=['','## ขอบเขต','',
 '- มือ 37,422 จุด; ข้อมูลอ้างอิงเป็น source30 ที่ตรวจพบมือครบและใช้ global canvas fit เดียวกับ output ไม่ปรับมือให้ตรงกันภายหลัง',
 '- Gap วัดข้อมือมือกับข้อมือใน pose payload ไม่ใช่การตรวจขอบภาพแขนจริง',
 '- ตัวคำนวณ fallback ใช้ double precision แทนการ cast source เป็น float32; ตาราง baseline ควรตรงกับรอบก่อนเมื่อปัดทศนิยมสองตำแหน่ง',
 '- ระดับขนาด 50 คือ scale=1+0.5*(scale เดิม-1) ไม่ใช่ย่อมือเหลือครึ่งหนึ่ง; ข้อมือ 75 คือผสมตำแหน่งไปทางข้อมือแขน 75%',
 '- ตรวจ baseline S0A1 และ S1A1 เทียบ capture เดิมทุกจุด tolerance 1e-6; ผ่าน',
 '- hash ไฟล์ production ที่ tracked ก่อน–หลังตรงกัน; ไม่เปลี่ยนระบบหลัก',
 '- ผลเลือกค่าจากชุดเดิม ไม่ใช่ held-out test และไม่ใช่เปอร์เซ็นต์ความเข้าใจของผู้ใช้',
 '- F1 แบบใหม่ยังไม่ได้วัด ไม่เติมจากการคาดเดา ผลเก่า S0A1=70.00%, S1A1=73.33% เป็นผลรอบก่อนเท่านั้น',
 '- อ่าน geometry_results.json สำหรับผลเครื่อง; เปิด compare.html ดูภาพกลางท่าคู่กับ baseline', '',
 '## งานที่ยังเหลือ','',
 'ต้องให้ PyTorch โหลดได้ภายใต้นโยบายที่ผู้ดูแลอนุญาตก่อน แล้วรัน step1b_evaluate.py evaluate เพื่อวัด F1 ใหม่ ไม่ควรนำแบบใดไปใช้จริงจาก PCK อย่างเดียว','']
(OUT/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(rows,indent=2))
