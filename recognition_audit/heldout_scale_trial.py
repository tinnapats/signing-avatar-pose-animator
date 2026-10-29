"""Frozen 30-new-gloss confirmation of S50A1 versus S1A1. Never tune here."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import zipfile
import recognition_100 as data_tools

HERE=Path(__file__).resolve().parent
OUT=HERE/'heldout_scale_trial'
VARIANTS=[dict(id='S1A1',scale=1,anchor=1),dict(id='S50A1',scale=.5,anchor=1)]

def prepare():
    from step1_evaluate import hashes
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    excluded={r['phrase'] for r in json.loads((HERE/'inputs.json').read_text())}
    for name in ['validation_trial','broad_pair_trial','recognition_100']:
        excluded.update(json.loads((HERE/name/'protocol.json').read_text())['words'])
    excluded.update(['motivate','continue','earn'])
    meta=json.loads((HERE/'kaggle_reference/WLASL_v0.3.json').read_bytes())
    with zipfile.ZipFile(HERE/'wlasl_metadata.zip') as z: released=json.loads(z.read('splits/asl2000.json'))
    ids={(r['gloss'],i['video_id']) for r in released for i in r['instances'] if i['split']=='test'}
    pool={r['gloss']:sorted([i for i in r['instances'] if i['split']=='test' and (r['gloss'],i['video_id']) in ids],key=lambda i:i['video_id']) for r in meta if r['gloss'] not in excluded}
    pool={k:v for k,v in pool.items() if v}
    words=random.Random(20261001).sample(sorted(pool),30)
    protocol=dict(seed=20261001,words=words,excluded=sorted(excluded),candidates={w:pool[w] for w in words},
        variants=VARIANTS,production_hashes=hashes(),primary='Per-clip mean 2D source-to-rendered point error normalized by source shoulder width; equal clip weighting',
        gate='No added pose-wrist gap >1e-8 shoulder widths; exact baseline payload reproduction. No production deployment from geometry alone.',
        analysis='Paired difference S50A1 minus S1A1; bootstrap 95% CI over clips, 10000 resamples seed20261002. Report all clips, wins/losses, coverage and PCK secondarily.',
        stopping='Exactly these 30 words; first available released-test video ID per word; no sample replacement based on score, no optional stopping, no parameter tuning.',
        limitation='New glosses relative to experiments listed; not guaranteed new signers or unseen by recognizer training. Geometry only, not human comprehension.',
        script_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'step1b_render.mjs',HERE/'geometry_100.py',HERE/'anchored_scale_trial.py',HERE/'paired_video_trial.py']})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));print('Frozen',words,flush=True)

def download():
    data_tools.OUT=OUT;data_tools.download()

def extract():
    data_tools.pilot.OUT=OUT;data_tools.pilot.extract()

def build():
    # Same build implementation; only frozen sample size differs.
    source=inspect.getsource(data_tools.build).replace('len(cases)==100','len(cases)==30')
    ns=dict(vars(data_tools),OUT=OUT);exec(compile(source,'heldout_build','exec'),ns);ns['build']()

def render():
    target=OUT/'comparison';target.mkdir(exist_ok=True)
    env=dict(os.environ,AUDIT_OUTPUT=str(target),AUDIT_INPUTS=str(OUT/'inputs.json'),AUDIT_VARIANTS=json.dumps(VARIANTS),AUDIT_REVIEW_FRAMES='1')
    subprocess.run(['C:/Users/newtu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe',str(HERE/'step1b_render.mjs')],env=env,check=True)
    captures=json.loads((target/'rendered.json').read_text())
    baseline=[dict(phrase=r['phrase'],samples=r['samples']) for r in captures if r['variant']=='S1A1']
    assert len(baseline)==30
    (OUT/'rendered.json').write_text(json.dumps(baseline))

def evaluate():
    import geometry_100 as geo
    import anchored_scale_trial as comparison
    import numpy as np
    from step1_evaluate import hashes
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert hashes()==protocol['production_hashes']
    for name,digest in protocol['script_hashes'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    geo.DATA=OUT;geo.OUT=OUT/'baseline_geometry';geo.main()
    target=OUT/'comparison'
    (target/'protocol.json').write_text(json.dumps(protocol,indent=2))
    source=inspect.getsource(comparison.evaluate)
    assert source.count("HERE/'geometry_100/results.json'")==2
    source=source.replace("HERE/'geometry_100/results.json'","HELDOUT/'baseline_geometry/results.json'").replace('len(lookup)==300','len(lookup)==60').replace('100 คลิปชุดพัฒนาเดิม','30 คลิปคำใหม่').replace('300 sequences','60 sequences')
    ns=dict(vars(comparison),DATA=OUT,OUT=target,HELDOUT=OUT,VARIANTS=VARIANTS)
    exec(compile(source,'heldout_comparison_same_metrics','exec'),ns);ns['evaluate']()
    result=json.loads((target/'results.json').read_text())
    differences=np.array([r['metrics']['S50A1']['distance_mean']-r['metrics']['S1A1']['distance_mean'] for r in result['perword']])
    assert len(differences)==30 and np.isfinite(differences).all()
    ci=np.quantile(np.random.default_rng(20261002).choice(differences,size=(10000,30)).mean(axis=1)*100,[.025,.975])
    stats=dict(n=30,mean_difference_shoulder_percentage_points=float(differences.mean()*100),bootstrap95=ci.tolist(),better=int((differences < -1e-9).sum()),worse=int((differences > 1e-9).sum()),unchanged=int((abs(differences)<=1e-9).sum()))
    (OUT/'statistics.json').write_text(json.dumps(stats,indent=2))
    lines=['# ยืนยันแบบกลางบน 30 คำใหม่','',
        'ตรึง S50A1 ก่อนเห็นผล ไม่เปลี่ยนข้อมือ ไม่ปรับรายคำ ไม่แก้ระบบหลัก ไม่มีคะแนน recognizer เป็นเกณฑ์คัดคำ', '',
        '| วิธี | ระยะผิดเฉลี่ยแบบให้น้ำหนักแต่ละคำเท่ากัน (% ไหล่) | PCK รวม | Wrist gap สูงสุด (% ไหล่) |','|---|---:|---:|---:|']
    for v in ['S1A1','S50A1']:
        avg=np.mean([r['metrics'][v]['distance_mean'] for r in result['perword']])*100;s=result['summaries'][v]
        lines.append(f"| {v} | {avg:.3f}% | {s['position_pck5_pct']:.2f}% | {s['wrist_gap_max']*100:.8f}% |")
    lines+=['',f"แบบกลางดีกว่า {stats['better']} คำ แย่กว่า {stats['worse']} คำ เท่ากัน {stats['unchanged']} คำ",
        f"ผลต่างเฉลี่ย (กลาง−เดิม) {stats['mean_difference_shoulder_percentage_points']:.3f} จุดเปอร์เซ็นต์ของไหล่; paired bootstrap 95% CI [{ci[0]:.3f}, {ci[1]:.3f}] ค่าลบคือ error ลดลง", '',
        'นี่เป็นการยืนยันด้านพิกัด 2D ของ candidate ที่เลือกไว้ ไม่ใช่ความเข้าใจภาษามือหรือความสมส่วนที่มนุษย์ให้คะแนน',
        '30 คำให้น้ำหนักเท่ากันสำหรับ primary metric ต่างจากตาราง pooled เดิมที่ถ่วงตามจำนวนจุด ห้ามเปรียบตัวเลขข้ามตารางตรง ๆ',
        'Bootstrap ระดับคลิปไม่คำนึงความสัมพันธ์ข้ามคลิปของผู้แสดงคนเดียวกัน ตัวอย่างขนาดเล็กและต้นทางอาจมี noise',
        'ชุดนี้เป็นคำใหม่ ไม่ใช่การรับรองว่าเป็นผู้แสดงใหม่ ไม่ได้สุ่มคำซ้ำจนผลดี และไม่เลือกคลิปจากคะแนน',
        'ยังไม่ deploy ต้องตรวจภาพและทดสอบผู้รู้ ASL ตามแผนมนุษย์ที่แยกไว้ก่อนอ้างผลความเข้าใจ', '']
    (OUT/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8')
    assert hashes()==protocol['production_hashes']
    (OUT/'evidence.json').write_text(json.dumps({str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'protocol.json',OUT/'manifest.json',OUT/'inputs.json',OUT/'rendered.json',target/'results.json',OUT/'statistics.json']},indent=2))
    print('\n'.join(lines))

if __name__=='__main__':globals()[sys.argv[1]]()
