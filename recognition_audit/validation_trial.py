"""Fixed candidate versus production on 60 disjoint words. Audit artifacts only."""
import json
import random
import sys
import math
import hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import step1_evaluate as base
from run_audit import HERE, ROOT, pipeline, csv_array

OUT = HERE/'validation_trial'

def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists(), 'Do not overwrite a frozen trial'
    excluded = [c['phrase'] for c in json.loads((HERE/'inputs.json').read_text())]
    clips = pipeline.discover_clips(ROOT/'SLclean.sqlite3')
    available = sorted({Path(n).stem for n in clips}-set(excluded))
    words = random.Random(20260927).sample(available, 60)
    protocol = dict(seed=20260927, words=words, excluded=excluded, variants={'S1A1':dict(scale=1,anchor=1),'S50A75':dict(scale=.5,anchor=.75)}, primary='paired top-1 recognition accuracy', secondary=['macro precision/recall/F1','hand PCK','wrist gap'], note='Frozen before preparing or scoring. Disjoint from tuning words, not guaranteed disjoint from model training.')
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    (OUT/'production_before.json').write_text(json.dumps(base.hashes()), encoding='utf-8')
    cases, fits = [], {}
    for name in words:
        affine = {}
        original = pipeline.fit_frames_to_canvas
        def capture(frames,width,height):
            a=np.array([[k['position']['x'],k['position']['y']] for f in frames for k in f['pose']['keypoints']])
            result=original(frames,width,height)
            b=np.array([[k['position']['x'],k['position']['y']] for f in result for k in f['pose']['keypoints']])
            scale,dx=np.linalg.lstsq(np.column_stack([a[:,0],np.ones(len(a))]),b[:,0],rcond=None)[0]
            dy=np.mean(b[:,1]-a[:,1]*scale)
            assert np.max(np.abs(b-(a*scale+[dx,dy])))<1e-6
            affine.update(scale=float(scale),offset=[float(dx),float(dy)])
            return result
        paths=pipeline.resolve_selected_paths(clips,ROOT/'SLclean.sqlite3',[name+'.csv'],'')
        with patch.object(pipeline,'fit_frames_to_canvas',capture):
            payload=pipeline.build_payload(ROOT/'SLclean.sqlite3',[name+'.csv'],'',30,513,513,0,0,2,1.2,2)
        source=csv_array(pipeline.concat_clips(paths,513,513,0,target_fps=30))
        indices=list(range(0,len(payload['frames']),2))
        assert len(source)==len(indices)
        cases.append(dict(phrase=name,payload=payload,source30=source.tolist(),frameIndices=indices))
        fits[name]=affine
        print('prepared',name,flush=True)
    assert base.hashes()==json.loads((OUT/'production_before.json').read_text())
    (OUT/'inputs.json').write_text(json.dumps(cases),encoding='utf-8')
    (OUT/'reference_fit.json').write_text(json.dumps(fits),encoding='utf-8')

def evaluate():
    # Reuse identical geometry/model definitions; replace only dataset/output paths
    # and omit historical capture comparison (new words have no old captures).
    source=(HERE/'step1_evaluate.py').read_text(encoding='utf-8')
    source=source.replace("(HERE/'inputs.json')", "(OUT/'inputs.json')")
    a=source.index("    previous={r['phrase']")
    b=source.index('    network,normalize,labels=load_model()',a)
    source=source[:a]+"    baseline_exact=None\n"+source[b:]
    scope={'__name__':'validation_evaluator'}
    exec(compile(source,'validation_evaluator','exec'),scope)
    scope['OUT']=OUT
    scope['NAMES']={'S1A1':'ระบบเดิม','S50A75':'ขนาด 50% / ยึดข้อมือ 75%'}
    scope['evaluate']()
    result=json.loads((OUT/'results.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text())
    rows=result['predictions']
    by={(r['variant'],r['label']):r['prediction']==r['label'] for r in rows}
    words=protocol['words']
    assert len(words)==60 and not set(words)&set(protocol['excluded'])
    assert len(by)==120
    gain=sum(by['S50A75',w] and not by['S1A1',w] for w in words)
    loss=sum(by['S1A1',w] and not by['S50A75',w] for w in words)
    n=gain+loss
    p=min(1,2*sum(math.comb(n,k) for k in range(min(gain,loss)+1))/2**n) if n else 1
    delta=np.array([int(by['S50A75',w])-int(by['S1A1',w]) for w in words])
    rng=np.random.default_rng(20260927)
    ci=np.percentile(delta[rng.integers(0,60,(10000,60))].mean(axis=1)*100,[2.5,97.5]).tolist()
    stats=dict(gain=gain,loss=loss,accuracy_delta_pp=float(delta.mean()*100),paired_bootstrap95_pp=ci,mcnemar_exact_p=p)
    result['paired_statistics']=stats
    result['evidence_sha256']={f:hashlib.sha256((OUT/f).read_bytes()).hexdigest() for f in ['protocol.json','inputs.json','rendered.json']}
    result['audit_script_sha256']={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['validation_trial.py','step1b_render.mjs','step1_evaluate.py','run_audit.py']}
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# ชุดยืนยันเพิ่มเติม: 60 คำใหม่','', 'กำหนดวิธีและสุ่มคำก่อนวัดผล ไม่ซ้ำ 30 คำที่ใช้เลือกค่า; ไม่รับรองว่าไม่ซ้ำชุดฝึก ST-GCN','', '| วิธี | ถูก/60 | Precision | Recall | F1 | PCK@5% | Gap ข้อมือเฉลี่ย (% ไหล่) |','|---|---:|---:|---:|---:|---:|---:|']
    for s in result['summaries']:
        lines.append(f"| {s['variant']} | {s['correct']} | {s['macro_precision']:.2f}% | {s['macro_recall']:.2f}% | {s['macro_f1']:.2f}% | {s['pck5']:.2f}% | {s['wrist_gap_mean_percent']:.2f}% |")
    lines += ['',f'ทายถูกเพิ่ม {gain} คำ แต่เสียจากเดิม {loss} คำ; Accuracy ต่าง {stats["accuracy_delta_pp"]:.2f} จุดเปอร์เซ็นต์', f'ช่วงความเชื่อมั่น paired bootstrap 95% ของผลต่าง: {ci[0]:.2f} ถึง {ci[1]:.2f} จุดเปอร์เซ็นต์ (10,000 resamples, seed 20260927)', f'Exact paired McNemar p = {p:.4f}', '', '## ข้อจำกัด', '- อ่านพิกัดมือจาก renderer จริง ร่วมกับลำตัว payload ไม่ใช่จำแนกภาพ avatar หรือความเข้าใจของมนุษย์', '- Macro เฉลี่ยเฉพาะ 60 target classes; zero division = 0; หนึ่งคลิปต่อคำ', '- ยังมีข้อจำกัดจากข้อมูลชุดเดียวกันและ train overlap ที่ไม่ทราบ', '- PCK คือความใกล้ต้นทาง ไม่ใช่เปอร์เซ็นต์ภาษามือถูกต้อง; gap วัดจุดข้อมือ ไม่ใช่ขอบผิวภาพ', '- ตรวจสูตรฉายกับจุดวาดทุกจุด tolerance 1e-6; ไม่มี historical capture สำหรับคำใหม่', '- Hash ไฟล์ tracked ก่อนและหลังเท่ากัน ไม่แก้ production', '']
    (OUT/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(stats),flush=True)

if __name__=='__main__':
    prepare() if sys.argv[1]=='prepare' else evaluate()
