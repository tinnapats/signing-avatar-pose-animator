"""Extend the original frozen 30-word ST-GCN protocol to 60 words."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import run_audit as audit
from repeat_original30 import production,sha

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'original_method60_20260930'

def prepare():
    assert not OUT.exists(),'Refuse to overwrite previous experiment'
    manifest=_local_json_loads((HERE/'manifest.json').read_text())
    for name in ['inputs.json','wlasl_stgcn.zip','wlasl_metadata.zip','upstream_st_gcn.py','upstream_graph_utils.py','upstream_pose_transforms.py']:
        assert sha(HERE/name)==manifest[name],name
    cases=_local_json_loads((HERE/'inputs.json').read_text())
    old=[c['phrase'] for c in cases];assert len(old)==len(set(old))==30
    db=ROOT/'SLclean.sqlite3';clips=audit.pipeline.discover_clips(db)
    available=sorted(Path(name).stem for name in clips)
    pool=[name for name in available if name not in old]
    new=random.Random(20260926).sample(pool,30)
    # Freeze selection before any new inference. Never replace misclassified clips.
    OUT.mkdir()
    provenance=dict(seed=20260926,original_words=old,added_words=new,pool=pool,
        selection='Keep original30, random.sample 30 from sorted remaining database keys; no prediction-based filtering',
        production_before=production(),original_results_sha256=sha(HERE/'results.json'),
        hashes={name:sha(HERE/name) for name in manifest},database_size=db.stat().st_size,
        database_mtime_ns=db.stat().st_mtime_ns)
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    # Same preprocessing calls and constants as original.prepare; retain old inputs verbatim.
    for label in new:
        paths=audit.pipeline.resolve_selected_paths(clips,db,[label+'.csv'],'')
        native=audit.csv_array(audit.pipeline.read_clip_csv(paths[0]))
        payload=audit.pipeline.build_payload(db,[label+'.csv'],'',30,513,513,0,0,2,1.2,2)
        source30=audit.csv_array(audit.pipeline.concat_clips(paths,513,513,0,target_fps=30))
        indexes=list(range(0,len(payload['frames']),2));assert len(indexes)==len(source30)
        cases.append(dict(phrase=label,payload=payload,frameIndices=indexes,native=native.tolist(),source30=source30.tolist()))
        print('prepared',label,len(native),flush=True)
    assert len(cases)==len({c['phrase'] for c in cases})==60
    assert production()==provenance['production_before']
    (OUT/'inputs.json').write_text(json.dumps(cases),encoding='utf-8')
    print('Frozen60:',old+new,flush=True)

def metrics(rows,labels):
    ps=[];rs=[];fs=[]
    for label in labels:
        tp=sum(r['label']==label and r['prediction']==label for r in rows)
        fp=sum(r['label']!=label and r['prediction']==label for r in rows)
        fn=sum(r['label']==label and r['prediction']!=label for r in rows)
        p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0
        ps.append(p);rs.append(r);fs.append(2*p*r/(p+r) if p+r else 0)
    return dict(n=len(rows),correct=sum(r['label']==r['prediction'] for r in rows),
        precision=100*float(np.mean(ps)),recall=100*float(np.mean(rs)),f1=100*float(np.mean(fs)))

def run():
    prov=_local_json_loads((OUT/'provenance.json').read_text())
    for name,h in prov['hashes'].items():assert sha(HERE/name)==h,'Audit dependency changed: '+name
    cases=_local_json_loads((OUT/'inputs.json').read_text());old_cases=_local_json_loads((HERE/'inputs.json').read_text())
    assert cases[:30]==old_cases
    captures=_local_json_loads((OUT/'rendered.json').read_text())
    assert len(captures)==60
    assert captures[:30]==_local_json_loads((HERE/'rendered.json').read_text()),'Original renderer changed'
    audit.run(OUT)
    results=_local_json_loads((OUT/'results.json').read_text());pred=results['predictions']
    assert len(pred)==420
    assert results['labels_evaluated']==prov['original_words']+prov['added_words']
    old=_local_json_loads((HERE/'results.json').read_text())
    lookup={(p['label'],p['variant']):p for p in pred}
    for p in old['predictions']:
        q=lookup[p['label'],p['variant']]
        assert all(p[k]==q[k] for k in ['prediction','top5','target_probability']),'Original30 changed'
    for label in results['labels_evaluated']:
        a=lookup[label,'source_native'];b=lookup[label,'corrupt_unselected_joints']
        assert a['top5']==b['top5'] and a['target_probability']==b['target_probability']
    assert production()==prov['production_before']
    assert sha(HERE/'results.json')==prov['original_results_sha256']
    db=ROOT/'SLclean.sqlite3'
    assert db.stat().st_size==prov['database_size'] and db.stat().st_mtime_ns==prov['database_mtime_ns']
    names=['source_native','source_30fps','processed_30fps','rendered_hands_payload_body','freeze_fingers','shuffle_frames','corrupt_unselected_joints']
    groups={}
    lines=['# ST-GCN วิธีเดิม ขยายเป็น 60 คำ — 30 กันยายน 2026','',
      'คง 30 คำเดิม เพิ่ม 30 คำสุ่มก่อนทำนายจากคีย์ที่เหลือในฐานข้อมูลด้วย seed 20260926 ไม่มีการคัดคำตามผล โมเดล checkpoint, input27 จุด x/y, normalization, preprocessing และ negative controls เหมือนเดิม ไม่ใช้ S50A1 หรือสูตร fallback ทดลอง','',
      'หนึ่งคลิปต่อคำ โมเดลจำแนกจาก 2,000 classes เสมอ ไม่ได้จำกัดให้เลือก 60 คำ macro เฉลี่ยเหนือ true labels ของกลุ่มนั้นและ zero_division=0; คำตอบนอกกลุ่มยังนับผิด','']
    for title,labels in [('รวม60',prov['original_words']+prov['added_words']),('เดิม30',prov['original_words']),('เพิ่ม30',prov['added_words'])]:
        lines += ['## '+title,'','| เงื่อนไข | ถูก/ทั้งหมด | Macro-P | Macro-R | Macro-F1 |','|---|---:|---:|---:|---:|']
        groups[title]={}
        for variant in names:
            rows=[p for p in pred if p['variant']==variant and p['label'] in labels]
            m=metrics(rows,labels);groups[title][variant]=m
            lines.append(f"| {variant} | {m['correct']}/{m['n']} | {m['precision']:.2f}% | {m['recall']:.2f}% | {m['f1']:.2f}% |")
        lines.append('')
    lines += ['## รายคำ','','| เป้าหมาย | ต้นทาง | ก่อน Renderer | มือ Renderer + ลำตัว payload |','|---|---|---|---|']
    for label in results['labels_evaluated']:
        lines.append('| '+label+' | '+' | '.join(lookup[label,v]['prediction'] for v in ['source_native','processed_30fps','rendered_hands_payload_body'])+' |')
    lines += ['','## การตรวจและข้อจำกัด','',
      '- 30 คำเดิม: inputs และพิกัด Renderer ตรงเดิม; prediction/Top5/target probability ตรง 210/210 รายการ',
      '- จุดที่โมเดลไม่อ่าน: ทำให้ผิดแล้วผลยังเหมือนต้นทางครบ 60/60 ไม่ใช่หลักฐานว่าท่านิ้วถูก',
      '- Production และผลเดิมไม่เปลี่ยน; ขนาดและเวลาปรับปรุงไฟล์ SQLite ไม่เปลี่ยน',
      '- 30 คำที่เพิ่มคือใหม่สำหรับชุดทดลองนี้ ไม่ได้ยืนยันว่าไม่เคยอยู่ในชุดฝึก หรือไม่เคยใช้ในการทดลองอื่นของโปรเจกต์',
      '- คะแนนรู้จำไม่ใช่เปอร์เซ็นต์ความเหมือนภาพหรือความเข้าใจ ASL และแถว Renderer ไม่อ่านภาพทั้งตัวหรือมือ SVG fallback ครบ',
      '- การขยายชุดอาจทำให้คะแนนลดได้ ไม่ปรับโมเดลหรือเปลี่ยนคำเพื่อรักษา 80% เดิม',
      '- ผล raw และ runtime อยู่ results.json; รายชื่อที่ตรึงก่อนทำนายอยู่ provenance.json; confusion แบบ sparse อยู่ confusion.json',
      '- สคริปต์ช่วยรันมีเวอร์ชันต่างจาก manifest แรก แต่โมเดล/ข้อมูลเดิมที่สำคัญตรวจ hash แล้ว และมีผล30เดิมเป็น regression control']
    (OUT/'REPORT_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'group_metrics.json').write_text(json.dumps(groups,indent=2),encoding='utf-8')
    verification=dict(prediction_count=420,original_predictions_exact=210,original_renderer_exact=True,
        unused_joint_controls_unchanged=60,production_unchanged=True,database_stat_unchanged=True,
        input_sha256=sha(OUT/'inputs.json'),capture_sha256=sha(OUT/'rendered.json'),
        runner_sha256=sha(Path(__file__)),checkpoint_sha256=results['checkpoint_sha256'])
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    confusion={v:[dict(actual=a,predicted=b,count=n) for (a,b),n in sorted(Counter((p['label'],p['prediction']) for p in pred if p['variant']==v).items())] for v in names}
    (OUT/'confusion.json').write_text(json.dumps(confusion,indent=2),encoding='utf-8')
    print(json.dumps(groups,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else run()
