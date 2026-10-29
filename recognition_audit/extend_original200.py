"""Same seven-condition protocol: frozen original60 + 140 preselected clips."""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import run_audit as audit
from repeat_original30 import production,sha
from extend_original60 import metrics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=HERE/'original_method60_20260930'
OUT=HERE/'original_method200_20260930'
NODE='C:/Users/newtu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'

def prepare():
    assert not OUT.exists(),'Refuse to overwrite an existing experiment'
    prior=json.loads((OLD/'provenance.json').read_text())
    assert production()==prior['production_before'],'Production differs from original60'
    for name,h in prior['hashes'].items():assert sha(HERE/name)==h,name
    cases=json.loads((OLD/'inputs.json').read_text());old=[c['phrase'] for c in cases]
    assert len(old)==len(set(old))==60
    db=ROOT/'SLclean.sqlite3';clips=audit.pipeline.discover_clips(db)
    pool=sorted(Path(n).stem for n in clips if Path(n).stem not in old)
    added=random.Random(20260926).sample(pool,140)
    OUT.mkdir();(OUT/'added140').mkdir()
    prov=dict(seed=20260926,original_words=old,added_words=added,pool=pool,
      production_before=production(),hashes=prior['hashes'],
      original60_hashes={n:sha(OLD/n) for n in ['inputs.json','rendered.json','results.json']},
      database_size=db.stat().st_size,database_mtime_ns=db.stat().st_mtime_ns,
      protocol='Keep original60 and select140 from sorted remaining database keys before inference; reuse verified original60 captures, freshly render140; infer all200 with unchanged run_audit.run')
    (OUT/'provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf-8')
    new=[]
    for i,label in enumerate(added):
        paths=audit.pipeline.resolve_selected_paths(clips,db,[label+'.csv'],'')
        native=audit.csv_array(audit.pipeline.read_clip_csv(paths[0]))
        payload=audit.pipeline.build_payload(db,[label+'.csv'],'',30,513,513,0,0,2,1.2,2)
        source30=audit.csv_array(audit.pipeline.concat_clips(paths,513,513,0,target_fps=30))
        indexes=list(range(0,len(payload['frames']),2));assert len(indexes)==len(source30)
        new.append(dict(phrase=label,payload=payload,frameIndices=indexes,native=native.tolist(),source30=source30.tolist()))
        print(f'prepared {i+1}/140 {label}',flush=True)
    assert production()==prov['production_before']
    (OUT/'added140/inputs.json').write_text(json.dumps(new),encoding='utf-8')
    (OUT/'inputs.json').write_text(json.dumps(cases+new),encoding='utf-8')

def render():
    env=os.environ.copy();env.update(AUDIT_OUTPUT=str(OUT/'added140'),AUDIT_INPUTS=str(OUT/'added140/inputs.json'))
    subprocess.run([NODE,str(HERE/'render_hands.mjs')],cwd=ROOT,env=env,check=True)
    old=json.loads((OLD/'rendered.json').read_text());new=json.loads((OUT/'added140/rendered.json').read_text())
    assert len(old)==60 and len(new)==140
    (OUT/'rendered.json').write_text(json.dumps(old+new),encoding='utf-8')

def run():
    p=json.loads((OUT/'provenance.json').read_text())
    assert production()==p['production_before']
    for n,h in p['hashes'].items():assert sha(HERE/n)==h,n
    for n,h in p['original60_hashes'].items():assert sha(OLD/n)==h,n
    cases=json.loads((OUT/'inputs.json').read_text())
    assert cases[:60]==json.loads((OLD/'inputs.json').read_text())
    assert [c['phrase'] for c in cases]==p['original_words']+p['added_words']
    del cases;gc.collect()
    captures=json.loads((OUT/'rendered.json').read_text());assert len(captures)==200
    assert [c['phrase'] for c in captures]==p['original_words']+p['added_words']
    del captures;gc.collect()
    audit.run(OUT)
    r=json.loads((OUT/'results.json').read_text());rows=r['predictions'];assert len(rows)==1400
    lookup={(x['label'],x['variant']):x for x in rows}
    for old in json.loads((OLD/'results.json').read_text())['predictions']:
        new=lookup[old['label'],old['variant']]
        assert all(new[k]==old[k] for k in ['prediction','top5','target_probability']),'Original60 changed'
    for label in r['labels_evaluated']:
        a=lookup[label,'source_native'];b=lookup[label,'corrupt_unselected_joints']
        assert a['top5']==b['top5'] and a['target_probability']==b['target_probability']
    assert production()==p['production_before']
    db=ROOT/'SLclean.sqlite3';assert db.stat().st_size==p['database_size'] and db.stat().st_mtime_ns==p['database_mtime_ns']
    variants=['source_native','source_30fps','processed_30fps','rendered_hands_payload_body','freeze_fingers','shuffle_frames','corrupt_unselected_joints']
    groups={};lines=['# ST-GCN วิธีเดิม รวม 200 คำ — 30 กันยายน 2026','',
      'คง 60 คำเดิมและสุ่มเพิ่ม 140 คำก่อนทำนาย seed 20260926 ไม่ปรับโมเดล preprocessing หรือ Renderer; ไม่ใช้ S50A1/fallback ทดลอง หนึ่งคลิปต่อคำ เลือกคำตอบจาก 2,000 classes เสมอ','',
      'ใช้พิกัด Renderer ของ60เดิมที่ตรวจ hash และไฟล์ production แล้ว พร้อมวาด140ที่เพิ่มใหม่ จากนั้น inference ใหม่ครบ200คำ ไม่ได้คัดลอกคะแนน60เดิมมาเฉลี่ย','']
    for title,labels in [('รวม200',p['original_words']+p['added_words']),('เดิม60',p['original_words']),('เพิ่ม140',p['added_words'])]:
        lines+=['## '+title,'','| เงื่อนไข | ถูก/ทั้งหมด | Accuracy | Macro-P | Macro-R | Macro-F1 |','|---|---:|---:|---:|---:|---:|'];groups[title]={}
        for v in variants:
            m=metrics([x for x in rows if x['variant']==v and x['label'] in labels],labels);groups[title][v]=m
            lines.append(f"| {v} | {m['correct']}/{m['n']} | {100*m['correct']/m['n']:.2f}% | {m['precision']:.2f}% | {m['recall']:.2f}% | {m['f1']:.2f}% |")
        lines.append('')
    lines+=['## รายคำ','','| คำเป้าหมาย | ต้นทาง | ก่อน Renderer | มือ Renderer + ลำตัว payload |','|---|---|---|---|']
    for label in r['labels_evaluated']:lines.append('| '+label+' | '+' | '.join(lookup[label,v]['prediction'] for v in variants if v in ['source_native','processed_30fps','rendered_hands_payload_body'])+' |')
    lines+=['','## ข้อจำกัดและการตรวจ','',
      '- 60คำเดิมทำนายใหม่ตรงผลเก่า prediction/Top5/probability ครบ420รายการ ไม่แก้ไฟล์ผลเดิมหรือ production; SQLite size/mtime ไม่เปลี่ยน',
      '- จุดนิ้วที่โมเดลไม่อ่าน: เปลี่ยนแล้วผลยังเหมือนต้นทาง200/200 ไม่ได้พิสูจน์ความถูกต้องของนิ้วทุกข้อ',
      '- Macro เฉลี่ยเหนือ true labels ของกลุ่มและ zero_division=0; คำตอบนอกกลุ่มยังผิด ไม่ใช่ macro เฉลี่ยทั้ง2000คลาส',
      '- 140คำใหม่สำหรับชุดนี้ ไม่รับรองว่าไม่เคยทดลองมาก่อนหรือไม่ซ้ำ training set ไม่ใช่ผลยืนยัน generalization',
      '- คะแนนจำแนกไม่ใช่เปอร์เซ็นต์ความเหมือนภาพหรือความเข้าใจ ASL; แถวRendererยังใช้ลำตัวpayload และไม่ตรวจภาพทั้งตัว/native fallback ครบ',
      '- Runtime inference อยู่ results.json เป็นเวลารอบนี้ ไม่ใช่ benchmarkควบคุมทุกปัจจัย',
      '- หลักฐานรายชื่อก่อนทำนาย: provenance.json; ผลรายรายการ: results.json; ตารางแยกกลุ่ม: group_metrics.json; confusion: confusion.json']
    (OUT/'REPORT_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'group_metrics.json').write_text(json.dumps(groups,indent=2),encoding='utf-8')
    confusion={v:[dict(actual=a,predicted=b,count=n) for (a,b),n in sorted(Counter((x['label'],x['prediction']) for x in rows if x['variant']==v).items())] for v in variants}
    (OUT/'confusion.json').write_text(json.dumps(confusion,indent=2),encoding='utf-8')
    verification=dict(predictions=1400,original60_exact=420,production_unchanged=True,database_stat_unchanged=True,
      input_sha256=sha(OUT/'inputs.json'),capture_sha256=sha(OUT/'rendered.json'),runner_sha256=sha(Path(__file__)),checkpoint_sha256=r['checkpoint_sha256'])
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(groups,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','render','run','all']);args=parser.parse_args()
    if args.stage=='all':
        prepare();gc.collect();render();gc.collect();run()
    else:globals()[args.stage]()
