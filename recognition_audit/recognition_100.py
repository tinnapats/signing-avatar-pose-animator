"""Original seven-condition ST-GCN evaluation on a frozen new 100-word sample."""
import hashlib
import json
import random
import sys
import urllib.error
import urllib.parse
import zipfile
from pathlib import Path
from inspect_kaggle import fetch,BASE
import paired_video_trial as pilot

HERE=Path(__file__).resolve().parent
OUT=HERE/'recognition_100'

def prepare():
    from step1_evaluate import hashes
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'Frozen protocol already exists'
    excluded={c['phrase'] for c in json.loads((HERE/'inputs.json').read_text())}
    for folder in ['validation_trial','broad_pair_trial']:
        excluded.update(json.loads((HERE/folder/'protocol.json').read_text())['words'])
    excluded.update(['motivate','continue','earn'])
    metadata=json.loads((HERE/'kaggle_reference/WLASL_v0.3.json').read_bytes())
    with zipfile.ZipFile(HERE/'wlasl_metadata.zip') as z:released=json.loads(z.read('splits/asl2000.json'))
    released_test={(r['gloss'],i['video_id']) for r in released for i in r['instances'] if i['split']=='test'}
    pool={r['gloss']:sorted([i for i in r['instances'] if i['split']=='test' and (r['gloss'],i['video_id']) in released_test],key=lambda i:i['video_id']) for r in metadata if r['gloss'] not in excluded}
    pool={k:v for k,v in pool.items() if v}
    words=random.Random(20260929).sample(sorted(pool),100)
    protocol=dict(seed=20260929,words=words,excluded=sorted(excluded),candidates={w:pool[w] for w in words},sample='100 new glosses, one first available test clip per gloss; test ID agrees with local released asl2000.json; no selection by model performance',conditions=['source_native','source_30fps','processed_30fps','rendered_hands_payload_body','freeze_fingers','shuffle_frames','corrupt_unselected_joints'],preprocess='MediaPipe Holistic 0.10.21; 30 FPS resample; production pipeline x2, Gaussian sigma1.2 radius2; renderer unchanged',macro_definition='Mean over selected true-label set, zero division=0, predictions outside selected set count as errors; not macro over all 2000 classes',production_hashes=hashes(),script_hashes={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['run_audit.py','render_hands.mjs','paired_video_trial.py']})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print('Frozen 100 words; released test IDs intersected.',flush=True)

def download():
    protocol=json.loads((OUT/'protocol.json').read_text())
    (OUT/'videos').mkdir(exist_ok=True)
    manifest=json.loads((OUT/'manifest.json').read_text()) if (OUT/'manifest.json').exists() else []
    done={i['word'] for i in manifest};failures=[]
    for word in protocol['words']:
        if word in done:continue
        for candidate in protocol['candidates'][word]:
            url=BASE+'/download/sttaseen/wlasl2000-resized/'+urllib.parse.quote('wlasl-complete/videos/'+candidate['video_id']+'.mp4',safe='')+'?datasetVersionNumber=1'
            try:data=fetch(url)
            except urllib.error.HTTPError as e:
                if e.code==404:
                    failures.append(dict(word=word,video_id=candidate['video_id'],reason='404'))
                    (OUT/'download_failures.json').write_text(json.dumps(failures,indent=2))
                    continue
                raise
            assert b'ftyp' in data[:64]
            path=OUT/'videos'/(word+'.mp4');path.write_bytes(data)
            manifest.append(dict(word=word,metadata=candidate,source_url=url,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),local_video=str(path),released_split_verified=True))
            (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            print('downloaded',len(manifest),word,candidate['video_id'],flush=True)
            break
        else:raise RuntimeError('No available clip for frozen word: '+word)

def extract():
    pilot.OUT=OUT;pilot.extract()

def build():
    from run_audit import pipeline,csv_array
    from step1_evaluate import hashes
    protocol=json.loads((OUT/'protocol.json').read_text());assert hashes()==protocol['production_hashes']
    cases=[]
    for item in json.loads((OUT/'manifest.json').read_text()):
        word=item['word'];path=OUT/'csv'/(word+'.csv')
        native=csv_array(pipeline.read_clip_csv(path))
        payload=pipeline.build_payload(OUT/'csv',[word+'.csv'],'',30,513,513,0,0,2,1.2,2)
        source=csv_array(pipeline.concat_clips([path],513,513,0,target_fps=30))
        indices=list(range(0,len(payload['frames']),2));assert len(source)==len(indices)
        cases.append(dict(phrase=word,payload=payload,native=native.tolist(),source30=source.tolist(),frameIndices=indices,video_id=item['metadata']['video_id']))
        print('built',len(cases),word,flush=True)
    assert len(cases)==100 and {c['phrase'] for c in cases}==set(protocol['words'])
    assert hashes()==protocol['production_hashes']
    (OUT/'inputs.json').write_text(json.dumps(cases),encoding='utf-8')

def evaluate():
    import run_audit
    from step1_evaluate import hashes
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert hashes()==protocol['production_hashes']
    for f,h in protocol['script_hashes'].items():assert hashlib.sha256((HERE/f).read_bytes()).hexdigest()==h
    captures=json.loads((OUT/'rendered.json').read_text())
    assert len(captures)==100 and {r['phrase'] for r in captures}==set(protocol['words'])
    run_audit.run(OUT)
    assert hashes()==protocol['production_hashes']
    report()

def report():
    from collections import Counter
    result=json.loads((OUT/'results.json').read_text());rows=result['predictions']
    assert len(rows)==700
    grouped={v:[r for r in rows if r['variant']==v] for v in {r['variant'] for r in rows}}
    assert all(len(r)==100 for r in grouped.values()) and len(grouped)==7
    native={r['label']:r for r in grouped['source_native']}
    for r in grouped['corrupt_unselected_joints']:
        assert r['top5']==native[r['label']]['top5']
        assert abs(r['target_probability']-native[r['label']]['target_probability'])<1e-8
    confusion={v:[dict(actual=a,predicted=b,count=n) for (a,b),n in sorted(Counter((r['label'],r['prediction']) for r in g).items())] for v,g in grouped.items()}
    (OUT/'confusion.json').write_text(json.dumps(confusion,indent=2),encoding='utf-8')
    names={'source_native':'ต้นทาง FPS เดิม','source_30fps':'ต้นทาง 30 FPS','processed_30fps':'หลังประมวลผล ก่อน Renderer','rendered_hands_payload_body':'มือหลัง Renderer + ลำตัว payload','freeze_fingers':'จงใจค้างรูปนิ้ว (FPS เดิม)','shuffle_frames':'จงใจสลับเฟรม (FPS เดิม)','corrupt_unselected_joints':'ทำจุดนิ้วที่โมเดลไม่ใช้ให้ผิด'}
    lines=['# ST-GCN / WLASL2000: 100 คำใหม่ วิธีประเมินเดิม','', 'หนึ่งคลิปต่อคำ สุ่มก่อนเห็นผล ยืนยัน video_id อยู่ใน test ทั้ง Kaggle metadata และ released asl2000.json ที่ใช้ร่วมกับ checkpoint แต่ยังไม่ใช่การตรวจประวัติการฝึกจริงทั้งหมด','', '| เงื่อนไข | ถูก/100 | Accuracy | Macro-P | Macro-R | Macro-F1 | Top-5 |','|---|---:|---:|---:|---:|---:|---:|']
    for v,name in names.items():
        s=next(s for s in result['summaries'] if s['variant']==v);correct=sum(r['label']==r['prediction'] for r in grouped[v]);assert abs(s['accuracy']-correct)<1e-8
        lines.append(f"| {name} | {correct} | "+' | '.join(f"{s[k]:.2f}%" for k in ['accuracy','macro_precision','macro_recall','macro_f1','top5'])+' |')
    lines+=['', '## การตีความ', '- วัดการรู้จำของโมเดล ไม่ใช่ความถูกต้องภาษามือหรือความเข้าใจของมนุษย์', '- ใช้ renderer ระบบเดิม ไม่มี PAIR หรือ IK ไม่แก้ production/SQLite', '- ต้นทางเป็นพิกัดที่สกัดใหม่จากวิดีโอ resized ด้วย MediaPipe ไม่ใช่ภาพมนุษย์เข้าโมเดลโดยตรง', '- Macro เฉลี่ยเฉพาะ 100 target labels; ค่ากรณีหารศูนย์เป็น 0 ไม่ใช่ macro บน 2,000 labels ทั้งหมด', '- เปรียบเทียบต้นทาง 30 FPS กับหลังประมวลผลและ renderer เพื่อควบคุมเวลา; ชุดค้างนิ้ว/สลับเฟรมเทียบกับต้นทาง FPS เดิมตามการทดลองแรก', '- การเพิ่มจำนวนคำไม่แก้ข้อจำกัดของโมเดล และห้ามเทียบคะแนนชุดนี้กับ 30 คำเดิมเป็นผลการปรับปรุง', '- จุดนิ้วบางจุดไม่ถูกป้อนเข้า ST-GCN; negative control ที่ทำจุดเหล่านี้ผิดยังต้องให้ผลเท่าต้นทาง แสดง blind spot ไม่ใช่ความทนทาน', '- หนึ่งคลิปต่อคำ ไม่มีการประเมินผู้ใช้หรือความทั่วไปข้ามผู้แสดงหลายคลิปต่อคำ', '', '## การตรวจสอบและหลักฐาน', '- 700 predictions ครบ 7 เงื่อนไข × 100 คลิป; มี confusion และผลรายคำใน JSON', '- รหัสวิดีโอ hash ไฟล์ และ coverage การตรวจพบมือใน manifest.json; รายการสุ่มและ hash production ใน protocol.json', '- อ้างอิง WLASL/Kaggle v1 และข้อกำหนด C-UDA ไม่รวมวิดีโอใน Git', '']
    (OUT/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'evidence.json').write_text(json.dumps({f:hashlib.sha256((OUT/f).read_bytes()).hexdigest() for f in ['protocol.json','manifest.json','inputs.json','rendered.json','results.json']},indent=2),encoding='utf-8')

if __name__=='__main__':globals()[sys.argv[1]]()
