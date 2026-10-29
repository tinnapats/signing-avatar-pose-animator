"""Frozen randomized new-word safety gate for PAIR; no production changes."""
import hashlib
import json
import random
import sys
from pathlib import Path
import paired_video_trial as pilot

HERE=Path(__file__).resolve().parent
OUT=HERE/'broad_pair_trial'

def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'Never replace the frozen sample'
    excluded={c['phrase'] for c in json.loads((HERE/'inputs.json').read_text())}
    excluded.update(json.loads((HERE/'validation_trial/protocol.json').read_text())['words'])
    excluded.update(pilot.WORDS)
    metadata=json.loads((HERE/'kaggle_reference/WLASL_v0.3.json').read_bytes())
    available=sorted(r['gloss'] for r in metadata if r['gloss'] not in excluded and any(i['split']=='test' for i in r['instances']))
    words=random.Random(20260928).sample(available,24)
    protocol=dict(words=words,excluded=sorted(excluded),seed=20260928,count=24,candidate='PAIR: no scale; shared mean wrist translation when both visible, per-side anchor fallback otherwise',variants=[dict(id='S1A1',scale=1,anchor=1),dict(id='PAIR',scale=0,anchor=1,pair=True)],selection='Random unseen glosses; first available test video by ascending ID; never select on scores',primary='Equal-clip mean absolute nearest inter-hand joint distance error vs newly extracted source, normalized by shoulder width',gate='Candidate must improve mean pair error AND mean wrist gap <= 0.02 shoulder width. Engineering screen, not an anatomical or user-validated threshold.',limits='Sample is not stratified by linguistic gesture type; WLASL test does not prove checkpoint training independence.',renderer_sha256=hashlib.sha256((HERE/'step1b_render.mjs').read_bytes()).hexdigest())
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print(json.dumps(protocol),flush=True)

def delegate(stage):
    protocol=json.loads((OUT/'protocol.json').read_text())
    pilot.OUT=OUT;pilot.WORDS=protocol['words']
    getattr(pilot,stage)()

def build():
    from run_audit import pipeline,csv_array
    from step1_evaluate import hashes
    before=hashes();cases=[]
    for item in json.loads((OUT/'manifest.json').read_text()):
        word=item['word'];path=OUT/'csv'/(word+'.csv')
        payload=pipeline.build_payload(OUT/'csv',[word+'.csv'],'',30,513,513,0,0,2,1.2,2)
        source=csv_array(pipeline.concat_clips([path],513,513,0,target_fps=30))
        indices=list(range(0,len(payload['frames']),2));assert len(source)==len(indices)
        cases.append(dict(phrase=word,payload=payload,source30=source.tolist(),frameIndices=indices,video_id=item['metadata']['video_id']))
        print('built',word,len(indices),flush=True)
    assert before==hashes()
    (OUT/'production_before.json').write_text(json.dumps(before),encoding='utf-8')
    (OUT/'inputs.json').write_text(json.dumps(cases),encoding='utf-8')

def evaluate():
    import numpy as np
    from run_audit import frame_array
    from step1_evaluate import hashes
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert hashlib.sha256((HERE/'step1b_render.mjs').read_bytes()).hexdigest()==protocol['renderer_sha256']
    cases=json.loads((OUT/'inputs.json').read_text())
    captures={(r['variant'],r['phrase']):r['samples'] for r in json.loads((OUT/'rendered.json').read_text())}
    assert len(captures)==48 and {c['phrase'] for c in cases}==set(protocol['words'])
    def near(a):return float(np.linalg.norm(a[33:54,None,:]-a[None,54:75,:],axis=2).min()/np.linalg.norm(a[11]-a[12]))
    rows=[]
    for case in cases:
        word=case['phrase'];src=np.asarray(case['source30'],dtype=float)
        frames=[case['payload']['frames'][i] for i in case['frameIndices']];payload=frame_array(frames).astype(float)
        pair=[];handframes=0
        for t,a in enumerate(src):
            if np.linalg.norm(a[11]-a[12])<1e-6:continue
            if not np.all(np.any(a[[11,12]]!=0,axis=1)):continue
            if all(np.all(np.any(a[o:o+21]!=0,axis=1)) for o in [33,54]) and all(len(captures[v,word][t]['hands'][s])==21 for v in ['S1A1','PAIR'] for s in ['left','right']):pair.append(t)
        for variant in ['S1A1','PAIR']:
            data=payload.copy();gaps=[];steps=[];previous={};fallback=0
            for t,sample in enumerate(captures[variant,word]):
                pose={k['part']:k for k in frames[t]['pose']['keypoints']}
                width=np.linalg.norm(payload[t,11]-payload[t,12])
                if width<1e-6:continue
                if any(len(sample['hands'][s])<21 for s in ['left','right']):fallback+=1
                for side,offset in [('left',33),('right',54)]:
                    out=sample['hands'][side];data[t,offset:offset+21]=0
                    for k,p in out.items():data[t,offset+int(k)]=p
                    wrist=pose[side+'Wrist']
                    if '0' in out and wrist['score']>=.2:
                        bp=np.array([wrist['position']['x'],wrist['position']['y']]);gaps.append(float(np.linalg.norm(data[t,offset]-bp)/width))
                        relative=(data[t,offset]-payload[t,[11,12]].mean(axis=0))/width
                        if side in previous and previous[side][0]==t-1:steps.append(float(np.linalg.norm(relative-previous[side][1])))
                        previous[side]=(t,relative)
            errors=[abs(near(data[t])-near(src[t])) for t in pair]
            rows.append(dict(word=word,variant=variant,video_id=case['video_id'],total_frames=len(src),pair_frames=len(pair),pair_error=float(np.mean(errors)) if errors else None,wrist_gap=float(np.mean(gaps)) if gaps else None,wrist_gap_p95=float(np.percentile(gaps,95)) if gaps else None,wrist_step_p95=float(np.percentile(steps,95)) if steps else None,frames_without_two_complete_rendered_hands=fallback))
    index={(r['variant'],r['word']):r for r in rows}
    eligible=[c['phrase'] for c in cases if index['S1A1',c['phrase']]['pair_error'] is not None and index['PAIR',c['phrase']]['pair_error'] is not None]
    diff={w:index['PAIR',w]['pair_error']-index['S1A1',w]['pair_error'] for w in eligible}
    summaries={v:{key:float(np.mean([r[key] for r in rows if r['variant']==v and r[key] is not None])) for key in ['pair_error','wrist_gap','wrist_gap_p95','wrist_step_p95']} for v in ['S1A1','PAIR']}
    passed=summaries['PAIR']['pair_error']<summaries['S1A1']['pair_error'] and summaries['PAIR']['wrist_gap']<=.02
    summary=dict(eligible_pair_clips=len(eligible),total_clips=len(cases),improved=sum(v < -1e-6 for v in diff.values()),worsened=sum(v>1e-6 for v in diff.values()),tied=sum(abs(v)<=1e-6 for v in diff.values()),worst_pair_change=sorted(diff.items(),key=lambda kv:kv[1],reverse=True)[:3],summaries=summaries,engineering_gate_passed=passed,production_unchanged=True)
    assert hashes()==json.loads((OUT/'production_before.json').read_text())
    (OUT/'results.json').write_text(json.dumps(dict(summary=summary,per_clip=rows),indent=2),encoding='utf-8')
    lines=['# ทดสอบ PAIR บน 24 คำใหม่ที่สุ่มก่อนวัดผล','', 'ล็อกกฎและ hash renderer ก่อนดาวน์โหลด/สกัดจุด ไม่ปรับตามผลแต่ละคำ ไม่ใช้ F1 เป็นผู้ตัดสินความถูกต้องทางภาษามือ','',f'ประเมินระยะระหว่างมือได้ {len(eligible)}/24 คลิป; คลิปอื่นไม่ถูกนับว่าไม่มีข้อผิดพลาด',f"ดีขึ้น {summary['improved']} / แย่ลง {summary['worsened']} / เท่าเดิม {summary['tied']} คลิป (เฉพาะ metric ระยะระหว่างมือ)",'','| วิธี | Error ระยะระหว่างมือ (% ไหล่) | Gap ข้อมือเฉลี่ย (% ไหล่) | ค่าเฉลี่ย P95 gap ต่อคลิป (% ไหล่) |','|---|---:|---:|---:|']
    for v,s in summaries.items():lines.append(f"| {v} | {100*s['pair_error']:.2f} | {100*s['wrist_gap']:.2f} | {100*s['wrist_gap_p95']:.2f} |")
    lines += ['', f"ผ่านเกณฑ์คัดกรองที่กำหนดก่อนทดลอง: {passed}", 'เกณฑ์: error ระหว่างมือลดลง และ gap ข้อมือเฉลี่ยไม่เกิน 2% ไหล่ เป็นเกณฑ์วิศวกรรมชั่วคราว ไม่ใช่เกณฑ์มนุษย์/กายวิภาค', '', '## ผลรายคลิป', '| คำ | เฟรมมือคู่ | Error เดิม → PAIR (% ไหล่) | Gap เดิม → PAIR (% ไหล่) |','|---|---:|---:|---:|']
    for c in cases:
        w=c['phrase'];a=index['S1A1',w];b=index['PAIR',w];err='N/A' if a['pair_error'] is None else f"{a['pair_error']*100:.2f} → {b['pair_error']*100:.2f}"
        gap='N/A' if a['wrist_gap'] is None or b['wrist_gap'] is None else f"{a['wrist_gap']*100:.2f} → {b['wrist_gap']*100:.2f}"
        lines.append(f"| {w} | {a['pair_frames']} | {err} | {gap} |")
    lines+=['', '## ข้อจำกัดและการตัดสิน', '- PAIR รักษาระยะโดยโครงสร้าง ค่า error ที่ดีขึ้นไม่ใช่หลักฐานว่าท่าถูกหรือดูดีกว่า', '- Source เป็น MediaPipe ที่อาจตรวจผิด ไม่ใช่ ground truth ที่คนกำกับจุด', '- สุ่มคำ ไม่ได้จัดสมดุลท่ามือเดียว/สัมผัส/บังกัน จึงยังไม่รับรองครอบคลุมทุกลักษณะ', '- Gap วัดจุด ไม่ใช่ผิวภาพ; ยังต้องตรวจการกระโดดตอนเข้าออก fallback และประเมินมนุษย์', '- WLASL test ไม่รับรองว่าไม่อยู่ในข้อมูลฝึกของ checkpoint; รอบนี้ไม่ใช้โมเดลรู้จำ', '- ค่าเฉลี่ยให้แต่ละคลิปน้ำหนักเท่ากัน; ไม่รวมคลิปที่ไม่มีข้อมูลใน metric นั้นและรายงาน N/A', '- ไม่แก้ production หรือ SQLite และไม่เผยแพร่วิดีโอใน Git', '', 'ข้อสรุป: ยังไม่แนะนำใช้แทน production หากไม่ผ่าน gate; อย่าปรับตามคำที่แพ้เพื่อทำให้ตารางนี้ดีขึ้น ต้องตั้งสมมติฐานใหม่ก่อนทดลองรอบถัดไป', '']
    (OUT/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='prepare':prepare()
    elif stage=='build':build()
    elif stage=='evaluate':evaluate()
    else:delegate(stage)
