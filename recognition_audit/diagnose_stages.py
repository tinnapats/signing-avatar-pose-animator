"""Read-only source/pipeline diagnostics, isolated output. Not linguistic truth."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json
from pathlib import Path
import numpy as np
import torch
from run_audit import HERE, PICK, load_model, frame_array
from step1_evaluate import hashes

OUT=HERE/'diagnosis'

def main():
    before=hashes()
    cases=_local_json_loads((HERE/'validation_trial/inputs.json').read_text())
    prior=_local_json_loads((HERE/'validation_trial/results.json').read_text())
    assert before==_local_json_loads((HERE/'validation_trial/production_before.json').read_text())
    lookup={(r['variant'],r['label']):r['prediction'] for r in prior['predictions']}
    network,normalize,labels=load_model()
    rows=[]
    for case in cases:
        word=case['phrase']
        source=np.asarray(case['source30'],dtype=np.float32)
        processed=frame_array([case['payload']['frames'][i] for i in case['frameIndices']])
        predictions={}
        for name,data in [('source30',source),('processed',processed)]:
            tensor=torch.from_numpy(data[:,PICK,:].transpose(2,0,1).copy())
            tensor=normalize({'frames':tensor})['frames'].unsqueeze(0)
            assert torch.isfinite(tensor).all()
            with torch.inference_mode(): logits=network(tensor)[0]
            assert torch.isfinite(logits).all()
            predictions[name]=labels[int(logits.argmax())]
        predictions['baseline']=lookup['S1A1',word]
        predictions['candidate']=lookup['S50A75',word]
        flags=[]
        if predictions['source30']!=word: flags.append('source_or_model_review')
        if predictions['source30']==word and predictions['processed']!=word: flags.append('pipeline_regression')
        if predictions['processed']==word and predictions['baseline']!=word: flags.append('renderer_regression')
        if predictions['baseline']!=word and predictions['candidate']==word: flags.append('candidate_recovery')
        if predictions['baseline']==word and predictions['candidate']!=word: flags.append('candidate_regression')
        valid=np.any(source!=0,axis=2)
        row=dict(word=word,predictions=predictions,flags=flags,source_hand_observed_fraction={side:float(valid[:,offset:offset+21].mean()) for side,offset in [('left',33),('right',54)]})
        rows.append(row)
        print(word,','.join(flags) or 'no_flag',flush=True)
    assert hashes()==before
    OUT.mkdir(exist_ok=True)
    totals={s:sum(r['predictions'][s]==r['word'] for r in rows) for s in ['source30','processed','baseline','candidate']}
    result=dict(rows=rows,correct_out_of_60=totals,production_unchanged=True,scope='source resampled 30 FPS, processed payload, renderer hands plus payload body. Flags are investigation leads, not causal proof.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# ตรวจจุดที่ผลรู้จำเปลี่ยนใน pipeline','', '| ขั้น | ถูก / 60 |','|---|---:|']
    lines += [f'| {s} | {n} |' for s,n in totals.items()]
    lines += ['', '## คำที่ควรตรวจต่อ', '', '| คำ | ต้นทาง 30 FPS | หลังประมวลผล | ระบบเดิม | ตัวทดลอง | ประเด็นตรวจ |','|---|---|---|---|---|---|']
    for r in rows:
        if r['flags']:
            p=r['predictions'];lines.append(f"| {r['word']} | {p['source30']} | {p['processed']} | {p['baseline']} | {p['candidate']} | {', '.join(r['flags'])} |")
    lines += ['', '## การตีความ', '- source_or_model_review: โมเดลผิดตั้งแต่พิกัดต้นทางที่ resample แล้ว อาจมาจากข้อมูล โมเดล การเตรียมอินพุต หรือความกำกวม ยังระบุสาเหตุไม่ได้', '- pipeline_regression: ต้นทางถูกแต่หลังประมวลผลผิด เป็นเบาะแสให้ตรวจ pipeline ไม่ใช่ข้อพิสูจน์ว่าท่าผิด', '- renderer_regression: payload ถูกแต่พิกัดมือหลังวาดผิด ควรตรวจ scale/anchor เป็นรายคำ', '- candidate_recovery/regression: ผลถูกผิดเปลี่ยนเมื่อใช้ตัวทดลอง', '- สัดส่วนจุดมือที่พบต่ำอาจเป็นมืออยู่นอกภาพหรือท่ามือเดียว ไม่ใช่ detection failure เสมอไป', '- ยังไม่มีวิดีโอต้นฉบับที่จับคู่คลิปยืนยันได้ จึงไม่ได้ตรวจความถูกต้องเทียบการแสดงของมนุษย์', '- ไม่ใช่การอ่าน pixels ของ Avatar และไม่ใช่ผลความเข้าใจของผู้ใช้', '- 60 คำนี้กลายเป็นชุดวิเคราะห์แล้ว ถ้าใช้เลือกค่าต่อควรหาชุดยืนยันใหม่', '']
    (OUT/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8')
    priority=[r for r in rows if 'renderer_regression' in r['flags'] or 'pipeline_regression' in r['flags']]
    priority.sort(key=lambda r: 'renderer_regression' not in r['flags'])
    selected=priority[:6]
    for r in rows:
        if len(selected)>=6:break
        if r not in selected and r['flags']:selected.append(r)
    (OUT/'review_cases.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    chosen={r['word'] for r in selected}
    (OUT/'inputs.json').write_text(json.dumps([c for c in cases if c['phrase'] in chosen]),encoding='utf-8')
    print(json.dumps(totals),flush=True)

if __name__=='__main__': main()
