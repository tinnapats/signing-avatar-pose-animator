"""Factorial renderer-only ablation. Production code is never edited."""
import argparse
import hashlib
import json
import subprocess
from unittest.mock import patch
import numpy as np
import torch
from run_audit import HERE, ROOT, PICK, load_model, frame_array, pipeline

OUT=HERE/'step1'
NAMES={'S0A0':'ไม่ปรับขนาด / ไม่ยึดข้อมือ','S1A0':'ปรับขนาดอย่างเดียว','S0A1':'ยึดข้อมืออย่างเดียว','S1A1':'เปิดทั้งสอง (ระบบปัจจุบัน)'}
BASELINE_TOLERANCE=0

def hashes():
    files=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files if (ROOT/f).is_file()}

def prepare():
    OUT.mkdir(exist_ok=True)
    (OUT/'production_before.json').write_text(json.dumps(hashes()),encoding='utf-8')
    cases=json.loads((HERE/'inputs.json').read_text()); refs={}
    for case in cases:
        affine={}; original=pipeline.fit_frames_to_canvas
        def capture(frames,width,height):
            before=np.array([[k['position']['x'],k['position']['y']] for f in frames for k in f['pose']['keypoints']])
            result=original(frames,width,height)
            after=np.array([[k['position']['x'],k['position']['y']] for f in result for k in f['pose']['keypoints']])
            scale,dx=np.linalg.lstsq(np.column_stack([before[:,0],np.ones(len(before))]),after[:,0],rcond=None)[0]
            dy=np.mean(after[:,1]-before[:,1]*scale)
            assert np.max(np.abs(after-(before*scale+[dx,dy])))<1e-6
            affine.update(scale=float(scale),offset=[float(dx),float(dy)])
            return result
        with patch.object(pipeline,'fit_frames_to_canvas',capture):
            payload=pipeline.build_payload(ROOT/'SLclean.sqlite3',[case['phrase']+'.csv'],'',30,513,513,0,0,2,1.2,2)
        # Guard against accidentally mixing a changed pipeline with old captures.
        assert payload['frames']==case['payload']['frames'],case['phrase']
        refs[case['phrase']]=affine
        print('reference',case['phrase'],flush=True)
    (OUT/'reference_fit.json').write_text(json.dumps(refs),encoding='utf-8')

def evaluate():
    cases=json.loads((HERE/'inputs.json').read_text()); fits=json.loads((OUT/'reference_fit.json').read_text())
    captures=json.loads((OUT/'rendered.json').read_text()); lookup={(r['variant'],r['phrase']):r['samples'] for r in captures}
    previous={r['phrase']:r['samples'] for r in json.loads((HERE/'rendered.json').read_text())}
    # Both-on must reproduce the prior production-mode audit exactly.
    baseline_exact=True
    for case in cases:
        current=lookup['S1A1',case['phrase']]; old=previous[case['phrase']]
        baseline_exact=baseline_exact and current==old
        assert len(current)==len(old)
        for a,b in zip(current,old):
            assert a['zoom']==b['zoom']
            for side in ['left','right']:
                assert a['hands'][side].keys()==b['hands'][side].keys()
                for key,point in a['hands'][side].items():
                    assert np.allclose(point,b['hands'][side][key],rtol=0,atol=BASELINE_TOLERANCE),case['phrase']
    network,normalize,labels=load_model(); predictions=[]; pooled={}; per_case=[]
    for variant in NAMES:
        pooled[variant]={k:[] for k in ['position','shape','wrist_gap']}
        for case in cases:
            name=case['phrase']; source=np.asarray(case['source30'],dtype=np.float32)
            fit=fits[name]; ref=source*fit['scale']+np.asarray(fit['offset'])
            frames=[case['payload']['frames'][i] for i in case['frameIndices']]
            data=frame_array(frames); samples=lookup[variant,name]
            assert len(ref)==len(samples)==len(data)
            errors={k:[] for k in pooled[variant]}
            for t,sample in enumerate(samples):
                shoulder=np.linalg.norm(ref[t,11]-ref[t,12])
                for side,offset in [('left',33),('right',54)]:
                    output=sample['hands'][side]; data[t,offset:offset+21]=0
                    for i,point in output.items(): data[t,offset+int(i)]=point
                    # Only fully observed reference hands and observed shoulders.
                    if shoulder<=1e-6 or not np.all(np.any(source[t,[11,12]]!=0,axis=1)): continue
                    if not np.all(np.any(source[t,offset:offset+21]!=0,axis=1)): continue
                    palm=np.linalg.norm(ref[t,offset+9]-ref[t,offset])
                    if palm<=1e-6: continue
                    for i in range(21):
                        point=output.get(str(i))
                        dist=np.linalg.norm(np.asarray(point)-ref[t,offset+i]) if point is not None else float('inf')
                        errors['position'].append(float(dist/shoulder))
                        if i:
                            dist=np.linalg.norm((np.asarray(point)-np.asarray(output['0']))-(ref[t,offset+i]-ref[t,offset])) if point is not None and '0' in output else float('inf')
                            errors['shape'].append(float(dist/palm))
                    pose={k['part']:k for k in frames[t]['pose']['keypoints']}
                    wrist=pose[side+'Wrist']
                    if '0' in output and wrist['score']>=.2:
                        body=np.array([wrist['position']['x'],wrist['position']['y']])
                        errors['wrist_gap'].append(float(np.linalg.norm(np.asarray(output['0'])-body)/shoulder))
            tensor=torch.from_numpy(data[:,PICK,:].transpose(2,0,1).copy())
            tensor=normalize({'frames':tensor})['frames'].unsqueeze(0)
            assert torch.isfinite(tensor).all()
            with torch.inference_mode(): probabilities=network(tensor)[0].softmax(0)
            top=probabilities.topk(5).indices.tolist()
            predictions.append(dict(variant=variant,label=name,prediction=labels[top[0]],top5=[labels[i] for i in top]))
            for key in errors: pooled[variant][key].extend(errors[key])
            per_case.append(dict(variant=variant,label=name,pck5=float(np.mean(np.array(errors['position'])<=.05)*100),points=len(errors['position'])))
        print('evaluated',variant,flush=True)
    summaries=[]
    for variant,e in pooled.items():
        ps=[p for p in predictions if p['variant']==variant]; metrics=[]
        for case in cases:
            name=case['phrase'];tp=sum(p['label']==name and p['prediction']==name for p in ps)
            fp=sum(p['label']!=name and p['prediction']==name for p in ps);fn=sum(p['label']==name and p['prediction']!=name for p in ps)
            p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0
            metrics.append([p,r,2*p*r/(p+r) if p+r else 0])
        mp,mr,mf=np.mean(metrics,axis=0)*100
        summaries.append(dict(variant=variant,points=len(e['position']),pck5=float(np.mean(np.array(e['position'])<=.05)*100),
            shape_pck20=float(np.mean(np.array(e['shape'])<=.2)*100),coverage=float(np.mean(np.isfinite(e['position']))*100),
            wrist_gap_mean_percent=float(np.mean(e['wrist_gap'])*100),wrist_gap_p95_percent=float(np.percentile(e['wrist_gap'],95)*100),wrist_gap_within2_percent=float(np.mean(np.array(e['wrist_gap'])<=.02)*100),correct=sum(p['label']==p['prediction'] for p in ps),
            macro_precision=float(mp),macro_recall=float(mr),macro_f1=float(mf)))
    before=json.loads((OUT/'production_before.json').read_text());after=hashes()
    assert before==after,'Tracked production files changed during audit'
    result=dict(summaries=summaries,predictions=predictions,per_case=per_case,production_files_unchanged=True,
        baseline_capture_exact_match=baseline_exact,baseline_capture_tolerance=BASELINE_TOLERANCE,reference='fully observed hands at source30 timestamps; same global fit as output, no per-hand fitted alignment',
        scope='hands captured from Paper.js strokes, body from payload; not full image or linguistic accuracy')
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# ขั้นที่ 1: แยกผลการปรับขนาดมือและการยึดข้อมือ','',
        'ทดสอบ 30 คำเดิมด้วย 4 เงื่อนไข เปลี่ยนเฉพาะ temporary browser bundle ไม่แก้ production Renderer; backend และโมเดลคงเดิม','',
        '| เงื่อนไข | PCK@5% ไหล่ | รูปนิ้ว PCK@20% ฝ่ามือ | ระยะข้อมือเฉลี่ย (% ไหล่) | ทายถูก/30 | Macro-P | Macro-R | Macro-F1 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {NAMES[s['variant']]} | {s['pck5']:.2f}% | {s['shape_pck20']:.2f}% | {s['wrist_gap_mean_percent']:.2f}% | {s['correct']} | {s['macro_precision']:.2f}% | {s['macro_recall']:.2f}% | {s['macro_f1']:.2f}% |")
    lines+=['','## อ่านผลอย่างไร','',
        '- PCK ยิ่งสูงยิ่งใกล้ตำแหน่งต้นทางตามเกณฑ์ที่ระบุ ไม่ใช่คะแนนความเข้าใจภาษามือ',
        '- รูปนิ้วหักตำแหน่งข้อมือออก แต่ไม่หักการเปลี่ยนขนาดมือ; เกณฑ์เป็น 20% ของระยะข้อมือถึงโคนนิ้วกลางต้นทาง',
        '- ระยะข้อมือคือช่องว่างระหว่างข้อมือของ detailed hand กับจุดข้อมือใน pose payload ไม่ใช่ช่องว่างจากขอบภาพแขนจริง; ต่ำลงไม่ได้รับรองว่าภาพเชื่อมเนียน',
        '- F1 จาก ST-GCN ใช้มือหลัง Renderer ร่วมกับลำตัวใน payload ไม่ใช่การอ่านภาพ Avatar ทั้งตัว',
        '- PCK รวมตามจำนวนจุด ส่วน Macro-F1 เฉลี่ยตาม 30 คำ; มีหนึ่งคลิปต่อคำและยังไม่ทราบ train overlap',
        '- การปิดทั้งสองทำให้ใกล้ payload โดยโครงสร้างอยู่แล้ว คะแนนที่สูงขึ้นจึงไม่ใช่หลักฐานเพียงพอว่าภาพดูดีกว่า ต้องดูสัดส่วนมือและการเชื่อมแขนประกอบ',
        '- ไม่ควรตีความความต่าง 1–2 คำในชุดเล็กว่าเป็นการปรับปรุงที่มีนัยสำคัญ', '',
        '## การตรวจสอบ','',
        '- เงื่อนไขเปิดทั้งสองให้พิกัดตรงกับ audit ระบบปัจจุบันเดิมทุกเฟรม',
        '- ตรวจพิกัดจากเส้นที่วาดกับสูตรฉายมือทุกจุด tolerance 1e-6',
        '- rebuild backend ให้ frames ตรง inputs เดิม ก่อนใช้ affine fit เป็นข้อมูลอ้างอิง',
        '- SHA-256 ไฟล์ tracked ก่อน–หลังเท่ากันทั้งหมด; รวมการแก้ launcher ที่มีอยู่ก่อนเริ่มโดยไม่เปลี่ยนมัน',
        '- ภาพแต่ละเงื่อนไขอยู่ในโฟลเดอร์นี้ เช่น S0A0_hello.png และ S1A1_hello.png','',
        'ทำซ้ำ: python recognition_audit/step1_evaluate.py prepare → node recognition_audit/step1_render.mjs → python recognition_audit/step1_evaluate.py evaluate','']
    (OUT/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','evaluate']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else evaluate()
