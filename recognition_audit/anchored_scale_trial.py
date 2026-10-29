"""Frozen scale-only experiment; original wrist anchoring remains at 100%."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from geometry_100 import metric,summarize
from run_audit import pipeline,frame_array
from step1_evaluate import hashes

HERE=Path(__file__).resolve().parent
DATA=HERE/'recognition_100'
OUT=HERE/'anchored_scale_trial'
VARIANTS=[dict(id='S1A1',scale=1,anchor=1),dict(id='S50A1',scale=.5,anchor=1),dict(id='S0A1',scale=0,anchor=1)]

def render():
    OUT.mkdir(exist_ok=True)
    protocol=dict(variants=VARIANTS,sample='same 100 development clips; no model-based selection',
        criteria='Report all three. Exact baseline reproduction; no added wrist gap; do not deploy from geometry alone.',
        production_hashes=hashes(),inputs_sha256=hashlib.sha256((DATA/'inputs.json').read_bytes()).hexdigest())
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    env=dict(os.environ,AUDIT_OUTPUT=str(OUT),AUDIT_INPUTS=str(DATA/'inputs.json'),AUDIT_VARIANTS=json.dumps(VARIANTS),AUDIT_REVIEW_FRAMES='1')
    subprocess.run(['C:/Users/newtu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe',str(HERE/'step1b_render.mjs')],env=env,check=True)

def evaluate():
    protocol=json.loads((OUT/'protocol.json').read_text()); assert hashes()==protocol['production_hashes']
    cases=json.loads((DATA/'inputs.json').read_text())
    fits={r['word']:r['global_fit'] for r in json.loads((HERE/'geometry_100/results.json').read_text())['rows']}
    baseline={r['phrase']:r['samples'] for r in json.loads((DATA/'rendered.json').read_text())}
    lookup={(r['variant'],r['phrase']):r['samples'] for r in json.loads((OUT/'rendered.json').read_text())}
    assert len(lookup)==300
    pooled={v['id']:[] for v in VARIANTS}; gaps={k:[] for k in pooled}; perword=[]
    for case in cases:
        word=case['phrase']; old=baseline[word]; current=lookup['S1A1',word]
        assert len(old)==len(current)
        for a,b in zip(old,current):
            for side in ['left','right']:
                assert a['hands'][side].keys()==b['hands'][side].keys()
                for key,point in a['hands'][side].items():assert np.linalg.norm(np.array(point)-b['hands'][side][key])<1e-6
        raw=pipeline.build_frames(pipeline.concat_clips([DATA/'csv'/(word+'.csv')],513,513,0,target_fps=30),513,513,0)
        fit=fits[word]; ref=frame_array(raw)*fit['scale']+fit['offset']; assert len(raw)==len(old)
        local={k:[] for k in pooled}
        for t,f in enumerate(raw):
            pose={k['part']:k for k in f['pose']['keypoints']}; shoulder=np.linalg.norm(ref[t,11]-ref[t,12])
            if shoulder<=1e-6 or min(pose[s]['score'] for s in ['leftShoulder','rightShoulder'])<=0:continue
            frame=case['payload']['frames'][case['frameIndices'][t]]
            outpose={k['part']:k for k in frame['pose']['keypoints']}
            for side,offset in [('left',33),('right',54)]:
                h=f.get('hands',{}).get(side,{})
                if not h.get('observed'):continue
                mask=np.array([k.get('score',0)>0 for k in h['keypoints']])
                for v in pooled:
                    pred=np.full((21,2),np.nan)
                    for key,point in lookup[v,word][t]['hands'][side].items():pred[int(key)]=point
                    m=metric(ref[t,offset:offset+21],pred,mask,shoulder);local[v].append(m);pooled[v].append(m)
                    wrist=outpose[side+'Wrist']
                    if wrist['score']>=.2 and np.isfinite(pred[0]).all():
                        gaps[v].append(float(np.linalg.norm(pred[0]-[wrist['position']['x'],wrist['position']['y']])/shoulder))
        perword.append(dict(word=word,metrics={k:summarize(v) for k,v in local.items()}))
    summaries={k:dict(**summarize(v),wrist_gap_mean=float(np.mean(gaps[k])),wrist_gap_max=float(np.max(gaps[k]))) for k,v in pooled.items()}
    for s in summaries.values():assert s['wrist_gap_max']<1e-8
    oldsum=json.loads((HERE/'geometry_100/results.json').read_text())['pooled']['source_to_rendered']
    for key in ['position_pck5_pct','distance_mean','bone_mean','angle_mean']:assert abs(summaries['S1A1'][key]-oldsum[key])<1e-8
    assert hashes()==protocol['production_hashes']
    (OUT/'results.json').write_text(json.dumps(dict(summaries=summaries,perword=perword,baseline_verified=True,production_unchanged=True),indent=2,allow_nan=False))
    lines=['# ทดลองขนาดมือ โดยยึดข้อมือกับแขนเหมือนเดิม','',
      '100 คลิปชุดพัฒนาเดิม วาดจริงใน browser sandbox รวม 300 sequences ตรวจ baseline ตรงทุกจุดกับ capture เดิม ไม่แก้ production', '',
      '| วิธี | PCK@5% ไหล่ | ระยะผิด/ไหล่ | รูปมือผิด/ฝ่ามือ | ความยาวท่อนผิด | มุมผิด | ช่องว่างข้อมือ/ไหล่ |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for k,s in summaries.items():lines.append(f"| {k} | {s['position_pck5_pct']:.2f}% | {100*s['distance_mean']:.2f}% | {100*s['shape_mean']:.2f}% | {100*s['bone_mean']:.2f}% | {s['angle_mean']:.2f}° | {100*s['wrist_gap_mean']:.6f}% |")
    lines+=['','S1A1 = ขนาดเดิม; S50A1 = ลดแรงปรับขนาดครึ่งหนึ่ง (ไม่ใช่ย่อมือครึ่งหนึ่ง); S0A1 = ไม่ปรับขนาดเพิ่ม ทั้งหมดใช้ข้อมือเดิม 100%',
      'ค่า error เป็นค่า 2D จากจุดที่สกัด ไม่ใช่ความถูกต้อง ASL หรือกระดูก 3D; PCK ไม่ใช่เปอร์เซ็นต์ที่ผู้ใช้เข้าใจ',
      'Wrist gap วัดเทียบข้อมือ pose payload ไม่ใช่ระยะถึงขอบผิวแขนในภาพ จึงต้องตรวจภาพด้วย',
      'การปิด scale อาจเข้าใกล้ source โดยนิยาม ไม่ยืนยันความสมส่วนกับแขนหรือความอ่านง่ายของท่า ห้ามใช้คะแนนนี้อย่างเดียวตัดสิน deploy',
      'เป็น development sensitivity study ไม่ใช่ held-out test ไม่เปลี่ยนระบบหลัก', '']
    (OUT/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines))

if __name__=='__main__':globals()[sys.argv[1]]()
