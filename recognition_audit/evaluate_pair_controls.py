"""Factorial controls for known-video hand-pair geometry (not linguistic scores)."""
import json
import os
from pathlib import Path
import numpy as np
from trace_paired_stages import HERE,OUT,PAIRED,measure
from run_audit import frame_array
from step1_evaluate import hashes

cases=json.loads((PAIRED/'inputs.json').read_text())
trace={r['word']:r for r in json.loads((OUT/'results.json').read_text())}
OUT=Path(os.environ.get('PAIR_CONTROL_OUTPUT',str(OUT)))
capture={(r['variant'],r['phrase']):r['samples'] for r in json.loads((OUT/'rendered.json').read_text())}
results=[]
for case in cases:
    word=case['phrase'];frames=[case['payload']['frames'][i] for i in case['frameIndices']]
    payload=frame_array(frames)
    source={r['frame']:r for r in trace[word]['stages'][0]['frames']}
    for variant in sorted({k[0] for k in capture}):
        sample=capture[variant,word];data=payload.copy();gaps=[];scales=[]
        for t in source:
            pose={k['part']:k for k in frames[t]['pose']['keypoints']}
            width=np.linalg.norm(payload[t,11]-payload[t,12])
            for side,offset in [('left',33),('right',54)]:
                hand=sample[t]['hands'][side];assert len(hand)==21
                data[t,offset:offset+21]=[hand[str(i)] for i in range(21)]
                wrist=pose[side+'Wrist']['position']
                gaps.append(float(np.linalg.norm(data[t,offset]-[wrist['x'],wrist['y']])/width))
                a=payload[t,offset:offset+21]-payload[t,offset];b=data[t,offset:offset+21]-data[t,offset]
                if np.sum(a*a)>1e-9:scales.append(float(np.sum(a*b)/np.sum(a*a)))
        values=[measure(data[t]) for t in source]
        error=[abs(v['nearest_joint_distance']-source[t]['nearest_joint_distance']) for t,v in zip(source,values)]
        results.append(dict(word=word,variant=variant,n=len(source),mean_nearest_distance=float(np.mean([v['nearest_joint_distance'] for v in values])),mean_abs_nearest_distance_error=float(np.mean(error)),mean_wrist_gap=float(np.mean(gaps)),median_scale=float(np.median(scales))))
assert hashes()==json.loads((PAIRED/'production_before.json').read_text())
(OUT/'controls_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
lines=['# แยกสาเหตุจาก renderer ด้วย 4 เงื่อนไข','', 'S = การปรับขนาดมือ, A = การยึดข้อมือกับแขน; 1 เปิด / 0 ปิด ทุกระยะหารความกว้างไหล่ ใช้เฟรมสองมือครบชุดเดียวกับ trace','', '| คำ | วิธี | จุดใกล้สุดเฉลี่ย | ความคลาดเคลื่อนระยะจากต้นทาง | Gap จุดข้อมือเฉลี่ย | Scale มัธยฐาน |','|---|---|---:|---:|---:|---:|']
for r in results:lines.append(f"| {r['word']} | {r['variant']} | {r['mean_nearest_distance']:.4f} | {r['mean_abs_nearest_distance_error']:.4f} | {r['mean_wrist_gap']:.4f} | {r['median_scale']:.3f} |")
lines+=['','การปิดการฉายมือย่อมใกล้ payload ขึ้นโดยโครงสร้าง แต่ไม่รับรองความสวยงามหรือการเชื่อมแขน อย่าตัดสินจาก metric เดียว ระยะจุดใกล้สุดไม่ใช่ระยะผิวสัมผัส','ไม่เปลี่ยน production; 3 คำเป็นชุดวิเคราะห์ ไม่ใช่ held-out evidence','']
(OUT/'CONTROLS_TH.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(results,indent=2))
