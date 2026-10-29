"""Reuse the audited stdlib geometry evaluator with IK-specific payloads."""
from pathlib import Path
audit_dir=Path(__file__).resolve().parent
code=(audit_dir/'step1b_geometry.py').read_text(encoding='utf-8').split("lines=['#")[0]
code=code.replace("OUT=HERE/'step1b'","OUT=HERE/'ik_trial'")
a=code.index('NAMES=');b=code.index('\ndef dist',a)
code=code[:a]+"NAMES={'BASELINE':'ระบบเดิม','IK':'รักษามือ + IK แขน 2D'}"+code[b:]
code=code.replace("(HERE/'inputs.json')","(OUT/'inputs.json')").replace('==7*len(cases)==210','==2*len(cases)==60')
code=code.replace("for variant in ['S0A1','S1A1']:","for variant in ['BASELINE']:").replace("old[variant,case['phrase']]","old['S1A1',case['phrase']]")
code=code.replace("frame=case['payload']['frames'][case['frameIndices'][t]]","frame=case['ikPayload' if variant=='IK' else 'payload']['frames'][case['frameIndices'][t]]")
exec(compile(code,'ik_geometry_adaptation','exec'),globals())
records=[r for case in json.loads((OUT/'arm_metrics.json').read_text()) for r in case['records']]
arm=dict(active_hand_frames=len(records),clamped_percent=100*sum(r['clamped'] for r in records)/len(records),mean_target_shift=statistics.mean(r['target_shift'] for r in records),p95_target_shift=percentile([r['target_shift'] for r in records],.95),mean_elbow_shift=statistics.mean(r['elbow_shift'] for r in records),old_mean_length_deviation_percent=statistics.mean([r[k] for r in records for k in ['old_upper_error','old_lower_error']])*100,new_max_length_error=max(r[k] for r in records for k in ['new_upper_error','new_lower_error']))
temporal={}
for mode in ['payload','ikPayload']:
    jumps=[];acceleration=[]
    for case in cases:
        for side in ['left','right']:
            track=[]
            for frame in case[mode]['frames']:
                pose={k['part']:k for k in frame['pose']['keypoints']}
                def xy(k):return [k['position']['x'],k['position']['y']]
                width=dist(xy(pose['leftShoulder']),xy(pose['rightShoulder']))
                if width<1e-6:continue
                e=xy(pose[side+'Elbow']);s=xy(pose[side+'Shoulder'])
                track.append([(e[i]-s[i])/width for i in [0,1]])
            jumps.extend(dist(track[i],track[i-1]) for i in range(1,len(track)))
            acceleration.extend(math.hypot(*[track[i][j]-2*track[i-1][j]+track[i-2][j] for j in [0,1]]) for i in range(2,len(track)))
    temporal[mode]=dict(jump_p95=percentile(jumps,.95),jump_max=max(jumps),mean_second_difference=statistics.mean(acceleration))
(OUT/'arm_summary.json').write_text(json.dumps(dict(arm=arm,temporal=temporal),indent=2))
lines=['# รักษามือ + IK แขน 2D เทียบระบบเดิม','',
'30 คำเดิม ไม่แก้ production; F1 ใหม่ยังติด Windows Application Control ที่บล็อก PyTorch','',
'| แบบ | PCK@5% ไหล่ | รูปนิ้ว PCK@20% ฝ่ามือ | Gap ข้อมือ (% ไหล่) |','|---|---:|---:|---:|']
for r in rows:lines.append(f"| {NAMES[r['variant']]} | {r['pck5']:.2f}% | {r['shape_pck20']:.2f}% | {r['gap_mean']:.2f}% |")
lines+=['','## วิธีต้นแบบ','',
'- analytic two-bone IK ในระนาบ 2D ยึดหัวไหล่เดิม เลือกข้อศอกโดยดูท่าต้นทางและเฟรมก่อนหน้า',
'- ใช้ median ความยาว upper/lower arm แบบ 2D ต่อคลิป ไม่ใช่ความยาว 3D มนุษย์หรือ rig ที่สอบเทียบข้ามคลิป',
'- จำกัดมุมภายในข้อศอก 15–165 องศา เป็นค่าทดลอง ไม่ใช่เกณฑ์กายวิภาคที่ผ่านการรับรอง',
'- ไม่ scale มือเพิ่ม เมื่อเป้าหมายเอื้อมไม่ถึงจะเลื่อนทั้งมือแบบ rigid translation ไม่เปลี่ยนรูปนิ้ว',
'- คงความยาวในระนาบภาพอาจขัดกับการหดสั้นจากมุมกล้อง จึงไม่ถือเป็น biomechanics 3D',
'','## ผลแขน','',
f"- จำกัดระยะเอื้อม {arm['clamped_percent']:.2f}% ของ {len(records)} hand-frames ที่ใช้ IK (รวมเฟรมเพิ่ม 60 FPS)",
f"- เลื่อนมือเฉลี่ย {arm['mean_target_shift']:.2f} หน่วยพิกัดก่อนซูม; P95 {arm['p95_target_shift']:.2f}",
f"- เลื่อนข้อศอกเฉลี่ย {arm['mean_elbow_shift']:.2f} หน่วยพิกัดก่อนซูม",
f"- deviation จาก median arm length ของ baseline เฉลี่ย {arm['old_mean_length_deviation_percent']:.2f}%; relative error สูงสุด IK {arm['new_max_length_error']:.3e}",
'- ความยาวคงที่เป็นผลจากข้อบังคับ ไม่ใช่หลักฐานว่าแขนถูกต้องกว่าเดิมทางกายวิภาค',
'','## การเปลี่ยนข้อศอกระหว่างเฟรม','',
'พิกัดข้อศอกเทียบหัวไหล่และหารความกว้างไหล่ ค่า jump ต่อเฟรมและ second difference ใช้ตรวจความเปลี่ยนแปลง ไม่ใช่ acceleration หน่วยเวลาจริง','',
'| แบบ | Jump P95 | Jump max | Mean second difference |','|---|---:|---:|---:|']
for mode,t in temporal.items():lines.append(f"| {mode} | {t['jump_p95']:.5f} | {t['jump_max']:.5f} | {t['mean_second_difference']:.5f} |")
lines+=['','## ข้อจำกัด','',
'- มืออ่านจาก Paper.js strokes หลังวาดจริง แต่ความยาวแขนและ gap วัด joint targets ไม่ใช่ขอบภาพผิวแขนจริง',
'- ต้องแยกผล IK จากการปิด scale มือ: control ปิด scale แต่ยึดข้อมือใน step1 ได้ PCK 29.37% และรูปนิ้ว 75.12%',
'- 37,422 จุด fully observed source30 hands; global fit เดิม ไม่ใช่ความเข้าใจของผู้ใช้',
'- ตรวจ baseline กับ capture เดิม tolerance 1e-6 และ hash tracked production ไม่เปลี่ยน',
'- ตรวจภาพกลางท่า 5 คำประกอบ ไม่ใช่การตรวจทุกเฟรมโดยคน ไม่ใช่ held-out evaluation',
'- ไม่ได้นำต้นแบบใช้จริงและไม่อ้าง F1 ใหม่','']
(OUT/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(geometry=rows,arm=arm,temporal=temporal),indent=2))
