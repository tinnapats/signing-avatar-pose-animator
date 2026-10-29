"""Trace hand-pair geometry at each production stage without editing production."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import copy
import json
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
from run_audit import HERE, pipeline, frame_array
from step1_evaluate import hashes

OUT=HERE/'paired_trace'
PAIRED=HERE/'paired_video'
STAGES=['build_frames','repair_hand_tracks','taper_hand_detection_edges','interpolate_frames','apply_gaussian_smoothing','apply_hand_gaussian_smoothing','stabilize_hand_bone_lengths','enforce_hand_articulation','fit_frames_to_canvas']

def measure(a):
    left=a[33:54];right=a[54:75]
    shoulder=float(np.linalg.norm(a[11]-a[12]))
    if shoulder<=1e-5:return None
    distances=np.linalg.norm(left[:,None,:]-right[None,:,:],axis=2)/shoulder
    return dict(nearest_joint_distance=float(distances.min()),wrist_distance=float(np.linalg.norm(left[0]-right[0])/shoulder),palm_center_distance=float(np.linalg.norm(left[[0,5,9,13,17]].mean(axis=0)-right[[0,5,9,13,17]].mean(axis=0))/shoulder))

def main():
    before=hashes();OUT.mkdir(exist_ok=True)
    cases=_local_json_loads((PAIRED/'inputs.json').read_text())
    captures={r['phrase']:r['samples'] for r in _local_json_loads((PAIRED/'rendered.json').read_text())}
    allrows=[]
    for case in cases:
        snapshots={}
        def wrapper(name,fn):
            def traced(*args,**kwargs):
                output=fn(*args,**kwargs)
                snapshots[name]=copy.deepcopy(output if isinstance(output,list) else args[0])
                return output
            return traced
        with ExitStack() as stack:
            for name in STAGES:stack.enter_context(patch.object(pipeline,name,wrapper(name,getattr(pipeline,name))))
            payload=pipeline.build_payload(PAIRED/'csv',[case['phrase']+'.csv'],'',30,513,513,0,0,2,1.2,2)
        assert payload['frames']==case['payload']['frames']
        n=len(case['frameIndices']);raw=frame_array(snapshots['build_frames'])
        valid=np.all(np.any(raw[:,33:75]!=0,axis=2),axis=1)&np.all(np.any(raw[:,[11,12]]!=0,axis=2),axis=1)
        arrays={}
        for name,frames in snapshots.items():
            indexes=list(range(n)) if len(frames)==n else case['frameIndices']
            arrays[name]=frame_array([frames[i] for i in indexes])
        rendered=arrays['fit_frames_to_canvas'].copy()
        for t,sample in enumerate(captures[case['phrase']]):
            for side,offset in [('left',33),('right',54)]:
                rendered[t,offset:offset+21]=0
                for key,point in sample['hands'][side].items():rendered[t,offset+int(key)]=point
            valid[t] &= all(len(sample['hands'][side])==21 for side in ['left','right'])
        arrays['renderer']=rendered
        stage_rows=[]
        for name,a in arrays.items():
            metrics=[dict(frame=int(t),time_sec=t/30,**measure(a[t])) for t in np.flatnonzero(valid) if measure(a[t]) is not None]
            stage_rows.append(dict(stage=name,frames=metrics,mean_nearest_joint_distance=float(np.mean([m['nearest_joint_distance'] for m in metrics])),mean_wrist_distance=float(np.mean([m['wrist_distance'] for m in metrics]))))
        allrows.append(dict(word=case['phrase'],observed_pair_frames=int(valid.sum()),stages=stage_rows))
    assert before==hashes()
    (OUT/'results.json').write_text(json.dumps(allrows,indent=2),encoding='utf-8')
    lines=['# ระยะระหว่างมือ แยกขั้นตอน','', 'ทุกค่าเป็นระยะ 2D หารความกว้างไหล่ในเฟรมนั้น ใช้เฉพาะเฟรมต้นทางที่พบครบ 21 จุดทั้งสองมือและ renderer แสดงครบ ไม่ใช่การวัดการสัมผัสผิวมือหรือความถูกต้องภาษามือ','']
    for row in allrows:
        lines += [f"## {row['word']} ({row['observed_pair_frames']} เฟรม)",'','| ขั้น | ระยะจุดใกล้สุดเฉลี่ย | ระยะข้อมือเฉลี่ย |','|---|---:|---:|']
        for s in row['stages']:lines.append(f"| {s['stage']} | {s['mean_nearest_joint_distance']:.4f} | {s['mean_wrist_distance']:.4f} |")
    lines+=['','ค่าเล็กไม่ได้แปลว่าดีกว่าเสมอ: บางท่าต้องแยกมือ เปรียบเทียบกับต้นทางและวิดีโอ ไม่ปรับให้มือชิดกันทุกเฟรม','Hash production ก่อน–หลังเท่าเดิม และ payload สร้างซ้ำตรงกับรอบก่อนทุกเฟรม','']
    (OUT/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8')
    for row in allrows:
        print(row['word'],[(s['stage'],round(s['mean_nearest_joint_distance'],4),round(s['mean_wrist_distance'],4)) for s in row['stages']])

if __name__=='__main__':main()
