"""Further diagnostics, not a fitted production correction."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json
import numpy as np
import wrist_coordinate_trace as trace
OUT=trace.OUT;pipe=trace.pipe;base=trace.base
r=_local_json_loads((OUT/'results.json').read_text());p=_local_json_loads((OUT/'protocol.json').read_text())
vectors=[];gau=[];examples=[]
for label in p['review']:
    d=_local_json_loads((OUT/(label.replace(' ','_')+'_trace.json')).read_text());a=d['aligned'];n=len(a['build'])
    for i in range(n):
        for side in ['left','right']:
            if not all(a[s][i]['sides'][side]['valid'] for s in p['stage_order']):continue
            v=np.array(a['build'][i]['sides'][side]['gap']);vectors.append(v)
            gau.append(np.linalg.norm(np.array(a['body_gaussian'][i]['sides'][side]['gap'])-a['interpolate'][i]['sides'][side]['gap']))
    if label in ['parents','remote control','chicken']:
        index={'parents':0,'remote control':18,'chicken':45}[label]
        final=a['fit'][index];record=dict(label=label,frame30=index,sides={})
        for side in ['left','right']:
            s=final['sides'][side]
            if not s['valid']:continue
            width=final['width'];hand=np.array(s['hand']);body=np.array(s['body']);nose=np.array(s['nose'])
            record['sides'][side]=dict(translation_shoulders=((body-hand)/width).tolist(),hand_wrist_to_nose_before=float(np.linalg.norm(hand-nose)/width),target_wrist_to_nose_after=float(np.linalg.norm(body-nose)/width))
        examples.append(record)
v=np.array(vectors);bias=v.mean(axis=0);norm=np.linalg.norm(v,axis=1);res=np.linalg.norm(v-bias,axis=1)
clips=pipe.discover_clips(base.ROOT/'SLclean.sqlite3');unitrows=[]
for label in p['labels']:
    path=pipe.resolve_selected_paths(clips,base.ROOT/'SLclean.sqlite3',[label+'.csv'],'')[0];df=pipe.read_clip_csv(path);est={}
    for part in ['pose','left_hand','right_hand']:
        sub=df[df.part==part];item={}
        for a,pixel in [('x','px'),('y','py')]:
            values=sub[(sub[a].abs()>.05)&sub[pixel].notna()]
            item[a]=float(np.median(values[pixel]/values[a])) if len(values) else None
        est[part]=item
    unitrows.append(dict(label=label,estimates=est))
ratios=[]
for row in unitrows:
    e=row['estimates']
    for side in ['left_hand','right_hand']:
        for a in ['x','y']:
            if e[side][a] is not None and e['pose'][a] and np.isfinite(e[side][a]):ratios.append(e[side][a]/e['pose'][a])
details=dict(diagnostic20=dict(common_pairs=len(v),mean_signed_gap=bias.tolist(),mean_distance=float(norm.mean()),mean_distance_after_removing_global_offset=float(res.mean()),p95_distance=float(np.percentile(norm,95)),body_gaussian_gap_vector_change_mean=float(np.mean(gau)),body_gaussian_gap_vector_change_max=float(max(gau))),
    pixel_units=dict(comparisons=len(ratios),median_hand_to_pose_ratio=float(np.median(ratios)),max_absolute_ratio_error=float(max(abs(np.array(ratios)-1))),count_over_5percent=int(np.sum(abs(np.array(ratios)-1)>.05)),note='Ratios inferred from px/x and py/y; agreement supports same units but cannot establish frame identity or detector accuracy.'),
    pixel_unit_rows=unitrows,examples=examples)
assert base.production()==p['production']
(OUT/'details.json').write_text(json.dumps(details,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in details.items() if k!='pixel_unit_rows'},indent=2))
