"""Review actual Canvas captures; native SVG has no invented 21-point skeleton."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json
import hashlib
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
HERE=Path(__file__).resolve().parent;OUT=HERE/'transition_chain_20261002'
cases=_local_json_loads((OUT/'inputs.json').read_text());caps=_local_json_loads((OUT/'rendered.json').read_text())
assert len(caps)==24
lookup={(c['phrase'],c['variant']):c for c in caps}
previous={c['phrase']:c for c in _local_json_loads((HERE/'combined_winners200/renderer/rendered.json').read_text()) if c['variant']=='S1A1'}
rows=[];examples=[]
for case in cases:
    name=case['phrase'];control=lookup[name,'control']['samples'];chain=lookup[name,'chain']['samples']
    old=previous[case['label']+'__no_hold_no_smoothing']['samples']
    assert len(control)==len(chain)==len(old)
    switched=set();native=unreferenced=changed=switch_events=0;factors=[];boundary_factor=[]
    for i,(a,b,o) in enumerate(zip(control,chain,old)):
        assert a['hands']==b['hands']==o['hands'], 'Detailed positions changed'
        for side in ['left','right']:
            x=a['fallback'][side];y=b['fallback'][side]
            assert x['nativePaths']==y['nativePaths']
            assert x['before']==y['before']
            assert x['factor']==1
            if y['nativePaths']:
                native+=1;unreferenced+=y['limit'] is None;factors.append(y['factor']);changed+=y['factor']<1
                if y['limit'] is not None and y['before'] is not None:assert y['after']<=y['limit']+1e-6
            assert abs(y['after']-y['before']*y['factor'])<1e-6 if y['after'] is not None else True
            if i and bool(a['fallback'][side]['nativePaths']) != bool(control[i-1]['fallback'][side]['nativePaths']):
                switch_events+=1;switched.add(i);boundary_factor.append(abs(y['factor']-chain[i-1]['fallback'][side]['factor']))
    pixel={}
    for v in ['control','chain']:
        differences=[]
        for i in sorted(switched):
            folder=OUT/'frames'/v/name
            a=np.asarray(Image.open(folder/f'{i-1:04d}.png').convert('RGB'),dtype=float)
            b=np.asarray(Image.open(folder/f'{i:04d}.png').convert('RGB'),dtype=float)
            differences.append(float(np.abs(a-b).mean()/255))
        pixel[v]=float(np.mean(differences)) if differences else None
    rows.append(dict(label=case['label'],native_hand_frames=native,no_reference_hand_frames=unreferenced,
        changed_native_hand_frames=changed,min_factor=min(factors) if factors else None,
        switch_video_frames=len(switched),switch_hand_events=switch_events,max_factor_jump=max(boundary_factor,default=0),boundary_canvas_difference=pixel))
    if changed and switched:
        # Deterministic most scaled native frame adjacent to a mode boundary.
        indices=sorted({j for i in switched for j in [i-1,i]})
        index=min(indices,key=lambda i:min(chain[i]['fallback'][s]['factor'] for s in ['left','right']))
        examples.append((min(chain[index]['fallback'][s]['factor'] for s in ['left','right']),case,index))
protocol=_local_json_loads((HERE/'bridge_transition_20261002/protocol.json').read_text())
for p,h in protocol['production'].items():assert hashlib.sha256((HERE.parent/p).read_bytes()).hexdigest()==h
result=dict(rows=rows,totals={k:sum(r[k] for r in rows) for k in ['native_hand_frames','no_reference_hand_frames','changed_native_hand_frames','switch_video_frames','switch_hand_events']},
    checks=dict(production_unchanged=True,detailed_endpoints_exact_to_prior=True,native_presence_unchanged=True,
        note='Recognizer input identical: reuse prior 5/12 prediction result, not new inference. Body unchanged by browser assertions. Fewer pixels alone does not prove better continuity or ASL correctness.'))
(OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
# Review four strongest effects, explicitly a diagnostic not representative sample.
for _,case,index in sorted(examples,key=lambda x:x[0])[:4]:
    selected=range(max(0,index-1),min(len(case['frameIndices']),index+2));width=360;height=385
    sheet=Image.new('RGB',(width*len(selected),height*2),'white');draw=ImageDraw.Draw(sheet)
    for row,v in enumerate(['control','chain']):
        for col,i in enumerate(selected):
            im=Image.open(OUT/'frames'/v/case['phrase']/f'{i:04d}.png').convert('RGB');im.thumbnail((360,360))
            sheet.paste(im,(col*width,row*height+25));draw.text((col*width+6,row*height+5),f"{case['label']} {v} frame {i}",fill='black')
    sheet.save(OUT/(case['label'].replace(' ','_')+'_comparison.png'))
print(json.dumps(result,indent=2))
