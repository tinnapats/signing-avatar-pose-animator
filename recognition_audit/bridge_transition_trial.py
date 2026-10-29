"""Isolated short-gap bridge screening on the existing diagnostic twelve words."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import copy
import json
import sys
from collections import Counter
import numpy as np
import novel_words30 as model
base = model.base
OUT = base.HERE / 'bridge_transition_20261002'
SOURCE = base.HERE / 'combined_winners200'

def visible(h):
    return h.get('score', 0) >= .001 and sum(k.get('score', 0) > 0 for k in h.get('keypoints', [])) >= 6

def complete(h):
    return visible(h) and len(h.get('keypoints', [])) == 21 and all(k.get('score', 0) > 0 for k in h['keypoints'])

def xyz(h):
    return np.array([[k['position'].get(a, 0) for a in ['x','y','z']] for k in h['keypoints']])

def prepare():
    assert not OUT.exists(), 'Do not overwrite an experiment'
    cases = [c for c in _local_json_loads((SOURCE/'review_inputs.json').read_text()) if c['variant'] == 'no_hold_no_smoothing']
    assert len(cases) == 12
    OUT.mkdir()
    protocol = dict(production=base.production(), source_hash=base.sha(SOURCE/'review_inputs.json'),
        labels=[c['label'] for c in cases], max_missing_frames=4, payload_fps=60,
        max_endpoint_wrist_travel_shoulders=.25,
        note='Diagnostic development12, not independent test. Only internally bounded invisible gaps, full21 endpoints, no observed frame replacement. 3D kinematic interpolation; no hold/extrapolation. Keep original pipeline/model/scale. Stop if no eligible gaps; do not raise gap limit to chase results.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    output=[];audit=[]
    for original in cases:
        original=copy.deepcopy(original);original['phrase']=original['label']+'__control';original['variant']='control'
        candidate=copy.deepcopy(original);candidate['phrase']=original['label']+'__bridge';candidate['variant']='bridge'
        frames=candidate['payload']['frames'];changes=[];runs=[]
        for side in ['left','right']:
            i=0
            while i<len(frames):
                if visible(frames[i]['hands'][side]):i+=1;continue
                start=i
                while i<len(frames) and not visible(frames[i]['hands'][side]):i+=1
                end=i;reason='eligible';n=end-start
                if start==0 or end==len(frames):reason='unbounded'
                elif n>4:reason='too_long'
                elif any(frames[j]['hands'][side].get('observed') for j in range(start,end)):reason='observed_do_not_replace'
                else:
                    a=frames[start-1]['hands'][side];b=frames[end]['hands'][side]
                    if not complete(a) or not complete(b):reason='incomplete_endpoints'
                    else:
                        pose={k['part']:k for k in frames[start-1]['pose']['keypoints']}
                        l=pose.get('leftShoulder',{});r=pose.get('rightShoulder',{})
                        if min(l.get('score',0),r.get('score',0))<.2:reason='unreliable_shoulders'
                        else:
                            width=np.hypot(l['position']['x']-r['position']['x'],l['position']['y']-r['position']['y'])
                            if width<=1e-6 or np.linalg.norm(xyz(b)[0,:2]-xyz(a)[0,:2])/width>.25:reason='wrist_travel'
                runs.append(dict(side=side,start=start,end=end,length=n,reason=reason))
                if reason!='eligible':continue
                for j in range(start,end):
                    t=(j-start+1)/(n+1);h=base.audit.pipeline.interpolate_hand_kinematically(a,b,t)
                    assert not h['observed'] and not h['flipHeld'] and complete(h)
                    # Verify each synthesized 3D bone is the endpoint-length interpolation.
                    for u,v in base.audit.pipeline.HAND_KINEMATIC_EDGES:
                        expected=(1-t)*np.linalg.norm(xyz(a)[v]-xyz(a)[u])+t*np.linalg.norm(xyz(b)[v]-xyz(b)[u])
                        assert abs(np.linalg.norm(xyz(h)[v]-xyz(h)[u])-expected)<1e-7
                    frames[j]['hands'][side]=h;changes.append([j,side])
        for j,(a,b) in enumerate(zip(original['payload']['frames'],frames)):
            assert {k:v for k,v in a.items() if k!='hands'}=={k:v for k,v in b.items() if k!='hands'}
            for side in ['left','right']:
                if [j,side] not in changes:assert a['hands'][side]==b['hands'][side]
        output.extend([original,candidate]);audit.append(dict(label=original['label'],runs=runs,changes=changes))
    assert base.production()==protocol['production']
    (OUT/'review_inputs.json').write_text(json.dumps(output),encoding='utf-8')
    (OUT/'gap_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(dict(reasons=dict(Counter(r['reason'] for a in audit for r in a['runs'])),changed_hand_frames=sum(len(a['changes']) for a in audit)),indent=2))

def score():
    protocol=_local_json_loads((OUT/'protocol.json').read_text());_,predict=model.load_predictor()
    cases={c['phrase']:c for c in _local_json_loads((OUT/'review_inputs.json').read_text())}
    captures=_local_json_loads((OUT/'renderer/rendered.json').read_text());assert len(captures)==24
    old={c['phrase']:c for c in _local_json_loads((SOURCE/'renderer/rendered.json').read_text()) if c['variant']=='S1A1'}
    rows=[]
    for cap in captures:
        c=cases[cap['phrase']];frames=[c['payload']['frames'][i] for i in c['frameIndices']]
        if c['variant']=='control':assert cap['samples']==old[c['label']+'__no_hold_no_smoothing']['samples']
        a=base.audit.frame_array(frames);last={};counts=Counter()
        for i,s in enumerate(cap['samples']):
            for side,offset in [('left',33),('right',54)]:
                d=bool(s['hands'][side]);n=bool(s['native'][side]);counts['detailed']+=d;counts['native']+=n
                state=(d,n)
                if side in last and state!=last[side]:counts['switches']+=1
                last[side]=state;a[i,offset:offset+21]=0
                for k,p in s['hands'][side].items():a[i,offset+int(k)]=p
        r=predict(a,c['label'],c['variant']);r.update(counts);rows.append(r)
    summaries={v:base.metrics([r for r in rows if r['variant']==v],protocol['labels']) for v in ['control','bridge']}
    totals={v:{k:sum(r[k] for r in rows if r['variant']==v) for k in ['native','detailed','switches']} for v in summaries}
    assert base.production()==protocol['production']
    result=dict(summaries=summaries,counts=totals,predictions=rows,production_unchanged=True,control_capture_exact=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':globals()[sys.argv[1]]()
