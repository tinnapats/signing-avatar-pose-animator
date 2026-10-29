"""Independent data invariants; no model scoring or parameter changes."""
import json
from pathlib import Path
import numpy as np
import bounded_forearm200 as trial
HERE=trial.HERE;OUT=HERE/'frozen100_20261005'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
p=read(OUT/'protocol.json');index=read(OUT/'case_index.json');assert len(index)==300
assert not set(p['labels'])&set(p['excluded']);assert len(set(p['labels']))==100
maxbone=0.;maxbody=0.;lengths=[]
for i in range(0,300,3):
    c=[read(Path(d['file'])) for d in index[i:i+3]]
    assert [x['method'] for x in c]==['BASE','SIZE_ONLY','BOUNDED_SIZE']
    assert all(x['frameIndices']==c[0]['frameIndices'] for x in c)
    assert c[0]['payload']==c[1]['payload']
    lengths.append(len(c[0]['frameIndices']))
    assert len(c[0]['payload']['frames'])==len(c[2]['payload']['frames'])
    for a,b in zip(c[0]['payload']['frames'],c[2]['payload']['frames']):
        assert a['hands']==b['hands'] and a.get('face')==b.get('face')
        pa={k['part']:k for k in a['pose']['keypoints']};pb={k['part']:k for k in b['pose']['keypoints']}
        for k in pa:
            if k not in ['leftWrist','rightWrist']:assert pa[k]==pb[k]
            else:assert pa[k]['score']==pb[k]['score']
        for s in ['left','right']:
            ea,wa=trial.xy(pa[s+'Elbow']),trial.xy(pa[s+'Wrist']);eb,wb=trial.xy(pb[s+'Elbow']),trial.xy(pb[s+'Wrist'])
            maxbone=max(maxbone,abs(np.linalg.norm(wa-ea)-np.linalg.norm(wb-eb)))
assert maxbone<1e-8 and trial.base.production()==p['production']
extra={str(HERE/f):trial.base.sha(HERE/f) for f in ['frozen100_render.mjs','combined_render.mjs','step1b_render.mjs','frozen100_verify.py']}
result=dict(labels=100,prior_excluded=len(p['excluded']),overlap=0,cases=300,frames30fps_per_variant=sum(lengths),min_frames=min(lengths),max_frames=max(lengths),max_forearm_length_error=maxbone,body_except_wrists_exact=True,hands_face_exact=True,source_and_size_payload_exact=True,renderer_hashes=extra,production_unchanged=True)
(OUT/'data_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)
