"""Frozen, model-input-only temporal normalization; no new rendered frames."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json
import math
import time
from pathlib import Path
import numpy as np
import novel_words30 as previous
base=previous.base;HERE=base.HERE;OUT=HERE/'equal_duration_20261002'

def select_indices(n,target):
    assert n>0 and target>1
    return np.floor(np.linspace(0,n-1,target)+.5).astype(int)

def arrays(case,cap):
    frames=[case['payload']['frames'][i] for i in case['frameIndices']]
    a=base.audit.frame_array(frames);assert len(a)==len(cap['samples'])
    for i,s in enumerate(cap['samples']):
        for side,offset in [('left',33),('right',54)]:
            a[i,offset:offset+21]=0
            for k,p in s['hands'][side].items():a[i,offset+int(k)]=p
    return a

def paired(rows,labels):
    by={(r['label'],r['variant']):r for r in rows};g=[];l=[]
    for label in labels:
        a=by[label,'original']['prediction']==label;b=by[label,'equal_frames']['prediction']==label
        if b and not a:g.append(label)
        if a and not b:l.append(label)
    n=len(g)+len(l);p=min(1.,2*sum(math.comb(n,k) for k in range(min(len(g),len(l))+1))/2**n) if n else 1.
    return dict(gained=g,lost=l,exact_mcnemar_p=p,bonferroni_three=min(1,p*3))

def main():
    assert not OUT.exists(),'Do not overwrite frozen experiment'
    for n in [1,2,17,80,200]:
        idx=select_indices(n,64);assert len(idx)==64 and idx[0]==0 and idx[-1]==n-1 and np.all(np.diff(idx)>=0)
        assert np.array_equal(select_indices(n,n),np.arange(n)) if n>1 else True
    before=base.production();source=HERE/'original_method200_20260930'
    cases=_local_json_loads((source/'inputs.json').read_text());assert len(cases)==200
    lengths=[len(c['frameIndices']) for c in cases];target=max(2,int(np.floor(np.median(lengths)+.5)))
    OUT.mkdir()
    sources=[source/'inputs.json',source/'rendered.json',source/'results.json',HERE/'novel_words30_20261001/review_inputs.json',HERE/'novel_words30_20261001/renderer/rendered.json',HERE/'novel_words30_20261001/renderer_results.json']
    protocol=dict(production=before,target_frames=target,selection='Rounded median frame count of development200, without looking at scores. No target-length sweep.',
        method='Nearest existing frame at uniformly spaced normalized times; endpoints retained. Select entire body and both hands together. Missing zeros selected unchanged; no coordinate interpolation, extrapolation, trimming, holding repair or Gaussian.',
        nominal_duration_seconds=(target-1)/30,source_fps=30,labels200=[c['phrase'] for c in cases],
        source_hashes={str(p.relative_to(HERE)):base.sha(p) for p in sources},checkpoint_hash=base.sha(HERE/'wlasl_stgcn.zip'),
        comparisons=['original200 production baseline','previous novel30 baseline','previous novel30 no_hold_no_smoothing'],
        note='Post-rendered detailed hands plus payload body, not RGB Avatar recognition. 30-word set previously inspected, not pristine holdout. No playback change. No model training. Bonferroni3 for paired tests.',
        original200_length_stats=dict(min=min(lengths),median=float(np.median(lengths)),max=max(lengths)))
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print('FROZEN TARGET',target,protocol['original200_length_stats'],flush=True)
    _,predict=previous.load_predictor();rows=[];summaries={}
    def run(group,cases,caps,saved):
        labels=[c.get('label',c['phrase']) for c in cases];assert len(set(labels))==len(labels)
        for i,c in enumerate(cases):
            label=c.get('label',c['phrase']);a=arrays(c,caps[c['phrase']]);idx=select_indices(len(a),target)
            for v,data in [('original',a),('equal_frames',a[idx])]:
                started=time.perf_counter();r=predict(data,label,v);r['runtime_ms']=(time.perf_counter()-started)*1000
                if v=='original':assert r['prediction']==saved[label],f'Control mismatch {group} {label}'
                r.update(group=group,input_frames=len(data),original_frames=len(a),unique_selected_frames=len(set(idx)),
                    missing_hand_fraction=float(np.mean([np.all(data[:,o:o+21]==0,axis=(1,2)) for o in [33,54]])))
                rows.append(r)
            if (i+1)%20==0:print(group,i+1,'/',len(cases),flush=True)
        current=[r for r in rows if r['group']==group]
        summaries[group]=dict(metrics={v:base.metrics([r for r in current if r['variant']==v],labels) for v in ['original','equal_frames']},paired=paired(current,labels))
        (OUT/'progress.json').write_text(json.dumps(dict(summaries=summaries,predictions=rows),indent=2),encoding='utf-8')
    caps={c['phrase']:c for c in _local_json_loads((source/'rendered.json').read_text())}
    saved={r['label']:r['prediction'] for r in _local_json_loads((source/'results.json').read_text())['predictions'] if r['variant']=='rendered_hands_payload_body'}
    run('original200',cases,caps,saved)
    del cases,caps
    source30=HERE/'novel_words30_20261001'
    cases=_local_json_loads((source30/'review_inputs.json').read_text());caps={c['phrase']:c for c in _local_json_loads((source30/'renderer/rendered.json').read_text())}
    savedrows=_local_json_loads((source30/'renderer_results.json').read_text())['predictions']
    for v in ['baseline','no_hold_no_smoothing']:
        run('prior30_'+v,[c for c in cases if c['variant']==v],caps,{r['label']:r['prediction'] for r in savedrows if r['variant']==v})
    assert base.production()==before
    for p in sources:assert base.sha(p)==protocol['source_hashes'][str(p.relative_to(HERE))]
    result=dict(protocol=protocol,summaries=summaries,predictions=rows,production_unchanged=True,all_original_top1_exact=True,source_artifacts_unchanged=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()
