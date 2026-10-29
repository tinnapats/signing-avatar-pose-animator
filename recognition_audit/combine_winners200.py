"""Predeclared interactions, not a per-word oracle or a new production pipeline."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json
from unittest.mock import patch
import numpy as np
import torch
import repair_parts200 as parts
base=parts.base
OUT=base.HERE/'combined_winners200'
VS=['no_repair_no_smoothing','no_hold_no_smoothing']

def main():
    assert not (OUT/'results.json').exists()
    before=base.production();assert before==_local_json_loads((parts.OUT/'protocol.json').read_text())['production']
    cases=_local_json_loads((base.SOURCE/'inputs.json').read_text());labels=[c['phrase'] for c in cases]
    controls=[]
    for folder,variants in [('processing_diagnosis200',['baseline','no_hand_smoothing']),('repair_parts200',['no_hold','no_repair'])]:
        controls.extend(r for r in _local_json_loads((base.HERE/folder/'results.json').read_text())['predictions'] if r['variant'] in variants)
    assert len(controls)==800 and len(labels)==200
    lookup={(r['label'],r['variant']):r for r in controls}
    OUT.mkdir(exist_ok=True);(OUT/'review_cases').mkdir(exist_ok=True)
    protocol=dict(labels=labels,variants=VS,review=parts.REVIEW,production=before,fps=30,
        prior_hashes={f:base.sha(base.HERE/f/'results.json') for f in ['processing_diagnosis200','repair_parts200','gated_hold200','gap_split200']},
        input_sha256=base.sha(base.SOURCE/'inputs.json'),checkpoint_sha256=base.sha(base.HERE/'wlasl_stgcn.zip'),
        renderer_scales=[1,0.5],note='Two fixed combinations. No Gaussian hand smoothing; retain body smoothing, taper and articulation. Development200. S50A1 renderer tested only on diagnostic12, not 200.')
    if (OUT/'protocol.json').exists():assert _local_json_loads((OUT/'protocol.json').read_text())==protocol
    else:(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    network,norm,vocab=base.audit.load_model()
    def predict(frames,label):
        a=base.audit.frame_array(frames);t=torch.from_numpy(a[:,base.audit.PICK,:].transpose(2,0,1).copy())
        t=norm({'frames':t})['frames'].unsqueeze(0);assert torch.isfinite(t).all()
        with torch.inference_mode():p=network(t)[0].softmax(0)
        top=p.topk(5)
        return dict(label=label,prediction=vocab[int(top.indices[0])],top5=[vocab[int(i)] for i in top.indices],target_probability=float(p[vocab.index(label)]))
    rows=[];done=0
    if (OUT/'progress.json').exists():
        ck=_local_json_loads((OUT/'progress.json').read_text());rows=ck['predictions'];done=ck['completed']
        assert len(rows)==2*done
        for i,r in enumerate(rows):assert r['label']==labels[i//2] and r['variant']==VS[i%2]
    for i,c in enumerate(cases):
        if i<done:continue
        label=c['phrase'];check=predict(c['payload']['frames'][::2],label)
        assert all(check[k]==lookup[label,'baseline'][k] for k in ['prediction','top5','target_probability'])
        review=[]
        if label in parts.REVIEW:
            review=[r for r in _local_json_loads((parts.OUT/'review_cases'/f'{label}.json').read_text()) if r['variant'] in ['baseline','no_hold','no_repair']]
        for v in VS:
            fn=(lambda frames,**kwargs:frames) if v=='no_repair_no_smoothing' else parts.make_repair(True,False)
            with patch.object(base.audit.pipeline,'repair_hand_tracks',fn):
                p=base.audit.pipeline.build_payload(base.ROOT/'SLclean.sqlite3',[label+'.csv'],'',30,513,513,0,0,2,1.2,2,hand_gaussian_sigma=0,hand_gaussian_radius=0)
            frames=p['frames'][::2];assert len(frames)==len(c['source30'])
            r=predict(frames,label);r.update(variant=v,geometry=base.geometry(frames),coverage=parts.coverage(frames));rows.append(r)
            if label in parts.REVIEW:review.append(dict(phrase=label+'__'+v,label=label,variant=v,payload=p,frameIndices=list(range(0,len(p['frames']),2)),coverage=parts.coverage(frames)))
        if review:(OUT/'review_cases'/f'{label}.json').write_text(json.dumps(review),encoding='utf-8')
        (OUT/'progress.json').write_text(json.dumps(dict(completed=i+1,predictions=rows)),encoding='utf-8')
        print(f'{i+1}/200 {label}: '+str([(r['variant'],r['prediction']) for r in rows[-2:]]),flush=True)
    combined=controls+rows
    summaries={v:base.metrics([r for r in combined if r['variant']==v],labels) for v in ['baseline','no_hand_smoothing','no_hold','no_repair']+VS}
    assert base.production()==before
    for f,h in protocol['prior_hashes'].items():assert base.sha(base.HERE/f/'results.json')==h
    (OUT/'results.json').write_text(json.dumps(dict(summaries=summaries,predictions=combined,production_unchanged=True,baseline_exact=True),indent=2),encoding='utf-8')
    review=[]
    for l in parts.REVIEW:review.extend(_local_json_loads((OUT/'review_cases'/f'{l}.json').read_text()))
    assert len(review)==60
    (OUT/'review_inputs.json').write_text(json.dumps(review),encoding='utf-8')
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()
