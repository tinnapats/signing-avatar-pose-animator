"""One fixed mild Gaussian candidate, short gap kept, hold disabled."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json
from unittest.mock import patch
import torch
import repair_parts200 as parts
base=parts.base;OUT=base.HERE/'mild_smoothing200'
VARIANT='no_hold_mild_smoothing'

def main():
    assert not (OUT/'results.json').exists()
    before=base.production();prior=base.HERE/'combined_winners200'
    assert before==_local_json_loads((prior/'protocol.json').read_text())['production']
    cases=_local_json_loads((base.SOURCE/'inputs.json').read_text());labels=[c['phrase'] for c in cases]
    old=_local_json_loads((prior/'results.json').read_text());controls=[r for r in old['predictions'] if r['variant'] in ['baseline','no_hold','no_hold_no_smoothing']]
    look={(r['label'],r['variant']):r for r in controls}
    OUT.mkdir(exist_ok=True);(OUT/'review_cases').mkdir(exist_ok=True)
    protocol=dict(labels=labels,review=parts.REVIEW,production=before,hand_sigma=0.6,hand_radius=2,fps=30,
        input_hash=base.sha(base.SOURCE/'inputs.json'),prior_hash=base.sha(prior/'results.json'),checkpoint_hash=base.sha(base.HERE/'wlasl_stgcn.zip'),
        screening=dict(min_correct=118,min_common_roughness_reduction=0.25,max_switches=44,max_native_hand_frames=1035,max_mean_bone_span=0.01),
        note='Fixed before results. Keep short-gap, taper, articulation; hold off. Engineering screening, not statistical or ASL certification. Known200 and selected12.')
    if (OUT/'protocol.json').exists():assert _local_json_loads((OUT/'protocol.json').read_text())==protocol
    else:(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    model,norm,vocab=base.audit.load_model()
    def predict(frames,label):
        a=base.audit.frame_array(frames);t=torch.from_numpy(a[:,base.audit.PICK,:].transpose(2,0,1).copy());t=norm({'frames':t})['frames'].unsqueeze(0)
        assert torch.isfinite(t).all()
        with torch.inference_mode():p=model(t)[0].softmax(0)
        top=p.topk(5)
        return dict(label=label,prediction=vocab[int(top.indices[0])],top5=[vocab[int(k)] for k in top.indices],target_probability=float(p[vocab.index(label)]))
    rows=[]
    if (OUT/'progress.json').exists():rows=_local_json_loads((OUT/'progress.json').read_text())['predictions']
    for i,r in enumerate(rows):assert r['label']==labels[i] and r['variant']==VARIANT
    for i,c in enumerate(cases):
        if i<len(rows):continue
        label=c['phrase'];check=predict(c['payload']['frames'][::2],label)
        assert all(check[k]==look[label,'baseline'][k] for k in ['prediction','top5','target_probability'])
        with patch.object(base.audit.pipeline,'repair_hand_tracks',parts.make_repair(True,False)):
            p=base.audit.pipeline.build_payload(base.ROOT/'SLclean.sqlite3',[label+'.csv'],'',30,513,513,0,0,2,1.2,2,hand_gaussian_sigma=0.6,hand_gaussian_radius=2)
        frames=p['frames'][::2];assert len(frames)==len(c['source30'])
        r=predict(frames,label);r.update(variant=VARIANT,geometry=base.geometry(frames),coverage=parts.coverage(frames));rows.append(r)
        if label in parts.REVIEW:
            review=[r for r in _local_json_loads((prior/'review_cases'/f'{label}.json').read_text()) if r['variant'] in ['baseline','no_hold','no_hold_no_smoothing']]
            review.append(dict(label=label,variant=VARIANT,phrase=label+'__'+VARIANT,payload=p,frameIndices=list(range(0,len(p['frames']),2)),coverage=parts.coverage(frames)))
            (OUT/'review_cases'/f'{label}.json').write_text(json.dumps(review),encoding='utf-8')
        (OUT/'progress.json').write_text(json.dumps(dict(predictions=rows)),encoding='utf-8')
        print(f'{i+1}/200 {label}: {r["prediction"]}',flush=True)
    combined=controls+rows;vs=['baseline','no_hold','no_hold_no_smoothing',VARIANT]
    summaries={v:base.metrics([r for r in combined if r['variant']==v],labels) for v in vs}
    assert base.production()==before and base.sha(prior/'results.json')==protocol['prior_hash']
    (OUT/'results.json').write_text(json.dumps(dict(summaries=summaries,predictions=combined,production_unchanged=True,baseline_exact=True),indent=2),encoding='utf-8')
    review=[]
    for l in parts.REVIEW:review.extend(_local_json_loads((OUT/'review_cases'/f'{l}.json').read_text()))
    (OUT/'review_inputs.json').write_text(json.dumps(review),encoding='utf-8')
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()
