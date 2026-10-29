"""Split short interpolation from hold/fade, keeping taper and production intact."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import copy
import inspect
import json
from unittest.mock import patch
import numpy as np
import torch
import diagnose_processing200 as base

OUT=base.HERE/'repair_parts200'
VARIANTS=['no_short_fill','no_hold']
# Diagnostic selection fixed before this run; not a representative validation set.
REVIEW=['yes','camera','never','recliner','parents','lord','scientist','head','approach','hurry','overwhelm','hello']
ORIGINAL=base.audit.pipeline.repair_hand_tracks

def make_repair(short=True,hold=True):
    source=inspect.getsource(ORIGINAL)
    needle='                if gap <= max_gap:\n'
    assert source.count(needle)==1
    if not short:
        # Do NOT set max_gap=0: that would incorrectly convert short gaps to holds.
        source=source.replace(needle,needle+'                    continue  # audit: leave short gaps unfilled\n')
    ns=dict(ORIGINAL.__globals__)
    exec(compile(source,'audit_repair_parts','exec'),ns)
    fn=ns['repair_hand_tracks']
    def repair(frames,max_gap_frames,fade_frames):
        return fn(frames,max_gap_frames=max_gap_frames,fade_frames=fade_frames if hold else 0)
    return repair

def test_repair():
    def frames():
        out=[]
        for i in range(13):
            score=1.0 if i in [2,4,10] else 0.0
            h=dict(score=score,keypoints=[dict(score=score,position=dict(x=float(i) if score else -9.,y=0.,z=0.)) for _ in range(21)])
            out.append(dict(hands=dict(left=copy.deepcopy(h),right=copy.deepcopy(h))))
        return out
    f=frames()
    assert make_repair()(copy.deepcopy(f),2,1)==ORIGINAL(copy.deepcopy(f),2,1)
    short=make_repair(True,False)(copy.deepcopy(f),2,1)
    hold=make_repair(False,True)(copy.deepcopy(f),2,1)
    off=make_repair(False,False)(copy.deepcopy(f),2,1)
    assert off==f
    assert short[3]['hands']['left']['keypoints'][0]['position']['x']==3.
    assert short[1]['hands']['left']['score']==short[5]['hands']['left']['score']==0
    assert hold[3]['hands']['left']['score']==0
    assert hold[1]['hands']['left']['score']>0 and hold[5]['hands']['left']['score']>0

def coverage(frames):
    hands=[f['hands'][s] for f in frames for s in ['left','right']]
    return dict(hand_frames=len(hands),observed=sum(bool(h.get('observed')) for h in hands),
        positive=sum(h.get('score',0)>0 for h in hands),
        synthetic_positive=sum(not h.get('observed') and h.get('score',0)>0 for h in hands),
        model_zero_hand_point_fraction=float(np.mean(np.all(base.audit.frame_array(frames)[:,33:75]==0,axis=2))))

def main():
    test_repair();print('Repair isolation tests passed',flush=True)
    assert not (OUT/'results.json').exists()
    old=_local_json_loads((base.HERE/'gap_split200/results.json').read_text())
    protocol0=_local_json_loads((base.OUT/'protocol.json').read_text())
    before=base.production();assert before==protocol0['production_before']
    assert base.sha(base.SOURCE/'inputs.json')==protocol0['source_inputs_hash']
    cases=_local_json_loads((base.SOURCE/'inputs.json').read_text());labels=[c['phrase'] for c in cases]
    assert labels==protocol0['labels'] and len(labels)==200
    OUT.mkdir(exist_ok=True);(OUT/'review_cases').mkdir(exist_ok=True)
    protocol=dict(variants=VARIANTS,labels=labels,review=REVIEW,production=before,
        prior_sha256=base.sha(base.HERE/'gap_split200/results.json'),
        checkpoint_sha256=base.sha(base.HERE/'wlasl_stgcn.zip'),fps=30,
        note='Taper stays ON. no_short_fill leaves internal short gaps untouched, does not convert them to holds; no_hold sets repair fade=0 only. Known development200.')
    if (OUT/'protocol.json').exists():assert _local_json_loads((OUT/'protocol.json').read_text())==protocol
    else:(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    network,normalize,vocab=base.audit.load_model()
    def predict(frames,label):
        a=base.audit.frame_array(frames)
        t=torch.from_numpy(a[:,base.audit.PICK,:].transpose(2,0,1).copy())
        t=normalize({'frames':t})['frames'].unsqueeze(0);assert torch.isfinite(t).all()
        with torch.inference_mode():p=network(t)[0].softmax(0)
        top=p.topk(5)
        return dict(label=label,prediction=vocab[int(top.indices[0])],top5=[vocab[int(i)] for i in top.indices],target_probability=float(p[vocab.index(label)]))
    lookup={(r['label'],r['variant']):r for r in old['predictions']}
    rows=[];done=0
    if (OUT/'progress.json').exists():
        ck=_local_json_loads((OUT/'progress.json').read_text());rows=ck['predictions'];done=ck['completed']
        assert len(rows)==2*done
        for i,r in enumerate(rows):assert r['label']==labels[i//2] and r['variant']==VARIANTS[i%2]
    for i,c in enumerate(cases):
        if i<done:continue
        label=c['phrase'];check=predict(c['payload']['frames'][::2],label)
        assert all(check[k]==lookup[label,'baseline'][k] for k in ['prediction','top5','target_probability'])
        payloads={'baseline':c['payload']}
        for v in VARIANTS:
            fn=make_repair(short=v!='no_short_fill',hold=v!='no_hold')
            with patch.object(base.audit.pipeline,'repair_hand_tracks',fn):
                p=base.audit.pipeline.build_payload(base.ROOT/'SLclean.sqlite3',[label+'.csv'],'',30,513,513,0,0,2,1.2,2)
            frames=p['frames'][::2];assert len(frames)==len(c['source30'])
            r=predict(frames,label);r.update(variant=v,geometry=base.geometry(frames),coverage=coverage(frames));rows.append(r);payloads[v]=p
        if label in REVIEW:
            with patch.object(base.audit.pipeline,'repair_hand_tracks',lambda frames,**kwargs:frames):
                p=base.audit.pipeline.build_payload(base.ROOT/'SLclean.sqlite3',[label+'.csv'],'',30,513,513,0,0,2,1.2,2)
            check=predict(p['frames'][::2],label)
            assert all(check[k]==lookup[label,'no_repair'][k] for k in ['prediction','top5','target_probability'])
            payloads['no_repair']=p
            review=[dict(phrase=label+'__'+v,label=label,variant=v,payload=p,frameIndices=list(range(0,len(p['frames']),2)),coverage=coverage(p['frames'][::2])) for v,p in payloads.items()]
            (OUT/'review_cases'/f'{label}.json').write_text(json.dumps(review),encoding='utf-8')
        (OUT/'progress.json').write_text(json.dumps(dict(completed=i+1,predictions=rows)),encoding='utf-8')
        print(f'{i+1}/200 {label}: '+str([(r['variant'],r['prediction']) for r in rows[-2:]]),flush=True)
    combined=[r for r in old['predictions'] if r['variant'] in ['baseline','no_repair']]+rows
    summaries={v:base.metrics([r for r in combined if r['variant']==v],labels) for v in ['baseline','no_short_fill','no_hold','no_repair']}
    assert base.production()==before
    assert base.sha(base.HERE/'gap_split200/results.json')==protocol['prior_sha256']
    (OUT/'results.json').write_text(json.dumps(dict(summaries=summaries,predictions=combined,production_unchanged=True,baseline_exact=True),indent=2),encoding='utf-8')
    review=[]
    for l in REVIEW:review.extend(_local_json_loads((OUT/'review_cases'/f'{l}.json').read_text()))
    assert len(review)==48
    (OUT/'review_inputs.json').write_text(json.dumps(review),encoding='utf-8')
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()
