"""Score actual detailed-hand endpoints for the predeclared diagnostic subset."""
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
import repair_parts200 as study

OUT=Path(os.environ.get('PARTS_OUTPUT',str(study.OUT)))
VS=json.loads(os.environ.get('PARTS_VARIANTS','["baseline","no_short_fill","no_hold","no_repair"]'))

def main():
    result=json.loads((OUT/'results.json').read_text())
    cases=json.loads((OUT/'review_inputs.json').read_text())
    captures=json.loads((OUT/'renderer/rendered.json').read_text())
    assert len(cases)==len(captures)==len(study.REVIEW)*len(VS)
    mapping={c['phrase']:c for c in cases}
    model,norm,vocab=study.base.audit.load_model()
    rows=[]
    original_captures={c['phrase']:c for c in json.loads((study.base.SOURCE/'rendered.json').read_text())}
    for capture in captures:
        c=mapping[capture['phrase']];label=c['label'];v=c['variant']
        frames=c['payload']['frames'][::2];samples=capture['samples'];assert len(samples)==len(frames)
        if v=='baseline':
            archived=original_captures[label]['samples']
            assert [s['hands'] for s in samples]==[s['hands'] for s in archived]
        a=study.base.audit.frame_array(frames)
        points=[];deltas=[];visible=0
        for i,s in enumerate(samples):
            pose={k['part']:k['position'] for k in frames[i]['pose']['keypoints']}
            shoulder=np.linalg.norm(np.array(list(pose['leftShoulder'][k] for k in ['x','y']))-np.array(list(pose['rightShoulder'][k] for k in ['x','y'])))
            for side,offset in [('left',33),('right',54)]:
                a[i,offset:offset+21]=0
                hand=s['hands'][side];visible+=bool(hand)
                for index,point in hand.items():
                    a[i,offset+int(index)]=point
                    if i and index in samples[i-1]['hands'][side] and shoulder>1e-8:
                        deltas.append(float(np.linalg.norm(np.array(point)-samples[i-1]['hands'][side][index])/shoulder))
        t=torch.from_numpy(a[:,study.base.audit.PICK,:].transpose(2,0,1).copy())
        t=norm({'frames':t})['frames'].unsqueeze(0);assert torch.isfinite(t).all()
        with torch.inference_mode():p=model(t)[0].softmax(0)
        rows.append(dict(label=label,variant=v,prediction=vocab[int(p.argmax())],
            target_probability=float(p[vocab.index(label)]),visible_detailed_hand_frames=visible,
            total_hand_frames=2*len(frames),p95_step_shoulder=float(np.percentile(deltas,95)) if deltas else None,
            detailed_visibility_switches=sum(bool(samples[i]['hands'][side])!=bool(samples[i-1]['hands'][side]) for i in range(1,len(samples)) for side in ['left','right']),
            observed_low_confidence=sum(bool(f['hands'][s].get('observed')) and 0<float(f['hands'][s].get('score',0))<0.05 for f in frames for s in ['left','right']),
            payload_coverage=c['coverage']))
    summaries={v:study.base.metrics([r for r in rows if r['variant']==v],study.REVIEW) for v in VS}
    assert study.base.production()==json.loads((OUT/'protocol.json').read_text())['production']
    (OUT/'renderer_review.json').write_text(json.dumps(dict(summaries=summaries,predictions=rows,
        baseline_capture_exact=True,production_unchanged=True,
        caveat='Selected12 diagnostic only. Actual detailed-hand endpoints + payload body, not RGB recognition; native fallback excluded.'),indent=2),encoding='utf-8')
    print(json.dumps(summaries,indent=2),flush=True)
    # Paired comparisons and observed geometry across all200 (not subset).
    stats={};pairs={}
    for v in VS:
        rr=[r for r in result['predictions'] if r['variant']==v]
        stats[v]=dict(bone_span_mean=float(np.mean([r['geometry']['mean_bone_span'] for r in rr if r['geometry']['mean_bone_span'] is not None])),
            bend_over115_mean=float(np.mean([r['geometry']['bend_over115_fraction'] for r in rr if r['geometry']['bend_over115_fraction'] is not None])))
    look={(r['label'],r['variant']):r for r in result['predictions']}
    labels=json.loads((OUT/'protocol.json').read_text())['labels']
    for i,a in enumerate(VS):
        for b in VS[i+1:]:
            gained=[l for l in labels if look[l,a]['prediction']!=l and look[l,b]['prediction']==l]
            lost=[l for l in labels if look[l,a]['prediction']==l and look[l,b]['prediction']!=l]
            n=len(gained)+len(lost)
            p=min(1,2*sum(math.comb(n,k) for k in range(min(len(gained),len(lost))+1))/2**n) if n else 1
            pairs[a+' -> '+b]=dict(gained=gained,lost=lost,p=p,bonferroni_adjusted=min(1,p*(len(VS)*(len(VS)-1)//2)))
    (OUT/'analysis.json').write_text(json.dumps(dict(geometry=stats,pairwise=pairs),indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':main()
