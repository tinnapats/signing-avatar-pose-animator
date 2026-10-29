"""Isolated repair/taper factorial follow-up; never modify production files."""
import json
import math
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import torch
import diagnose_processing200 as prior

OUT = prior.HERE / 'gap_split200'
VARIANTS = ['no_repair', 'no_taper']

def main():
    assert not (OUT/'results.json').exists(), 'Refuse to overwrite finished experiment'
    previous = json.loads((prior.OUT/'results.json').read_text())
    old_protocol = json.loads((prior.OUT/'protocol.json').read_text())
    before = prior.production()
    assert before == old_protocol['production_before']
    assert prior.sha(prior.SOURCE/'inputs.json') == old_protocol['source_inputs_hash']
    cases = json.loads((prior.SOURCE/'inputs.json').read_text())
    labels = [c['phrase'] for c in cases]
    assert labels == old_protocol['labels'] and len(labels) == 200
    protocol = dict(labels=labels, variants=VARIANTS, production=before,
        previous_results_sha256=prior.sha(prior.OUT/'results.json'),
        inputs_sha256=old_protocol['source_inputs_hash'],
        checkpoint_sha256=prior.sha(prior.HERE/'wlasl_stgcn.zip'),
        fps=30, note='Known development200. Reuse baseline/both-off from prior; fresh two single-off conditions. Before renderer, no retraining.')
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text()) == protocol
    else:
        (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    network, normalize, vocabulary = prior.audit.load_model()
    def predict(frames, label):
        a = prior.audit.frame_array(frames)
        t = torch.from_numpy(np.asarray(a,dtype=np.float32)[:,prior.audit.PICK,:].transpose(2,0,1).copy())
        t = normalize({'frames':t})['frames'].unsqueeze(0)
        assert torch.isfinite(t).all()
        with torch.inference_mode(): p = network(t)[0].softmax(0)
        top = p.topk(5)
        return dict(label=label,prediction=vocabulary[int(top.indices[0])],
            top5=[vocabulary[int(i)] for i in top.indices],target_probability=float(p[vocabulary.index(label)]))
    baseline = {r['label']:r for r in previous['predictions'] if r['variant']=='baseline'}
    rows=[];completed=0
    if (OUT/'progress.json').exists():
        saved=json.loads((OUT/'progress.json').read_text());rows=saved['predictions'];completed=saved['completed']
        assert len(rows)==completed*2
        for i,r in enumerate(rows):
            assert r['label']==labels[i//2] and r['variant']==VARIANTS[i%2]
    for i,c in enumerate(cases):
        if i<completed:continue
        label=c['phrase']
        # Re-infer each archived baseline to validate current model and input path.
        check=predict(c['payload']['frames'][::2],label)
        assert all(check[k]==baseline[label][k] for k in ['prediction','top5','target_probability'])
        for variant in VARIANTS:
            name='repair_hand_tracks' if variant=='no_repair' else 'taper_hand_detection_edges'
            with patch.object(prior.audit.pipeline,name,lambda frames,**kwargs:frames):
                payload=prior.audit.pipeline.build_payload(prior.ROOT/'SLclean.sqlite3',[label+'.csv'],'',30,513,513,0,0,2,1.2,2)
            frames=payload['frames'][::2]
            assert len(frames)==len(c['source30'])
            r=predict(frames,label)
            r.update(variant=variant,geometry=prior.geometry(frames))
            rows.append(r)
        (OUT/'progress.json').write_text(json.dumps(dict(completed=i+1,predictions=rows)),encoding='utf-8')
        print(f'{i+1}/200 {label}: '+str([(r['variant'],r['prediction']) for r in rows[-2:]]),flush=True)
    combined=[r for r in previous['predictions'] if r['variant'] in ['baseline','no_gap_handling']]+rows
    summaries={}
    for v in ['baseline','no_repair','no_taper','no_gap_handling']:
        rr=[r for r in combined if r['variant']==v]
        m=prior.metrics(rr,labels)
        m['gained_words']=[r['label'] for r in rr if r['prediction']==r['label'] and baseline[r['label']]['prediction']!=r['label']]
        m['lost_words']=[r['label'] for r in rr if r['prediction']!=r['label'] and baseline[r['label']]['prediction']==r['label']]
        summaries[v]=m
    # All six pairwise comparisons; development-only even with multiplicity control.
    pairs={};vs=list(summaries)
    lookup={(r['label'],r['variant']):r['prediction']==r['label'] for r in combined}
    for i,a in enumerate(vs):
        for b in vs[i+1:]:
            gains=sum(not lookup[l,a] and lookup[l,b] for l in labels)
            losses=sum(lookup[l,a] and not lookup[l,b] for l in labels)
            n=gains+losses
            p=min(1,2*sum(math.comb(n,k) for k in range(min(gains,losses)+1))/2**n) if n else 1
            pairs[a+' -> '+b]=dict(gains=gains,losses=losses,p=p,bonferroni6=min(1,p*6))
    assert prior.production()==before
    assert prior.sha(prior.OUT/'results.json')==protocol['previous_results_sha256']
    result=dict(summaries=summaries,pairwise=pairs,predictions=combined,
        production_unchanged=True,baseline_predictions_exact=True,
        caveat='Pre-renderer known200; two control conditions reused. Not human comprehension or independent validation.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(summaries=summaries,pairwise=pairs),indent=2),flush=True)

if __name__=='__main__':main()
