"""Frozen new-word pilot; not certified unseen model-training data or signers."""
import argparse
import json
import random
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import repair_parts200 as parts
base=parts.base;OUT=base.HERE/'novel_words30_20261001'
VS=['baseline','no_hold_no_smoothing']

def load_predictor():
    model,norm,vocab=base.audit.load_model()
    def predict(a,label,variant):
        t=torch.from_numpy(a[:,base.audit.PICK,:].transpose(2,0,1).copy());t=norm({'frames':t})['frames'].unsqueeze(0);assert torch.isfinite(t).all()
        with torch.inference_mode():p=model(t)[0].softmax(0)
        return dict(label=label,variant=variant,prediction=vocab[int(p.argmax())])
    return vocab,predict

def prepare():
    assert not OUT.exists(),'Do not replace frozen evaluation'
    before=base.production();vocab,predict=load_predictor();vocab=set(vocab);seen=set();sources={}
    def collect(x,field=''):
        # Candidate pools and model vocabulary are NOT previously evaluated labels.
        names={'label','phrase','actual','gloss','labels_evaluated','labels','original_words','added_words','review','words'}
        if isinstance(x,str) and field in names and x in vocab:seen.add(x)
        elif isinstance(x,dict):
            for k,v in x.items():
                if k not in {'pool','excluded','available','vocabulary','top5','prediction','predicted'}:collect(v,k)
        elif isinstance(x,list):
            for v in x:collect(v,field)
    files={base.HERE/'results.json'}
    for pattern in ['*/results.json','*/protocol.json','*/provenance.json','*/diagnostics.json']:
        files.update(base.HERE.glob(pattern))
    for file in sorted(files):
        if file.exists():collect(json.loads(file.read_text(encoding='utf-8-sig')));sources[str(file.relative_to(base.HERE))]=base.sha(file)
    clips=base.audit.pipeline.discover_clips(base.ROOT/'SLclean.sqlite3')
    pool=sorted({Path(c).stem for c in clips}&vocab-seen)
    print(f'Excluded reported labels: {len(seen)}; eligible pool: {len(pool)}',flush=True)
    assert len(pool)>=30
    labels=random.Random(20261001).sample(pool,30)
    OUT.mkdir();(OUT/'cases').mkdir()
    protocol=dict(labels=labels,seed=20261001,pool=pool,excluded=sorted(seen),exclusion_files=sources,
        production=before,variants=VS,fps=30,checkpoint_hash=base.sha(base.HERE/'wlasl_stgcn.zip'),
        note='Fixed selected candidate from development200 vs baseline. Labels absent from inspected prior reports. Not proof of unseen source clips, signers, training set, or all historical experiments. No tuning/reselection.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print('Frozen labels: '+str(labels),flush=True)
    rows=[];cases=[]
    for i,label in enumerate(labels):
        for v in VS:
            if v=='baseline':p=base.audit.pipeline.build_payload(base.ROOT/'SLclean.sqlite3',[label+'.csv'],'',30,513,513,0,0,2,1.2,2)
            else:
                with patch.object(base.audit.pipeline,'repair_hand_tracks',parts.make_repair(True,False)):
                    p=base.audit.pipeline.build_payload(base.ROOT/'SLclean.sqlite3',[label+'.csv'],'',30,513,513,0,0,2,1.2,2,hand_gaussian_sigma=0,hand_gaussian_radius=0)
            frames=p['frames'][::2]
            c=dict(label=label,phrase=label+'__'+v,variant=v,payload=p,frameIndices=list(range(0,len(p['frames']),2)))
            cases.append(c);r=predict(base.audit.frame_array(frames),label,v);r['geometry']=base.geometry(frames);rows.append(r)
        assert len(cases[-1]['frameIndices'])==len(cases[-2]['frameIndices'])
        (OUT/'cases'/f'{i:02d}.json').write_text(json.dumps(cases[-2:]),encoding='utf-8')
        print(f'{i+1}/30 {label}: '+str([r['prediction'] for r in rows[-2:]]),flush=True)
    assert base.production()==before
    (OUT/'review_inputs.json').write_text(json.dumps(cases),encoding='utf-8')
    (OUT/'pre_results.json').write_text(json.dumps(dict(predictions=rows,summaries={v:base.metrics([r for r in rows if r['variant']==v],labels) for v in VS}),indent=2),encoding='utf-8')

def score():
    protocol=json.loads((OUT/'protocol.json').read_text());labels=protocol['labels'];_,predict=load_predictor()
    cases={c['phrase']:c for c in json.loads((OUT/'review_inputs.json').read_text())};captures=json.loads((OUT/'renderer/rendered.json').read_text());assert len(captures)==60
    rows=[]
    for cap in captures:
        assert cap['variant']=='S1A1'
        c=cases[cap['phrase']];frames=c['payload']['frames'][::2];a=base.audit.frame_array(frames);native=detailed=switches=0;last={}
        for i,s in enumerate(cap['samples']):
            for side,offset in [('left',33),('right',54)]:
                d=bool(s['hands'][side]);n=bool(s['native'][side]);native+=n;detailed+=d
                state=(d,n)
                if side in last and last[side]!=state:switches+=1
                last[side]=state;a[i,offset:offset+21]=0
                for k,p in s['hands'][side].items():a[i,offset+int(k)]=p
        r=predict(a,c['label'],c['variant']);r.update(native=native,detailed=detailed,switches=switches);rows.append(r)
    assert base.production()==protocol['production']
    summaries={v:base.metrics([r for r in rows if r['variant']==v],labels) for v in VS}
    (OUT/'renderer_results.json').write_text(json.dumps(dict(summaries=summaries,predictions=rows,production_unchanged=True),indent=2),encoding='utf-8')
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','score']);a=p.parse_args();globals()[a.stage]()
