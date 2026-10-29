"""Frozen best-pipeline 200-word rendered comparison; isolated and resumable."""
import json
import math
import sys
import time
from unittest.mock import patch
import numpy as np
import torch
import compare_slgcn_best as pilot
import repair_parts200 as parts
from equal_duration_trial import arrays
base=pilot.base;HERE=base.HERE;OUT=HERE/'slgcn200_20261002';OLD=HERE/'combined_winners200'

def check():
    p=json.loads((OUT/'protocol.json').read_text());assert base.production()==p['production']
    for name,h in p['hashes'].items():assert base.sha(HERE/name)==h,name
    db=base.ROOT/'SLclean.sqlite3';assert [db.stat().st_size,db.stat().st_mtime_ns]==p['db_stat']
    return p

def prepare():
    oldp=json.loads((OLD/'protocol.json').read_text());labels=oldp['labels'];assert len(labels)==200
    if not OUT.exists():
        OUT.mkdir();(OUT/'cases').mkdir()
        names=['combined_winners200/results.json','combined_winners200/protocol.json','combined_winners200/review_inputs.json','combined_winners200/renderer/rendered.json','wlasl_stgcn.zip','wlasl_metadata.zip','slgcn_trial_20261002/wlasl_slgcn.zip','slgcn_trial_20261002/upstream_decoupled_gcn.py','slgcn_trial_20261002/upstream_graph_utils.py','compare_slgcn_best.py','compare_slgcn200.py','slgcn200_render.mjs']
        db=base.ROOT/'SLclean.sqlite3'
        protocol=dict(labels=labels,production=base.production(),hashes={f:base.sha(HERE/f) for f in names},db_stat=[db.stat().st_size,db.stat().st_mtime_ns],
            pipeline='no_hold_no_smoothing; short fill/body Gaussian/taper/articulation/bone controls unchanged; S1A1; original durations at30FPS; no cap/fixed68',
            primary='post-rendered detailed hands plus payload body, not full RGB',secondary='pre-render payload',negative='whole-frame temporal shuffle seed20261002 per clip',models=['ST-GCN','SL-GCN'],
            note='Same development200 selected before experiment. Not independent holdout; checkpoint training overlap unknown. No tuning, ensemble, sample exclusion, or word routing. Two paired tests with Bonferroni2.')
        (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    p=check();assert labels==p['labels']
    reviews={c['label']:c for c in json.loads((OLD/'review_inputs.json').read_text()) if c['variant']=='no_hold_no_smoothing'}
    allcases=[]
    for i,label in enumerate(labels):
        path=OUT/'cases'/f'{i:03d}.json'
        if path.exists():c=json.loads(path.read_text());assert c['label']==label
        else:
            with patch.object(base.audit.pipeline,'repair_hand_tracks',parts.make_repair(True,False)):
                payload=base.audit.pipeline.build_payload(base.ROOT/'SLclean.sqlite3',[label+'.csv'],'',30,513,513,0,0,2,1.2,2,hand_gaussian_sigma=0,hand_gaussian_radius=0)
            c=dict(label=label,phrase=label+'__no_hold_no_smoothing',variant='no_hold_no_smoothing',payload=payload,frameIndices=list(range(0,len(payload['frames']),2)))
            if label in reviews:
                assert c['payload']['frames']==reviews[label]['payload']['frames'] and c['frameIndices']==reviews[label]['frameIndices'],'Prior12 payload changed'
            path.write_text(json.dumps(c),encoding='utf-8')
        allcases.append(c)
        if (i+1)%10==0:print('prepared',i+1,'/200',flush=True)
    check();(OUT/'inputs.json').write_text(json.dumps(allcases),encoding='utf-8')
    (OUT/'input_hash.json').write_text(json.dumps(dict(sha256=base.sha(OUT/'inputs.json'))),encoding='utf-8')

def score():
    protocol=check();assert not (OUT/'results.json').exists()
    assert base.sha(OUT/'inputs.json')==json.loads((OUT/'input_hash.json').read_text())['sha256']
    caphash=base.sha(OUT/'renderer/rendered.json');capture_path=OUT/'capture_hash.json'
    if capture_path.exists():assert json.loads(capture_path.read_text())['sha256']==caphash
    else:capture_path.write_text(json.dumps(dict(sha256=caphash)),encoding='utf-8')
    caps={c['phrase']:c for c in json.loads((OUT/'renderer/rendered.json').read_text())};assert len(caps)==200
    oldcaps={c['phrase']:c for c in json.loads((OLD/'renderer/rendered.json').read_text()) if c['variant']=='S1A1' and c['phrase'].endswith('__no_hold_no_smoothing')}
    for name,old in oldcaps.items():assert caps[name]['samples']==old['samples'],'Prior diagnostic capture changed'
    oldpre={r['label']:r for r in json.loads((OLD/'results.json').read_text())['predictions'] if r['variant']=='no_hold_no_smoothing'}
    baseline,norm,vocab=base.audit.load_model();candidate,_=pilot.load_slgcn();rows=[];done=0
    progress=OUT/'progress.json'
    if progress.exists():
        saved=json.loads(progress.read_text());rows=saved['predictions'];done=saved['completed'];assert len(rows)==done*6
        for i,r in enumerate(rows):assert r['label']==protocol['labels'][i//6]
    for i,label in enumerate(protocol['labels']):
        if i<done:continue
        c=json.loads((OUT/'cases'/f'{i:03d}.json').read_text());assert c['label']==label
        pre=base.audit.frame_array([c['payload']['frames'][j] for j in c['frameIndices']]);post=arrays(c,caps[c['phrase']]);idx=np.random.default_rng(20261002).permutation(len(post))
        for stage,a in [('pre',pre),('post',post),('shuffled',post[idx])]:
            t=torch.from_numpy(a[:,base.audit.PICK,:].transpose(2,0,1).copy());t=norm({'frames':t})['frames'].unsqueeze(0);assert torch.isfinite(t).all()
            for name,model in [('ST-GCN',baseline),('SL-GCN',candidate)]:
                start=time.perf_counter()
                with torch.inference_mode():p=model(t)[0].softmax(0)
                assert torch.isfinite(p).all() and p.shape==(2000,)
                top=p.topk(5);r=dict(label=label,model=name,stage=stage,prediction=vocab[int(top.indices[0])],top5=[vocab[int(k)] for k in top.indices],target_probability=float(p[vocab.index(label)]),frames=len(a),runtime_ms=(time.perf_counter()-start)*1000)
                if name=='ST-GCN' and stage=='pre':assert all(r[k]==oldpre[label][k] for k in ['prediction','top5','target_probability']),label
                rows.append(r)
        progress.write_text(json.dumps(dict(completed=i+1,predictions=rows)),encoding='utf-8')
        if (i+1)%10==0:print('scored',i+1,'/200',flush=True)
    summaries={f'{m}_{s}':base.metrics([r for r in rows if r['model']==m and r['stage']==s],protocol['labels']) for m in protocol['models'] for s in ['pre','post','shuffled']}
    comparisons={}
    for stage in ['pre','post']:
        by={(r['label'],r['model']):r for r in rows if r['stage']==stage};g=[];l=[]
        for label in protocol['labels']:
            a=by[label,'ST-GCN']['prediction']==label;b=by[label,'SL-GCN']['prediction']==label
            if b and not a:g.append(label)
            if a and not b:l.append(label)
        n=len(g)+len(l);p=min(1,2*sum(math.comb(n,k) for k in range(min(len(g),len(l))+1))/2**n) if n else 1
        comparisons[stage]=dict(gained=g,lost=l,exact_mcnemar_p=p,bonferroni_two=min(1,p*2))
    check();assert base.sha(OUT/'renderer/rendered.json')==caphash
    result=dict(summaries=summaries,paired=comparisons,predictions=rows,production_unchanged=True,pre200_exact=True,prior12_capture_exact=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(summaries=summaries,paired=comparisons),indent=2))

if __name__=='__main__':globals()[sys.argv[1]]()
