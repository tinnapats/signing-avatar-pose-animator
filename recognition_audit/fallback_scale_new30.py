"""Frozen scale-only validation on labels absent from inspected experiment reports."""
import sys,json,random
from pathlib import Path
from unittest.mock import patch
import numpy as np
import fallback_components as comp
import repair_parts200 as parts
base=comp.base;HERE=comp.HERE;OUT=HERE/'fallback_scale_new30_20261004';read=comp.read;write=comp.write
def prepare():
    assert not OUT.exists()
    clips=base.audit.pipeline.discover_clips(base.ROOT/'SLclean.sqlite3');available={Path(c).stem for c in clips};seen=set();files={}
    fields={'label','phrase','actual','gloss','labels_evaluated','labels','original_words','added_words','review','words','excluded'}
    def collect(x,key=''):
        if isinstance(x,str) and key in fields:
            label=x.split('__')[0]
            if label in available:seen.add(label)
        elif isinstance(x,list):
            for v in x:collect(v,key)
        elif isinstance(x,dict):
            for k,v in x.items():
                if k not in {'pool','available','vocabulary','top5','prediction','predicted'}:collect(v,k)
    paths=set()
    for pattern in ['results.json','protocol.json','*/results.json','*/protocol.json','*/provenance.json','*/diagnostics.json','*/case_index.json']:paths.update(HERE.glob(pattern))
    for p in sorted(paths):collect(read(p));files[str(p.relative_to(HERE))]=base.sha(p)
    pool=sorted(available-seen);assert len(pool)>=30
    labels=random.Random(20261004).sample(pool,30);OUT.mkdir();(OUT/'cases').mkdir()
    db=base.ROOT/'SLclean.sqlite3'
    protocol=dict(labels=labels,seed=20261004,excluded=sorted(seen),pool=pool,inspected=files,production=base.production(),db_stat=[db.stat().st_size,db.stat().st_mtime_ns],
      rules='OFF vs SCALE only, unchanged formula/window/limits from fallback_components.30 random labels absent from inspected prior reports. No screening for switches or recognition scores, no replacement. Original durations,30FPS sampling. No new model inference.',
      caveat='New labels relative to inspected local reports, not proven unseen signers, original videos, model training, or all historical work. No ASL ground truth.',
      review='Largest OFF size ratio switch, largest worsening ON-OFF center switch, median OFF size switch; fixed ranking, no manual favorable selection.',
      hashes={n:base.sha(HERE/n) for n in ['fallback_boundary_adjust.js','fallback_components_render.mjs','fallback_scale_new30_render.mjs']})
    write(OUT/'protocol.json',protocol);index=[]
    for i,label in enumerate(labels):
        with patch.object(base.audit.pipeline,'repair_hand_tracks',parts.make_repair(True,False)):
            payload=base.audit.pipeline.build_payload(db,[label+'.csv'],'',30,513,513,0,0,2,1.2,2,hand_gaussian_sigma=0,hand_gaussian_radius=0)
        c=dict(label=label,phrase=label+'__no_hold_no_smoothing',payload=payload,frameIndices=list(range(0,len(payload['frames']),2)))
        dest=OUT/'cases'/f'{i:02d}.json';dest.write_text(json.dumps(c,separators=(',',':')),encoding='utf-8');index.append(dict(label=label,phrase=c['phrase'],file=str(dest),sha256=base.sha(dest)))
        print('prepared',i+1,label,flush=True)
    assert protocol['production']==base.production();assert [db.stat().st_size,db.stat().st_mtime_ns]==protocol['db_stat'];write(OUT/'case_index.json',index)
def analyse():
    p=read(OUT/'protocol.json');assert p['production']==base.production()
    for n,h in p['hashes'].items():assert base.sha(HERE/n)==h
    cases=read(OUT/'case_index.json')
    for c in cases:assert base.sha(Path(c['file']))==c['sha256']
    caps={(c['phrase'],c['variant']):c for c in read(OUT/'renderer/rendered.json')};assert len(caps)==60
    rows=[];modified=0;frames=0
    for c in cases:
        a,b=[caps[c['phrase'],v]['samples'] for v in ['OFF','SCALE']];assert len(a)==len(b)
        for x,y in zip(a,b):
            frames+=1;assert x['index']==y['index'] and x['hands']==y['hands'] and x['body']==y['body']
            for side in ['left','right']:
                assert [n['opacity'] for n in x['native'][side]]==[n['opacity'] for n in y['native'][side]]
                op=y['ops'].get(side)
                if op:assert op['rotation']==0 and 1<=op['distance']<=6 and .85<=op['scale']<=1.15;modified+=1
                else:assert x['geometry'][side]==y['geometry'][side]
        for i in range(1,len(a)):
            for side in ['left','right']:
                s0,s1=comp.prev.old.state(a[i-1],side),comp.prev.old.state(a[i],side);kind='switch' if s0!=s1 else s1
                if any(x['geometry'][side] is None for x in [a[i-1],a[i],b[i-1],b[i]]):continue
                rows.append(dict(label=c['label'],phrase=c['phrase'],side=side,before=a[i-1]['index'],after=a[i]['index'],kind=kind,OFF=comp.metric(a[i-1],a[i],side),SCALE=comp.metric(b[i-1],b[i],side)))
    summary={}
    for kind in ['switch','native','detailed']:
        rr=[r for r in rows if r['kind']==kind]
        summary[kind]=dict(n=len(rr),words=len({r['label'] for r in rr}),variants={v:{m:dict(mean=float(np.mean([r[v][m] for r in rr])),median=float(np.median([r[v][m] for r in rr]))) for m in ['center','size']} for v in ['OFF','SCALE']},changes={m:dict(improved=sum(r['SCALE'][m]<r['OFF'][m]-1e-9 for r in rr),worse=sum(r['SCALE'][m]>r['OFF'][m]+1e-9 for r in rr)) for m in ['center','size']}) if rr else dict(n=0)
    switches=[r for r in rows if r['kind']=='switch'];word_diffs=[]
    for label in p['labels']:
        rr=[r for r in switches if r['label']==label]
        if rr:word_diffs.append(dict(label=label,n=len(rr),**{m:float(np.mean([r['SCALE'][m]-r['OFF'][m] for r in rr])) for m in ['center','size']}))
    confidence={}
    for m in ['center','size']:
        d=np.array([r[m] for r in word_diffs]);rng=np.random.default_rng(20261004);boot=np.mean(rng.choice(d,(5000,len(d)),replace=True),axis=1)
        confidence[m]=dict(mean_word_difference=float(d.mean()),bootstrap95=np.percentile(boot,[2.5,97.5]).tolist(),words=len(d),note='Exploratory percentile bootstrap resamples words, not frames; negative favors SCALE. Conditional on words with switches, not semantic accuracy.')
    selected=[]
    if switches:
        ordered=sorted(switches,key=lambda r:r['OFF']['size'])
        for r in [ordered[-1],max(switches,key=lambda r:r['SCALE']['center']-r['OFF']['center']),ordered[len(ordered)//2]]:
            if r not in selected:selected.append(r)
    review=[]
    for i,r in enumerate(selected):
        original=next(c for c in cases if c['label']==r['label']);c=read(Path(original['file']));c['phrase']=f'review{i}_'+r['label'];c['frameIndices']=[r['before'],r['after']];dest=OUT/'cases'/f'review{i}.json';dest.write_text(json.dumps(c,separators=(',',':')),encoding='utf-8');review.append(dict(file=str(dest),**r))
    write(OUT/'review_index.json',review);assert p['production']==base.production()
    result=dict(summary=summary,word_differences=word_diffs,confidence=confidence,rows=rows,review=selected,modified_hand_frames=modified,sampled_frames=frames,production_unchanged=True,invariants='Detailed endpoints/body/opacity/visibility/frame indices unchanged; no-op frames identical; rotation0; limits passed.')
    write(OUT/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['rows','word_differences']},indent=2),flush=True)
if __name__=='__main__':{'prepare':prepare,'analyse':analyse}[sys.argv[1]]()
