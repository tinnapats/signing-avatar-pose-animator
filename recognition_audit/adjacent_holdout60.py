"""Frozen new-label validation. No production modifications or model inference."""
import sys,json,random
from pathlib import Path
from unittest.mock import patch
import numpy as np
import fallback_scale_new30 as prev
base=prev.base;HERE=prev.HERE;read=prev.read;write=prev.write
OUT=HERE/'adjacent_holdout60_20261004';VS=['OFF','OLD','NEW']
FROZEN=['fallback_boundary_adjust.js','render_bounds_scale_render.mjs','adjacent_reference_render.mjs','fallback_boundary_render.mjs','boundary_render.mjs','combined_render.mjs','step1b_render.mjs','adjacent_holdout60_render.mjs']

def verify():
    p=read(OUT/'protocol.json');assert p['production']==base.production()
    for f,h in p['hashes'].items():assert base.sha(HERE/f)==h,f
    db=base.ROOT/'SLclean.sqlite3';assert [db.stat().st_size,db.stat().st_mtime_ns]==p['db_stat']
    return p

def prepare():
    assert not OUT.exists(),'Never overwrite a frozen experiment'
    db=base.ROOT/'SLclean.sqlite3';clips=base.audit.pipeline.discover_clips(db)
    available={Path(c).stem for c in clips};seen=set();sources={}
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
    files=set()
    for pattern in ['results.json','protocol.json','*/results.json','*/protocol.json','*/provenance.json','*/diagnostics.json','*/case_index.json']:
        files.update(HERE.glob(pattern))
    for f in sorted(files):collect(read(f));sources[str(f.relative_to(HERE))]=base.sha(f)
    pool=sorted(available-seen);assert len(pool)>=60
    labels=random.Random(2026100460).sample(pool,60)
    OUT.mkdir();(OUT/'cases').mkdir()
    write(OUT/'protocol.json',dict(labels=labels,seed=2026100460,pool=pool,excluded=sorted(seen),inspected=sources,production=base.production(),db_stat=[db.stat().st_size,db.stat().st_mtime_ns],hashes={f:base.sha(HERE/f) for f in FROZEN},
        rules='60 random labels absent from inspected local reports, no replacement or tuning. Same no_hold_no_smoothing payload. Every payload frame60FPS, original duration. OFF / nearest observed rendered bounds / nearest drawable rendered bounds. Frozen caps .85..1.15,6 frames,Smoothstep,past wins ties,rotation0. Future allowed offline. No model inference.',
        metrics='Primary: all native/detailed switches, size larger/smaller strokeBounds diagonal, wrist-relative bbox center step / shoulder width. Controls: all native/native and detailed/detailed pairs. Report means,medians,p95,max,word averages and regressions. Word bootstrap exploratory.',
        runtime='performance.now around synchronous renderFrame including view.update; excludes capture extraction/IPC, warm20 frames per method. Rotate method order by word. Single headless run, not sustained FPS or real-time guarantee.',
        review='Fixed rankings: largest size improvement NEW-OLD, largest center regression NEW-OLD, median OFF size switch; deduplicate. Capture +/-12 frames around boundary at60FPS.',
        caveat='Unseen labels relative to inspected files, not proof of unseen videos/signers/training or all historical experiments. No ASL ground truth. No predeclared clinical/perceptual acceptance threshold: no automatic deployment.'))
    index=[]
    for i,label in enumerate(labels):
        with patch.object(base.audit.pipeline,'repair_hand_tracks',prev.parts.make_repair(True,False)):
            payload=base.audit.pipeline.build_payload(db,[label+'.csv'],'',30,513,513,0,0,2,1.2,2,hand_gaussian_sigma=0,hand_gaussian_radius=0)
        c=dict(label=label,phrase=label+'__no_hold_no_smoothing',payload=payload,frameIndices=list(range(len(payload['frames']))))
        dest=OUT/'cases'/f'{i:02d}.json';write(dest,c)
        index.append(dict(label=label,phrase=c['phrase'],file=str(dest),sha256=base.sha(dest)))
        print('Prepared',i+1,label,len(c['frameIndices']),flush=True)
    write(OUT/'case_index.json',index)
    write(OUT/'canary_index.json',read(HERE/'adjacent_reference_20261004/case_index.json')[:3])
    verify();print('Frozen labels',len(labels),'excluded',len(seen),'pool',len(pool),flush=True)

def canary():
    verify();old={(c['phrase'],c['variant']):c['samples'] for c in read(HERE/'adjacent_reference_20261004/renderer/rendered.json')};n=0
    caps=read(OUT/'canary/rendered.json');assert len(caps)==9
    for c in caps:
        expected=old[c['phrase'],c['variant']];assert len(expected)==len(c['samples'])
        for a,b in zip(expected,c['samples']):assert a=={k:v for k,v in b.items() if k!='renderMs'};n+=1
    write(OUT/'canary_validation.json',dict(exact_samples=n,passed=True));print('Exact canary samples',n)

def stats(a):
    a=np.asarray(a);return dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max())) if len(a) else None

def analyse():
    p=verify();assert read(OUT/'canary_validation.json')['passed']
    cases=read(OUT/'case_index.json');caps={(c['phrase'],c['variant']):c['samples'] for c in read(OUT/'renderer/rendered.json')};assert len(caps)==180
    rows=[];ops={v:0 for v in VS};error=0;times={v:[] for v in VS};frame_count=0
    for d in cases:
        assert base.sha(Path(d['file']))==d['sha256'];a=caps[d['phrase'],'OFF'];frame_count+=len(a)
        for v in VS:
            b=caps[d['phrase'],v];assert len(a)==len(b)
            for i,(x,y) in enumerate(zip(a,b)):
                assert x['index']==y['index']==i and x['hands']==y['hands'] and x['body']==y['body']
                assert np.isfinite(y['renderMs']) and y['renderMs']>=0;times[v].append(y['renderMs'])
                for side in ['left','right']:
                    assert [q['opacity'] for q in x['native'][side]]==[q['opacity'] for q in y['native'][side]]
                    op=y['ops'].get(side)
                    if op:
                        ops[v]+=1;assert op['rotation']==0 and .85<=op['scale']<=1.15 and 1<=op['distance']<=6
                        ref=a[op['referenceIndex']];assert prev.comp.prev.old.state(ref,side)=='detailed'
                        bb=ref['geometry'][side]['bounds'];diag=float(np.linalg.norm(np.array(bb[2:])-bb[:2]));error=max(error,abs(diag-op['target']));assert abs(diag-op['target'])<1e-6
                    else:assert x['geometry'][side]==y['geometry'][side]
        for i in range(1,len(a)):
            for side in ['left','right']:
                s0,s1=[prev.comp.prev.old.state(a[j],side) for j in [i-1,i]]
                if any(caps[d['phrase'],v][j]['geometry'][side] is None for v in VS for j in [i-1,i]):continue
                kind='switch' if {s0,s1}=={'native','detailed'} else s1 if s0==s1 else 'other'
                rows.append(dict(label=d['label'],phrase=d['phrase'],side=side,index=i,kind=kind,**{v:prev.comp.metric(caps[d['phrase'],v][i-1],caps[d['phrase'],v][i],side) for v in VS}))
    summary={};wordrows=[]
    for kind in ['switch','native','detailed','other']:
        rr=[r for r in rows if r['kind']==kind]
        summary[kind]=dict(n=len(rr),words=len({r['label'] for r in rr}),variants={v:{m:dict(stats=stats([r[v][m] for r in rr]),better_vs_off=sum(r[v][m]<r['OFF'][m]-1e-9 for r in rr),worse_vs_off=sum(r[v][m]>r['OFF'][m]+1e-9 for r in rr),better_vs_old=sum(r[v][m]<r['OLD'][m]-1e-9 for r in rr),worse_vs_old=sum(r[v][m]>r['OLD'][m]+1e-9 for r in rr)) for m in ['center','size']} for v in VS})
        for label in p['labels']:
            ww=[r for r in rr if r['label']==label]
            if ww:wordrows.append(dict(label=label,kind=kind,n=len(ww),**{m:float(np.mean([r['NEW'][m]-r['OLD'][m] for r in ww])) for m in ['center','size']}))
    sw=[r for r in rows if r['kind']=='switch'];selected=[]
    if sw:
        ranked=sorted(sw,key=lambda r:r['OFF']['size'])
        for r in [min(sw,key=lambda r:r['NEW']['size']-r['OLD']['size']),max(sw,key=lambda r:r['NEW']['center']-r['OLD']['center']),ranked[len(ranked)//2]]:
            if r not in selected:selected.append(r)
    reviews=[]
    for i,r in enumerate(selected):
        c=read(Path(next(d['file'] for d in cases if d['label']==r['label'])))
        c['phrase']=f'review{i}_'+r['label'];c['frameIndices']=list(range(max(0,r['index']-13),min(len(c['payload']['frames']),r['index']+13)))
        dest=OUT/'cases'/f'review{i}.json';write(dest,c);reviews.append(dict(file=str(dest),**r))
    write(OUT/'review_index.json',reviews)
    word_summary={};confidence={}
    for kind in ['switch','native']:
        ww=[r for r in wordrows if r['kind']==kind]
        word_summary[kind]={m:dict(better=sum(r[m]<-1e-9 for r in ww),worse=sum(r[m]>1e-9 for r in ww),unchanged=sum(abs(r[m])<=1e-9 for r in ww)) for m in ['center','size']}
        for m in ['center','size']:
            a=np.array([r[m] for r in ww])
            if len(a):confidence[kind+'_'+m]=dict(mean=float(a.mean()),bootstrap95=np.percentile(np.random.default_rng(2026100460).choice(a,(5000,len(a)),replace=True).mean(axis=1),[2.5,97.5]).tolist(),words=len(a))
    verify()
    result=dict(summary=summary,rows=rows,word_differences=wordrows,word_summary=word_summary,confidence=confidence,modified_hand_frames=ops,max_reference_target_error=error,frames_per_method=frame_count,production_unchanged=True,runtime={v:dict(**stats(times[v]),fraction_over_16_667ms=float(np.mean(np.array(times[v])>1000/60))) for v in VS},review=selected,words_without_switch=sorted(set(p['labels'])-{r['label'] for r in sw}))
    write(OUT/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['rows','word_differences']},indent=2))

def review_check():
    verify();caps={(c['phrase'],c['variant']):c['samples'] for c in read(OUT/'renderer/rendered.json')};n=0
    desc=read(OUT/'review_index.json')
    for c in read(OUT/'visual_review/rendered.json'):
        d=next(d for d in desc if read(Path(d['file']))['phrase']==c['phrase']);full=caps[d['phrase'],c['variant']]
        for s in c['samples']:
            assert {k:v for k,v in s.items() if k!='renderMs'}=={k:v for k,v in full[s['index']].items() if k!='renderMs'};n+=1
    write(OUT/'review_validation.json',dict(exact_samples=n,passed=True));print('Review exact samples',n)

if __name__=='__main__':{'prepare':prepare,'canary':canary,'analyse':analyse,'review_check':review_check}[sys.argv[1]]()
