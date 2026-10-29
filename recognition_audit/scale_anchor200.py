"""Exact affine counterfactuals from actual200; browser validation on changed20."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json
import math
import sys
import numpy as np
from pathlib import Path
import novel_words30 as prev
from equal_duration_trial import arrays
base=prev.base;HERE=base.HERE;SRC=HERE/'slgcn200_20261002';OUT=HERE/'scale_anchor200_20261003'
VS={'A':(0,0),'B':(1,0),'C':(0,1),'D':(1,1)}

def variants(c,cap):
    original=arrays(c,cap);out={v:original.copy() for v in VS};maxerr=0;scales=[]
    for i,s in enumerate(cap['samples']):
        f=c['payload']['frames'][c['frameIndices'][i]];pose={k['part']:k for k in f['pose']['keypoints']}
        for side,offset in [('left',33),('right',54)]:
            points=s['hands'][side]
            if not points:continue
            ids=sorted(map(int,points));h=f['hands'][side]
            raw=np.array([[k['position']['x'],k['position']['y']] for k in h['keypoints']]);w=raw[0]
            body=pose[side+'Wrist'];anchor=np.array([body['position']['x'],body['position']['y']]) if body['score']>=.2 else w
            drawn=np.array([points[str(k)] for k in ids]);x=raw[ids]-w;y=drawn-anchor
            denom=np.sum(x*x);assert denom>1e-10
            scale=float(np.sum(x*y)/denom);assert scale>0
            err=float(np.max(np.linalg.norm(anchor+scale*x-drawn,axis=1)));assert err<1e-6
            maxerr=max(maxerr,err);scales.append(scale)
            for v,(size,attach) in VS.items():
                if v=='D':continue # Preserve original float32 input exactly.
                out[v][i,offset+np.array(ids)]=w+attach*(anchor-w)+(scale if size else 1)*x
    return out,maxerr,scales

def geometry(c,a):
    wrist=[];ratio=[];bone=[];observed=0;rough=[]
    for side,offset in [('left',33),('right',54)]:
        track=[];valid=[];widths=[]
        for i,index in enumerate(c['frameIndices']):
            f=c['payload']['frames'][index];p={k['part']:k for k in f['pose']['keypoints']}
            pos=lambda k:np.array([p[k]['position']['x'],p[k]['position']['y']])
            width=np.linalg.norm(pos('leftShoulder')-pos('rightShoulder'));q=a[i,offset:offset+21].astype(float)
            ok=not np.any(np.all(q==0,axis=1)) and width>1e-6
            track.append(q-q[0]);valid.append(ok);widths.append(width)
            if not ok:continue
            if p[side+'Wrist']['score']>=.2:wrist.append(float(np.linalg.norm(q[0]-pos(side+'Wrist'))/width))
            arm=np.linalg.norm(pos(side+'Wrist')-pos(side+'Elbow'))
            chains=[[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
            lengths=[sum(np.linalg.norm(q[b]-q[t]) for t,b in zip(ch,ch[1:])) for ch in chains]
            if arm>1e-6 and min(p[side+'Wrist']['score'],p[side+'Elbow']['score'])>=.2:ratio.append(max(lengths)/arm)
            if f['hands'][side].get('observed'):
                observed+=1;bone.extend(np.linalg.norm(q[b]-q[t]) for ch in chains for t,b in zip(ch,ch[1:]))
        for i in range(1,len(track)-1):
            if all(valid[i-1:i+2]):rough.append(float(np.linalg.norm(track[i+1]-2*track[i]+track[i-1],axis=1).mean()/widths[i]))
    mean=lambda x:float(np.mean(x)) if x else None
    return dict(wrist_to_pose_wrist_shoulders=mean(wrist),hand_chain_to_projected_forearm=mean(ratio),wrist_relative_second_difference_shoulders=mean(rough),observed_hand_frames=observed,
        note='2D projection/proxy metrics, not anatomical accuracy, not actual SVG arm endpoint, not jerk. Hand/forearm projection affected by foreshortening.')

def prepare():
    assert not OUT.exists();OUT.mkdir();(OUT/'arrays').mkdir()
    prior=_local_json_loads((SRC/'results.json').read_text());labels=_local_json_loads((SRC/'protocol.json').read_text())['labels']
    by={(r['label'],r['stage']):r for r in prior['predictions'] if r['model']=='ST-GCN'}
    review=[l for l in labels if (by[l,'pre']['prediction']==l)!=(by[l,'post']['prediction']==l)];assert len(review)==20
    files=[SRC/'inputs.json',SRC/'renderer/rendered.json',SRC/'results.json',HERE/'wlasl_stgcn.zip']
    protocol=dict(labels=labels,review=review,variants=VS,production=base.production(),hashes={str(p.relative_to(HERE)):base.sha(p) for p in files},
        primary='ST-GCN accuracy/F1 on unchanged200; three paired comparisons A/B/C vs D, Bonferroni3',
        method='D actual rendered captures. A/B/C analytically reconstructed from exact affine projection; browser verification on preselected changed20, not fresh800 Canvas captures.',
        scope='Development, one clip/word, no production changes. Same visibility/body/native fallback across conditions; original30FPS. No ASL correctness inference. No tuning.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    caps={c['phrase']:c for c in _local_json_loads((SRC/'renderer/rendered.json').read_text())};selected=[];geoms=[];maxerr=0
    for i,label in enumerate(labels):
        c=_local_json_loads((SRC/'cases'/f'{i:03d}.json').read_text());assert c['label']==label
        variants_a,err,scales=variants(c,caps[c['phrase']]);maxerr=max(maxerr,err)
        np.savez_compressed(OUT/'arrays'/f'{i:03d}.npz',**variants_a)
        for v,a in variants_a.items():geoms.append(dict(label=label,variant=v,**geometry(c,a)))
        if label in review:selected.append(c)
    assert base.production()==protocol['production']
    (OUT/'review_inputs.json').write_text(json.dumps(selected),encoding='utf-8')
    (OUT/'geometry.json').write_text(json.dumps(dict(rows=geoms,max_affine_reconstruction_error=maxerr),indent=2),encoding='utf-8')
    print('Prepared200, browser review20, max affine error',maxerr,flush=True)

def score():
    p=_local_json_loads((OUT/'protocol.json').read_text());assert base.production()==p['production']
    for name,h in p['hashes'].items():assert base.sha(HERE/name)==h
    captures=_local_json_loads((OUT/'renderer/rendered.json').read_text());assert len(captures)==80
    cases={c['phrase']:c for c in _local_json_loads((OUT/'review_inputs.json').read_text())};worst=0
    oldcaps={c['phrase']:c for c in _local_json_loads((SRC/'renderer/rendered.json').read_text())}
    for cap in captures:
        c=cases[cap['phrase']];i=p['labels'].index(c['label']);expected=np.load(OUT/'arrays'/f'{i:03d}.npz')[cap['variant']]
        actual=arrays(c,cap);error=float(np.max(np.abs(actual-expected)));worst=max(worst,error);assert error<.0001
        for a,b in zip(cap['samples'],oldcaps[c['phrase']]['samples']):
            assert a['native']==b['native']
            for side in ['left','right']:assert set(a['hands'][side])==set(b['hands'][side])
        if cap['variant']=='D':assert cap['samples']==oldcaps[c['phrase']]['samples']
    _,predict=prev.load_predictor();old={r['label']:r for r in _local_json_loads((SRC/'results.json').read_text())['predictions'] if r['model']=='ST-GCN' and r['stage']=='post'}
    rows=[]
    for i,label in enumerate(p['labels']):
        data=np.load(OUT/'arrays'/f'{i:03d}.npz')
        for v in VS:
            r=predict(data[v],label,v)
            if v=='D':assert r['prediction']==old[label]['prediction']
            rows.append(r)
        if (i+1)%25==0:print('scored',i+1,flush=True)
    summaries={v:base.metrics([r for r in rows if r['variant']==v],p['labels']) for v in VS};paired={}
    for v in ['A','B','C']:
        by={(r['label'],r['variant']):r['prediction'] for r in rows};g=[];l=[]
        for label in p['labels']:
            a=by[label,'D']==label;b=by[label,v]==label
            if b and not a:g.append(label)
            if a and not b:l.append(label)
        n=len(g)+len(l);pv=min(1,2*sum(math.comb(n,k) for k in range(min(len(g),len(l))+1))/2**n) if n else 1
        paired[v]=dict(gained=g,lost=l,p=pv,bonferroni3=min(1,pv*3))
    geom=_local_json_loads((OUT/'geometry.json').read_text());avg={v:{k:float(np.mean([r[k] for r in geom['rows'] if r['variant']==v and r[k] is not None])) for k in ['wrist_to_pose_wrist_shoulders','hand_chain_to_projected_forearm','wrist_relative_second_difference_shoulders']} for v in VS}
    assert base.production()==p['production']
    result=dict(summaries=summaries,paired=paired,geometry_word_means=avg,predictions=rows,browser_max_error=worst,production_unchanged=True,control200_top1_exact=True,native_unchanged_in_review=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='predictions'},indent=2))

if __name__=='__main__':globals()[sys.argv[1]]()
