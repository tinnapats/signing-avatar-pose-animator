"""Coverage audit and frozen masking controls; never invent fallback joints."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json
from collections import Counter
import numpy as np
import novel_words30 as prev
from equal_duration_trial import arrays
base=prev.base; HERE=base.HERE; SRC=HERE/'slgcn200_20261002'; PAIR=HERE/'combined_winners200'; OUT=HERE/'visibility_audit_20261003'
def read(p):return _local_json_loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def stats(c,cap):
    counts=Counter(); tracks={side:[] for side in ['left','right']};observed_mask=np.zeros((len(cap['samples']),2),dtype=bool)
    for i,(index,s) in enumerate(zip(c['frameIndices'],cap['samples'])):
        f=c['payload']['frames'][index]
        for j,side in enumerate(['left','right']):
            detail=bool(s['hands'][side]);native=bool(s['native'][side]);observed=bool(f['hands'][side].get('observed',False))
            state=('both' if native else ('detailed_observed' if observed else 'detailed_synthetic')) if detail else ('native_only' if native else 'neither')
            counts[state]+=1;tracks[side].append(state);observed_mask[i,j]=detail and observed
    transitions=Counter()
    for t in tracks.values():
        for a,b in zip(t,t[1:]):
            if a!=b:transitions[a+' -> '+b]+=1
    total=sum(counts.values());return dict(label=c['label'],counts=dict(counts),hand_frames=total,native_fraction=counts['native_only']/total,transitions=dict(transitions)),observed_mask
def mask(a,keep):
    a=a.copy()
    for j,offset in enumerate([33,54]):a[~keep[:,j],offset:offset+21]=0
    return a
def main():
    assert not OUT.exists();OUT.mkdir()
    files=[SRC/'renderer/rendered.json',SRC/'results.json',PAIR/'renderer/rendered.json',PAIR/'review_inputs.json',HERE/'wlasl_stgcn.zip']
    protocol=dict(production=base.production(),hashes={str(p.relative_to(HERE)):base.sha(p) for p in files},
      scope='Reuse verified actual Canvas captures, no new rendering or production change. Fixed200 candidate and diagnostic12 baseline/candidate.',
      rules='Hand-frame = one side at one sampled time, 30FPS. Observed means payload flag, not verified ground truth. Native visibility from actual tagged SVG paths. Never construct native hand joints.',
      controls='200 full vs zero synthetic detailed hand blocks, preserve frame count/body/time. Diagnostic12 baseline vs candidate, full and same observed-hand visibility intersection. Zero is existing missing convention, not image removal. Scores remain whole-clip recognition; no per-frame F1.',
      interpretation='Exploratory development data. Masking controls change model inputs, not user-visible animation; not proof of human comprehension or causal benefit. No tuning.')
    write(OUT/'protocol.json',protocol)
    caps={c['phrase']:c for c in read(SRC/'renderer/rendered.json')};old={r['label']:r['prediction'] for r in read(SRC/'results.json')['predictions'] if r['model']=='ST-GCN' and r['stage']=='post'}
    _,predict=prev.load_predictor();rows=[];coverage=[];labels=[];removed=0
    for n,path in enumerate(sorted((SRC/'cases').glob('*.json'))):
        c=read(path);labels.append(c['label']);cap=caps[c['phrase']];s,observed=stats(c,cap);a=arrays(c,cap)
        full=predict(a,c['label'],'full');assert full['prediction']==old[c['label']]
        limited=mask(a,observed);rows.extend([full,predict(limited,c['label'],'observed_only')]);s['correct']=full['prediction']==c['label'];coverage.append(s)
        removed+=int(np.any(a!=limited,axis=(1,2)).sum())
        if (n+1)%40==0:print('200 audit scored',n+1,flush=True)
    metrics={v:base.metrics([r for r in rows if r['variant']==v],labels) for v in ['full','observed_only']}
    totals=Counter();transitions=Counter()
    for s in coverage:totals.update(s['counts']);transitions.update(s['transitions'])
    by={(r['label'],r['variant']):r['prediction']==r['label'] for r in rows}
    changes={name:[l for l in labels if condition(by[l,'full'],by[l,'observed_only'])] for name,condition in [('gained',lambda a,b:not a and b),('lost',lambda a,b:a and not b)]}
    groups={str(ok):dict(words=sum(s['correct']==ok for s in coverage),mean_word_native_fraction=float(np.mean([s['native_fraction'] for s in coverage if s['correct']==ok]))) for ok in [True,False]}
    # Same visibility intersection within each word, never shorten or concatenate frames.
    cases=read(PAIR/'review_inputs.json');paircaps=read(PAIR/'renderer/rendered.json');lookup={c['phrase']:c for c in cases};pairs={}
    for cap in paircaps:
        if cap['variant']!='S1A1':continue
        c=lookup[cap['phrase']]
        v=c.get('variant')
        if v not in ['baseline','no_hold_no_smoothing']:
            v=next((v for v in ['no_hold_no_smoothing','baseline'] if c['phrase'].endswith('__'+v)),None)
        if v:pairs.setdefault(c['label'],{})[v]=(c,cap)
    assert len(pairs)==12 and all(len(x)==2 for x in pairs.values())
    pairrows=[];paircoverage=[]
    for label,pair in pairs.items():
        c0,s0=pair['baseline'];c1,s1=pair['no_hold_no_smoothing'];assert c0['frameIndices']==c1['frameIndices']
        g0,m0=stats(c0,s0);g1,m1=stats(c1,s1);keep=m0&m1
        for v,(c,cap) in pair.items():
            a=arrays(c,cap);pairrows.extend([predict(a,label,v+'__full'),predict(mask(a,keep),label,v+'__common_observed')])
        paircoverage.append(dict(label=label,baseline=g0,candidate=g1,common_observed_hand_frames=int(keep.sum())))
    pm={v:base.metrics([r for r in pairrows if r['variant']==v],list(pairs)) for v in sorted({r['variant'] for r in pairrows})}
    assert base.production()==protocol['production']
    result=dict(metrics200=metrics,totals200=dict(totals),transitions200=dict(transitions),changed_time_frames=removed,changes200=changes,descriptive_groups200=groups,per_word_coverage=coverage,predictions200=rows,metrics12=pm,coverage12=paircoverage,predictions12=pairrows,production_unchanged=True,baseline_predictions_reproduced=200)
    write(OUT/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_word_coverage','predictions200','coverage12','predictions12']},indent=2),flush=True)
if __name__=='__main__':main()
