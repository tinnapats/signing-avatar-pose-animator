"""Measure visible stroke bounds across all previously observed hand switches."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json,sys
from pathlib import Path
import numpy as np
import novel_words30 as prev
base=prev.base;HERE=base.HERE;SRC=HERE/'slgcn200_20261002';OUT=HERE/'boundary_audit_20261003'
def read(p):return _local_json_loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def state(s,side):
    d=bool(s['hands'][side]);n=bool(s['native'][side]);return 'detailed' if d and not n else 'native' if n and not d else 'other'
def prepare():
    assert not OUT.exists();OUT.mkdir();(OUT/'cases').mkdir()
    protocol=dict(production=base.production(),capture_sha=base.sha(SRC/'renderer/rendered.json'),
      scope='All native/detailed switches found in existing200 capture, sampled30FPS; unchanged candidate no_hold_no_smoothing. Re-render only windows; no model inference, no production changes.',
      metrics='Union visible strokeBounds for each hand, both renderer types. Camera undone; bbox center minus pose wrist normalized by shoulder width. Step is change of wrist-relative center / mean shoulder width. Size jump is larger/smaller bbox diagonal ratio. Not joint error, semantic similarity, bone length, or jerk.',
      selection='For each switch at original sample i retain i-2..i+1 clipped. Controls are adjacent same-type frame pairs in the selected windows; deduplicate by word/side/time. Descriptive matched-neighborhood comparison, not causal. Review top center jump, top size ratio, and median center event, unique events.')
    write(OUT/'protocol.json',protocol)
    caps={c['phrase']:c for c in read(SRC/'renderer/rendered.json')};index=[];events=[]
    for p in sorted((SRC/'cases').glob('*.json')):
        c=read(p);cap=caps[c['phrase']];selected=set()
        for side in ['left','right']:
            for i in range(1,len(cap['samples'])):
                a,b=state(cap['samples'][i-1],side),state(cap['samples'][i],side)
                if {a,b}=={'native','detailed'}:
                    events.append(dict(label=c['label'],side=side,before=c['frameIndices'][i-1],after=c['frameIndices'][i],direction=a+'->'+b))
                    selected.update(range(max(0,i-2),min(len(cap['samples']),i+2)))
        if not selected:continue
        c['frameIndices']=[c['frameIndices'][i] for i in sorted(selected)]
        dest=OUT/'cases'/p.name;dest.write_text(json.dumps(c,separators=(',',':')),encoding='utf-8');index.append(dict(label=c['label'],phrase=c['phrase'],file=str(dest)))
    assert len(events)==592;write(OUT/'events.json',events);write(OUT/'case_index.json',index)
    print('Prepared words',len(index),'switches',len(events),flush=True)
def analyse():
    protocol=read(OUT/'protocol.json');assert base.production()==protocol['production'];assert base.sha(SRC/'renderer/rendered.json')==protocol['capture_sha']
    cases={d['phrase']:d for d in read(OUT/'case_index.json')};old={c['phrase']:c for c in read(SRC/'renderer/rendered.json')};eventkeys={(e['label'],e['side'],e['after']) for e in read(OUT/'events.json')};rows=[]
    caps=read(OUT/'renderer/rendered.json')
    for cap in caps:
        d=cases[cap['phrase']];c=read(Path(d['file']));prior={s['index']:s for s in old[c['phrase']]['samples']}
        for s in cap['samples']:
            assert s['hands']==prior[s['index']]['hands'] and s['native']==prior[s['index']]['native']
        for a,b in zip(cap['samples'],cap['samples'][1:]):
            if b['index']-a['index']!=2:continue
            for side in ['left','right']:
                ga,gb=a['geometry'][side],b['geometry'][side]
                if ga is None or gb is None:continue
                width=(ga['shoulderWidth']+gb['shoulderWidth'])/2
                if width<=1e-6:continue
                ba,bb=np.array(ga['bounds']),np.array(gb['bounds']);ca=(ba[:2]+ba[2:])/2;cb=(bb[:2]+bb[2:])/2
                da=np.linalg.norm(ba[2:]-ba[:2]);db=np.linalg.norm(bb[2:]-bb[:2]);assert min(da,db)>1e-6
                relative=np.linalg.norm((cb-np.array(gb['wrist']))-(ca-np.array(ga['wrist'])))/width
                switched=(c['label'],side,b['index']) in eventkeys
                rows.append(dict(label=c['label'],phrase=c['phrase'],side=side,before=a['index'],after=b['index'],switch=switched,direction=state(a,side)+'->'+state(b,side),center_step=float(relative),size_ratio=float(max(da,db)/min(da,db))))
    switches=[r for r in rows if r['switch']];assert len(switches)==592
    controls=[r for r in rows if not r['switch'] and r['direction'] in ['native->native','detailed->detailed']]
    def summary(items):
        return dict(n=len(items),**{k:dict(mean=float(np.mean([r[k] for r in items])),median=float(np.median([r[k] for r in items])),p95=float(np.percentile([r[k] for r in items],95))) for k in ['center_step','size_ratio']})
    ordered=sorted(switches,key=lambda r:r['center_step']);selected=[]
    for r in [ordered[-1],max(switches,key=lambda r:r['size_ratio']),ordered[len(ordered)//2]]:
        if r not in selected:selected.append(r)
    review=[]
    for n,r in enumerate(selected):
        c=read(Path(cases[r['phrase']]['file']));c['frameIndices']=[r['before'],r['after']];c['phrase']=f"review{n}_"+r['label'];dest=OUT/'cases'/f'review{n}.json';dest.write_text(json.dumps(c,separators=(',',':')),encoding='utf-8');review.append(dict(file=str(dest),**r))
    write(OUT/'review_index.json',review)
    result=dict(switches=summary(switches),nearby_controls=summary(controls),native_controls=summary([r for r in controls if r['direction']=='native->native']),detailed_controls=summary([r for r in controls if r['direction']=='detailed->detailed']),words=len(caps),rows=rows,review=selected,production_unchanged=True)
    write(OUT/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
if __name__=='__main__':{'prepare':prepare,'analyse':analyse}[sys.argv[1]]()
