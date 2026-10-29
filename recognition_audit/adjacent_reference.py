import sys
import numpy as np
import render_bounds_scale as prev
base=prev.base; read=prev.read; write=prev.write
OUT=prev.HERE/'adjacent_reference_20261004'
VS=['OFF','OLD','NEW']

def prepare():
    assert not OUT.exists(); OUT.mkdir()
    write(OUT/'protocol.json',dict(production=base.production(),labels=read(prev.OUT/'protocol.json')['labels'],rules='Same 30 development words, all 60FPS frames. OFF no scale. OLD camera-aware actual bounds nearest observed. NEW nearest drawable detailed hand (including interpolated), past wins ties. Same 6-frame window, .85..1.15 cap, Smoothstep decay, wrist pivot, zero rotation. Offline future reference allowed. No word-specific rules. No production changes.',script_hashes={f:base.sha(prev.HERE/f) for f in ['adjacent_reference.py','adjacent_reference_render.mjs','render_bounds_scale_render.mjs','fallback_boundary_adjust.js']}))
    for name in ['case_index.json','review_index.json']:write(OUT/name,read(prev.OUT/name))
    print('Prepared frozen adjacent-reference experiment')

def analyse():
    assert read(OUT/'protocol.json')['production']==base.production()
    caps={(c['phrase'],c['variant']):c['samples'] for c in read(OUT/'renderer/rendered.json')}
    old={(c['phrase'],c['variant']):c['samples'] for c in read(prev.OUT/'renderer/rendered.json')}
    assert len(caps)==90
    rows=[]; ops={v:0 for v in VS}; replay=0; error=0
    for d in read(OUT/'case_index.json'):
        phrase=d['phrase']; a=caps[phrase,'OFF']
        for v in VS:
            b=caps[phrase,v]; assert len(a)==len(b)
            for i,(x,y) in enumerate(zip(a,b)):
                assert x['index']==y['index']==i and x['hands']==y['hands'] and x['body']==y['body']
                for side in ['left','right']:
                    assert [t['opacity'] for t in x['native'][side]]==[t['opacity'] for t in y['native'][side]]
                    op=y['ops'].get(side)
                    if op:
                        ops[v]+=1; assert op['rotation']==0 and .85<=op['scale']<=1.15 and 1<=op['distance']<=6
                        ref=a[op['referenceIndex']]
                        assert prev.prev.comp.prev.old.state(ref,side)=='detailed'
                        bounds=ref['geometry'][side]['bounds']; diag=float(np.linalg.norm(np.array(bounds[2:])-bounds[:2]))
                        error=max(error,abs(diag-op['target'])); assert abs(diag-op['target'])<1e-6
                    else:assert x['geometry'][side]==y['geometry'][side]
            if v in ['OFF','OLD']:
                previous=old[phrase,'OFF' if v=='OFF' else 'NEW']; assert b==previous; replay+=len(b)
        for i in range(1,len(a)):
            for side in ['left','right']:
                s0,s1=[prev.prev.comp.prev.old.state(a[j],side) for j in [i-1,i]]
                if any(caps[phrase,v][j]['geometry'][side] is None for v in VS for j in [i-1,i]):continue
                rows.append(dict(label=d['label'],side=side,index=i,kind='switch' if s0!=s1 else s1,**{v:prev.prev.comp.metric(caps[phrase,v][i-1],caps[phrase,v][i],side) for v in VS}))
    summary={}
    for kind in ['switch','native','detailed']:
        rr=[r for r in rows if r['kind']==kind]
        summary[kind]=dict(n=len(rr),variants={v:{m:dict(mean=float(np.mean([r[v][m] for r in rr])),median=float(np.median([r[v][m] for r in rr])),better=sum(r[v][m]<r['OFF'][m]-1e-9 for r in rr),worse=sum(r[v][m]>r['OFF'][m]+1e-9 for r in rr),better_than_old=sum(r[v][m]<r['OLD'][m]-1e-9 for r in rr),worse_than_old=sum(r[v][m]>r['OLD'][m]+1e-9 for r in rr)) for m in ['center','size']} for v in VS})
    assert read(OUT/'protocol.json')['production']==base.production()
    result=dict(summary=summary,rows=rows,modified_hand_frames=ops,previous_samples_exact=replay,max_reference_target_error=error,production_unchanged=True)
    write(OUT/'results.json',result)
    print(__import__('json').dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':{'prepare':prepare,'analyse':analyse}[sys.argv[1]]()
