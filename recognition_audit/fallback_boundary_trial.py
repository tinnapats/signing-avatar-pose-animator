import json,sys
from pathlib import Path
import numpy as np
import boundary_audit as old
base=old.base;HERE=old.HERE;SRC=old.SRC;OUT=HERE/'fallback_boundary_20261003'
read=old.read;write=old.write
def prepare():
    assert not OUT.exists();OUT.mkdir();(OUT/'cases').mkdir()
    labels=['yes','camera','never','recliner','parents','lord','scientist','head','approach','hurry','overwhelm','hello','image']
    write(OUT/'protocol.json',dict(labels=labels,production=base.production(),rules='13 fixed diagnostic words. OFF vs ON; native-only similarity transform about body wrist, reference nearest observed hand within6 payload frames=100ms, past wins equal distance; rotation<=45degrees, scale .85..1.15, smoothstep decay to zero outside window. Offline future reference allowed. No change of detailed hands, native visibility/opacity, frame timing/body. Not new hand pose reconstruction. One fixed candidate, no tuning.',hashes={n:base.sha(HERE/n) for n in ['fallback_boundary_adjust.js','fallback_boundary_render.mjs']}))
    index=[]
    for p in sorted((SRC/'cases').glob('*.json')):
        c=read(p)
        if c['label'] not in labels:continue
        dest=OUT/'cases'/p.name;dest.write_text(json.dumps(c,separators=(',',':')),encoding='utf-8');index.append(dict(file=str(dest),phrase=c['phrase'],label=c['label']))
    assert len(index)==13;write(OUT/'case_index.json',index)
    write(OUT/'review_index.json',read(old.OUT/'review_index.json'))
    print('Prepared13 fixed words; production unchanged')
def analyse():
    p=read(OUT/'protocol.json');assert p['production']==base.production()
    captures=read(OUT/'renderer/rendered.json');off={c['phrase']:c for c in captures if c['variant']=='OFF'};on={c['phrase']:c for c in captures if c['variant']=='ON'};assert len(off)==len(on)==13
    prior={c['phrase']:c for c in read(SRC/'renderer/rendered.json')};rows=[];ops=0;frames=0
    for phrase,a in off.items():
        b=on[phrase];assert len(a['samples'])==len(b['samples']);label=phrase.split('__')[0]
        for x,y,z in zip(a['samples'],b['samples'],prior[phrase]['samples']):
            frames+=1;assert x['hands']==y['hands']==z['hands'];assert x['native']==z['native'];assert x['body']==y['body'];assert x['index']==y['index']
            for side in ['left','right']:
                assert len(x['native'][side])==len(y['native'][side]);assert [q['opacity'] for q in x['native'][side]]==[q['opacity'] for q in y['native'][side]]
                op=y['ops'].get(side)
                if op:assert 1<=op['distance']<=6 and abs(op['rotation'])<=45 and .85<=op['scale']<=1.15;ops+=1
                else:assert x['geometry'][side]==y['geometry'][side]
        for i in range(1,len(a['samples'])):
            for side in ['left','right']:
                x0,x1=a['samples'][i-1:i+1];kind='switch' if old.state(x0,side)!=old.state(x1,side) else 'native' if old.state(x1,side)=='native' else 'detailed'
                scores={}
                for variant,c in [('OFF',a),('ON',b)]:
                    u,v=c['samples'][i-1:i+1];g,h=u['geometry'][side],v['geometry'][side]
                    if not g or not h:continue
                    q,r=np.array(g['bounds']),np.array(h['bounds']);w=(g['shoulderWidth']+h['shoulderWidth'])/2
                    center=np.linalg.norm((r[:2]+r[2:])/2-np.array(h['wrist'])-((q[:2]+q[2:])/2-np.array(g['wrist'])))/w
                    sizes=[np.linalg.norm(t[2:]-t[:2]) for t in [q,r]]
                    scores[variant]=dict(center=float(center),size=float(max(sizes)/min(sizes)))
                if len(scores)==2:rows.append(dict(label=label,side=side,index=x1['index'],kind=kind,**scores))
    summary={}
    for kind in ['switch','native','detailed']:
        rr=[r for r in rows if r['kind']==kind];summary[kind]=dict(n=len(rr),**{v:{m:dict(mean=float(np.mean([r[v][m] for r in rr])),median=float(np.median([r[v][m] for r in rr]))) for m in ['center','size']} for v in ['OFF','ON']},center_improved=sum(r['ON']['center']<r['OFF']['center']-1e-9 for r in rr),center_worse=sum(r['ON']['center']>r['OFF']['center']+1e-9 for r in rr))
    write(OUT/'results.json',dict(summary=summary,modified_native_hand_frames=ops,sampled_time_frames=frames,rows=rows,invariants='Detailed endpoints, non-hand geometry, native visibility/opacity, frame count and indices exact; OFF reproduces previous hand/native captures; non-adjusted frames identical.',production_unchanged=True))
    print(json.dumps(dict(summary=summary,modified_native_hand_frames=ops,sampled_time_frames=frames),indent=2))
if __name__=='__main__':{'prepare':prepare,'analyse':analyse}[sys.argv[1]]()
