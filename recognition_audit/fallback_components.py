import sys,json
import numpy as np
import fallback_boundary_trial as prev
base=prev.base;HERE=prev.HERE;OUT=HERE/'fallback_components_20261003';read=prev.read;write=prev.write
VS=['OFF','ROTATE','SCALE','BOTH']
def prepare():
    assert not OUT.exists();OUT.mkdir()
    write(OUT/'protocol.json',dict(production=base.production(),labels=read(prev.OUT/'protocol.json')['labels'],variants=VS,rules='Same fixed13,30FPS, original durations, 6-frame reference window,45degree rotation cap,15percent scale cap,Smoothstep decay. OFF vs rotation only vs scale only vs both. No parameter changes or tuning. Whole-clip native continuity checked; not semantic evaluation. Reproduce OFF/BOTH previous captures exactly.',previous_capture_sha=base.sha(prev.OUT/'renderer/rendered.json')))
    for name in ['case_index.json','review_index.json']:write(OUT/name,read(prev.OUT/name))
    print('Prepared fixed four-way ablation')
def metric(a,b,side):
    g,h=a['geometry'][side],b['geometry'][side]
    q,r=np.array(g['bounds']),np.array(h['bounds']);w=(g['shoulderWidth']+h['shoulderWidth'])/2
    center=np.linalg.norm((r[:2]+r[2:])/2-np.array(h['wrist'])-((q[:2]+q[2:])/2-np.array(g['wrist'])))/w
    sizes=[np.linalg.norm(t[2:]-t[:2]) for t in [q,r]]
    return dict(center=float(center),size=float(max(sizes)/min(sizes)))
def analyse():
    p=read(OUT/'protocol.json');assert p['production']==base.production();assert p['previous_capture_sha']==base.sha(prev.OUT/'renderer/rendered.json')
    captures={(c['phrase'],c['variant']):c for c in read(OUT/'renderer/rendered.json')};assert len(captures)==52
    old={(c['phrase'],c['variant']):c for c in read(prev.OUT/'renderer/rendered.json')};rows=[];modified={v:0 for v in VS}
    phrases=[phrase for phrase,v in captures if v=='OFF']
    for phrase in phrases:
        off=captures[phrase,'OFF']['samples'];assert off==old[phrase,'OFF']['samples'];assert captures[phrase,'BOTH']['samples']==old[phrase,'ON']['samples']
        for v in VS:
            ss=captures[phrase,v]['samples'];assert len(ss)==len(off)
            for a,b in zip(off,ss):
                assert a['index']==b['index'] and a['hands']==b['hands'] and a['body']==b['body']
                for side in ['left','right']:
                    assert [n['opacity'] for n in a['native'][side]]==[n['opacity'] for n in b['native'][side]]
                    op=b['ops'].get(side)
                    if op:
                        assert 1<=op['distance']<=6 and abs(op['rotation'])<=45 and .85<=op['scale']<=1.15
                        if v=='ROTATE':assert op['scale']==1
                        if v=='SCALE':assert op['rotation']==0
                        modified[v]+=1
                    else:assert a['geometry'][side]==b['geometry'][side]
        for i in range(1,len(off)):
            for side in ['left','right']:
                a,b=off[i-1:i+1];s0,s1=prev.old.state(a,side),prev.old.state(b,side);kind='switch' if s0!=s1 else s1
                row=dict(label=phrase.split('__')[0],side=side,index=b['index'],kind=kind)
                for v in VS:row[v]=metric(*captures[phrase,v]['samples'][i-1:i+1],side)
                rows.append(row)
    summary={}
    for kind in ['switch','native','detailed']:
        rr=[r for r in rows if r['kind']==kind];summary[kind]=dict(n=len(rr),variants={v:dict(**{m:dict(mean=float(np.mean([r[v][m] for r in rr])),median=float(np.median([r[v][m] for r in rr]))) for m in ['center','size']},improved=sum(r[v]['center']<r['OFF']['center']-1e-9 for r in rr),worse=sum(r[v]['center']>r['OFF']['center']+1e-9 for r in rr)) for v in VS})
    write(OUT/'results.json',dict(summary=summary,rows=rows,modified=modified,production_unchanged=True,off_both_exact_previous=True,invariants='Detailed endpoints/body/opacity/indices unchanged. Same eligibility141 per active option. Scale-only rotation0; rotation-only scale1.'))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':{'prepare':prepare,'analyse':analyse}[sys.argv[1]]()
