import sys,json
from pathlib import Path
import numpy as np
import fallback_scale_new30 as prev
base=prev.base;HERE=prev.HERE;OUT=HERE/'render_bounds_scale_camera_20261004';read=prev.read;write=prev.write;VS=['OFF','OLD','NEW']
def prepare():
    assert not OUT.exists();OUT.mkdir();(OUT/'cases').mkdir()
    write(OUT/'protocol.json',dict(production=base.production(),labels=read(prev.OUT/'protocol.json')['labels'],rules='Same30 now development, three-way OFF/old landmark-target/new actually-rendered strokeBounds target. All payload frames60FPS; same duration, reference<=6frames,scale limits .85..1.15,Smoothstep,rotation0. Reference drawn temporarily before camera, removed before final draw. No tuning/no per-word rules. Compare size AND center AND within-native movement.',hashes={f:base.sha(HERE/f) for f in ['fallback_boundary_adjust.js','render_bounds_scale_render.mjs']},review='Same bald/long/lip events as prior diagnosis; no favorable reselection.'))
    index=[]
    for i,d in enumerate(read(prev.OUT/'case_index.json')):
        c=read(Path(d['file']));c['frameIndices']=list(range(len(c['payload']['frames'])));dest=OUT/'cases'/f'{i:02d}.json';dest.write_text(json.dumps(c,separators=(',',':')),encoding='utf-8');index.append(dict(file=str(dest),label=c['label'],phrase=c['phrase']))
    write(OUT/'case_index.json',index);write(OUT/'review_index.json',read(prev.OUT/'review_index.json'));print('Prepared30 at60FPS, same rules')
def analyse():
    p=read(OUT/'protocol.json');assert p['production']==base.production()
    caps={(c['phrase'],c['variant']):c['samples'] for c in read(OUT/'renderer/rendered.json')};assert len(caps)==90
    prior={(c['phrase'],c['variant']):c['samples'] for c in read(prev.OUT/'renderer/rendered.json')};rows=[];replay=0;target_error=0;ops={v:0 for v in VS}
    for d in read(OUT/'case_index.json'):
        phrase=d['phrase'];a=caps[phrase,'OFF']
        for v in VS:
            b=caps[phrase,v];assert len(a)==len(b)
            for i,(x,y) in enumerate(zip(a,b)):
                assert x['index']==y['index']==i and x['hands']==y['hands'] and x['body']==y['body']
                for side in ['left','right']:
                    assert [t['opacity'] for t in x['native'][side]]==[t['opacity'] for t in y['native'][side]]
                    op=y['ops'].get(side)
                    if op:
                        ops[v]+=1;assert op['rotation']==0 and .85<=op['scale']<=1.15 and 1<=op['distance']<=6
                        if v=='NEW':
                            ref=a[op['referenceIndex']]['geometry'][side]['bounds'];diag=float(np.linalg.norm(np.array(ref[2:])-ref[:2]));err=abs(diag-op['target']);target_error=max(target_error,err);assert err<1e-6
                    else:assert x['geometry'][side]==y['geometry'][side]
            if v in ['OFF','OLD']:
                old=prior[phrase,'OFF' if v=='OFF' else 'SCALE'];assert len(b[::2])==len(old)
                for x,y in zip(b[::2],old):
                    for key in ['hands','native','geometry','body','index']:assert x[key]==y[key]
                    replay+=1
        for i in range(1,len(a)):
            for side in ['left','right']:
                s0,s1=prev.comp.prev.old.state(a[i-1],side),prev.comp.prev.old.state(a[i],side);kind='switch' if s0!=s1 else s1
                if any(caps[phrase,v][j]['geometry'][side] is None for v in VS for j in [i-1,i]):continue
                rows.append(dict(label=d['label'],side=side,index=i,kind=kind,**{v:prev.comp.metric(caps[phrase,v][i-1],caps[phrase,v][i],side) for v in VS}))
    summary={}
    for kind in ['switch','native','detailed']:
        rr=[r for r in rows if r['kind']==kind];summary[kind]=dict(n=len(rr),variants={v:{m:dict(mean=float(np.mean([r[v][m] for r in rr])),median=float(np.median([r[v][m] for r in rr])),better=sum(r[v][m]<r['OFF'][m]-1e-9 for r in rr),worse=sum(r[v][m]>r['OFF'][m]+1e-9 for r in rr)) for m in ['center','size']} for v in VS})
    # Compare exactly the old75 30FPS event pairs as secondary, do not mix with60FPS.
    secondary=[]
    for e in [r for r in read(prev.OUT/'results.json')['rows'] if r['kind']=='switch']:
        secondary.append(dict(label=e['label'],**{v:prev.comp.metric(caps[e['phrase'],v][e['before']],caps[e['phrase'],v][e['after']],e['side']) for v in VS}))
    secondary_summary={v:{m:dict(mean=float(np.mean([r[v][m] for r in secondary])),median=float(np.median([r[v][m] for r in secondary])),worse=sum(r[v][m]>r['OFF'][m]+1e-9 for r in secondary)) for m in ['center','size']} for v in VS}
    result=dict(summary60=summary,old75_at30fps=secondary_summary,rows=rows,modified_hand_frames=ops,old_capture_samples_exact=replay,max_reference_target_error=target_error,production_unchanged=True)
    assert p['production']==base.production();write(OUT/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':{'prepare':prepare,'analyse':analyse}[sys.argv[1]]()
