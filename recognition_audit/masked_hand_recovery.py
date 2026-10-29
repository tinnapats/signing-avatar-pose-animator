"""Frozen synthetic gaps in raw database landmarks; no renderer/production edits."""
import sys,json,random,time,copy
from pathlib import Path
import numpy as np
import adjacent_holdout60 as trial
HERE=trial.HERE;OUT=HERE/'masked_hand_recovery_20261005';pipe=trial.base.audit.pipeline
METHODS=['HOLD','HOLD_SHAPE','LINEAR','SMOOTHSTEP','KINEMATIC']
EDGES=pipe.HAND_KINEMATIC_EDGES
CHAINS=[[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
TRIPLES=[(a,b,c) for chain in CHAINS for a,b,c in zip(chain,chain[1:],chain[2:])]
def hand(a):return dict(score=1.,keypoints=[dict(score=1.,landmarkId=i,position=dict(zip(['x','y','z'],map(float,p)))) for i,p in enumerate(a)])
def lengths(a):return np.stack([np.linalg.norm(a[...,b,:]-a[...,p,:],axis=-1) for p,b in EDGES],axis=-1)
def angles(a):
    vals=[]
    for p,q,r in TRIPLES:
        u=a[...,p,:]-a[...,q,:];v=a[...,r,:]-a[...,q,:];den=np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)
        vals.append(np.degrees(np.arccos(np.clip(np.sum(u*v,axis=-1)/np.maximum(den,1e-12),-1,1))))
    return np.stack(vals,axis=-1)
def predict(a,b,n,method):
    result=[]
    for i in range(n):
        t=(i+1)/(n+1)
        if method=='HOLD':v=a.copy()
        elif method=='HOLD_SHAPE':v=a-a[0]+((1-t)*a[0]+t*b[0])
        elif method=='LINEAR':v=(1-t)*a+t*b
        elif method=='SMOOTHSTEP':u=t*t*(3-2*t);v=(1-u)*a+u*b
        else:
            h=pipe.interpolate_hand_kinematically(hand(a),hand(b),t);v=np.array([[k['position'][axis] for axis in ['x','y','z']] for k in h['keypoints']])
        result.append(v)
    return np.array(result)
def prepare():
    assert not OUT.exists();p=trial.verify();db=trial.base.ROOT/'SLclean.sqlite3';clips=pipe.discover_clips(db);rng=random.Random(20261005);windows=[];skips=[]
    OUT.mkdir()
    protocol=dict(labels=p['labels'],seed=20261005,gap_frames=[3,6,12],selection='For every word/hand/gap length choose one random eligible contiguous raw CSV window with both endpoints and all hidden frames complete21 finite points, nonzero bones, increasing near-uniform timestamps. No score-based selection. Windows may overlap across lengths; cluster by word. Missing words/lengths retained as exclusions.',methods=METHODS,information='HOLD sees preceding endpoint only. Other methods see preceding/following endpoint; HOLD_SHAPE holds relative shape but linearly moves wrist. Predictors never receive hidden samples. All future-assisted methods offline.',reference='Raw SQLite CSV x/y/z before animation processing. Coordinates normalized as stored; z relative to wrist, not calibrated physical3D. No source images inspected; reference may contain detector noise. No claim ASL correctness.',metrics='Primary mean XY joint error / endpoint mean wrist-to-middle-MCP XY length; wrist-relative XY shape error; PCK@.1 same length; 3D coordinate bone relative absolute error vs withheld raw; angle MAE degrees. Equal window weights, separate lengths. No-fill has0% coverage and undefined errors, not zeros. Runtime excludes IO.',production=trial.base.production(),pipeline_sha=trial.base.sha(trial.base.ROOT/'export_pose_animator_sequence.py'),script_sha=trial.base.sha(Path(__file__)),db_stat=p['db_stat'])
    trial.write(OUT/'protocol.json',protocol)
    for label in p['labels']:
        path=pipe.resolve_selected_paths(clips,db,[label+'.csv'],'')[0];df=pipe.read_clip_csv(path)
        for side in ['left','right']:
            track={}
            for f,g in df[df.part==side+'_hand'].groupby('frame'):
                g=g.sort_values('landmark_id')
                if g.landmark_id.tolist()!=list(range(21)):continue
                a=g[['x','y','z']].to_numpy(float);ts=float(g.time_sec.iloc[0])
                if not np.isfinite(a).all() or not np.isfinite(ts) or np.min(lengths(a))<1e-8:continue
                track[int(f)]=(a,ts)
            for n in [3,6,12]:
                candidates=[]
                for start in sorted(track):
                    ids=list(range(start,start+n+2))
                    if not all(i in track for i in ids):continue
                    dt=np.diff([track[i][1] for i in ids])
                    if np.min(dt)<=0 or np.max(dt)/np.min(dt)>1.05:continue
                    a,b=track[start][0],track[start+n+1][0];scale=(np.linalg.norm(a[9,:2]-a[0,:2])+np.linalg.norm(b[9,:2]-b[0,:2]))/2
                    if scale>1e-6:candidates.append(start)
                if not candidates:skips.append(dict(label=label,side=side,gap=n,reason='no eligible complete raw window'));continue
                start=rng.choice(candidates);a=track[start][0];b=track[start+n+1][0]
                windows.append(dict(label=label,side=side,gap=n,start=start,timestamps=[track[i][1] for i in range(start,start+n+2)],left=a.tolist(),right=b.tolist(),hidden=[track[i][0].tolist() for i in range(start+1,start+n+1)]))
        print('Selected',label,flush=True)
    trial.write(OUT/'windows.json',windows);trial.write(OUT/'selection.json',dict(windows=len(windows),skipped=skips,windows_sha=trial.base.sha(OUT/'windows.json')));trial.verify()
def run():
    assert not (OUT/'results.json').exists();trial.verify();p=trial.read(OUT/'protocol.json');assert p['script_sha']==trial.base.sha(Path(__file__))
    assert trial.base.sha(OUT/'windows.json')==trial.read(OUT/'selection.json')['windows_sha'];rows=[]
    # Unit checks: endpoint input arrays unchanged; equal endpoints reproduce shape.
    windows=trial.read(OUT/'windows.json');assert windows
    a=np.array(windows[0]['left'])
    for m in METHODS:assert np.allclose(predict(a,a,3,m),np.repeat(a[None],3,axis=0),atol=1e-8)
    for w in windows:
        a=np.array(w['left']);b=np.array(w['right']);truth=np.array(w['hidden']);n=w['gap'];scale=(np.linalg.norm(a[9,:2]-a[0,:2])+np.linalg.norm(b[9,:2]-b[0,:2]))/2
        for m in METHODS:
            ac=a.copy();bc=b.copy();start=time.perf_counter();pred=predict(ac,bc,n,m);elapsed=(time.perf_counter()-start)*1000
            assert np.array_equal(ac,a) and np.array_equal(bc,b) and np.isfinite(pred).all()
            errors=np.linalg.norm(pred[:,:,:2]-truth[:,:,:2],axis=-1)/scale
            shape=np.linalg.norm((pred[:,:,:2]-pred[:,0:1,:2])-(truth[:,:,:2]-truth[:,0:1,:2]),axis=-1)/scale
            bone=np.abs(lengths(pred)-lengths(truth))/np.maximum(lengths(truth),1e-8)
            rows.append(dict(label=w['label'],side=w['side'],gap=n,start=w['start'],method=m,hidden_duration_s=float(np.median(np.diff(w['timestamps']))*n),xy_error=float(errors.mean()),shape_error=float(shape[:,1:].mean()),pck_01=float((errors<=.1).mean()),bone_error_pct=float(bone.mean()*100),angle_mae_deg=float(np.abs(angles(pred)-angles(truth)).mean()),runtime_ms=elapsed))
    summary={}
    for n in [3,6,12]:
        summary[str(n)]={}
        for m in METHODS:
            rr=[r for r in rows if r['gap']==n and r['method']==m]
            summary[str(n)][m]=dict(n=len(rr),words=len({r['label'] for r in rr}),**{k:float(np.mean([r[k] for r in rr])) for k in ['hidden_duration_s','xy_error','shape_error','pck_01','bone_error_pct','angle_mae_deg','runtime_ms']}) if rr else dict(n=0)
    trial.verify();assert trial.base.production()==p['production']
    trial.write(OUT/'results.json',dict(summary=summary,rows=rows,production_unchanged=True,no_fill=dict(coverage=0,errors=None),unit_checks_passed=True))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
