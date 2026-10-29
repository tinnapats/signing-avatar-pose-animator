"""Read-only200 raw audit + exact stage tracing for fixed diagnostic20."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import repair_parts200 as parts
base=parts.base;pipe=base.audit.pipeline;HERE=base.HERE;OUT=HERE/'wrist_coordinate_trace_20261003';SRC=HERE/'slgcn200_20261002'

def snapshot(frames):
    result=[]
    for f in frames:
        pose={p['part']:p for p in f['pose']['keypoints']}
        xy=lambda k:np.array([k['position']['x'],k['position']['y']],dtype=float)
        width=float(np.linalg.norm(xy(pose['leftShoulder'])-xy(pose['rightShoulder'])))
        item=dict(width=width,sides={})
        for side in ['left','right']:
            h=f['hands'][side];w=h['keypoints'][0];p=pose[side+'Wrist']
            ok=h.get('observed',False) and h['score']>.001 and w['score']>0 and p['score']>=.2 and width>1e-6
            item['sides'][side]=dict(valid=bool(ok),hand=xy(w).tolist(),body=xy(p).tolist(),nose=xy(pose['nose']).tolist(),gap=((xy(w)-xy(p))/max(width,1e-6)).tolist(),pose_score=p['score'])
        result.append(item)
    return result

def main():
    assert not OUT.exists();OUT.mkdir();before=base.production()
    protocol0=_local_json_loads((HERE/'scale_anchor200_20261003/protocol.json').read_text());labels=protocol0['labels'];review=protocol0['review']
    p=dict(labels=labels,review=review,production=before,scope='Diagnostic only. Raw wrist0 vs pose15/16 in aspect-corrected coordinates; full200. Stage trace fixed changed20. No confidence gating fixes, no landmark swap, no model inference, no production edits.',
        stage_order=['build','repair','taper','interpolate','body_gaussian','hand_gaussian','bones','articulation','fit'])
    (OUT/'protocol.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
    clips=pipe.discover_clips(base.ROOT/'SLclean.sqlite3');rawrows=[];traces=[]
    for i,label in enumerate(labels):
        path=pipe.resolve_selected_paths(clips,base.ROOT/'SLclean.sqlite3',[label+'.csv'],'')[0]
        df=pipe.read_clip_csv(path);aspect=pipe.estimate_source_aspect(df);raw={};low=total=missingvis=0
        for side,wid in [('left',15),('right',16)]:
            hand=df[(df.part==side+'_hand')&(df.landmark_id==0)].drop_duplicates('frame').set_index('frame')
            pose=df[(df.part=='pose')&(df.landmark_id==wid)].drop_duplicates('frame').set_index('frame')
            left=df[(df.part=='pose')&(df.landmark_id==11)].drop_duplicates('frame').set_index('frame')
            right=df[(df.part=='pose')&(df.landmark_id==12)].drop_duplicates('frame').set_index('frame')
            ids=hand.index.intersection(pose.index).intersection(left.index).intersection(right.index)
            if not len(ids):raw[side]=dict(n=0);continue
            xy=lambda t:np.asarray(t.loc[ids,['x','y']],dtype=float)*[aspect,1]
            width=np.linalg.norm(xy(left)-xy(right),axis=1);valid=np.isfinite(width)&(width>1e-6)
            gap=np.linalg.norm(xy(hand)-xy(pose),axis=1)/np.maximum(width,1e-6)
            visibility=np.asarray(pose.loc[ids,'visibility'],dtype=float);finite=np.isfinite(visibility)&valid
            total+=int(valid.sum());low+=int(np.sum(finite&(visibility<.2)));missingvis+=int(np.sum(valid&~finite))
            raw[side]=dict(n=int(valid.sum()),mean_gap_shoulders=float(np.mean(gap[valid])),median_gap_shoulders=float(np.median(gap[valid])),p95_gap_shoulders=float(np.percentile(gap[valid],95)),
                visibility_below_point2=int(np.sum(finite&(visibility<.2))),visibility_missing=int(np.sum(valid&~finite)),
                gap_when_visibility_ge_point5=float(np.mean(gap[valid&(visibility>=.5)])) if np.any(valid&(visibility>=.5)) else None)
        rawrows.append(dict(label=label,aspect=aspect,sides=raw,total_pairs=total,low_visibility_pairs=low,missing_visibility_pairs=missingvis))
        if label in review:
            shots={}
            def capture(name,fn,mutates=False):
                def wrapped(*args,**kwargs):
                    value=fn(*args,**kwargs);shots[name]=snapshot(args[0] if mutates else value);return value
                return wrapped
            mapping={'build':'build_frames','taper':'taper_hand_detection_edges','interpolate':'interpolate_frames','body_gaussian':'apply_gaussian_smoothing','hand_gaussian':'apply_hand_gaussian_smoothing','bones':'stabilize_hand_bone_lengths','articulation':'enforce_hand_articulation','fit':'fit_frames_to_canvas'}
            with ExitStack() as stack:
                stack.enter_context(patch.object(pipe,'repair_hand_tracks',capture('repair',parts.make_repair(True,False))))
                for name,fn in mapping.items():stack.enter_context(patch.object(pipe,fn,capture(name,getattr(pipe,fn),name in ['bones','articulation'])))
                payload=pipe.build_payload(base.ROOT/'SLclean.sqlite3',[label+'.csv'],'',30,513,513,0,0,2,1.2,2,hand_gaussian_sigma=0,hand_gaussian_radius=0)
            old=_local_json_loads((SRC/'cases'/f'{i:03d}.json').read_text());assert payload['frames']==old['payload']['frames'],label
            n=len(shots['build']);aligned={name:values if name in ['build','repair','taper'] else values[::2] for name,values in shots.items()};assert all(len(v)==n for v in aligned.values())
            common=[(j,side) for j in range(n) for side in ['left','right'] if all(aligned[s][j]['sides'][side]['valid'] for s in p['stage_order'])]
            metrics={}
            for name in p['stage_order']:
                vals=[np.linalg.norm(aligned[name][j]['sides'][side]['gap']) for j,side in common]
                metrics[name]=dict(common_pairs=len(vals),mean_gap_shoulders=float(np.mean(vals)) if vals else None,max_gap_shoulders=float(max(vals)) if vals else None)
            changes={}
            for a,b in zip(p['stage_order'],p['stage_order'][1:]):
                vals=[np.linalg.norm(np.array(aligned[a][j]['sides'][side]['gap'])-aligned[b][j]['sides'][side]['gap']) for j,side in common]
                changes[a+'->'+b]=dict(mean_gap_vector_change=float(np.mean(vals)) if vals else None,max_gap_vector_change=float(max(vals)) if vals else None)
            # Preserve per-frame evidence for face/two-hand relationships, not ASL truth.
            trace=dict(label=label,metrics=metrics,changes=changes,aligned=aligned)
            (OUT/(label.replace(' ','_')+'_trace.json')).write_text(json.dumps(trace),encoding='utf-8');traces.append(dict(label=label,metrics=metrics,changes=changes))
        if (i+1)%20==0:print('audited',i+1,'/200',flush=True)
    assert base.production()==before
    summary={s:float(np.mean([r['metrics'][s]['mean_gap_shoulders'] for r in traces if r['metrics'][s]['mean_gap_shoulders'] is not None])) for s in p['stage_order']}
    allmeans=[np.mean([v['mean_gap_shoulders'] for v in r['sides'].values() if v['n']]) for r in rawrows if any(v['n'] for v in r['sides'].values())]
    totals={k:sum(r[k] for r in rawrows) for k in ['total_pairs','low_visibility_pairs','missing_visibility_pairs']}
    result=dict(raw200=rawrows,trace20=traces,trace20_word_means=summary,raw200_word_mean_gap=float(np.mean(allmeans)),raw_totals=totals,production_unchanged=True,trace_payload_exact=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['raw200','trace20']},indent=2))

if __name__=='__main__':main()
