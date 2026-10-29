"""Diagnostic decomposition on known failures, not a new validation set."""
import json
from pathlib import Path
import numpy as np
from run_audit import pipeline,frame_array
from step1_evaluate import hashes

HERE=Path(__file__).resolve().parent
DATA=HERE/'heldout_scale_trial'
OUT=HERE/'method_review'
WORDS=['president','toothbrush','crazy','cracker','politics']

def main():
    OUT.mkdir(exist_ok=True);before=hashes()
    cases=json.loads((DATA/'inputs.json').read_text());cases=[c for c in cases if c['phrase'] in WORDS]
    assert len(cases)==5
    captures={(r['variant'],r['phrase']):r['samples'] for r in json.loads((DATA/'comparison/rendered.json').read_text())}
    fits={r['word']:r['global_fit'] for r in json.loads((DATA/'baseline_geometry/results.json').read_text())['rows']}
    manifest={r['word']:r for r in json.loads((DATA/'manifest.json').read_text())}
    allrows=[]
    for c in cases:
        word=c['phrase'];raw=pipeline.build_frames(pipeline.concat_clips([DATA/'csv'/(word+'.csv')],513,513,0,target_fps=30),513,513,0)
        fit=fits[word];ref=frame_array(raw)*fit['scale']+fit['offset']; processed=frame_array([c['payload']['frames'][i] for i in c['frameIndices']])
        variants={k:[] for k in ['S1A1','S50A1']};frame_rows=[]
        tracks={k:dict(left=[],right=[]) for k in variants}
        for t,f in enumerate(raw):
            shoulder=np.linalg.norm(ref[t,11]-ref[t,12]);pose={p['part']:p for p in f['pose']['keypoints']}
            frame_metrics={k:[] for k in variants}
            for side,offset in [('left',33),('right',54)]:
                h=f.get('hands',{}).get(side,{})
                valid=shoulder>1e-6 and min(pose[s]['score'] for s in ['leftShoulder','rightShoulder'])>0 and h.get('observed') and len(h.get('keypoints',[]))==21 and all(k.get('score',0)>0 for k in h['keypoints'])
                for v in variants:
                    out=captures[v,word][t]['hands'][side]
                    if not valid or len(out)!=21:
                        tracks[v][side].append(None);continue
                    a=ref[t,offset:offset+21];b=np.array([out[str(i)] for i in range(21)])
                    translation=(b[0]-a[0])/shoulder
                    shape=((b-b[0])-(a-a[0]))/shoulder
                    error=(b-a)/shoulder
                    mse=float(np.mean(np.sum(error**2,axis=1)))
                    trans2=float(np.dot(translation,translation));shape2=float(np.mean(np.sum(shape**2,axis=1)));cross=float(np.mean(2*shape@translation))
                    assert abs(mse-trans2-shape2-cross)<1e-10
                    q=processed[t,offset:offset+21];denom=np.sum((q-q[0])**2)
                    scale=float(np.sum((b-b[0])*(q-q[0]))/denom) if denom>0 else None
                    row=dict(frame=t,time_sec=t/30,side=side,position_error=float(np.linalg.norm(error,axis=1).mean()),
                        wrist_error=float(np.linalg.norm(translation)),wrist_relative_error=float(np.linalg.norm(shape[1:],axis=1).mean()),
                        mse=mse,translation2=trans2,shape2=shape2,cross=cross,scale=scale)
                    variants[v].append(row);frame_metrics[v].append(row['position_error'])
                    tracks[v][side].append(dict(frame=t,points=(b/shoulder).tolist(),relative=((b-b[0])/shoulder).tolist()))
            if all(frame_metrics.values()):frame_rows.append(dict(frame=t,time_sec=t/30,delta=float(np.mean(frame_metrics['S50A1'])-np.mean(frame_metrics['S1A1']))))
        temporal={}
        for v,sides in tracks.items():
            absolute=[];relative=[];gaps=0
            for track in sides.values():
                for i in range(1,len(track)):
                    if track[i] is None or track[i-1] is None:continue
                    # Normalize both displacement samples by the current frame shoulder, no resampling.
                    s0=np.linalg.norm(ref[i-1,11]-ref[i-1,12]);s1=np.linalg.norm(ref[i,11]-ref[i,12])
                    b=np.array(track[i]['points']);a=np.array(track[i-1]['points'])*s0/s1
                    rb=np.array(track[i]['relative']);ra=np.array(track[i-1]['relative'])*s0/s1
                    absolute.extend(np.linalg.norm(b-a,axis=1).tolist());relative.extend(np.linalg.norm(rb-ra,axis=1).tolist())
                gaps+=sum((track[i] is None)!=(track[i-1] is None) for i in range(1,len(track)))
            temporal[v]=dict(consecutive_point_pairs=len(absolute),position_step_p95=float(np.quantile(absolute,.95)) if absolute else None,
                position_step_max=float(max(absolute)) if absolute else None,relative_step_p95=float(np.quantile(relative,.95)) if relative else None,observed_boundary_count=gaps)
        summaries={v:{k:float(np.mean([r[k] for r in values])) for k in ['position_error','wrist_error','wrist_relative_error','mse','translation2','shape2','cross']} for v,values in variants.items()}
        sides={v:{side:dict(scale=float(np.median([r['scale'] for r in values if r['side']==side])),observed_frames=sum(r['side']==side for r in values)) for side in ['left','right'] if any(r['side']==side for r in values)} for v,values in variants.items()}
        worst=max(frame_rows,key=lambda r:r['delta']); chosen=sorted(set([round((len(raw)-1)*f) for f in [.1,.3,.5,.7,.9]]+[worst['frame']]))
        allrows.append(dict(word=word,frames=len(raw),extraction=manifest[word]['extraction'],source_video=manifest[word]['local_video'],
            summary=summaries,side_stats=sides,temporal=temporal,worst_frame=worst,review_frames=chosen,per_frame=variants))
        print(word,json.dumps(dict(summary=summaries,temporal=temporal,worst=worst)),flush=True)
    assert hashes()==before
    (OUT/'diagnostics.json').write_text(json.dumps(dict(rows=allrows,production_hashes=before,scope='Five known regressions, diagnostic only; all temporal metrics 30Hz, observed consecutive frames only'),indent=2),encoding='utf-8')
    (OUT/'inputs.json').write_text(json.dumps(cases),encoding='utf-8')

if __name__=='__main__':main()
