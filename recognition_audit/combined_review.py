"""Compare combinations and record native fallback without inventing landmarks."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json
import math
import os
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import combine_winners200 as study
base=study.base;OUT=Path(os.environ.get('STUDY_OUTPUT',str(study.OUT)))

def main():
    result=_local_json_loads((OUT/'results.json').read_text())
    cases={r['phrase']:r for r in _local_json_loads((OUT/'review_inputs.json').read_text())}
    captured=_local_json_loads((OUT/'renderer/rendered.json').read_text());assert len(captured)==len(cases)*int(os.environ.get('STUDY_SCALE_COUNT','2'))
    originals={c['phrase']:c for c in _local_json_loads((base.SOURCE/'inputs.json').read_text()) if c['phrase'] in study.parts.REVIEW}
    archived={c['phrase']:c for c in _local_json_loads((base.SOURCE/'rendered.json').read_text())}
    model,norm,vocab=base.audit.load_model();rows=[]
    for cap in captured:
        c=cases[cap['phrase']];label=c['label'];variant=c['variant']+'__'+cap['variant'];samples=cap['samples'];frames=c['payload']['frames'][::2]
        assert len(samples)==len(frames)
        if c['variant']=='baseline' and cap['variant']=='S1A1':assert [s['hands'] for s in samples]==[s['hands'] for s in archived[label]['samples']]
        a=base.audit.frame_array(frames);source=np.asarray(originals[label]['source30']);states=Counter();transitions=Counter();native_diags=[];change_steps=[];errors=[];previous={}
        for i,sample in enumerate(samples):
            pose={k['part']:k for k in frames[i]['pose']['keypoints']}
            shoulder=float(np.linalg.norm(np.array([pose['leftShoulder']['position'][k] for k in ['x','y']])-np.array([pose['rightShoulder']['position'][k] for k in ['x','y']])))
            source_width=float(np.linalg.norm(source[i,11]-source[i,12]))
            source_center=(source[i,11]+source[i,12])/2
            pose_center=np.mean([[pose[n]['position'][k] for k in ['x','y']] for n in ['leftShoulder','rightShoulder']],axis=0)
            for side,offset in [('left',33),('right',54)]:
                hand=sample['hands'][side];native=sample['native'][side];d=bool(hand);n=bool(native)
                state='both' if d and n else 'detailed' if d else 'native' if n else 'missing';states[state]+=1
                a[i,offset:offset+21]=0
                for k,p in hand.items():
                    a[i,offset+int(k)]=p
                    ref=source[i,offset+int(k)]
                    if originals[label]['payload']['frames'][2*i]['hands'][side].get('observed') and np.any(ref!=0) and source_width>1e-8 and shoulder>1e-8:
                        errors.append(float(np.linalg.norm((np.array(p)-pose_center)/shoulder-(ref-source_center)/source_width)))
                boxes=[p['bounds'] for p in native]
                if d:
                    xy=np.array(list(hand.values()));boxes.append([*xy.min(axis=0),*xy.max(axis=0)])
                bounds=None
                if boxes:
                    bb=np.array(boxes);bounds=np.array([min(bb[:,0]),min(bb[:,1]),max(bb[:,2]),max(bb[:,3])])
                    if n and shoulder>1e-8:native_diags.append(float(np.linalg.norm(bounds[2:]-bounds[:2])/shoulder))
                if side in previous:
                    last,oldbounds=previous[side]
                    if last!=state:
                        transitions[last+'->'+state]+=1
                        if bounds is not None and oldbounds is not None and shoulder>1e-8:
                            change_steps.append(float(np.linalg.norm((bounds[:2]+bounds[2:]-oldbounds[:2]-oldbounds[2:])/2)/shoulder))
                previous[side]=(state,bounds)
        t=torch.from_numpy(a[:,base.audit.PICK,:].transpose(2,0,1).copy());t=norm({'frames':t})['frames'].unsqueeze(0);assert torch.isfinite(t).all()
        with torch.inference_mode():p=model(t)[0].softmax(0)
        rows.append(dict(label=label,variant=variant,prediction=vocab[int(p.argmax())],states=dict(states),transitions=dict(transitions),
            max_native_diagonal_shoulder=max(native_diags) if native_diags else None,
            max_type_change_bbox_center_step=max(change_steps) if change_steps else None,
            mean_shoulder_aligned_point_distance=float(np.mean(errors)) if errors else None,point_samples=len(errors)))
    variants=list(dict.fromkeys(r['variant'] for r in rows));summaries={}
    # Common observed three-frame windows across all ten rendered conditions.
    # Wrist-relative second differences, not jerk and not semantic accuracy.
    capture_map={(cases[c['phrase']]['label'],cases[c['phrase']]['variant']+'__'+c['variant']):c['samples'] for c in captured}
    roughness=[]
    for label in study.parts.REVIEW:
        n=len(capture_map[label,variants[0]]);acc={v:[] for v in variants}
        raw=np.asarray(originals[label]['source30']);width=np.linalg.norm(raw[:,11]-raw[:,12],axis=1)
        for side in ['left','right']:
            for joint in range(1,21):
                tracks={v:np.full((n,2),np.nan) for v in variants}
                for t in range(n):
                    if not originals[label]['payload']['frames'][2*t]['hands'][side].get('observed') or width[t]<=1e-8:continue
                    for v in variants:
                        h=capture_map[label,v][t]['hands'][side]
                        if str(joint) in h and '0' in h:tracks[v][t]=(np.array(h[str(joint)])-h['0'])/width[t]
                valid=np.logical_and.reduce([np.isfinite(tracks[v]).all(axis=1) for v in variants])
                windows=valid[:-2]&valid[1:-1]&valid[2:]
                for v in variants:
                    a=tracks[v];acc[v].extend(np.linalg.norm((a[2:]-2*a[1:-1]+a[:-2])[windows],axis=1).tolist())
        for v in variants:roughness.append(dict(label=label,variant=v,common_windows=len(acc[v]),mean_second_difference=float(np.mean(acc[v])) if acc[v] else None))
    for v in variants:
        rr=[r for r in rows if r['variant']==v];m=base.metrics(rr,study.parts.REVIEW);states=Counter();trans=Counter()
        for r in rr:states.update(r['states']);trans.update(r['transitions'])
        m.update(states=dict(states),transitions=dict(trans),mean_word_shoulder_aligned_point_distance=float(np.mean([r['mean_shoulder_aligned_point_distance'] for r in rr if r['mean_shoulder_aligned_point_distance'] is not None])))
        vals=[r['mean_second_difference'] for r in roughness if r['variant']==v and r['mean_second_difference'] is not None]
        m['mean_word_common_second_difference']=float(np.mean(vals)) if vals else None
        summaries[v]=m
    # Four predeclared incremental comparisons, not all-pairs searching.
    pairs={};lookup={(r['label'],r['variant']):r for r in result['predictions']};labels=_local_json_loads((OUT/'protocol.json').read_text())['labels']
    comparisons=_local_json_loads(os.environ['STUDY_COMPARISONS']) if 'STUDY_COMPARISONS' in os.environ else [('no_repair','no_repair_no_smoothing'),('no_hold','no_hold_no_smoothing'),('no_hand_smoothing','no_repair_no_smoothing'),('no_hand_smoothing','no_hold_no_smoothing')]
    for a,b in comparisons:
        gained=[l for l in labels if lookup[l,a]['prediction']!=l and lookup[l,b]['prediction']==l]
        lost=[l for l in labels if lookup[l,a]['prediction']==l and lookup[l,b]['prediction']!=l]
        n=len(gained)+len(lost);p=min(1,2*sum(math.comb(n,k) for k in range(min(len(gained),len(lost))+1))/2**n) if n else 1
        pairs[a+' -> '+b]=dict(gained=gained,lost=lost,p=p,bonferroni_adjusted=min(1,p*len(comparisons)))
    geom={}
    for v in result['summaries']:
        rr=[r for r in result['predictions'] if r['variant']==v]
        geom[v]=dict(bone_span=float(np.mean([r['geometry']['mean_bone_span'] for r in rr if r['geometry']['mean_bone_span'] is not None])),bend_over115=float(np.mean([r['geometry']['bend_over115_fraction'] for r in rr if r['geometry']['bend_over115_fraction'] is not None])))
    assert base.production()==_local_json_loads((OUT/'protocol.json').read_text())['production']
    (OUT/'renderer_review.json').write_text(json.dumps(dict(summaries=summaries,predictions=rows,common_roughness=roughness,baseline_capture_exact=True,production_unchanged=True,
        caveat='Diagnostic12, actual detailed endpoints plus payload body; native paths measured as bounds/state, NOT assigned fake 21 points or recognized.'),indent=2),encoding='utf-8')
    (OUT/'analysis.json').write_text(json.dumps(dict(pairwise=pairs,geometry=geom),indent=2),encoding='utf-8')
    print(json.dumps(dict(renderer=summaries,pairwise=pairs,geometry=geom),indent=2),flush=True)

if __name__=='__main__':main()
