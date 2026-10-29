"""Explain scale regressions without tuning or changing renderer inputs."""
import json,sys
from pathlib import Path
import numpy as np
import fallback_scale_new30 as trial
base=trial.base;HERE=trial.HERE;SRC=trial.OUT;OUT=HERE/'scale_failure_diagnosis_20261004';read=trial.read;write=trial.write
def visible(h):return bool(h and h.get('score',0)>=.001 and sum(bool(k and k.get('position') and k.get('score',0)>0) for k in h.get('keypoints',[]))>=6)
def pos(k):return np.array([k['position']['x'],k['position']['y']],float)
def diag(bounds):return float(np.linalg.norm(np.array(bounds[2:])-bounds[:2]))
def main():
    assert not OUT.exists();OUT.mkdir();before=base.production()
    write(OUT/'protocol.json',dict(production=before,scope='Explain all75 switches from fixed new30; compare18 worse to57 better. No formula changes, no rerender, no recognition. Replay exact reference and scale with assertions. Direction classification uses actual adjacent detailed bounds only as diagnostic target, not ground truth or production rule.',hashes={n:base.sha(SRC/n) for n in ['results.json','renderer/rendered.json','case_index.json']},source_formula_sha=base.sha(HERE/'fallback_boundary_adjust.js')))
    index={c['phrase']:c for c in read(SRC/'case_index.json')};caps={(c['phrase'],c['variant']):c for c in read(SRC/'renderer/rendered.json')};events=[r for r in read(SRC/'results.json')['rows'] if r['kind']=='switch'];rows=[];maxerr=0
    for phrase,d in index.items():
        c=read(Path(d['file']));frames=c['payload']['frames'];off={s['index']:s for s in caps[phrase,'OFF']['samples']};on={s['index']:s for s in caps[phrase,'SCALE']['samples']};scales={}
        for side in ['left','right']:
            found=[]
            for i,s in off.items():
                pts=s['hands'][side]
                if not pts:continue
                raw=np.array([pos(k) for k in frames[i]['hands'][side]['keypoints']]);ids=sorted(map(int,pts));p={k['part']:k for k in frames[i]['pose']['keypoints']};anchor=pos(p[side+'Wrist']) if p[side+'Wrist']['score']>=.2 else raw[0];x=raw[ids]-raw[0];y=np.array([pts[str(k)] for k in ids])-anchor
                if (x*x).sum()>1e-8:found.append(float((x*y).sum()/(x*x).sum()))
            scales[side]=float(np.median(found)) if found else None
            assert not found or max(abs(s-scales[side]) for s in found)<1e-6
        for e in [r for r in events if r['phrase']==phrase]:
            side=e['side'];a,b=e['before'],e['after'];native=a if off[a]['native'][side] else b;detail=b if native==a else a
            op=on[native]['ops'].get(side);assert op is not None
            ref=None
            for distance in range(1,7):
                for j in [native-distance,native+distance]:
                    if 0<=j<len(frames):
                        h=frames[j]['hands'][side]
                        if h.get('observed') and visible(h) and len(h['keypoints'])==21:ref=j;break
                if ref is not None:break
            assert ref is not None and abs(ref-native)==op['distance']
            n=diag(off[native]['geometry'][side]['bounds']);dn=diag(off[detail]['geometry'][side]['bounds']);after=diag(on[native]['geometry'][side]['bounds'])
            def target(i):
                points=np.array([pos(k) for k in frames[i]['hands'][side]['keypoints']]);return float(np.linalg.norm(points.max(0)-points.min(0))*scales[side])
            tr=target(ref);td=target(detail);t=1-op['distance']/7;weight=t*t*(3-2*t);pred=1+(np.clip(tr/n,.85,1.15)-1)*weight;err=abs(pred-op['scale']);maxerr=max(maxerr,err);assert err<1e-10
            start=end=native
            while start>0 and not visible(frames[start-1]['hands'][side]):start-=1
            while end+1<len(frames) and not visible(frames[end+1]['hands'][side]):end+=1
            bad=e['SCALE']['size']>e['OFF']['size']+1e-9;wrong=(after-n)*(dn-n)<0
            rows.append(dict(**e,worse=bad,native_frame=native,detailed_frame=detail,reference_frame=ref,reference_distance=op['distance'],reference_equals_adjacent=(ref==detail),reference_observed=True,adjacent_observed=bool(frames[detail]['hands'][side].get('observed')),scale=op['scale'],native_diagonal=n,detailed_diagonal=dn,adjusted_native_diagonal=after,reference_landmark_target=tr,adjacent_landmark_target=td,reference_target_to_adjacent_render=tr/dn,adjacent_landmark_to_render=td/dn,wrong_direction=bool(wrong),needed_scale=dn/n,overshot=bool((n-dn)*(after-dn)<0),reference_changes_target_direction=bool((tr-n)*(td-n)<0),gap_frames=end-start+1,gap_kind='whole' if start==0 and end==len(frames)-1 else 'leading' if start==0 else 'trailing' if end==len(frames)-1 else 'internal'))
    assert len(rows)==75 and sum(r['worse'] for r in rows)==18
    summary={}
    for bad in [False,True]:
        rr=[r for r in rows if r['worse']==bad];summary['worse' if bad else 'better']=dict(n=len(rr),words=sorted({r['label'] for r in rr}),wrong_direction=sum(r['wrong_direction'] for r in rr),overshot=sum(r['overshot'] for r in rr),reference_same=sum(r['reference_equals_adjacent'] for r in rr),reference_changes_target_direction=sum(r['reference_changes_target_direction'] for r in rr),shrink=sum(r['scale']<1 for r in rr),grow=sum(r['scale']>1 for r in rr),gap_kinds={k:sum(r['gap_kind']==k for r in rr) for k in ['leading','trailing','internal','whole']},median={k:float(np.median([r[k] for r in rr])) for k in ['reference_distance','gap_frames','reference_target_to_adjacent_render','adjacent_landmark_to_render']},original_size_median=float(np.median([r['OFF']['size'] for r in rr])))
    assert before==base.production();result=dict(summary=summary,max_scale_replay_error=maxerr,rows=rows,production_unchanged=True);write(OUT/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
