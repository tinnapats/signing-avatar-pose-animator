"""Pixel-only hand identity experiment. References accessed only after decisions are saved."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
O=ROOT/'tracking3'
SETTINGS={'confidence':.30,'swap_margin_shoulder_units':.15,
          'temporal_weight':.35,'history_max_age_frames':3,
          'forearm_to_upperarm_range':[.2,5.0],
          'variants':['raw','body_anchor','body_anchor_temporal'],
          'body_source':'SDPose unchanged; anatomical left 5-7-9, right 6-8-10',
          'hand_source':'SDPose COCO WholeBody left 91:112 right 112:133',
          'references_used_in_assignment':False}

def assign(k,s,previous=None,temporal=False):
    """Only reorder the two complete hands; never synthesize coordinates or change body."""
    k=np.asarray(k); s=np.asarray(s)
    hands=k[91:133].reshape(2,21,2)
    hs=s[91:133].reshape(2,21)
    conf=SETTINGS['confidence']
    scale=max(float(np.linalg.norm(k[5]-k[6])),1.)
    valid_hand=(hs[:,0]>=conf)&(hs.mean(axis=1)>=conf)
    valid_body=all(s[j]>=conf for j in (5,6,7,8,9,10))
    ratios=[]
    for a,b,c in ((5,7,9),(6,8,10)):
        ratio=float(np.linalg.norm(k[b]-k[c])/max(np.linalg.norm(k[a]-k[b]),1.))
        ratios.append(ratio)
        valid_body &= .2<=ratio<=5.
    meta={'swapped':False,'body_valid':bool(valid_body),'hand_valid':valid_hand.tolist(),'arm_ratios':ratios}
    if not valid_hand.all() or not valid_body:
        meta['reason']='insufficient_current_evidence'
        return k.copy(),s.copy(),meta
    costs=[]
    for perm in ((0,1),(1,0)):
        wrists=hands[list(perm),0]
        cost=float(np.linalg.norm(wrists-k[[9,10]],axis=1).sum()/scale)
        if temporal and previous is not None:
            cost+=SETTINGS['temporal_weight']*float(np.linalg.norm(wrists-previous,axis=1).sum()/scale)
        costs.append(cost)
    swap=costs[0]-costs[1]>SETTINGS['swap_margin_shoulder_units']
    out=k.copy(); scores=s.copy()
    if swap:
        out[91:112],out[112:133]=k[112:133].copy(),k[91:112].copy()
        scores[91:112],scores[112:133]=s[112:133].copy(),s[91:112].copy()
    meta.update(swapped=bool(swap),cost_identity=costs[0],cost_swap=costs[1],reason='cost_comparison')
    return out,scores,meta

def tests():
    k=np.zeros((133,2)); s=np.ones(133)
    k[[5,7,9]]=[[0,0],[0,10],[0,20]]
    k[[6,8,10]]=[[20,0],[20,10],[20,20]]
    k[91:112]=[20,20]; k[112:133]=[0,20]
    out,sc,m=assign(k,s); assert m['swapped']; assert np.array_equal(out[:91],k[:91])
    assert np.array_equal(out[91:112],k[112:133]); assert np.array_equal(sc,s)
    out2,_,m2=assign(out,s); assert not m2['swapped']; assert np.array_equal(out,out2)
    shifted,_,ms=assign(k+37,s); assert ms['swapped']; assert np.allclose(shifted,out+37)
    ss=s.copy(); ss[91:133]=.1
    unchanged,_,m3=assign(k,ss); assert not m3['swapped']; assert np.array_equal(unchanged,k)
    _,_,m4=assign(k,s,previous=out[[91,112]],temporal=True); assert m4['swapped']
    ss=s.copy(); ss[91:112]=.7; ss[112:133]=.9
    _,swapped_scores,_=assign(k,ss)
    assert np.allclose(swapped_scores[91:112],.9) and np.allclose(swapped_scores[112:133],.7)
    ss=s.copy(); ss[9]=.1
    unchanged,_,m5=assign(k,ss); assert not m5['swapped']; assert np.array_equal(unchanged,k)
    print('Synthetic identity tests passed',flush=True)

def main():
    tests()
    assert json.loads((O/'protocol.json').read_text())['status']=='three_clip_inference_completed'
    rows=json.loads((O/'results.json').read_text())
    outdir=O/'identity_trial'; outdir.mkdir(exist_ok=True)
    if (outdir/'decisions.json').exists(): raise RuntimeError('Preserve existing decisions')
    state={}; decisions=[]; arrays={v:[] for v in SETTINGS['variants']}; scores={v:[] for v in SETTINGS['variants']}
    for r in rows:
        k=np.array(r['keypoints']); s=np.array(r['scores'])
        decision={'id':r['id'],'frame':r['frame'],'sourceIndex':r['sourceIndex'],'tag':r['tag']}
        for variant in SETTINGS['variants']:
            key=(variant,r['id'])
            previous=state.get(key)
            prev=previous[1] if previous and r['frame']-previous[0]<=3 else None
            if variant=='raw': out,ss,meta=k.copy(),s.copy(),{'swapped':False}
            else: out,ss,meta=assign(k,s,prev,variant=='body_anchor_temporal')
            if all(ss[b]>=.3 and ss[b:b+21].mean()>=.3 for b in (91,112)):
                state[key]=(r['frame'],out[[91,112]].copy())
            assert np.array_equal(out[:91],k[:91])
            arrays[variant].append(out); scores[variant].append(ss); decision[variant]=meta
        decisions.append(decision)
    (outdir/'settings.json').write_text(json.dumps(SETTINGS,indent=2),encoding='utf-8')
    (outdir/'decisions.json').write_text(json.dumps(decisions,indent=2),encoding='utf-8')
    np.savez_compressed(outdir/'predictions.npz',**{v:np.array(a) for v,a in arrays.items()},
                        **{v+'_scores':np.array(a) for v,a in scores.items()})
    # Evaluation is separate and occurs only after all reference-free decisions are persisted.
    B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007')
    refs={c['id']:{f['index']:f for f in c['samples']} for c in json.loads((B/'coordinate_reference.json').read_text())}
    report={'frames':len(rows),'recognition_run':False,'metrics':{},'per_clip':{},'changes':[]}
    for variant in SETTINGS['variants']:
        err=[]; byclip={}; wrist=[]; nhand=0
        for i,r in enumerate(rows):
            ref=refs[r['id']][r['sourceIndex']]
            assert ref['pngMatches'] and ref['pixelRatio']==1
            for side,base in (('left',91),('right',112)):
                h=ref['hands'][side]
                if not all(str(j) in h for j in range(21)): continue
                gt=np.array([h[str(j)] for j in range(21)])
                e=np.linalg.norm(arrays[variant][i][base:base+21]-gt,axis=1)
                err.extend(e.tolist()); wrist.append(float(e[0])); nhand+=1
                byclip.setdefault(r['id'],[]).extend(e.tolist())
                if variant!='raw' and decisions[i][variant]['swapped']:
                    e0=np.linalg.norm(arrays['raw'][i][base:base+21]-gt,axis=1)
                    report['changes'].append({'tag':r['tag'],'variant':variant,'side':side,
                        'old_mean_error':float(e0.mean()),'new_mean_error':float(e.mean())})
        def stat(e):
            e=np.array(e); return {'rmse_px':float(np.sqrt(np.mean(e**2))),'pck10_pct':float(np.mean(e<=10)*100),'n_points':len(e)}
        report['metrics'][variant]={**stat(err),'reference_hand_frames':nhand,
            'wrist_rmse_px':float(np.sqrt(np.mean(np.square(wrist)))),
            'swapped_frames':sum(d[variant]['swapped'] for d in decisions)}
        report['per_clip'][variant]={c:stat(e) for c,e in byclip.items()}
    (outdir/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['metrics'],indent=2))

if __name__=='__main__': main()
