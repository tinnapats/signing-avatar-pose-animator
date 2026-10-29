"""One predeclared conservative hold candidate; isolated from production."""
import copy
import inspect
import json
import math
from collections import Counter
from unittest.mock import patch
import numpy as np
import torch
import repair_parts200 as previous

base=previous.base
OUT=base.HERE/'gated_hold200'
RULE=dict(max_hold_frames=1,min_hand_point_score=0.8,min_pose_score=0.5,max_wrist_motion_shoulder=0.15)

def allowed(frames,side,target,source):
    if target<0 or target>=len(frames):return False,'outside'
    if abs(target-source)>RULE['max_hold_frames']:return False,'duration'
    hand=frames[source]['hands'][side]
    if not hand.get('observed') or len(hand.get('keypoints',[]))!=21 or any(k.get('score',0)<RULE['min_hand_point_score'] for k in hand['keypoints']):return False,'hand_confidence'
    poses=[{k['part']:k for k in frames[i]['pose']['keypoints']} for i in [source,target]]
    name=side+'Wrist';required=[name,'leftShoulder','rightShoulder']
    if any(n not in p or p[n].get('score',0)<RULE['min_pose_score'] for p in poses for n in required):return False,'pose_confidence'
    def xy(k):return np.array([k['position']['x'],k['position']['y']],dtype=float)
    width=float(np.mean([np.linalg.norm(xy(p['leftShoulder'])-xy(p['rightShoulder'])) for p in poses]))
    movement=float(np.linalg.norm(xy(poses[0][name])-xy(poses[1][name])))
    if not math.isfinite(width+movement) or width<=1e-8 or movement/width>RULE['max_wrist_motion_shoulder']:return False,'wrist_motion'
    return True,'allowed'

def gated_repair(frames,max_gap_frames,fade_frames):
    frozen=copy.deepcopy(frames);cache={}
    def gate(side,target,source):
        key=(side,target,source)
        if key not in cache:
            cache[key]=allowed(frozen,side,target,source)
            COUNTS[cache[key][1]]+=1
        return cache[key][0]
    source=inspect.getsource(previous.ORIGINAL)
    needle='                target = keypoints[index]\n'
    assert source.count(needle)==1
    source=source.replace(needle,'                if not _audit_allow(side,index,source_index):\n                    return\n'+needle)
    ns=dict(previous.ORIGINAL.__globals__);ns['_audit_allow']=gate
    exec(compile(source,'audit_gated_hold','exec'),ns)
    return ns['repair_hand_tracks'](frames,max_gap_frames,fade_frames)

COUNTS=Counter()

def test():
    frames=[]
    for i in range(10):
        score=1. if i in [2,4,9] else 0.
        h=dict(observed=bool(score),score=score,keypoints=[dict(score=score,position=dict(x=float(i),y=0.,z=0.)) for _ in range(21)])
        pose=[dict(part=n,score=1.,position=dict(x=x,y=0.)) for n,x in [('leftShoulder',0.),('rightShoulder',100.),('leftWrist',20.),('rightWrist',80.)]]
        frames.append(dict(hands=dict(left=copy.deepcopy(h),right=copy.deepcopy(h)),pose=dict(keypoints=pose)))
    assert allowed(frames,'left',1,2)==(True,'allowed')
    assert not allowed(frames,'left',0,2)[0]
    bad=copy.deepcopy(frames);bad[1]['pose']['keypoints'][2]['position']['x']=50.
    assert allowed(bad,'left',1,2)==(False,'wrist_motion')
    bad=copy.deepcopy(frames);bad[2]['hands']['left']['keypoints'][3]['score']=0.2
    assert allowed(bad,'left',1,2)==(False,'hand_confidence')
    out=gated_repair(copy.deepcopy(frames),4,3)
    assert out[3]['hands']['left']['keypoints'][0]['position']['x']==3. # short fill retained
    assert out[1]['hands']['left']['score']>0 and out[0]['hands']['left']['score']==0
    for i in [2,4,9]:assert out[i]['hands']==frames[i]['hands']
    COUNTS.clear()

def main():
    test();print('Gating tests passed',flush=True)
    assert not (OUT/'results.json').exists()
    before=base.production();oldprotocol=json.loads((previous.OUT/'protocol.json').read_text());assert before==oldprotocol['production']
    cases=json.loads((base.SOURCE/'inputs.json').read_text());labels=[c['phrase'] for c in cases]
    assert labels==oldprotocol['labels']
    old=json.loads((previous.OUT/'results.json').read_text());lookup={(r['label'],r['variant']):r for r in old['predictions']}
    OUT.mkdir(exist_ok=True);(OUT/'review_cases').mkdir(exist_ok=True)
    protocol=dict(rule=RULE,labels=labels,review=previous.REVIEW,production=before,fps=30,
        inputs_sha256=base.sha(base.SOURCE/'inputs.json'),prior_sha256=base.sha(previous.OUT/'results.json'),
        checkpoint_sha256=base.sha(base.HERE/'wlasl_stgcn.zip'),
        note='Single development candidate fixed before inference; short gaps and taper preserved. Hold capped at one 30fps frame, only confident observed hand and stable confident body wrist. Existing first-frame decay retained; no wrist translation.')
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    network,norm,vocab=base.audit.load_model()
    def predict(frames,label):
        a=base.audit.frame_array(frames);t=torch.from_numpy(a[:,base.audit.PICK,:].transpose(2,0,1).copy())
        t=norm({'frames':t})['frames'].unsqueeze(0);assert torch.isfinite(t).all()
        with torch.inference_mode():p=network(t)[0].softmax(0)
        top=p.topk(5)
        return dict(label=label,prediction=vocab[int(top.indices[0])],top5=[vocab[int(i)] for i in top.indices],target_probability=float(p[vocab.index(label)]))
    rows=[]
    if (OUT/'progress.json').exists():rows=json.loads((OUT/'progress.json').read_text())['predictions']
    for i,r in enumerate(rows):assert r['label']==labels[i] and r['variant']=='gated_hold1'
    for i,c in enumerate(cases):
        if i<len(rows):continue
        label=c['phrase'];check=predict(c['payload']['frames'][::2],label)
        assert all(check[k]==lookup[label,'baseline'][k] for k in ['prediction','top5','target_probability'])
        COUNTS.clear()
        with patch.object(base.audit.pipeline,'repair_hand_tracks',gated_repair):
            payload=base.audit.pipeline.build_payload(base.ROOT/'SLclean.sqlite3',[label+'.csv'],'',30,513,513,0,0,2,1.2,2)
        frames=payload['frames'][::2];assert len(frames)==len(c['source30'])
        r=predict(frames,label);r.update(variant='gated_hold1',geometry=base.geometry(frames),coverage=previous.coverage(frames),gate_counts=dict(COUNTS))
        if label in previous.REVIEW:
            review=[x for x in json.loads((previous.OUT/'review_cases'/f'{label}.json').read_text()) if x['variant'] in ['baseline','no_hold']]
            review.append(dict(phrase=label+'__gated_hold1',label=label,variant='gated_hold1',payload=payload,frameIndices=list(range(0,len(payload['frames']),2)),coverage=previous.coverage(frames)))
            (OUT/'review_cases'/f'{label}.json').write_text(json.dumps(review),encoding='utf-8')
        rows.append(r);(OUT/'progress.json').write_text(json.dumps(dict(predictions=rows)),encoding='utf-8')
        print(f'{i+1}/200 {label}: {r["prediction"]}',flush=True)
    combined=[r for r in old['predictions'] if r['variant'] in ['baseline','no_hold']]+rows
    summaries={v:base.metrics([r for r in combined if r['variant']==v],labels) for v in ['baseline','no_hold','gated_hold1']}
    assert base.production()==before and base.sha(previous.OUT/'results.json')==protocol['prior_sha256']
    (OUT/'results.json').write_text(json.dumps(dict(summaries=summaries,predictions=combined,production_unchanged=True,baseline_exact=True),indent=2),encoding='utf-8')
    review=[]
    for l in previous.REVIEW:review.extend(json.loads((OUT/'review_cases'/f'{l}.json').read_text()))
    (OUT/'review_inputs.json').write_text(json.dumps(review),encoding='utf-8')
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()
