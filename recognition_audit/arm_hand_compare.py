"""Isolated, frozen two-condition arm/hand experiment; no production edits."""
import json, sys, copy, math
import numpy as np
import novel_words30 as prev
from equal_duration_trial import arrays
from scale_anchor200 import variants
base=prev.base
HERE=base.HERE
SRC=HERE/'slgcn200_20261002'
OUT=HERE/'arm_hand_compare_20261003'

def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def xy(k): return np.array([k['position']['x'],k['position']['y']],dtype=float)

def prepare():
    assert not (OUT/'renderer/rendered.json').exists()
    OUT.mkdir(exist_ok=True)
    (OUT/'cases').mkdir(exist_ok=True)
    cases=[json.loads(p.read_text()) for p in sorted((SRC/'cases').glob('*.json'))]
    caps={c['phrase']:c for c in json.loads((SRC/'renderer/rendered.json').read_text())}
    protocol=dict(labels=[c['label'] for c in cases],production=base.production(),
      source_hashes={str(p.relative_to(HERE)):base.sha(p) for p in [SRC/'inputs.json',SRC/'renderer/rendered.json',HERE/'wlasl_stgcn.zip']},
      methods=['HAND_TO_BODY','ARM_TO_HAND'],fps=30,
      rule='Planar two-bone IK, shoulder fixed, per-frame original projected segment lengths. Clamp reach to triangle interval (epsilon 1e-6); nearest original elbow with 0.6 temporal squared-distance cost. Reset temporal preference on missing/invalid hand. Never shift hand to clamped endpoint. Freeze original detailed-hand scales in both conditions.',
      evaluation='Actual Canvas hand endpoints and renderer skeleton shoulders/elbows; nose/eyes from payload. Not full image recognition. Same 200 development words, one clip each, unknown training overlap. No parameter search; no production changes.')
    write(OUT/'protocol.json',protocol)
    inputs=[]; diagnostics=[]
    for c in cases:
        cap=caps[c['phrase']]; scales={}
        for side in ['left','right']:
            found=[]
            for s,index in zip(cap['samples'],c['frameIndices']):
                pts=s['hands'][side]
                if not pts: continue
                f=c['payload']['frames'][index];h=f['hands'][side];raw=np.array([xy(k) for k in h['keypoints']]);ids=sorted(map(int,pts))
                pose={k['part']:k for k in f['pose']['keypoints']};w=xy(pose[side+'Wrist']) if pose[side+'Wrist']['score']>=.2 else raw[0]
                x=raw[ids]-raw[0];y=np.array([pts[str(i)] for i in ids])-w
                scale=float((x*y).sum()/(x*x).sum());found.append(scale)
            scales[side]=float(np.median(found)) if found else 1
            assert not found or max(abs(v-scales[side]) for v in found)<1e-6
        for method in protocol['methods']:
            d=copy.deepcopy(c);d['phrase']=c['label']+'__'+method;d['method']=method;d['payload']['auditScales']=scales
            previous={};clamped=0;solved=0;maxerr=0
            if method=='ARM_TO_HAND':
                for f in d['payload']['frames']:
                    p={k['part']:k for k in f['pose']['keypoints']}
                    for side in ['left','right']:
                        h=f.get('hands',{}).get(side)
                        keys=[p[side+k] for k in ['Shoulder','Elbow','Wrist']]
                        if not h or len(h.get('keypoints',[]))!=21 or h.get('score',0)<.001 or min(k['score'] for k in keys)<.2 or h['keypoints'][0].get('score',0)<=0:
                            previous.pop(side,None);continue
                        s,e,w=map(xy,keys);target=xy(h['keypoints'][0]);l1=np.linalg.norm(e-s);l2=np.linalg.norm(w-e)
                        if min(l1,l2)<1e-5: previous.pop(side,None);continue
                        delta=target-s;dist=np.linalg.norm(delta);u=delta/dist if dist>1e-9 else (w-s)/max(np.linalg.norm(w-s),1e-9)
                        if np.linalg.norm(u)<.5:u=np.array([1.,0.])
                        reach=float(np.clip(dist,abs(l1-l2)+1e-6,l1+l2-1e-6));wnew=s+u*reach
                        along=(l1*l1-l2*l2+reach*reach)/(2*reach);height=math.sqrt(max(0,l1*l1-along*along));perp=np.array([-u[1],u[0]])
                        choices=[s+u*along+sign*perp*height for sign in [-1,1]]
                        def cost(q):return np.sum((q-e)**2)+(.6*np.sum((q-s-previous[side])**2) if side in previous else 0)
                        enew=min(choices,key=cost);previous[side]=enew-s
                        for k,q in [(keys[1],enew),(keys[2],wnew)]:k['position']['x'],k['position']['y']=map(float,q)
                        err=max(abs(np.linalg.norm(enew-s)-l1),abs(np.linalg.norm(wnew-enew)-l2));maxerr=max(maxerr,err);assert err<1e-5
                        clamped+=abs(reach-dist)>1e-5;solved+=1
            diagnostics.append(dict(label=c['label'],method=method,solved=solved,clamped=int(clamped),max_length_error=maxerr))
            name=f'{len(inputs):03d}.json'
            with (OUT/'cases'/name).open('w',encoding='utf-8') as stream:json.dump(d,stream,ensure_ascii=False,separators=(',',':'))
            inputs.append(dict(label=d['label'],phrase=d['phrase'],method=method,file=str(OUT/'cases'/name)))
        if len(inputs)%40==0:print('Prepared cases',len(inputs),flush=True)
    write(OUT/'case_index.json',inputs)
    write(OUT/'ik_diagnostics.json',diagnostics)
    assert base.production()==protocol['production']
    print('Prepared',len(inputs),'cases',flush=True)

def score():
    p=json.loads((OUT/'protocol.json').read_text());assert base.production()==p['production']
    for name,h in p['source_hashes'].items():assert base.sha(HERE/name)==h
    cases={c['phrase']:c for c in json.loads((OUT/'case_index.json').read_text())}
    caps=json.loads((OUT/'renderer/rendered.json').read_text());assert len(caps)==400
    _,predict=prev.load_predictor();rows=[];geo=[];max_body_error=0
    for n,cap in enumerate(caps):
        c=json.loads(__import__('pathlib').Path(cases[cap['phrase']]['file']).read_text());orig=json.loads((SRC/'cases'/f"{p['labels'].index(c['label']):03d}.json").read_text());a=arrays(c,cap);gap=[];length_error=[];tracks={side:[] for side in ['left','right']};widths=[]
        assert c['frameIndices']==orig['frameIndices']
        assert len(c['payload']['frames'])==len(orig['payload']['frames'])
        assert all(f.get('hands')==o.get('hands') for f,o in zip(c['payload']['frames'],orig['payload']['frames']))
        for i,(index,sample) in enumerate(zip(c['frameIndices'],cap['samples'])):
            raw={k['part']:k for k in orig['payload']['frames'][index]['pose']['keypoints']}
            current={k['part']:k for k in c['payload']['frames'][index]['pose']['keypoints']};b=sample['arms']
            for part,num in [('leftShoulder',11),('rightShoulder',12),('leftElbow',13),('rightElbow',14)]:a[i,num]=b[part]
            for part,q in b.items():max_body_error=max(max_body_error,float(np.linalg.norm(np.array(q)-xy(current[part]))))
            width=np.linalg.norm(np.array(b['leftShoulder'])-b['rightShoulder']);widths.append(width)
            for side in ['left','right']:
                shoulder,elbow,wrist=[np.array(b[side+k]) for k in ['Shoulder','Elbow','Wrist']];tracks[side].append(elbow)
                pts=sample['hands'][side]
                if '0' in pts and width>1e-6:gap.append(float(np.linalg.norm(wrist-pts['0'])/width))
                for q,r,k1,k2 in [(shoulder,elbow,'Shoulder','Elbow'),(elbow,wrist,'Elbow','Wrist')]:
                    expected=np.linalg.norm(xy(raw[side+k1])-xy(raw[side+k2]))
                    if expected>1e-6:length_error.append(abs(np.linalg.norm(q-r)/expected-1))
        rough=[]
        for t in tracks.values():
            for i in range(1,len(t)-1):
                if widths[i]>1e-6:rough.append(float(np.linalg.norm(t[i+1]-2*t[i]+t[i-1])/widths[i]))
        geo.append(dict(label=c['label'],method=c['method'],wrist_gap=float(np.mean(gap)) if gap else None,bone_relative_error=float(np.mean(length_error)),elbow_second_difference=float(np.mean(rough))))
        rows.append(predict(a,c['label'],c['method']))
        if (n+1)%40==0:print('Scored',n+1,flush=True)
    metrics={m:base.metrics([r for r in rows if r['variant']==m],p['labels']) for m in p['methods']}
    summary={m:{key:float(np.mean([r[key] for r in geo if r['method']==m and r[key] is not None])) for key in ['wrist_gap','bone_relative_error','elbow_second_difference']} for m in p['methods']}
    by={(r['label'],r['variant']):r['prediction']==r['label'] for r in rows};g=sum(not by[l,'HAND_TO_BODY'] and by[l,'ARM_TO_HAND'] for l in p['labels']);loss=sum(by[l,'HAND_TO_BODY'] and not by[l,'ARM_TO_HAND'] for l in p['labels']);n=g+loss
    prob=min(1.,2*sum(math.comb(n,k) for k in range(min(g,loss)+1))/2**n) if n else 1.
    old={r['label']:r['prediction'] for r in json.loads((SRC/'results.json').read_text())['predictions'] if r['model']=='ST-GCN' and r['stage']=='post'}
    baseline_matches=sum(r['prediction']==old[r['label']] for r in rows if r['variant']=='HAND_TO_BODY')
    result=dict(metrics=metrics,geometry=summary,gained=g,lost=loss,paired_exact_p=prob,max_renderer_payload_arm_error=max_body_error,baseline_prediction_matches_previous=baseline_matches,predictions=rows,per_word_geometry=geo,
      capture_sha256=base.sha(OUT/'renderer/rendered.json'),script_hashes={name:base.sha(HERE/name) for name in ['arm_hand_compare.py','arm_hand_render.mjs']})
    assert base.production()==p['production'];write(OUT/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['predictions','per_word_geometry']},indent=2),flush=True)

if __name__=='__main__': {'prepare':prepare,'score':score}[sys.argv[1]]()
