"""Fixed elbow, bounded forearm rotation: exploratory input-only model audit."""
import json, math
from pathlib import Path
import numpy as np
import scale_anchor200 as prior
from novel_words30 import load_predictor

base=prior.base;HERE=prior.HERE;SRC=prior.SRC;OUT=HERE/'bounded_forearm200_20261005'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def xy(k):return np.array([k['position']['x'],k['position']['y']],float)

def targets(c):
    tracks={s:[] for s in ['left','right']};angles={s:0. for s in tracks};err=0
    fps=c['payload'].get('fps',60);assert fps==60, fps
    for f in c['payload']['frames']:
        p={k['part']:k for k in f['pose']['keypoints']}
        for side in tracks:
            e,w=xy(p[side+'Elbow']),xy(p[side+'Wrist']);v=w-e;length=np.linalg.norm(v)
            h=f['hands'][side];goal=0.
            valid=min(p[side+k]['score'] for k in ['Shoulder','Elbow','Wrist'])>=.2 and h.get('score',0)>.001 and len(h.get('keypoints',[]))==21 and h['keypoints'][0].get('score',0)>0 and length>1e-6
            if valid:
                d=xy(h['keypoints'][0])-e
                goal=float(np.clip(math.atan2(v[0]*d[1]-v[1]*d[0],np.dot(v,d)),-math.pi/12,math.pi/12))
            # 15-degree offset, 90 degrees/sec correction velocity, EMA alpha .25.
            angle=angles[side]+float(np.clip(.25*(goal-angles[side]),-math.pi/2/fps,math.pi/2/fps))
            angles[side]=angle
            rot=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
            new=e+rot@v
            err=max(err,abs(np.linalg.norm(new-e)-length))
            tracks[side].append(dict(wrist=new.tolist(),original=w.tolist(),angle=angle,valid=bool(valid)))
    assert err<1e-8
    return tracks,err

def main():
    assert not OUT.exists();before=base.production()
    cases=read(SRC/'inputs.json');caps={c['phrase']:c for c in read(SRC/'renderer/rendered.json')}
    labels=[c['label'] for c in cases];assert len(labels)==200
    files=[SRC/'inputs.json',SRC/'renderer/rendered.json',SRC/'results.json',HERE/'wlasl_stgcn.zip',HERE/'wlasl_metadata.zip',Path(__file__),HERE/'scale_anchor200.py',HERE/'novel_words30.py',HERE/'run_audit.py',HERE/'upstream_pose_transforms.py',HERE/'upstream_st_gcn.py',HERE/'upstream_graph_utils.py']
    hashes={str(p):base.sha(p) for p in files};OUT.mkdir()
    vs=['BASE','BOUNDED','BOUNDED_PAYLOAD_SIZE']
    write(OUT/'protocol.json',dict(production=before,hashes=hashes,labels=labels,variants=vs,
      rule='Full60FPS pose wrist rotation around fixed elbow; shoulders/elbows unchanged. Preserve per-frame 2D forearm length. Offset <=15deg, EMA alpha .25, correction rate <=90deg/sec. Decay correction toward zero on unavailable target; no new hand landmarks. Existing detailed hands attached to new pose wrist only when pose wrist score>=.2. Keep production scale or payload size as two predetermined candidates. Sample same30FPS and durations. No tuning after scores. Input-only screening from verified affine capture, NOT new Canvas or RGB evaluation. Unknown 3D anatomy and source correctness. Existing development200, not independent test.'))
    _,predict=load_predictor();rows=[];geometry=[];alltargets=[]
    saved={r['label']:r['prediction'] for r in read(SRC/'results.json')['predictions'] if r['model']=='ST-GCN' and r['stage']=='post'}
    for n,c in enumerate(cases):
        cap=caps[c['phrase']];arr,_,_=prior.variants(c,cap);tracks,lengtherr=targets(c)
        data={'BASE':arr['D'],'BOUNDED':arr['D'].copy(),'BOUNDED_PAYLOAD_SIZE':arr['C'].copy()}
        gaps={v:[] for v in vs};dist={v:[] for v in vs};wristtracks={v:{s:[] for s in tracks} for v in vs};widths=[]
        for i,(index,sample) in enumerate(zip(c['frameIndices'],cap['samples'])):
            f=c['payload']['frames'][index];p={k['part']:k for k in f['pose']['keypoints']};width=max(np.linalg.norm(xy(p['leftShoulder'])-xy(p['rightShoulder'])),1e-6);widths.append(width)
            for side,offset in [('left',33),('right',54)]:
                t=tracks[side][index];new=np.array(t['wrist']);w=np.array(t['original']);delta=new-w;pts=sample['hands'][side];ids=sorted(map(int,pts))
                if ids and p[side+'Wrist']['score']>=.2:
                    for v in vs[1:]:data[v][i,offset+np.array(ids)]+=delta
                for v in vs:
                    wristtracks[v][side].append(w if v=='BASE' else new)
                    if '0' in pts and p[side+'Wrist']['score']>=.2:
                        a=data[v][i,offset];target=xy(f['hands'][side]['keypoints'][0]);end=w if v=='BASE' else new
                        gaps[v].append(float(np.linalg.norm(a-end)/width));dist[v].append(float(np.linalg.norm(a-target)/width))
        for v,a in data.items():
            assert np.array_equal(a[:,:33],arr['D'][:,:33]) # all model body joints unchanged
            r=predict(a,c['label'],v)
            if v=='BASE':assert r['prediction']==saved[c['label']]
            rows.append(r)
            rough=[]
            for side in tracks:
                q=np.array(wristtracks[v][side]);rough.extend(np.linalg.norm(q[2:]-2*q[1:-1]+q[:-2],axis=1)/np.array(widths[1:-1]))
            geometry.append(dict(label=c['label'],variant=v,attachment_gap=float(np.mean(gaps[v])) if gaps[v] else None,hand_target_distance=float(np.mean(dist[v])) if dist[v] else None,wrist_second_difference=float(np.mean(rough)),max_length_error=lengtherr))
        alltargets.append(dict(label=c['label'],tracks=tracks))
        if (n+1)%20==0:write(OUT/'progress.json',dict(completed=n+1,predictions=rows));print('Scored',n+1,'/200',flush=True)
    summary={v:base.metrics([r for r in rows if r['variant']==v],labels) for v in vs}
    geo={v:{k:float(np.mean([r[k] for r in geometry if r['variant']==v and r[k] is not None])) for k in ['attachment_gap','hand_target_distance','wrist_second_difference']} for v in vs}
    d={(r['label'],r['variant']):r['prediction']==r['label'] for r in rows}
    paired={v:dict(gained=[l for l in labels if d[l,v] and not d[l,'BASE']],lost=[l for l in labels if not d[l,v] and d[l,'BASE']]) for v in vs[1:]}
    assert base.production()==before and all(base.sha(p)==hashes[str(p)] for p in files)
    write(OUT/'targets.json',alltargets);write(OUT/'results.json',dict(summary=summary,geometry=geo,paired=paired,predictions=rows,per_word_geometry=geometry,production_unchanged=True))
    print(json.dumps(dict(summary=summary,geometry=geo,paired=paired),indent=2),flush=True)
if __name__=='__main__':main()
