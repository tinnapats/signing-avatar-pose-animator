"""Pixels-only 30-word pilot, detection and scoring in separate environments."""
import sys,json,hashlib,time,collections
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit')
OUT=HERE/'pixels30_20261007'
sys.path.insert(0,str(HERE))
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def longest(values):
    best=run=0
    for x in values:
        run=run+1 if x else 0;best=max(best,run)
    return best
def detect():
    import cv2,mediapipe as mp,numpy as np
    manifest=read(OUT/'capture_manifest.json');protocol=read(OUT/'protocol.json')
    assert [m['label'] for m in manifest]==protocol['labels'] and len(manifest)==30
    for folder in ['detected','arrays','review']: (OUT/folder).mkdir(exist_ok=True)
    def pack(lm):return [[p.x,p.y,p.z,p.visibility] for p in lm.landmark] if lm else []
    def process(images):
        rows=[]
        settings=dict(model_complexity=1,min_detection_confidence=.5,min_tracking_confidence=.5)
        with mp.solutions.holistic.Holistic(static_image_mode=False,smooth_landmarks=True,enable_segmentation=False,refine_face_landmarks=False,**settings) as holistic,mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=2,**settings) as hands:
            for i,rgb in enumerate(images):
                start=time.perf_counter();a=holistic.process(rgb);b=hands.process(rgb)
                rows.append(dict(index=i,pose=pack(a.pose_landmarks),pose_world=pack(a.pose_world_landmarks),face=pack(a.face_landmarks),left=pack(a.left_hand_landmarks),right=pack(a.right_hand_landmarks),
                    hands=[pack(x) for x in b.multi_hand_landmarks or []],handedness=[dict(label=x.classification[0].label,score=x.classification[0].score) for x in b.multi_handedness or []],elapsed_ms=(time.perf_counter()-start)*1000))
        return rows
    def rgba_rgb(p):
        im=cv2.imread(str(p),cv2.IMREAD_UNCHANGED);assert im is not None
        if im.shape[2]==4:
            a=im[:,:,3:4].astype(np.float32)/255;im=np.rint(im[:,:,:3]*a+255*(1-a)).astype(np.uint8)
        return cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    def summary(rows):
        n=len(rows)
        return dict(frames=n,pose=sum(bool(r['pose']) for r in rows),left=sum(bool(r['left']) for r in rows),right=sum(bool(r['right']) for r in rows),
          any_hand=sum(bool(r['left'] or r['right']) for r in rows),both_hands=sum(bool(r['left'] and r['right']) for r in rows),
          hands_detector_any=sum(bool(r['hands']) for r in rows),hands_detector_both=sum(len(r['hands'])==2 for r in rows),
          longest_no_left=longest([not r['left'] for r in rows]),longest_no_right=longest([not r['right'] for r in rows]),
          longest_no_pose=longest([not r['pose'] for r in rows]),mean_detection_ms=float(np.mean([r['elapsed_ms'] for r in rows])))
    summaries=[]
    for m in manifest:
        target=OUT/'detected'/(m['id']+'.json');paths=[OUT/'frames'/m['id']/f['file'] for f in m['frames']]
        assert all(sha(p)==f['sha256'] for p,f in zip(paths,m['frames']))
        if target.exists():rows=read(target)['rows']
        else:
            rows=process(rgba_rgb(p) for p in paths);save(target,dict(id=m['id'],frames_sha256=sha(OUT/('capture_'+m['id']+'.json')),rows=rows,mediapipe=mp.__version__,opencv=cv2.__version__))
        assert len(rows)==len(paths)
        primary=np.zeros((len(rows),75,2),np.float32);secondary=primary.copy();added=collections.Counter()
        width,height=m['width'],m['height']
        for i,r in enumerate(rows):
            if r['pose']:
                for j,p in enumerate(r['pose']):
                    if p[3]>0:primary[i,j]=[p[0]*width,p[1]*height]
            for side,offset in [('left',33),('right',54)]:
                if r[side]:primary[i,offset:offset+21]=np.array(r[side],np.float32)[:,:2]*[width,height]
            secondary[i]=primary[i];fallback={}
            for lm,h in zip(r['hands'],r['handedness']):
                side='right' if h['label']=='Left' else 'left'
                if h['score']>=.5 and (side not in fallback or h['score']>fallback[side][0]):fallback[side]=(h['score'],lm)
            for side,offset in [('left',33),('right',54)]:
                if not r[side] and side in fallback:
                    secondary[i,offset:offset+21]=np.array(fallback[side][1],np.float32)[:,:2]*[width,height];added[side]+=1
        assert np.isfinite(primary).all() and np.isfinite(secondary).all()
        np.savez_compressed(OUT/'arrays'/(m['id']+'.npz'),primary=primary,secondary=secondary)
        s=summary(rows);s.update(id=m['id'],label=m['label'],fallback_added=dict(added),detected_sha256=sha(target),arrays_sha256=sha(OUT/'arrays'/(m['id']+'.npz')),
          secondary_any_hand=int(np.any(secondary[:,33:]!=0,axis=(1,2)).sum()),secondary_both_hands=int((np.any(secondary[:,33:54]!=0,axis=(1,2))&np.any(secondary[:,54:]!=0,axis=(1,2))).sum()))
        summaries.append(s);save(OUT/'detection_summary.json',dict(per_clip=summaries,completed=len(summaries)))
        # Six equally spaced original/overlay frames per clip, independent of score.
        panels=[]
        for idx in np.rint(np.linspace(0,len(rows)-1,6)).astype(int):
            rgb=rgba_rgb(paths[idx]);bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);overlay=bgr.copy();r=rows[idx]
            for key,edges,col in [('pose',mp.solutions.pose.POSE_CONNECTIONS,(120,120,120)),('left',mp.solutions.hands.HAND_CONNECTIONS,(213,155,91)),('right',mp.solutions.hands.HAND_CONNECTIONS,(49,125,237))]:
                pts=r[key]
                if not pts:continue
                for a,b in edges:
                    pa,pb=pts[a],pts[b];cv2.line(overlay,(round(pa[0]*width),round(pa[1]*height)),(round(pb[0]*width),round(pb[1]*height)),col,2)
                for p in pts:cv2.circle(overlay,(round(p[0]*width),round(p[1]*height)),3,col,-1)
            panels.append(np.concatenate([cv2.resize(bgr,(256,256)),cv2.resize(overlay,(256,256))],axis=0))
        sheet=np.concatenate(panels,axis=1);sheet=cv2.copyMakeBorder(sheet,40,35,0,0,cv2.BORDER_CONSTANT,value=(255,255,255));cv2.putText(sheet,m['label']+' | top: Canvas | bottom: Holistic image detections',(15,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(30,30,30),1)
        cv2.putText(sheet,'6 uniformly spaced frames; full PNG and every detected landmark retained.',(15,570),cv2.FONT_HERSHEY_SIMPLEX,.65,(30,30,30),1);cv2.imwrite(str(OUT/'review'/(m['id']+'.png')),sheet)
        print('detected',m['id'],m['label'],s['frames'],'pose',s['pose'],'any hand',s['any_hand'],'both',s['both_hands'],flush=True)
    # Controls only use imagery; no label inference is needed for these short windows.
    if not (OUT/'controls.json').exists():
        human=read(HERE/'paired_video/manifest.json')[0];p=Path(human['local_video']);assert sha(p)==human['sha256'];cap=cv2.VideoCapture(str(p));frames=[]
        for _ in range(26):
            ok,img=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        cap.release();assert len(frames)==26
        human_rows=process(frames);blank_rows=process([np.full((808,808,3),255,np.uint8) for _ in range(26)])
        save(OUT/'controls.json',dict(human_source=str(p),human_sha256=sha(p),human=summary(human_rows),blank=summary(blank_rows),human_rows=human_rows,blank_rows=blank_rows))
    save(OUT/'detection_summary.json',dict(per_clip=summaries,completed=30,versions=dict(mediapipe=mp.__version__,opencv=cv2.__version__),total_frames=sum(s['frames'] for s in summaries)))

def score():
    import numpy as np,torch
    import run_audit as audit
    assert not (OUT/'results.json').exists(),'Do not replace completed result'
    p=read(OUT/'protocol.json');summary=read(OUT/'detection_summary.json');assert summary['completed']==30
    model,norm,vocab=audit.load_model();rows=[]
    for m in summary['per_clip']:
        file=OUT/'arrays'/(m['id']+'.npz');assert sha(file)==m['arrays_sha256'];z=np.load(file)
        primary=z['primary'];nohands=primary.copy();nohands[:,33:]=0
        variants={'pixels_holistic':primary,'pixels_holistic_hands':z['secondary'],
          'pixels_shuffled':primary[np.random.default_rng(20261007).permutation(len(primary))],'pixels_zero_hands':nohands}
        for variant,a in variants.items():
            raw=torch.from_numpy(a[:,audit.PICK,:].transpose(2,0,1).copy());tensor=norm({'frames':raw})['frames'].unsqueeze(0);assert torch.isfinite(tensor).all()
            start=time.perf_counter()
            with torch.inference_mode():prob=model(tensor)[0].softmax(0)
            top=prob.topk(5);rows.append(dict(id=m['id'],label=m['label'],variant=variant,prediction=vocab[int(top.indices[0])],top5=[vocab[int(i)] for i in top.indices],top5_probabilities=top.values.tolist(),target_probability=float(prob[vocab.index(m['label'])]),runtime_ms=(time.perf_counter()-start)*1000))
        print('scored',m['id'],m['label'],[(r['variant'],r['prediction']) for r in rows[-4:]],flush=True)
    old=read(HERE/'slgcn200_20261002/results.json')
    for r in old['predictions']:
        if r['model']=='ST-GCN' and r['label'] in p['labels'] and r['stage'] in ['pre','post']:
            rr=dict(r);rr['variant']='archived_'+r['stage'];rows.append(rr)
    def metrics(rr):
        assert len(rr)==30 and set(r['label'] for r in rr)==set(p['labels']);ps=[];rs=[];fs=[]
        for l in p['labels']:
            tp=sum(r['label']==l and r['prediction']==l for r in rr);fp=sum(r['label']!=l and r['prediction']==l for r in rr);fn=sum(r['label']==l and r['prediction']!=l for r in rr)
            prec=tp/(tp+fp) if tp+fp else 0;rec=tp/(tp+fn) if tp+fn else 0;ps.append(prec);rs.append(rec);fs.append(2*prec*rec/(prec+rec) if prec+rec else 0)
        return dict(n=30,correct=sum(r['label']==r['prediction'] for r in rr),accuracy=100*sum(r['label']==r['prediction'] for r in rr)/30,precision=float(np.mean(ps)*100),recall=float(np.mean(rs)*100),f1=float(np.mean(fs)*100),top5=100*sum(r['label'] in r['top5'] for r in rr)/30)
    scores={v:metrics([r for r in rows if r['variant']==v]) for v in sorted(set(r['variant'] for r in rows))}
    for f,h in p['production'].items():assert sha(HERE.parent/f)==h,'Production changed'
    result=dict(summary=scores,predictions=rows,checkpoint_sha256=sha(HERE/'wlasl_stgcn.zip'),production_unchanged=True,protocol_sha256=sha(OUT/'protocol.json'))
    save(OUT/'results.json',result);print(json.dumps(scores,indent=2),flush=True)

if __name__=='__main__':{'detect':detect,'score':score}[sys.argv[1]]()
