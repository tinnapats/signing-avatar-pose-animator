"""Change only Holistic complexity; frozen videos, ST-GCN and native FPS."""
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
DATA = HERE/'recognition_100'
OUT = HERE/'preprocessing_control'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def extract():
    import cv2
    import mediapipe as mp
    import numpy as np
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((DATA/'manifest.json').read_text())
    protocol = dict(change='Holistic model_complexity 1 -> 2 only', sample='Same frozen 100 clips, all frames, original FPS',
        no_tuning=True, mediapipe=mp.__version__, opencv=cv2.__version__,
        inputs_sha256=digest(DATA/'inputs.json'), results_sha256=digest(DATA/'results.json'),
        manifest_sha256=digest(DATA/'manifest.json'), extractor_sha256=digest(Path(__file__)),
        settings=dict(static_image_mode=False, model_complexity=2, smooth_landmarks=True, enable_segmentation=False,
                      refine_face_landmarks=False, min_detection_confidence=.5, min_tracking_confidence=.5))
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text()) == protocol
    else:
        (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    for i,item in enumerate(manifest):
        target=OUT/(item['metadata']['video_id']+'.npz')
        if target.exists():
            print('cached',i+1,item['word'],flush=True)
            continue
        path=Path(item['local_video']); assert digest(path)==item['sha256']
        cap=cv2.VideoCapture(str(path)); assert cap.isOpened()
        fps=cap.get(cv2.CAP_PROP_FPS); frames=[]; counts=dict(pose=0,left_hand=0,right_hand=0)
        start=time.perf_counter()
        with mp.solutions.holistic.Holistic(**protocol['settings']) as model:
            while True:
                ok,image=cap.read()
                if not ok: break
                result=model.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                points=np.zeros((75,2),dtype=np.float32)
                for part,offset,count in [('pose',0,33),('left_hand',33,21),('right_hand',54,21)]:
                    landmarks=getattr(result,part+'_landmarks')
                    if landmarks:
                        counts[part]+=1
                        points[offset:offset+count]=[[p.x,p.y] for p in landmarks.landmark]
                frames.append(points)
        cap.release()
        assert len(frames)==item['extraction']['frames'] and abs(fps-item['extraction']['fps'])<1e-8
        np.savez_compressed(target, points=np.asarray(frames), fps=fps)
        (OUT/(target.stem+'.json')).write_text(json.dumps(dict(word=item['word'],video_id=target.stem,
            frames=len(frames),fps=fps,detected_frames=counts,runtime_seconds=time.perf_counter()-start,
            video_sha256=item['sha256'],points_sha256=digest(target)),indent=2))
        print('extracted',i+1,'/100',item['word'],counts,flush=True)

def evaluate():
    import numpy as np
    import torch
    import run_audit as audit
    from step1_evaluate import hashes
    before=hashes()
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert digest(DATA/'inputs.json')==protocol['inputs_sha256']
    assert digest(DATA/'results.json')==protocol['results_sha256']
    cases=json.loads((DATA/'inputs.json').read_text())
    old=json.loads((DATA/'results.json').read_text())
    manifest=json.loads((DATA/'manifest.json').read_text())
    assert len(cases)==len(manifest)==100
    network,normalize,labels=audit.load_model()
    assert digest(HERE/'wlasl_stgcn.zip')==old['checkpoint_sha256']
    rows=[]
    for case,item in zip(cases,manifest):
        assert case['phrase']==item['word']
        word=case['phrase']; video_id=item['metadata']['video_id']
        new=np.load(OUT/(video_id+'.npz'))['points']
        native=np.asarray(case['native'],dtype=np.float32)
        assert native.shape==new.shape
        for variant,data in [('complexity1',native),('complexity2',new)]:
            tensor=torch.from_numpy(data[:,audit.PICK,:].transpose(2,0,1).copy())
            tensor=normalize({'frames':tensor})['frames'].unsqueeze(0)
            assert torch.isfinite(tensor).all()
            with torch.inference_mode(): probs=network(tensor)[0].softmax(0)
            top=probs.topk(5)
            row=dict(label=word,video_id=video_id,variant=variant,prediction=labels[int(top.indices[0])],
                     top5=[labels[int(i)] for i in top.indices],target_probability=float(probs[labels.index(word)]))
            if variant=='complexity1':
                prior=next(r for r in old['predictions'] if r['label']==word and r['variant']=='source_native')
                assert row['top5']==prior['top5']
                assert abs(row['target_probability']-prior['target_probability'])<1e-6
            rows.append(row)
        print('evaluated',word,rows[-2]['prediction'],rows[-1]['prediction'],flush=True)
    summaries=[]
    for variant in ['complexity1','complexity2']:
        selected=[r for r in rows if r['variant']==variant]; ps=[];rs=[];fs=[]
        for label in [c['phrase'] for c in cases]:
            tp=sum(r['label']==label==r['prediction'] for r in selected)
            fp=sum(r['label']!=label and r['prediction']==label for r in selected)
            fn=sum(r['label']==label and r['prediction']!=label for r in selected)
            p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
            ps.append(p);rs.append(r);fs.append(2*p*r/(p+r) if p+r else 0)
        summaries.append(dict(variant=variant,correct=sum(r['label']==r['prediction'] for r in selected),
            top5=sum(r['label'] in r['top5'] for r in selected),macro_precision=100*float(np.mean(ps)),
            macro_recall=100*float(np.mean(rs)),macro_f1=100*float(np.mean(fs))))
    gain=[rows[i]['label'] for i in range(0,len(rows),2) if rows[i]['prediction']!=rows[i]['label'] and rows[i+1]['prediction']==rows[i]['label']]
    loss=[rows[i]['label'] for i in range(0,len(rows),2) if rows[i]['prediction']==rows[i]['label'] and rows[i+1]['prediction']!=rows[i]['label']]
    assert hashes()==before
    result=dict(summaries=summaries,gained=gain,lost=loss,predictions=rows,production_unchanged=True,
                baseline_reproduced=True,checkpoint_sha256=old['checkpoint_sha256'])
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(summaries,indent=2),flush=True)
    print('gained',gain,'lost',loss,flush=True)

if __name__=='__main__':
    globals()[sys.argv[1]]()
