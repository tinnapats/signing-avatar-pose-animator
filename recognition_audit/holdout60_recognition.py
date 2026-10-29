"""Real ST-GCN forward passes for every frozen renderer method, no retraining."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import adjacent_holdout60 as trial
from extend_original60 import metrics

HERE=trial.HERE;OUT=HERE/'holdout60_recognition_20261004';audit=trial.base.audit
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()

def main():
    assert not OUT.exists(),'Do not overwrite recognition evidence'
    p=trial.verify();labels=p['labels'];OUT.mkdir()
    inputs=trial.OUT/'case_index.json';rendered=trial.OUT/'renderer/rendered.json'
    files=[inputs,rendered,HERE/'wlasl_stgcn.zip',HERE/'wlasl_metadata.zip',HERE/'run_audit.py',HERE/'upstream_pose_transforms.py',HERE/'upstream_st_gcn.py',HERE/'upstream_graph_utils.py',Path(__file__)]
    hashes={str(f):trial.base.sha(f) for f in files}
    trial.write(OUT/'protocol.json',dict(labels=labels,hashes=hashes,production=trial.base.production(),model='OpenHands released ST-GCN WLASL2000, same checkpoint/normalizer, strict load, eval, no training',conditions=['OFF','OLD','NEW'],fps={'30':'primary: every other frame, historical recognizer protocol','60':'secondary: all renderer frames, same original clip duration'},input='Rendered detailed hand endpoints + payload body, 27 selected x/y points, shoulder normalization. Native SVG hand silhouettes NOT consumed. Missing detailed hands zero-filled exactly as historical adapter.',metrics='Accuracy, macro precision/recall/F1 over fixed60 target labels, zero division0; classifier still predicts all2000 classes. One clip per word. No semantic ground truth or training-overlap guarantee. Run actual forward for every method even when identical.'))
    caps={(c['phrase'],c['variant']):c['samples'] for c in trial.read(rendered)};assert len(caps)==180
    model,norm,vocab=audit.load_model();assert all(l in vocab for l in labels)
    rows=[];checks=[]
    for ci,c in enumerate(trial.read(inputs)):
        assert trial.base.sha(Path(c['file']))==c['sha256'];case=trial.read(Path(c['file']));frames=case['payload']['frames']
        arrays={}
        for variant in trial.VS:
            a=audit.frame_array(frames)
            ss=caps[c['phrase'],variant];assert len(ss)==len(frames)
            for i,s in enumerate(ss):
                for side,offset in [('left',33),('right',54)]:
                    a[i,offset:offset+21]=0
                    for index,point in s['hands'][side].items():a[i,offset+int(index)]=point
            arrays[variant]=a
        checks.append(dict(label=c['label'],arrays_equal=all(np.array_equal(arrays['OFF'],arrays[v]) for v in trial.VS)))
        for fps,stride in [(30,2),(60,1)]:
            for variant in trial.VS:
                a=arrays[variant][::stride]
                t=torch.from_numpy(a[:,audit.PICK,:].transpose(2,0,1).copy());t=norm({'frames':t})['frames'].unsqueeze(0);assert torch.isfinite(t).all()
                if not rows:
                    with torch.inference_mode():model(t)
                start=time.perf_counter()
                with torch.inference_mode():prob=model(t)[0].softmax(0)
                elapsed=(time.perf_counter()-start)*1000;assert prob.shape==(2000,) and torch.isfinite(prob).all()
                top=prob.topk(5)
                rows.append(dict(label=c['label'],variant=variant,fps=fps,frames=len(a),prediction=vocab[int(top.indices[0])],top5=[vocab[int(x)] for x in top.indices],target_probability=float(prob[vocab.index(c['label'])]),tensor_sha256=digest(t.numpy()),probabilities_sha256=digest(prob.numpy()),inference_ms=elapsed))
        trial.write(OUT/'progress.json',dict(completed=ci+1,predictions=rows,input_checks=checks))
        print('Scored',ci+1,'/60',c['label'],flush=True)
    summaries={}
    for fps in [30,60]:
        for variant in trial.VS:
            rr=[r for r in rows if r['fps']==fps and r['variant']==variant]
            summaries[f'{fps}fps_{variant}']=dict(**metrics(rr,labels),accuracy=100*sum(r['label']==r['prediction'] for r in rr)/len(rr),top5=100*sum(r['label'] in r['top5'] for r in rr)/len(rr))
    agreement={}
    for fps in [30,60]:
        groups=[[next(r for r in rows if r['label']==l and r['fps']==fps and r['variant']==v) for v in trial.VS] for l in labels]
        agreement[str(fps)]=dict(identical_tensors=sum(len({r['tensor_sha256'] for r in g})==1 for g in groups),identical_full_probability_vectors=sum(len({r['probabilities_sha256'] for r in g})==1 for g in groups),identical_predictions=sum(len({r['prediction'] for r in g})==1 for g in groups))
    trial.verify()
    for f,h in hashes.items():assert trial.base.sha(Path(f))==h
    result=dict(summaries=summaries,agreement=agreement,input_checks=checks,predictions=rows,actual_forward_passes=len(rows),production_unchanged=True)
    trial.write(OUT/'results.json',result)
    print(json.dumps(dict(summaries=summaries,agreement=agreement,actual_forward_passes=len(rows)),indent=2),flush=True)

if __name__=='__main__':main()
