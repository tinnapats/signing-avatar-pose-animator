"""Fixed training-augmentation-scale five-view averaging; development only."""
import json,math
from pathlib import Path
import torch
import run_audit as audit
from equal_duration_trial import arrays
from repeat_original30 import production,sha
from extend_original60 import metrics
HERE=Path(__file__).resolve().parent;SRC=HERE/'slgcn200_20261002';OUT=HERE/'model_tta200_20261005'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    assert not OUT.exists();before=production();OUT.mkdir()
    files=[SRC/'inputs.json',SRC/'renderer/rendered.json',SRC/'results.json',HERE/'wlasl_stgcn.zip',HERE/'wlasl_metadata.zip',HERE/'run_audit.py',HERE/'upstream_pose_transforms.py',Path(__file__)]
    p=dict(production=before,hashes={str(f):sha(f) for f in files},views=['identity','rotation_minus_0.1rad','rotation_plus_0.1rad','shear_minus_0.1','shear_plus_0.1'],
      rules='Fixed five normalized-input views using row-vector matrix convention from upstream training transforms. Angle/shear magnitude from checkpoint training std0.1, not tuned. Identity plus two rotations and two shears, equal arithmetic mean softmax probabilities across all2000 classes. Do not select per-word view or report best-view oracle. Same200 post-rendered captures30FPS; no changes to FPS/missing conventions/renderer/model weights. No independent holdout evaluation this round. Single-view results retained only for provenance; candidate is fixed five-view mean.')
    write(OUT/'protocol.json',p);cases=read(SRC/'inputs.json');caps={x['phrase']:x for x in read(SRC/'renderer/rendered.json')};labels=[c['label'] for c in cases];saved={x['label']:x for x in read(SRC/'results.json')['predictions'] if x['model']=='ST-GCN' and x['stage']=='post'}
    model,norm,vocab=audit.load_model();rows=[];views=[]
    def row(prob,label,variant):
        top=prob.topk(5);return dict(label=label,variant=variant,prediction=vocab[int(top.indices[0])],top5=[vocab[int(i)] for i in top.indices],target_probability=float(prob[vocab.index(label)]))
    for i,c in enumerate(cases):
        a=arrays(c,caps[c['phrase']]);raw=torch.from_numpy(a[:,audit.PICK,:].transpose(2,0,1).copy());t=norm({'frames':raw})['frames'];ts=[t]
        for angle in [-.1,.1]:
            m=torch.tensor([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]],dtype=torch.float32);ts.append((t.permute(1,2,0)@m).permute(2,0,1))
        for s in [-.1,.1]:
            m=torch.tensor([[1.,s],[0.,1.]],dtype=torch.float32);ts.append((t.permute(1,2,0)@m).permute(2,0,1))
        probs=[]
        for v,x in zip(p['views'],ts):
            assert torch.isfinite(x).all()
            with torch.inference_mode():prob=model(x.unsqueeze(0))[0].softmax(0)
            probs.append(prob);views.append(row(prob,c['label'],v))
        baseline=row(probs[0],c['label'],'BASE');assert all(baseline[k]==saved[c['label']][k] for k in ['prediction','top5','target_probability'])
        rows.extend([baseline,row(torch.stack(probs).mean(0),c['label'],'TTA5')])
        if (i+1)%20==0:print('Scored',i+1,'/200',flush=True)
    summary={v:metrics([r for r in rows if r['variant']==v],labels) for v in ['BASE','TTA5']};d={(r['label'],r['variant']):r['prediction']==r['label'] for r in rows};g=[l for l in labels if d[l,'TTA5'] and not d[l,'BASE']];lost=[l for l in labels if d[l,'BASE'] and not d[l,'TTA5']]
    assert production()==before and all(sha(f)==p['hashes'][str(f)] for f in files)
    result=dict(summary=summary,gained=g,lost=lost,predictions=rows,views=views,inferences=1000,production_unchanged=True,baseline_exact=True);write(OUT/'results.json',result);print(json.dumps({k:result[k] for k in ['summary','gained','lost']},indent=2),flush=True)
if __name__=='__main__':main()
