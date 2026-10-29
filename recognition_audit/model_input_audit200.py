"""Frozen preprocessing diagnostics only, not production or Avatar changes."""
import ast,json,zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import run_audit as audit
from repeat_original30 import production,sha
from extend_original60 import metrics
from equal_duration_trial import arrays
HERE=Path(__file__).resolve().parent;SRC=HERE/'slgcn200_20261002';OUT=HERE/'model_input_audit200_20261005'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    assert not OUT.exists();before=production();cases=read(SRC/'inputs.json');caps={c['phrase']:c for c in read(SRC/'renderer/rendered.json')};labels=[c['label'] for c in cases];assert len(labels)==200
    # Extract upstream collate method without importing unrelated torchvision.
    tree=ast.parse((HERE/'upstream_base.py').read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='BaseIsolatedDataset');fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='collate_fn');fn.decorator_list=[]
    namespace={'torch':torch,'F':F};exec(compile(ast.Module(body=[fn],type_ignores=[]),'upstream_collate','exec'),namespace);collate=namespace['collate_fn']
    transforms=audit.module('audit_input_transforms','upstream_pose_transforms.py');assert audit.PICK==transforms.PoseSelect.KEYPOINT_PRESETS['mediapipe_holistic_minimal_27']
    files=[SRC/'inputs.json',SRC/'renderer/rendered.json',SRC/'results.json',HERE/'wlasl_stgcn.zip',HERE/'wlasl_metadata.zip',HERE/'upstream_base.py',HERE/'upstream_pose_transforms.py',HERE/'upstream_st_gcn.py',HERE/'run_audit.py',Path(__file__)]
    OUT.mkdir();protocol=dict(production=before,labels=labels,hashes={str(p):sha(p) for p in files},variants=['BASE','ZERO_MISSING_AFTER_NORM','BATCH64','BATCH64_REVERSED'],
       rules='Same frozen200 post-render captures; same30FPS and durations, minimal27xy, model and clip-level shoulder normalization. Missing-zero control changes only originally absent hand joint tensor entries AFTER normalize; not claimed training-compatible. Batch64 uses actual upstream collate AFTER per-clip normalize and pads end with zeros to group maximum. Reversed changes group membership only, preserves each clip frame order. Batch ordering is diagnostic, never choose per-word/best batch. No FPS sweep, renderer change, model training or new holdout. All variants predeclared; baseline top5 and target probability must reproduce exactly.')
    write(OUT/'protocol.json',protocol);(OUT/'checkpoint_config.yaml').write_bytes(zipfile.ZipFile(HERE/'wlasl_stgcn.zip').read('wlasl/st_gcn/config.yaml'))
    model,norm,vocab=audit.load_model();tensors=[];masked=[];diagnostics=[];rows=[]
    previous={r['label']:r for r in read(SRC/'results.json')['predictions'] if r['model']=='ST-GCN' and r['stage']=='post'}
    for c in cases:
        a=arrays(c,caps[c['phrase']]);raw=torch.from_numpy(a[:,audit.PICK,:].transpose(2,0,1).copy());valid=(raw!=0).any(dim=0)
        t=norm({'frames':raw.clone()})['frames'];assert torch.isfinite(t).all();z=t.clone();missing=~valid;missing[:,:7]=False;z[:,missing]=0
        assert torch.equal(z[:,:,:7],t[:,:,:7]);tensors.append(t);masked.append(z)
        missing_norm=t[:,missing];diagnostics.append(dict(label=c['label'],frames=t.shape[1],missing_hand_joints=int(missing.sum()),missing_norm_abs_mean=float(missing_norm.abs().mean()) if missing.any() else 0))
    def record(p,i,v):
        top=p.topk(5);r=dict(label=labels[i],variant=v,prediction=vocab[int(top.indices[0])],top5=[vocab[int(k)] for k in top.indices],target_probability=float(p[vocab.index(labels[i])]))
        if v=='BASE':assert all(r[k]==previous[labels[i]][k] for k in ['prediction','top5','target_probability'])
        rows.append(r)
    for v,ts in [('BASE',tensors),('ZERO_MISSING_AFTER_NORM',masked)]:
        for i,t in enumerate(ts):
            with torch.inference_mode():p=model(t.unsqueeze(0))[0].softmax(0)
            record(p,i,v)
        print(v,'complete',flush=True)
    groups=[]
    for v,order in [('BATCH64',list(range(200))),('BATCH64_REVERSED',list(reversed(range(200))))]:
        for start in range(0,200,64):
            ids=order[start:start+64];items=[dict(frames=tensors[i],label=torch.tensor(i),file=labels[i],dataset_name='audit') for i in ids];batch=collate(items)['frames'];assert torch.isfinite(batch).all()
            for j,i in enumerate(ids):assert torch.equal(batch[j,:,:tensors[i].shape[1]],tensors[i])
            with torch.inference_mode():ps=model(batch).softmax(-1)
            for p,i in zip(ps,ids):record(p,i,v)
            groups.append(dict(variant=v,labels=[labels[i] for i in ids],padded_frames=batch.shape[2]));print(v,start+len(ids),flush=True)
    summaries={v:metrics([r for r in rows if r['variant']==v],labels) for v in protocol['variants']};by={(r['label'],r['variant']):r for r in rows};paired={}
    for v in protocol['variants'][1:]:paired[v]=dict(gained=[l for l in labels if by[l,v]['prediction']==l and by[l,'BASE']['prediction']!=l],lost=[l for l in labels if by[l,v]['prediction']!=l and by[l,'BASE']['prediction']==l])
    changed=[l for l in labels if by[l,'BATCH64']['prediction']!=by[l,'BATCH64_REVERSED']['prediction']]
    assert production()==before and all(sha(p)==protocol['hashes'][str(p)] for p in files)
    write(OUT/'results.json',dict(summary=summaries,paired=paired,predictions=rows,diagnostics=diagnostics,batch_groups=groups,batch_order_changed_top1=changed,production_unchanged=True,baseline_exact=True))
    print(json.dumps(dict(summary=summaries,paired=paired,batch_order_changed_top1=changed),indent=2),flush=True)
if __name__=='__main__':main()
