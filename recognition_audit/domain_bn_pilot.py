"""Unlabelled calibration200 -> disjoint evaluation100; immutable base model."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import copy,json
from pathlib import Path
import numpy as np
import torch
import run_audit as audit
from equal_duration_trial import arrays
from repeat_original30 import production,sha
from extend_original60 import metrics
HERE=Path(__file__).resolve().parent;SRC=HERE/'slgcn200_20261002';TEST=HERE/'frozen100_20261005';OUT=HERE/'domain_bn_pilot_20261005'
def read(p):return _local_json_loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    assert not OUT.exists();before=production();traincases=read(SRC/'inputs.json');tcaps={c['phrase']:c for c in read(SRC/'renderer/rendered.json')};trainlabels=[c['label'] for c in traincases];tp=read(TEST/'protocol.json');testlabels=tp['labels'];assert len(trainlabels)==200 and len(testlabels)==100 and not set(trainlabels)&set(testlabels)
    files=[SRC/'inputs.json',SRC/'renderer/rendered.json',TEST/'protocol.json',TEST/'case_index.json',TEST/'results.json',HERE/'wlasl_stgcn.zip',HERE/'wlasl_metadata.zip',HERE/'run_audit.py',HERE/'upstream_st_gcn.py',HERE/'upstream_pose_transforms.py',HERE/'upstream_graph_utils.py',Path(__file__)]
    OUT.mkdir();protocol=dict(production=before,hashes={str(f):sha(f) for f in files},calibration_labels=trainlabels,evaluation_labels=testlabels,variants=['BASE','INPUT_BN','ALL_BN'],
      rules='AdaBN-inspired pilot, not exact paper reproduction.200 post-render BASE no_hold_no_smoothing clips for unlabeled statistics only.100 label-disjoint previously inspected BASE clips evaluation only, not pristine final test. Freeze before fitting; no label loss/gradients/pseudo-labels or hyperparameter selection. INPUT_BN: replace input BN moments by frame-weighted calibration normalized coordinates pooled across200, sample variance unbiased. ALL_BN: reset all BN running stats, one calibration pass in stored order, momentum=None, batch1 unpadded, equal-clip average within BN; non-BN modules eval. This is not identical to global pooled moments in deeper layers. Save adapted statistics before loading evaluation tensors. Score fixed checkpoint2000 classes at30FPS, all learnable parameters identical, evaluator only, no Renderer modification.',source='https://arxiv.org/abs/1603.04779')
    write(OUT/'protocol.json',protocol);model,norm,vocab=audit.load_model();models={'BASE':model,'INPUT_BN':copy.deepcopy(model),'ALL_BN':copy.deepcopy(model)}
    original={k:v.clone() for k,v in model.named_parameters()};tensors=[]
    for c in traincases:
        a=arrays(c,tcaps[c['phrase']]);t=norm({'frames':torch.from_numpy(a[:,audit.PICK,:].transpose(2,0,1).copy())})['frames'];assert torch.isfinite(t).all();tensors.append(t)
    # data_bn sees flattened V*C channels with T samples.
    joined=torch.cat([t.permute(2,0,1).reshape(54,-1).double() for t in tensors],dim=1)
    bn=models['INPUT_BN'][0].data_bn
    with torch.no_grad():bn.running_mean.copy_(joined.mean(1));bn.running_var.copy_(joined.var(1,unbiased=True));bn.num_batches_tracked.fill_(200)
    allbn=[m for m in models['ALL_BN'].modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
    for m in allbn:m.reset_running_stats();m.momentum=None;m.train()
    for i,t in enumerate(tensors):
        with torch.inference_mode():y=models['ALL_BN'](t.unsqueeze(0));assert torch.isfinite(y).all()
        if (i+1)%40==0:print('Calibrated',i+1,'/200',flush=True)
    del traincases,tcaps,tensors,joined
    for name,m in models.items():
        m.eval();assert all(torch.equal(v,original[k]) for k,v in m.named_parameters())
        if name!='BASE':torch.save({k:v for k,v in m.state_dict().items() if k.endswith(('running_mean','running_var','num_batches_tracked'))},OUT/(name+'_statistics.pt'))
    stats_hashes={name:sha(OUT/(name+'_statistics.pt')) for name in ['INPUT_BN','ALL_BN']};write(OUT/'calibration_complete.json',dict(statistics_hashes=stats_hashes,learnable_parameters_unchanged=True,calibration_only=True))
    # Evaluation data loaded only after statistics have been frozen and saved.
    desc=[d for d in read(TEST/'case_index.json') if d['method']=='BASE'];assert len(desc)==100
    previous={r['label']:r for r in read(TEST/'results.json')['predictions'] if r['variant']=='BASE'};rows=[];testhashes={}
    for i,d in enumerate(desc):
        path=Path(d['file']);assert sha(path)==d['sha256'];capfile=TEST/'renderer'/('capture_'+path.stem+'.json');testhashes[str(capfile)]=sha(capfile);c=read(path);cap=read(capfile);assert c['label'] in testlabels and c['label'] not in trainlabels
        a=arrays(c,cap)
        for j,s in enumerate(cap['samples']):
            for name,num in [('leftShoulder',11),('rightShoulder',12),('leftElbow',13),('rightElbow',14)]:a[j,num]=s['arms'][name]
        t=norm({'frames':torch.from_numpy(a[:,audit.PICK,:].transpose(2,0,1).copy())})['frames'].unsqueeze(0)
        for name,m in models.items():
            with torch.inference_mode():p=m(t)[0].softmax(0)
            assert torch.isfinite(p).all();top=p.topk(5);r=dict(label=c['label'],variant=name,prediction=vocab[int(top.indices[0])],top5=[vocab[int(k)] for k in top.indices],target_probability=float(p[vocab.index(c['label'])]))
            if name=='BASE':assert r['prediction']==previous[c['label']]['prediction']
            rows.append(r)
        if (i+1)%20==0:print('Evaluated',i+1,'/100',flush=True)
    summaries={v:metrics([r for r in rows if r['variant']==v],testlabels) for v in models};by={(r['label'],r['variant']):r['prediction']==r['label'] for r in rows}
    paired={v:dict(gained=[l for l in testlabels if by[l,v] and not by[l,'BASE']],lost=[l for l in testlabels if by[l,'BASE'] and not by[l,v]]) for v in ['INPUT_BN','ALL_BN']}
    assert production()==before and all(sha(f)==protocol['hashes'][str(f)] for f in files)
    assert all(sha(OUT/(v+'_statistics.pt'))==h for v,h in stats_hashes.items())
    write(OUT/'results.json',dict(summary=summaries,paired=paired,predictions=rows,evaluation_capture_hashes=testhashes,learnable_parameters_unchanged=True,production_unchanged=True,calibration_eval_label_overlap=0));print(json.dumps(dict(summary=summaries,paired=paired),indent=2),flush=True)
if __name__=='__main__':main()
