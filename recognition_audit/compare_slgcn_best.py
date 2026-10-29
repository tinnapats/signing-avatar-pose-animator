"""Change only recognizer on frozen best-pipeline data; no production change."""
import io
import json
import math
import time
import zipfile
import numpy as np
import torch
import yaml
import novel_words30 as prior
from equal_duration_trial import arrays
base=prior.base;HERE=base.HERE;OUT=HERE/'slgcn_trial_20261002';DATA=HERE/'novel_words30_20261001'

def load_slgcn():
    torch.set_num_threads(4)
    with zipfile.ZipFile(OUT/'wlasl_slgcn.zip') as z:
        cfg=yaml.safe_load(z.read('wlasl/sl_gcn/config.yaml'))
        safe=[(type('UnusedCheckpointCallback',(),{}),'pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint'),
              (type('UnusedStoppingCallback',(),{}),'pytorch_lightning.callbacks.early_stopping.EarlyStopping')]
        with torch.serialization.safe_globals(safe):
            ck=torch.load(io.BytesIO(z.read('wlasl/sl_gcn/epoch=169-step=75819.ckpt')),map_location='cpu',weights_only=True)
    transforms=cfg['data']['valid_pipeline']['transforms']
    assert transforms==[{'PoseSelect':{'preset':'mediapipe_holistic_minimal_27'}},{'CenterAndScaleNormalize':{'reference_points_preset':'shoulder_mediapipe_holistic_minimal_27','scale_factor':1}}]
    ns={};exec(compile((OUT/'upstream_graph_utils.py').read_text(),str(OUT/'upstream_graph_utils.py'),'exec'),ns)
    src=(OUT/'upstream_decoupled_gcn.py').read_text().replace('from omegaconf import OmegaConf','').replace('from .graph_utils import SpatialGraph','').replace('OmegaConf.to_container(graph_args)','dict(graph_args)')
    exec(compile(src,str(OUT/'upstream_decoupled_gcn.py'),'exec'),ns)
    encoder=ns['DecoupledGCN'](in_channels=2,**cfg['model']['encoder']['params'])
    sd=ck['state_dict'];encoder.load_state_dict({k.removeprefix('model.encoder.'):v for k,v in sd.items() if k.startswith('model.encoder.')},strict=True)
    assert cfg['model']['decoder']=={'type':'fc','params':{'dropout_ratio':0}}
    decoder=torch.nn.Linear(256,2000)
    decoder.load_state_dict({k.removeprefix('model.decoder.classifier.'):v for k,v in sd.items() if k.startswith('model.decoder.classifier.')},strict=True)
    assert not [k for k in sd if not k.startswith(('model.encoder.','model.decoder.classifier.'))]
    return torch.nn.Sequential(encoder,decoder).eval(),cfg

def main():
    assert not (OUT/'protocol.json').exists(),'Do not overwrite frozen protocol'
    before=base.production()
    cases=[c for c in json.loads((DATA/'review_inputs.json').read_text()) if c['variant']=='no_hold_no_smoothing'];assert len(cases)==30
    caps={c['phrase']:c for c in json.loads((DATA/'renderer/rendered.json').read_text()) if c['variant']=='S1A1'}
    oldpre={r['label']:r['prediction'] for r in json.loads((DATA/'pre_results.json').read_text())['predictions'] if r['variant']=='no_hold_no_smoothing'}
    oldpost={r['label']:r['prediction'] for r in json.loads((DATA/'renderer_results.json').read_text())['predictions'] if r['variant']=='no_hold_no_smoothing'}
    files=[DATA/'review_inputs.json',DATA/'renderer/rendered.json',OUT/'wlasl_slgcn.zip',OUT/'upstream_decoupled_gcn.py',OUT/'upstream_graph_utils.py',HERE/'wlasl_stgcn.zip',HERE/'wlasl_metadata.zip',HERE/'compare_slgcn_best.py']
    protocol=dict(labels=[c['label'] for c in cases],pipeline='no_hold_no_smoothing; retain short fill, body Gaussian, taper, articulation, bone controls; S1A1; original durations at30FPS; no fixed68 or cap',
        primary='post-rendered detailed hands + payload body',secondary='pre-render payload',negative_control='post-rendered entire frame order shuffled, fixed seed20261002 per clip',
        models=['ST-GCN','SL-GCN'],production=before,hashes={str(p.relative_to(HERE)):base.sha(p) for p in files},
        sources=['https://openhands.ai4bharat.org/en/latest/instructions/models.html','https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/wlasl_slgcn.zip','https://github.com/AI4Bharat/OpenHands/blob/main/openhands/models/encoder/graph/decoupled_gcn.py'],
        scope='One clip/word, previously examined30 not pristine holdout; unknown checkpoint training overlap. No fine tuning, no per-word routing, no candidate selection by results. No whole-image recognition.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    baseline,norm,vocab=base.audit.load_model();candidate,cfg=load_slgcn()
    (OUT/'checkpoint_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    rows=[]
    for i,c in enumerate(cases):
        label=c['label'];pre=base.audit.frame_array([c['payload']['frames'][j] for j in c['frameIndices']]);post=arrays(c,caps[c['phrase']])
        idx=np.random.default_rng(20261002).permutation(len(post))
        for stage,a in [('pre',pre),('post',post),('shuffled',post[idx])]:
            t=torch.from_numpy(a[:,base.audit.PICK,:].transpose(2,0,1).copy());t=norm({'frames':t})['frames'].unsqueeze(0);assert torch.isfinite(t).all()
            for name,model in [('ST-GCN',baseline),('SL-GCN',candidate)]:
                started=time.perf_counter()
                with torch.inference_mode():p=model(t)[0].softmax(0)
                assert p.shape==(2000,) and torch.isfinite(p).all()
                if i==0 and stage=='post':
                    with torch.inference_mode():assert torch.equal(p,model(t)[0].softmax(0)),'Nondeterministic eval'
                top=p.topk(5);r=dict(label=label,model=name,stage=stage,prediction=vocab[int(top.indices[0])],top5=[vocab[int(k)] for k in top.indices],target_probability=float(p[vocab.index(label)]),runtime_ms=(time.perf_counter()-started)*1000,frames=len(a))
                if name=='ST-GCN' and stage!='shuffled':assert r['prediction']==(oldpre if stage=='pre' else oldpost)[label]
                rows.append(r)
        print(i+1,label,[(r['model'],r['prediction']) for r in rows[-4:-2]],flush=True)
        (OUT/'progress.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    summaries={f'{m}_{s}':base.metrics([r for r in rows if r['model']==m and r['stage']==s],protocol['labels']) for m in protocol['models'] for s in ['pre','post','shuffled']}
    comparisons={}
    for stage in ['pre','post']:
        by={(r['label'],r['model']):r for r in rows if r['stage']==stage};g=[];l=[]
        for label in protocol['labels']:
            a=by[label,'ST-GCN']['prediction']==label;b=by[label,'SL-GCN']['prediction']==label
            if b and not a:g.append(label)
            if a and not b:l.append(label)
        n=len(g)+len(l);p=min(1,2*sum(math.comb(n,k) for k in range(min(len(g),len(l))+1))/2**n) if n else 1
        comparisons[stage]=dict(gained=g,lost=l,exact_mcnemar_p=p,bonferroni_two=min(1,p*2))
    assert base.production()==before
    for p in files:assert base.sha(p)==protocol['hashes'][str(p.relative_to(HERE))]
    result=dict(summaries=summaries,paired=comparisons,predictions=rows,production_unchanged=True,original_top1_exact=True,strict_checkpoint_load=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(summaries=summaries,paired=comparisons),indent=2))

if __name__=='__main__':main()
