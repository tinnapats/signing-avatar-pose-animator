"""Frozen input-only diagnostic confirmation; never changes production."""
import json
from collections import Counter
import numpy as np
import scale_anchor200 as old
import visibility_audit200 as visibility
from novel_words30 import load_predictor

base=old.base; HERE=old.HERE; SRC=old.SRC
OUT=HERE/'renderer_components_confirm_20261005'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    assert not OUT.exists(), 'Never overwrite completed or partial evidence'
    before=base.production()
    cases=read(SRC/'inputs.json'); caps={x['phrase']:x for x in read(SRC/'renderer/rendered.json')}
    saved={(x['label'],x['stage']):x for x in read(SRC/'results.json')['predictions'] if x['model']=='ST-GCN'}
    labels=[c['label'] for c in cases];assert len(labels)==len(set(labels))==200
    files=[SRC/'inputs.json',SRC/'renderer/rendered.json',SRC/'results.json',HERE/'wlasl_stgcn.zip',HERE/'wlasl_metadata.zip',HERE/'scale_anchor200.py',HERE/'visibility_audit200.py',HERE/'equal_duration_trial.py',HERE/'novel_words30.py',HERE/'run_audit.py',HERE/'upstream_st_gcn.py',HERE/'upstream_pose_transforms.py',HERE/'upstream_graph_utils.py',__import__('pathlib').Path(__file__)]
    hashes={str(p):base.sha(p) for p in files}
    variants=['rendered','payload_wrist','payload_size','payload_wrist_size','payload_shape','observed_only']
    OUT.mkdir()
    write(OUT/'protocol.json',dict(labels=labels,production=before,hashes=hashes,variants=variants,fps=30,
      rules='Same development200 and duration, model and normalization. Reuse frozen actual captures. Input-only diagnostic coordinates, not new rendered Avatar. Wrist=B, size=C, both=A from verified affine decomposition. Shape: verify all drawn points equal payload wrist-relative shape times a single positive isotropic scale, max residual <1e-6; if so retain exact baseline float32 to avoid numerical artifacts. This is an algebraic identity control, not evidence that all 3D joints or earlier stages are correct. Observed-only masks synthetic detailed hands, never invents native SVG joints. No label-specific routing or tuning.'))
    _,predict=load_predictor();rows=[];checks=[];coverage=Counter()
    for i,c in enumerate(cases):
        cap=caps[c['phrase']];arr,err,scales=old.variants(c,cap)
        # variants() verifies every rendered joint is a translated uniformly
        # scaled copy of payload XY. Restoring that shape is exactly identity.
        assert err<1e-6
        stats,keep=visibility.stats(c,cap);coverage.update(stats['counts'])
        data=dict(rendered=arr['D'],payload_wrist=arr['B'],payload_size=arr['C'],payload_wrist_size=arr['A'],payload_shape=arr['D'].copy(),observed_only=visibility.mask(arr['D'],keep))
        for name,a in data.items():
            assert np.array_equal(a[:,:33],arr['D'][:,:33]) and a.shape==arr['D'].shape
            r=predict(a,c['label'],name)
            if name=='rendered':assert r['prediction']==saved[c['label'],'post']['prediction']
            rows.append(r)
        checks.append(dict(label=c['label'],shape_residual=err,shape_exact=np.array_equal(data['payload_shape'],arr['D']),coverage=stats))
        if (i+1)%20==0:
            write(OUT/'progress.json',dict(completed=i+1,predictions=rows));print('Scored',i+1,'/200',flush=True)
    summary={};paired={};d={(x['label'],x['variant']):x['prediction'] for x in rows}
    for v in variants:
        rr=[r for r in rows if r['variant']==v];summary[v]=base.metrics(rr,labels)
        paired[v]=dict(gained=[l for l in labels if d[l,v]==l and d[l,'rendered']!=l],lost=[l for l in labels if d[l,v]!=l and d[l,'rendered']==l])
    assert base.production()==before
    assert all(base.sha(p)==hashes[str(p)] for p in files)
    write(OUT/'results.json',dict(summary=summary,paired=paired,predictions=rows,checks=checks,coverage=dict(coverage),production_unchanged=True,source_hashes_unchanged=True,inferences=len(rows)))
    print(json.dumps(dict(summary=summary,paired=paired,coverage=dict(coverage)),indent=2),flush=True)

if __name__=='__main__':main()
