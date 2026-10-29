"""Development ablation only: frozen200, unchanged ST-GCN; no production edits."""
from collections import Counter
from contextlib import ExitStack
import gc
import json
import math
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import run_audit as audit
from extend_original60 import metrics
from repeat_original30 import production,sha

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
SOURCE=HERE/'original_method200_20260930';OUT=HERE/'processing_diagnosis200'
VARIANTS=['baseline','no_hand_smoothing','no_body_face_smoothing','no_articulation','no_gap_handling']
CHAINS=[[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
EDGES=[(a,b) for c in CHAINS for a,b in zip(c,c[1:])]

def geometry(frames):
    cvs=[];angles=[];hand_frames=0
    for side in ['left','right']:
        points=[]
        for f in frames:
            h=f.get('hands',{}).get(side,{})
            if not h.get('observed') or h.get('score',0)<=0 or len(h.get('keypoints',[]))!=21:continue
            if any(k.get('score',0)<=0 for k in h['keypoints']):continue
            p=np.array([[k['position'].get(a,0) for a in ['x','y','z']] for k in h['keypoints']]);points.append(p)
        if not points:continue
        p=np.array(points);hand_frames+=len(p)
        lengths=np.stack([np.linalg.norm(p[:,b]-p[:,a],axis=1) for a,b in EDGES],axis=1)
        med=np.median(lengths,axis=0);valid=med>1e-8
        cvs.extend(((np.percentile(lengths,95,axis=0)-np.percentile(lengths,5,axis=0))[valid]/med[valid]).tolist())
        for c in CHAINS:
            for a,b,d in zip(c,c[1:],c[2:]):
                u=p[:,b]-p[:,a];v=p[:,d]-p[:,b];den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1);ok=den>1e-8
                angles.extend(np.degrees(np.arccos(np.clip(np.sum(u[ok]*v[ok],axis=1)/den[ok],-1,1))).tolist())
    return dict(observed_hand_frames=hand_frames,mean_bone_span=float(np.mean(cvs)) if cvs else None,
      bend_over115_fraction=float(np.mean(np.array(angles)>115.0001)) if angles else None)

def main():
    assert not (OUT/'results.json').exists(),'Finished results already exist'
    before=production();cases=json.loads((SOURCE/'inputs.json').read_text());saved=json.loads((SOURCE/'results.json').read_text())
    assert len(cases)==200;labels=[c['phrase'] for c in cases];old={(r['label'],r['variant']):r for r in saved['predictions']}
    OUT.mkdir(exist_ok=True);protocol=dict(variants=VARIANTS,labels=labels,production_before=before,
      source_inputs_hash=sha(SOURCE/'inputs.json'),source_results_hash=sha(SOURCE/'results.json'),
      note='Known200 development set; one component disabled per candidate, except gap group includes repair+taper. Not independent validation. No renderer changes or model training.')
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text())==protocol,'Protocol or production changed'
    else:(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    network,normalize,vocabulary=audit.load_model();assert len(vocabulary)==2000
    assert all(l in vocabulary for l in labels)
    def predict(data,label):
        t=torch.from_numpy(np.asarray(data,dtype=np.float32)[:,audit.PICK,:].transpose(2,0,1).copy())
        t=normalize({'frames':t})['frames'].unsqueeze(0);assert torch.isfinite(t).all()
        with torch.inference_mode():p=network(t)[0].softmax(0)
        top=p.topk(5)
        return dict(label=label,prediction=vocabulary[int(top.indices[0])],top5=[vocabulary[int(i)] for i in top.indices],target_probability=float(p[vocabulary.index(label)]))
    predict(cases[0]['native'],labels[0])
    transitions={}
    for a,b in [('source_native','source_30fps'),('source_30fps','processed_30fps'),('processed_30fps','rendered_hands_payload_body')]:
        transitions[a+' -> '+b]={kind:[] for kind in ['correct_to_wrong','wrong_to_correct','both_correct','both_wrong']}
        for l in labels:
            x=old[l,a]['prediction']==l;y=old[l,b]['prediction']==l
            kind=('both_correct' if y else 'correct_to_wrong') if x else ('wrong_to_correct' if y else 'both_wrong')
            transitions[a+' -> '+b][kind].append(l)
    (OUT/'stage_transitions.json').write_text(json.dumps(transitions,indent=2),encoding='utf-8')
    predictions=[];checks=[];completed=0
    if (OUT/'progress.json').exists():
        progress=json.loads((OUT/'progress.json').read_text());completed=progress['completed'];predictions=progress['predictions']
        assert len(predictions)==completed*len(VARIANTS)
        for i,r in enumerate(predictions):
            assert r['label']==labels[i//len(VARIANTS)] and r['variant']==VARIANTS[i%len(VARIANTS)]
            if r['variant']=='baseline':assert all(r[k]==old[r['label'],'processed_30fps'][k] for k in ['prediction','top5','target_probability'])
        print(f'Resuming from completed {completed}/200',flush=True)
    for ci,c in enumerate(cases):
        l=c['phrase'];check=dict(label=l)
        for name in ['native','source30']:
            a=np.asarray(c[name],dtype=np.float32);assert np.isfinite(a).all()
            shoulder=np.linalg.norm(a[:,11]-a[:,12],axis=1)
            check[name]=dict(frames=len(a),zero_shoulder_frames=int(np.sum(shoulder<1e-8)),
              missing_selected_fraction=float(np.mean(np.all(a[:,audit.PICK]==0,axis=2))))
        checks.append(check)
        if ci<completed:continue
        for variant in VARIANTS:
            kw={};sigma=1.2;radius=2
            if variant=='no_hand_smoothing':kw.update(hand_gaussian_sigma=0,hand_gaussian_radius=0)
            if variant=='no_body_face_smoothing':sigma=0;radius=0
            if variant=='no_articulation':kw['articulate_hand_joints']=False
            with ExitStack() as stack:
                if variant=='no_gap_handling':
                    stack.enter_context(patch.object(audit.pipeline,'repair_hand_tracks',lambda frames,**kwargs:frames))
                    stack.enter_context(patch.object(audit.pipeline,'taper_hand_detection_edges',lambda frames,**kwargs:frames))
                payload=audit.pipeline.build_payload(ROOT/'SLclean.sqlite3',[l+'.csv'],'',30,513,513,0,0,2,sigma,radius,**kw)
            frames=payload['frames'][::2];assert len(frames)==len(c['source30'])
            if variant=='baseline':assert payload['frames']==c['payload']['frames'],'Rebuilt baseline differs: '+l
            result=predict(audit.frame_array(frames),l);result.update(variant=variant,geometry=geometry(frames))
            if variant=='baseline':
                prior=old[l,'processed_30fps'];assert all(result[k]==prior[k] for k in ['prediction','top5','target_probability'])
            predictions.append(result)
        print(f'{ci+1}/200 {l}: '+str([(r['variant'],r['prediction']) for r in predictions[-len(VARIANTS):]]),flush=True)
        if (ci+1)%20==0:(OUT/'progress.json').write_text(json.dumps(dict(completed=ci+1,predictions=predictions)),encoding='utf-8')
    summaries={}
    for v in VARIANTS:
        rows=[r for r in predictions if r['variant']==v];m=metrics(rows,labels)
        gains=[r['label'] for r in rows if r['prediction']==r['label'] and old[r['label'],'processed_30fps']['prediction']!=r['label']]
        losses=[r['label'] for r in rows if r['prediction']!=r['label'] and old[r['label'],'processed_30fps']['prediction']==r['label']]
        n=len(gains)+len(losses);p=min(1,2*sum(math.comb(n,k) for k in range(min(len(gains),len(losses))+1))/2**n) if n else 1
        m.update(gained_words=gains,lost_words=losses,mcnemar_exact_p=p,
          mean_bone_span=float(np.mean([r['geometry']['mean_bone_span'] for r in rows if r['geometry']['mean_bone_span'] is not None])),
          mean_bend_over115_fraction=float(np.mean([r['geometry']['bend_over115_fraction'] for r in rows if r['geometry']['bend_over115_fraction'] is not None])))
        summaries[v]=m
    assert production()==before;assert sha(SOURCE/'results.json')==protocol['source_results_hash']
    result=dict(summaries=summaries,predictions=predictions,input_checks=checks,baseline_exact=True,production_unchanged=True,
      caveat='Exploratory200. McNemar unadjusted for multiple comparisons. Geometry constraints are imposed, not ASL correctness. Before-renderer only.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()
