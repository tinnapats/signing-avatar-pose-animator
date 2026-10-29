"""Real paired ST-GCN inference of full source clips with synthetic recovered gaps."""
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
import masked_hand_recovery as recovery
from extend_original60 import metrics
trial=recovery.trial;HERE=trial.HERE;SRC=recovery.OUT;OUT=HERE/'masked_hand_recognition_20261005';audit=trial.base.audit
def main():
    assert not OUT.exists();trial.verify();OUT.mkdir();windows=trial.read(SRC/'windows.json')
    labels=sorted({w['label'] for w in windows});hashes={str(p):trial.base.sha(p) for p in [SRC/'windows.json',SRC/'protocol.json',HERE/'wlasl_stgcn.zip',HERE/'wlasl_metadata.zip',HERE/'run_audit.py',HERE/'masked_hand_recovery.py',Path(__file__)]}
    trial.write(OUT/'protocol.json',dict(labels=labels,hashes=hashes,methods=['UNMASKED']+recovery.METHODS,scope='Full raw source clips before renderer, replace only selected masked hand samples by endpoint-only recovery. All other body/hand coordinates unchanged. Released ST-GCN WLASL2000 strict load, same27 xy and shoulder normalization, no retraining. Keep original frame order/rate/duration; no resampling. Not final avatar RGB.',metrics='Accuracy,macro P/R/F1 over target labels within each gap group; zero division0; unrestricted2000 predictions. One modified full clip per selected gap window; repeated words/windows not independent clips. UNMASKED predictions cached per source word, other methods real forward per window.',production=trial.base.production()))
    model,norm,vocab=audit.load_model();assert all(l in vocab for l in labels)
    db=trial.base.ROOT/'SLclean.sqlite3';clips=audit.pipeline.discover_clips(db);source={};baseline={};rows=[];calls=0
    def predict(a,label):
        nonlocal calls
        t=torch.from_numpy(np.asarray(a,dtype=np.float32)[:,audit.PICK,:].transpose(2,0,1).copy());t=norm({'frames':t})['frames'].unsqueeze(0);assert torch.isfinite(t).all()
        start=time.perf_counter()
        with torch.inference_mode():p=model(t)[0].softmax(0)
        elapsed=(time.perf_counter()-start)*1000;calls+=1;top=p.topk(5)
        return dict(prediction=vocab[int(top.indices[0])],top5=[vocab[int(i)] for i in top.indices],target_probability=float(p[vocab.index(label)]),tensor_sha256=hashlib.sha256(t.numpy().tobytes()).hexdigest(),runtime_ms=elapsed)
    for label in labels:
        df=audit.pipeline.read_clip_csv(audit.pipeline.resolve_selected_paths(clips,db,[label+'.csv'],'')[0]);a=audit.csv_array(df);ids=sorted(df.frame.unique());source[label]=(a,{int(x):i for i,x in enumerate(ids)});baseline[label]=predict(a,label)
    for wi,w in enumerate(windows):
        label=w['label'];a,indices=source[label];frameids=[indices[i] for i in range(w['start']+1,w['start']+w['gap']+1)];offset=33 if w['side']=='left' else 54
        assert np.allclose(a[frameids,offset:offset+21,:],np.array(w['hidden'])[:,:,:2],atol=1e-7)
        base=dict(label=label,gap=w['gap'],side=w['side'],start=w['start'],frames=len(a))
        rows.append(dict(**base,method='UNMASKED',**baseline[label]))
        for method in recovery.METHODS:
            b=a.copy();pred=recovery.predict(np.array(w['left']),np.array(w['right']),w['gap'],method)
            b[frameids,offset:offset+21,:]=pred[:,:,:2]
            keep=np.ones(a.shape,dtype=bool);keep[frameids,offset:offset+21,:]=False;assert np.array_equal(a[keep],b[keep])
            rows.append(dict(**base,method=method,**predict(b,label)))
        if (wi+1)%20==0 or wi+1==len(windows):
            trial.write(OUT/'progress.json',dict(completed=wi+1,predictions=rows,actual_forward_passes=calls));print('Scored windows',wi+1,'/',len(windows),flush=True)
    summary={};paired={}
    for gap in [3,6,12,'all']:
        rr=[r for r in rows if gap=='all' or r['gap']==gap];ls=sorted({r['label'] for r in rr});summary[str(gap)]={};paired[str(gap)]={}
        for method in ['UNMASKED']+recovery.METHODS:
            mm=[r for r in rr if r['method']==method];summary[str(gap)][method]=dict(**metrics(mm,ls),accuracy=100*sum(r['prediction']==r['label'] for r in mm)/len(mm),top5=100*sum(r['label'] in r['top5'] for r in mm)/len(mm),words=len(ls))
            paired[str(gap)][method]=dict(gained=sum(r['prediction']==r['label'] and baseline[r['label']]['prediction']!=r['label'] for r in mm),lost=sum(r['prediction']!=r['label'] and baseline[r['label']]['prediction']==r['label'] for r in mm))
    trial.verify()
    for f,h in hashes.items():assert trial.base.sha(Path(f))==h
    result=dict(summary=summary,paired=paired,predictions=rows,unique_unmasked=[dict(label=l,**baseline[l]) for l in labels],actual_forward_passes=calls,production_unchanged=True)
    trial.write(OUT/'results.json',result);print(json.dumps(dict(summary=summary,paired=paired,actual_forward_passes=calls),indent=2),flush=True)
if __name__=='__main__':main()
