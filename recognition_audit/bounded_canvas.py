import copy,json,sys
from pathlib import Path
import numpy as np
import bounded_forearm200 as trial
from equal_duration_trial import arrays
from novel_words30 import load_predictor
base=trial.base;HERE=trial.HERE;OUT=HERE/'bounded_canvas_20261005'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def prepare():
    assert not OUT.exists();OUT.mkdir();(OUT/'cases').mkdir()
    labels=['parents','remote control','chicken','sofa','cough','class']
    write(OUT/'protocol.json',dict(labels=labels,production=base.production(),selection='Prior visual diagnostics parents/remote control/chicken plus all three changed top1 correctness outcomes between size-only and bounded+size: sofa/cough/class. Deliberate diagnostic sample, not representative or holdout.',source_hash=base.sha(trial.OUT/'results.json'),modes=['SIZE_ONLY','BOUNDED_SIZE']))
    cases=read(trial.SRC/'inputs.json');index=[]
    for c in cases:
        if c['label'] not in labels:continue
        tracks,_=trial.targets(c)
        for mode in ['SIZE_ONLY','BOUNDED_SIZE']:
            d=copy.deepcopy(c);d['method']=mode;d['phrase']=c['label']+'__'+mode
            if mode=='BOUNDED_SIZE':
                for i,f in enumerate(d['payload']['frames']):
                    p={k['part']:k for k in f['pose']['keypoints']}
                    for s in ['left','right']:
                        p[s+'Wrist']['position']['x'],p[s+'Wrist']['position']['y']=tracks[s][i]['wrist']
            path=OUT/'cases'/f'{len(index):02d}.json';write(path,d)
            index.append(dict(file=str(path),phrase=d['phrase'],label=c['label']))
    write(OUT/'case_index.json',index);assert base.production()==read(OUT/'protocol.json')['production']
def score():
    _,predict=load_predictor();p=read(OUT/'protocol.json');rows=[];maxerr=0
    desc={d['phrase']:d for d in read(OUT/'case_index.json')}
    originals={c['label']:c for c in read(trial.SRC/'inputs.json')}
    oldcaps={c['phrase']:c for c in read(trial.SRC/'renderer/rendered.json')}
    expected={ (r['label'],'BOUNDED_SIZE'):r['prediction'] for r in read(trial.OUT/'results.json')['predictions'] if r['variant']=='BOUNDED_PAYLOAD_SIZE'}
    for r in read(HERE/'renderer_components_confirm_20261005/results.json')['predictions']:
        if r['variant']=='payload_size':expected[r['label'],'SIZE_ONLY']=r['prediction']
    for cap in read(OUT/'renderer/rendered.json'):
        c=read(Path(desc[cap['phrase']]['file']));orig=originals[c['label']];a=arrays(c,cap)
        control=trial.prior.variants(orig,oldcaps[orig['phrase']])[0]['C'];tracks,_=trial.targets(orig)
        for i,(sample,j) in enumerate(zip(cap['samples'],c['frameIndices'])):
            pose={k['part']:k for k in c['payload']['frames'][j]['pose']['keypoints']}
            for name,num in [('leftShoulder',11),('rightShoulder',12),('leftElbow',13),('rightElbow',14)]:
                a[i,num]=sample['arms'][name]
            for name,q in sample['arms'].items():assert np.max(np.abs(np.array(q)-trial.xy(pose[name])))<1e-6
            if c['method']=='BOUNDED_SIZE':
                for s,o in [('left',33),('right',54)]:
                    ids=sorted(map(int,sample['hands'][s]));t=tracks[s][j]
                    if ids and pose[s+'Wrist']['score']>=.2:control[i,o+np.array(ids)]+=np.array(t['wrist'])-t['original']
        err=float(np.max(np.abs(a-control)));maxerr=max(maxerr,err);assert err<1e-4
        r=predict(a,c['label'],c['method']);assert r['prediction']==expected[c['label'],c['method']];rows.append(r)
    assert len(rows)==12 and base.production()==p['production'];assert base.sha(trial.OUT/'results.json')==p['source_hash']
    write(OUT/'results.json',dict(predictions=rows,max_float32_error=maxerr,all_top1_match=True,production_unchanged=True));print(rows,flush=True)
if __name__=='__main__':{'prepare':prepare,'score':score}[sys.argv[1]]()
