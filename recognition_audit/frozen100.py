"""Prospective frozen 100-label comparison; isolated from production."""
import copy,json,random,sys,zipfile,math
from pathlib import Path
from unittest.mock import patch
import numpy as np
import bounded_forearm200 as trial
import repair_parts200 as parts
from equal_duration_trial import arrays
from novel_words30 import load_predictor
base=trial.base;HERE=trial.HERE;OUT=HERE/'frozen100_20261005'
VS=['BASE','SIZE_ONLY','BOUNDED_SIZE']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def prepare():
    assert not OUT.exists(),'Do not replace frozen experiment'
    vocab=set(v['gloss'] for v in json.loads(zipfile.ZipFile(HERE/'wlasl_metadata.zip').read('splits/asl2000.json')))
    seen=set();hashes={}
    names={'label','phrase','actual','gloss','labels_evaluated','labels','original_words','added_words','review','words'}
    def collect(x,k=''):
        if isinstance(x,str) and k in names:
            s=x.split('__')[0]
            if s in vocab:seen.add(s)
        elif isinstance(x,dict):
            for key,val in x.items():
                if key not in {'pool','excluded','available','vocabulary','top5','prediction','predicted','hashes','source_hashes','production','production_before'}:collect(val,key)
        elif isinstance(x,list):
            for val in x:collect(val,k)
    files={HERE/'results.json',HERE/'inputs.json'}
    # Reports/manifests and prediction artifacts only; avoid environments and
    # enormous raw frame files. Record exactly which evidence was inspected.
    for folder in [HERE]+[p for p in HERE.iterdir() if p.is_dir() and not p.name.endswith('_env') and p.name!='__pycache__']:
        for f in folder.glob('*.json'):
            if any(t in f.name for t in ['protocol','provenance','manifest','result','diagnostic','case_index']):files.add(f)
    for f in sorted(files):
        if f.exists():collect(read(f));hashes[str(f)]=base.sha(f)
    available={Path(x).stem for x in base.audit.pipeline.discover_clips(base.ROOT/'SLclean.sqlite3')}
    pool=sorted((available&vocab)-seen);assert len(pool)>=100
    labels=random.Random(20261005100).sample(pool,100)
    OUT.mkdir();(OUT/'cases').mkdir()
    db=base.ROOT/'SLclean.sqlite3';stat=db.stat()
    frozen=[Path(__file__),HERE/'bounded_forearm200.py',HERE/'run_audit.py',HERE/'repair_parts200.py',HERE/'wlasl_stgcn.zip',HERE/'wlasl_metadata.zip']
    protocol=dict(labels=labels,variants=VS,seed=20261005100,excluded=sorted(seen),pool=pool,exclusion_files=hashes,production=base.production(),code_hashes={str(f):base.sha(f) for f in frozen},database=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns),
      rules='100 labels absent from inspected historical manifests/results; not guaranteed absent from ST-GCN training or all unrecorded experiments. No replacement by model outcome. Same no_hold_no_smoothing, payload60FPS sampled30FPS, original durations, frozen bounded15deg/90degsec/EMA.25. BASE production scale anchor; SIZE_ONLY scale0 anchor1; BOUNDED_SIZE same size plus fixed wrist rotation. All three actual Canvas captures. No hyperparameter selection after new scores. Macro over100 true labels, 2000 output classes, zero_division0. Primary paired against BASE (two comparisons); size vs bounded secondary. Not RGB semantic or ASL user validation.')
    write(OUT/'protocol.json',protocol);print('Excluded',len(seen),'eligible',len(pool),'frozen100',flush=True)
    index=[]
    for i,label in enumerate(labels):
        with patch.object(base.audit.pipeline,'repair_hand_tracks',parts.make_repair(True,False)):
            p=base.audit.pipeline.build_payload(db,[label+'.csv'],'',30,513,513,0,0,2,1.2,2,hand_gaussian_sigma=0,hand_gaussian_radius=0)
        c=dict(label=label,payload=p,frameIndices=list(range(0,len(p['frames']),2)))
        tracks,_=trial.targets(c)
        for mode in VS:
            d=copy.deepcopy(c);d.update(method=mode,phrase=label+'__'+mode)
            if mode=='BOUNDED_SIZE':
                for j,f in enumerate(d['payload']['frames']):
                    pose={k['part']:k for k in f['pose']['keypoints']}
                    for s in ['left','right']:pose[s+'Wrist']['position']['x'],pose[s+'Wrist']['position']['y']=tracks[s][j]['wrist']
            path=OUT/'cases'/f'{len(index):03d}.json';write(path,d)
            index.append(dict(file=str(path),label=label,phrase=d['phrase'],method=mode,sha256=base.sha(path)))
        write(OUT/'case_index.json',index)
        if (i+1)%10==0:print('Prepared',i+1,'/100',flush=True)
    assert base.production()==protocol['production']
    write(OUT/'prepared.json',dict(cases=len(index),production_unchanged=True))
def score():
    p=read(OUT/'protocol.json');assert base.production()==p['production']
    assert all(base.sha(Path(f))==h for f,h in p['code_hashes'].items())
    db=(base.ROOT/'SLclean.sqlite3').stat();assert (db.st_size,db.st_mtime_ns)==(p['database']['size'],p['database']['mtime_ns'])
    _,predict=load_predictor();rows=[];geo=[]
    descriptors=read(OUT/'case_index.json');assert len(descriptors)==300
    for i,d in enumerate(descriptors):
        path=Path(d['file']);assert base.sha(path)==d['sha256'];c=read(path);cap=read(OUT/'renderer'/('capture_'+path.stem+'.json'))
        assert cap['phrase']==c['phrase'];a=arrays(c,cap);widths=[];wr={s:[] for s in ['left','right']};gap=[];native=0;detailed=0;switches=0;previous={}
        for j,(idx,sample) in enumerate(zip(c['frameIndices'],cap['samples'])):
            pose={k['part']:k for k in c['payload']['frames'][idx]['pose']['keypoints']};b=sample['arms'];width=max(np.linalg.norm(np.array(b['leftShoulder'])-b['rightShoulder']),1e-6);widths.append(width)
            for name,num in [('leftShoulder',11),('rightShoulder',12),('leftElbow',13),('rightElbow',14)]:a[j,num]=b[name]
            for name,q in b.items():assert np.max(np.abs(np.array(q)-trial.xy(pose[name])))<1e-6
            for side in wr:
                wr[side].append(b[side+'Wrist']);pts=sample['hands'][side];state=bool(pts);detailed+=state;native+=bool(sample['native'][side])
                switches+=side in previous and previous[side]!=state;previous[side]=state
                if '0' in pts and pose[side+'Wrist']['score']>=.2:gap.append(float(np.linalg.norm(np.array(pts['0'])-b[side+'Wrist'])/width))
        rough=[]
        for q in wr.values():
            q=np.array(q);rough.extend(np.linalg.norm(q[2:]-2*q[1:-1]+q[:-2],axis=1)/np.array(widths[1:-1]))
        rows.append(predict(a,c['label'],c['method']));geo.append(dict(label=c['label'],variant=c['method'],gap=float(np.mean(gap)) if gap else None,wrist_second_difference=float(np.mean(rough)),native_hand_frames=native,detailed_hand_frames=detailed,switches=int(switches)))
        if (i+1)%30==0:write(OUT/'score_progress.json',dict(completed=i+1));print('Scored',i+1,'/300',flush=True)
    summaries={v:base.metrics([r for r in rows if r['variant']==v],p['labels']) for v in VS};by={(r['label'],r['variant']):r['prediction']==r['label'] for r in rows};paired={}
    for a,b in [('BASE','SIZE_ONLY'),('BASE','BOUNDED_SIZE'),('SIZE_ONLY','BOUNDED_SIZE')]:
        g=[l for l in p['labels'] if by[l,b] and not by[l,a]];loss=[l for l in p['labels'] if by[l,a] and not by[l,b]];n=len(g)+len(loss)
        prob=min(1.,2*sum(math.comb(n,k) for k in range(min(len(g),len(loss))+1))/2**n) if n else 1.
        paired[a+'->'+b]=dict(gained=g,lost=loss,exact_p=prob,bonferroni_two=min(1,2*prob) if a=='BASE' else None)
    assert base.production()==p['production']
    write(OUT/'results.json',dict(summary=summaries,paired=paired,predictions=rows,geometry=geo,production_unchanged=True));print(json.dumps(dict(summary=summaries,paired=paired),indent=2),flush=True)
if __name__=='__main__':{'prepare':prepare,'score':score}[sys.argv[1]]()
