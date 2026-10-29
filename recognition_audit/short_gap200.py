
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import sys,json,inspect,copy,math
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import repair_parts200 as parts
from equal_duration_trial import arrays
base=parts.base;HERE=base.HERE;SRC=HERE/'slgcn200_20261002';OUT=HERE/'short_gap200_20261005'
def read(p):return _local_json_loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def smooth_repair():
    code=inspect.getsource(parts.ORIGINAL);needle='                        t = float(step) / float(gap + 1)'
    assert code.count(needle)==1
    code=code.replace(needle,needle+'\n                        t = t*t*(3.0-2.0*t)  # isolated audit only')
    ns=dict(parts.ORIGINAL.__globals__);exec(compile(code,'audit_smooth_short_gap','exec'),ns);fn=ns['repair_hand_tracks']
    return lambda frames,max_gap_frames,fade_frames:fn(frames,max_gap_frames=max_gap_frames,fade_frames=0)
def verify():
    p=read(OUT/'protocol.json');assert p['production']==base.production()
    for f,h in p['hashes'].items():assert base.sha(Path(f))==h
    return p
def prepare():
    assert not OUT.exists();OUT.mkdir();(OUT/'cases').mkdir();prior=read(SRC/'protocol.json');assert prior['production']==base.production()
    files=[SRC/'inputs.json',SRC/'renderer/rendered.json',SRC/'results.json',base.ROOT/'export_pose_animator_sequence.py',HERE/'repair_parts200.py',HERE/'wlasl_stgcn.zip',Path(__file__)]
    write(OUT/'protocol.json',dict(labels=prior['labels'],production=base.production(),hashes={str(f):base.sha(f) for f in files},rules='Baseline no_hold_no_smoothing already uses LINEAR short gap repair; explicit Linear control must equal baseline. Smoothstep changes only t within short-gap branch. No changes to eligibility, confidence, long gaps, Gaussian/body, articulation, stabilization, FPS, scale/anchor.200 known development words30FPS; same released ST-GCN; primary post detailed-rendered hands+payload body, secondary pre. Exact cached captures reused ONLY for exact identical payload frames. Not RGB, not independent holdout.'))
    # Synthetic test makes change observable without relying on classification.
    def toy():return [dict(hands={s:dict(keypoints=[dict(score=1. if i in [0,3] else 0.,position=dict(x=float(i),y=float(i),z=0.)) for _ in range(21)]) for s in ['left','right']}) for i in range(4)]
    old=parts.make_repair(True,False)(toy(),4,3);new=smooth_repair()(toy(),4,3)
    assert old[1]['hands']['left']['keypoints'][0]['position']['x']==1.
    assert np.isclose(new[1]['hands']['left']['keypoints'][0]['position']['x'],7/9)
    index=[];render=[];checks=[]
    for i,c in enumerate(read(SRC/'inputs.json')):
        label=c['label']
        with patch.object(base.audit.pipeline,'repair_hand_tracks',parts.make_repair(True,False)):
            linear=base.audit.pipeline.build_payload(base.ROOT/'SLclean.sqlite3',[label+'.csv'],'',30,513,513,0,0,2,1.2,2,hand_gaussian_sigma=0,hand_gaussian_radius=0)
        assert linear['frames']==c['payload']['frames'],'Linear/baseline mismatch '+label
        with patch.object(base.audit.pipeline,'repair_hand_tracks',smooth_repair()):
            payload=base.audit.pipeline.build_payload(base.ROOT/'SLclean.sqlite3',[label+'.csv'],'',30,513,513,0,0,2,1.2,2,hand_gaussian_sigma=0,hand_gaussian_radius=0)
        assert len(payload['frames'])==len(linear['frames']);changed=0
        for a,b in zip(linear['frames'],payload['frames']):
            assert a['pose']==b['pose'] and a.get('face')==b.get('face')
            for side in ['left','right']:
                assert a['hands'][side]['score']==b['hands'][side]['score']
            changed+=a!=b
        test=dict(label=label,phrase=c['phrase'],payload=payload,frameIndices=c['frameIndices']);dest=OUT/'cases'/f'{i:03d}.json';write(dest,test)
        index.append(dict(file=str(dest),label=label,sha256=base.sha(dest)));checks.append(dict(label=label,linear_exact=True,changed_frames=changed))
        if changed:render.append(test)
        if (i+1)%20==0:print('Prepared',i+1,'/200 changed clips',len(render),flush=True)
    write(OUT/'case_index.json',index);write(OUT/'render_inputs.json',render);write(OUT/'checks.json',checks);verify();print('Only changed clips require rendering:',len(render),flush=True)
def score():
    p=verify();assert not (OUT/'results.json').exists();cases=read(SRC/'inputs.json');oldcaps={c['phrase']:c for c in read(SRC/'renderer/rendered.json')};changed=read(OUT/'render_inputs.json');newcaps={c['phrase']:c for c in read(OUT/'renderer/rendered.json')} if changed else {};assert len(newcaps)==len(changed)
    oldresults={(r['label'],r['stage']):r for r in read(SRC/'results.json')['predictions'] if r['model']=='ST-GCN'}
    model,norm,vocab=base.audit.load_model();rows=[]
    def predict(a,label):
        t=torch.from_numpy(a[:,base.audit.PICK,:].transpose(2,0,1).copy());t=norm({'frames':t})['frames'].unsqueeze(0);assert torch.isfinite(t).all()
        with torch.inference_mode():p=model(t)[0].softmax(0)
        top=p.topk(5);return dict(prediction=vocab[int(top.indices[0])],top5=[vocab[int(i)] for i in top.indices],target_probability=float(p[vocab.index(label)]))
    for i,(c,d) in enumerate(zip(cases,read(OUT/'case_index.json'))):
        assert base.sha(Path(d['file']))==d['sha256'];candidate=read(Path(d['file']));label=c['label'];assert label==candidate['label']
        cap=newcaps.get(c['phrase'])
        if cap is None:assert candidate['payload']['frames']==c['payload']['frames'];cap=oldcaps[c['phrase']]
        assert len(cap['samples'])==len(oldcaps[c['phrase']]['samples'])
        for variant,case,capture in [('BASE_LINEAR',c,oldcaps[c['phrase']]),('SMOOTHSTEP',candidate,cap)]:
            for stage,a in [('pre',base.audit.frame_array([case['payload']['frames'][j] for j in case['frameIndices']])),('post',arrays(case,capture))]:
                r=predict(a,label)
                if variant=='BASE_LINEAR':assert all(r[k]==oldresults[label,stage][k] for k in r),'Baseline prediction mismatch '+label
                rows.append(dict(label=label,variant=variant,stage=stage,**r))
        if (i+1)%20==0:write(OUT/'progress.json',dict(completed=i+1,predictions=rows));print('Scored',i+1,'/200',flush=True)
    summary={};paired={}
    for stage in ['pre','post']:
        for variant in ['BASE_LINEAR','SMOOTHSTEP']:
            rr=[r for r in rows if r['stage']==stage and r['variant']==variant];summary[stage+'_'+variant]=dict(**base.metrics(rr,p['labels']),accuracy=100*sum(r['prediction']==r['label'] for r in rr)/200)
        by={(r['label'],r['variant']):r for r in rows if r['stage']==stage};g=[];l=[]
        for label in p['labels']:
            a=by[label,'BASE_LINEAR']['prediction']==label;b=by[label,'SMOOTHSTEP']['prediction']==label
            if b and not a:g.append(label)
            if a and not b:l.append(label)
        n=len(g)+len(l);prob=min(1.,2*sum(math.comb(n,k) for k in range(min(len(g),len(l))+1))/2**n) if n else 1.
        paired[stage]=dict(gained=g,lost=l,exact_mcnemar_p=prob)
    verify();write(OUT/'results.json',dict(summary=summary,paired=paired,predictions=rows,linear_equals_baseline_200=True,cached_render_reuse=200-len(newcaps),new_render_clips=len(newcaps),production_unchanged=True));print(json.dumps(dict(summary=summary,paired=paired),indent=2),flush=True)
if __name__=='__main__':{'prepare':prepare,'score':score}[sys.argv[1]]()
