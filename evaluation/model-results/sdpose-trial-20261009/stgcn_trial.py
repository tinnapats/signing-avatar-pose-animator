
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import sys,json,hashlib,importlib.util,time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit')
B=A/'pixels30_20261007'; P=ROOT/'tracking3'; D=P/'rtmw_anchor_trial'
O=D/'stgcn'; O.mkdir(exist_ok=True)
assert not (O/'results.json').exists(), 'Preserve results'
sys.path.insert(0,str(A))
import run_audit as audit
spec=importlib.util.spec_from_file_location('rtm_eval',ROOT.parent/'rtm-eval.py')
ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=_local_json_loads((P/'results.json').read_text()); z=np.load(D/'predictions.npz')
variants=['raw','sd_anchor','sd_temporal','rtmw_anchor','rtmw_temporal']
protocol={'status':'registered','python':sys.version,'torch':torch.__version__,
 'checkpoint_archive_sha256':sha(A/'wlasl_stgcn.zip'),'metadata_sha256':sha(A/'wlasl_metadata.zip'),
 'checkpoint_member':'wlasl/st_gcn/epoch=212-step=95210.ckpt',
 'predictions_sha256':sha(D/'predictions.npz'),'audit_script_sha256':sha(A/'run_audit.py'),
 'converter_sha256':sha(ROOT.parent/'rtm-eval.py'),'variants':variants,'frames':len(rows),
 'input':'Final Avatar pixels -> SDPose -> identity variants; SDPose body identical in all primary variants. RTMW used ONLY to assign hands.',
 'normalization':'Existing clip-level shoulder CenterAndScaleNormalize',
 'confidence':'Existing convert(): body >=.30; hand mean and wrist >=.30. Missing -> zero as prior evaluator; no new repair.',
 'time':'Original 30FPS captured sequence; no resampling, trimming, interpolation or batch padding',
 'classes':2000,'macro_average':'Three true labels only, zero division 0; not 2000-class macro',
 'control':'RTMW raw from old cache compared with previous rtm-eval top5 and probabilities',
 'production_edited':False}
(O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
model,norm,vocab=audit.load_model(); assert len(vocab)==2000
probe=np.arange(266,dtype=np.float32).reshape(1,133,2)
expected=[0,1,2,5,6,7,8]+[91+j for j in [0,4,5,8,9,12,13,16,17,20]]+[112+j for j in [0,4,5,8,9,12,13,16,17,20]]
assert np.array_equal(ev.convert(probe,np.ones((1,133)))[:,audit.PICK],probe[:,expected])
pred={v:[] for v in variants}; controls=[]; arrays={}; logits={}
old=_local_json_loads((B/'rtm_trial_20261008/results.json').read_text())
for cid,label in [('000','abdomen'),('001','hello'),('002','love')]:
    indices=[i for i,r in enumerate(rows) if r['id']==cid]
    assert [rows[i]['frame'] for i in indices]==list(range(len(indices)))
    cache=np.load(B/'rtm_trial_20261008'/f'{cid}.npz')
    assert np.allclose(cache['times'],[rows[i]['time'] for i in indices])
    clip_inputs={v:ev.convert(z[v][indices],z[v+'_scores'][indices]) for v in variants}
    clip_inputs['rtmw_control']=ev.convert(cache['rtmw_det'],cache['rtmw_det_scores'])
    body=None
    for v,a in clip_inputs.items():
        tensor=norm({'frames':torch.from_numpy(a[:,audit.PICK].transpose(2,0,1).copy())})['frames']
        assert torch.isfinite(tensor).all()
        if v!='rtmw_control':
            if body is None: body=tensor[:,:,:7].clone()
            assert torch.equal(body,tensor[:,:,:7]), 'Body normalization changed between variants'
        arrays[f'{cid}_{v}']=tensor.numpy()
        start=time.perf_counter()
        with torch.inference_mode(): raw=model(tensor.unsqueeze(0))[0]; p=raw.softmax(0)
        top=p.topk(5); order=torch.argsort(p,descending=True).tolist()
        rr={'label':label,'prediction':vocab[int(top.indices[0])],
            'top5':[vocab[int(j)] for j in top.indices],'top5_probability':[float(j) for j in top.values],
            'target_probability':float(p[vocab.index(label)]),'target_rank':order.index(vocab.index(label))+1,
            'frames':len(indices),'runtime_ms_including_first_run':1000*(time.perf_counter()-start)}
        logits[f'{cid}_{v}']=raw.numpy()
        if v=='rtmw_control':
            before=next(r for r in old['predictions']['rtmw_det'] if r['label']==label)
            assert before['top5']==rr['top5'], (before,rr)
            assert abs(before['target_probability']-rr['target_probability'])<1e-5
            controls.append(rr)
        else: pred[v].append(rr)
        print(label,v,rr['prediction'],'target rank',rr['target_rank'],flush=True)
np.savez_compressed(O/'model_inputs.npz',**arrays)
np.savez_compressed(O/'logits.npz',**logits)
result={'summary':{v:ev.metrics(rr) for v,rr in pred.items()},'predictions':pred,'controls':controls,
        'control_reproduced':True,'frame_count':len(rows),'clip_count':3}
(O/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
protocol['status']='completed'; protocol['control_reproduced']=True
(O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
lines=['# ST-GCN: SDPose hands ก่อน–หลังจับคู่','',
'รันโมเดล ST-GCN checkpoint เดิมจริง ทายได้อิสระ 2,000 คำ ใช้ 3 คลิป abdomen/hello/love รวม 212 เฟรม ไม่ฝึกใหม่ ไม่เลือกคำตอบจากแค่สามคำ',
'ผลหลักตรึงลำตัว SDPose ทุกวิธี RTMW ใช้ตัดสินฝั่งมือเท่านั้น จึงแยกผลของการจับคู่ ไม่ใช่การเปลี่ยนลำตัวทั้งชุดให้โมเดลรู้จำ',
'ใช้ตัวแปลง 133→75→27 จุดและ normalization ตัวเดิม ไม่มีปรับ FPS/ความยาว/การเติมข้อมูล; confidence และ missing zero เหมือนตัวประเมินเดิม',
'ตรวจ RTMW baseline เดิมบนสามคลิปซ้ำได้ top5 เหมือนเดิมและ target probability คลาดเคลื่อน <1e-5',
'','| วิธี | ถูก/3 | Accuracy | Macro-P | Macro-R | Macro-F1 | Top5 |','|---|---:|---:|---:|---:|---:|---:|']
for v,s in result['summary'].items(): lines.append(f"| {v} | {s['correct']}/3 | {s['accuracy']:.2f}% | {s['precision']:.2f}% | {s['recall']:.2f}% | {s['f1']:.2f}% | {s['top5']:.2f}% |")
lines+=['','Macro เฉลี่ยเฉพาะ 3 true labels, zero_division=0 หนึ่งคลิปต่อคำ ผลมีขนาดตัวอย่างน้อยมากและไม่ใช่ test set อิสระ',
'','## คำทายรายคลิป','| คำจริง | วิธี | คำทาย | อันดับคำจริง | ความน่าจะเป็นคำจริง |','|---|---|---|---:|---:|']
for v,rs in pred.items():
    for r in rs: lines.append(f"| {r['label']} | {v} | {r['prediction']} | {r['target_rank']} | {r['target_probability']:.6f} |")
lines+=['','โมเดลรับ 7 จุดลำตัวและมือข้างละ 10 จุด ไม่ได้ใช้มือครบ 21 จุดเพื่อทายคำ คะแนนนี้ไม่ใช่เปอร์เซ็นต์ความเหมือนของภาพหรือความเข้าใจของผู้ใช้',
'เก็บพิกัด normalized ที่เข้าโมเดลจริง, logits ทั้ง 2000 คลาส, top5, target rank, checksum และ protocol ครบ ไม่มี production edits']
(O/'REPORT_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(result['summary']),flush=True)
