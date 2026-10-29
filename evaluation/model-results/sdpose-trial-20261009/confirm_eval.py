
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import sys,json,hashlib,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
from track_hands import assign,tests,SETTINGS
ROOT=Path(__file__).resolve().parent
P=ROOT/'confirmation3'; O=P/'stgcn'; O.mkdir(exist_ok=True)
assert not (O/'protocol.json').exists()
assert _local_json_loads((P/'protocol.json').read_text())['status']=='three_clip_inference_completed'
A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit')
B=A/'pixels30_20261007'; R=B/'rtm_trial_20261008'
sys.path.insert(0,str(A))
import run_audit as audit
spec=importlib.util.spec_from_file_location('rtm_eval',ROOT.parent/'rtm-eval.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
rows=_local_json_loads((P/'results.json').read_text())
manifest=_local_json_loads((B/'capture_manifest.json').read_text())[3:6]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol={'status':'registered','settings':SETTINGS,'selection':'manifest entries 3:6, no score selection; new to this SDPose experiment only',
 'clips':[{ 'id':c['id'],'label':c['label'],'frames':len(c['frames'])} for c in manifest],
 'hashes':{str(p):sha(p) for p in [P/'results.json',ROOT/'track_hands.py',Path(__file__),A/'wlasl_stgcn.zip',ROOT.parent/'rtm-eval.py']},
 'rules':'Original timestamps, confidence .30, missing zero, same shoulder normalization, no tuning or resampling. RTMW body changes normalization too.',
 'production_edited':False}
(O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
tests(); model,norm,vocab=audit.load_model()
old=_local_json_loads((R/'results.json').read_text())
pred={}; decisions=[]; inputs={}; logits={}
for clip in manifest:
 cid=clip['id']; label=clip['label']; rr=[r for r in rows if r['id']==cid]
 cache=np.load(R/f'{cid}.npz')
 assert len(rr)==len(clip['frames']) and np.allclose(cache['times'],[r['time'] for r in rr])
 arrays={v:[] for v in ['sdpose','hybrid_raw','hybrid_anchor','hybrid_temporal','rtmw']}
 scores={v:[] for v in arrays}; history={}
 for i,r in enumerate(rr):
  assert r['frame']==i and r['sha256']==clip['frames'][i]['sha256']
  k=np.array(r['keypoints']); s=np.array(r['scores'])
  for v in arrays:
   work=k.copy(); ws=s.copy()
   if v=='rtmw': work=cache['rtmw_det'][i].copy();ws=cache['rtmw_det_scores'][i].copy()
   elif v.startswith('hybrid'):work[:91]=cache['rtmw_det'][i,:91];ws[:91]=cache['rtmw_det_scores'][i,:91]
   info={'swapped':False}
   if v in ('hybrid_anchor','hybrid_temporal'):
    prior=history.get(v); prev=prior[1] if prior and i-prior[0]<=3 else None
    work,ws,info=assign(work,ws,prev,v=='hybrid_temporal')
    if all(ws[b]>=.3 and ws[b:b+21].mean()>=.3 for b in (91,112)):history[v]=(i,work[[91,112]].copy())
   arrays[v].append(work);scores[v].append(ws)
   decisions.append({'id':cid,'frame':i,'variant':v,**info})
 # Save assignment evidence before recognition; no renderer reference is loaded.
 (O/'decisions.json').write_text(json.dumps(decisions,indent=2),encoding='utf-8')
 for v in arrays:
  a=ev.convert(np.array(arrays[v]),np.array(scores[v]))
  t=norm({'frames':torch.from_numpy(a[:,audit.PICK].transpose(2,0,1).copy())})['frames']
  assert torch.isfinite(t).all()
  with torch.inference_mode(): out=model(t.unsqueeze(0))[0];prob=out.softmax(0)
  top=prob.topk(5);target=vocab.index(label)
  result={'label':label,'prediction':vocab[int(top.indices[0])],'top5':[vocab[int(j)] for j in top.indices],
   'target_rank':torch.argsort(prob,descending=True).tolist().index(target)+1,'target_probability':float(prob[target])}
  if v=='rtmw':
   before=next(x for x in old['predictions']['rtmw_det'] if x['label']==label)
   assert before['top5']==result['top5'] and abs(before['target_probability']-result['target_probability'])<1e-5
  pred.setdefault(v,[]).append(result);inputs[f'{cid}_{v}']=t.numpy();logits[f'{cid}_{v}']=out.numpy()
  print(label,v,result['prediction'],result['target_rank'],flush=True)
result={'summary':{v:ev.metrics(r) for v,r in pred.items()},'predictions':pred,'rtmw_control_reproduced':True}
(O/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
np.savez_compressed(O/'inputs.npz',**inputs);np.savez_compressed(O/'logits.npz',**logits)
protocol['status']='completed';(O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
lines=['# ตรวจวิธีผสมบน 3 คลิปถัดไป','',str(protocol['clips']),'',
 'เลือกตามลำดับ manifest ก่อนดูคะแนน ไม่ซ้ำ abdomen/hello/love แต่ยังมาจากชุดพัฒนาเดิม ไม่ใช่ test set อิสระ',
 'คงเวลา confidence และการจัดการข้อมูลหายเดิม ไม่แก้ production RTMW เปลี่ยนทั้งลำตัว confidence และค่า normalization ที่คำนวณจากไหล่',
 '| วิธี | ถูก/3 | Accuracy % | Precision % | Recall % | F1 % | Top5 % |','|---|---:|---:|---:|---:|---:|---:|']
for v,m in result['summary'].items():lines.append(f"| {v} | {m['correct']}/3 | {m['accuracy']:.2f} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['top5']:.2f} |")
lines+=['','| คำจริง | วิธี | คำทาย | อันดับคำจริง |','|---|---|---|---:|']
for v,rs in pred.items():
 for r in rs:lines.append(f"| {r['label']} | {v} | {r['prediction']} | {r['target_rank']} |")
lines+=['','Macro เฉลี่ยเฉพาะ 3 true labels; หนึ่งคลิปต่อคำ คะแนนนี้ไม่ใช่ความเหมือนภาพหรือความเข้าใจ ASL ตรวจ RTMW control ซ้ำตรงกับผลเก่าแล้ว']
(O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(result['summary']),flush=True)
