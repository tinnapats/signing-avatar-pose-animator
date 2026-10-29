import sys,json,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;O=ROOT/'stage_inputs_20261011'
assert not (O/'mask_control.json').exists()
A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit');B=A/'pixels30_20261007'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sys.path.insert(0,str(A));import run_audit as audit
spec=importlib.util.spec_from_file_location('ev',ROOT.parent/'rtm-eval.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
model,norm,vocab=audit.load_model();refs={c['id']:c for c in read(B/'coordinate_reference.json')};pred={};inputs={};logits={}
for c in read(B/'capture_manifest.json'):
 cid=c['id'];z=np.load(B/'rtm_trial_20261008'/f'{cid}.npz');raw=ev.convert(z['rtmw_det'],z['rtmw_det_scores']);masked=raw.copy();coordinate=raw.copy()
 for i,r in enumerate(refs[cid]['samples']):
  for side,b in [('left',33),('right',54)]:
   h=r['hands'][side]
   if not h:masked[i,b:b+21]=0
   elif len(h)==21 and np.any(raw[i,b:b+21]):coordinate[i,b:b+21]=np.array([h[str(j)] for j in range(21)])
 for v,a in [('reference_presence_mask_only',masked),('reference_coordinates_keep_detection_presence',coordinate)]:
  t=norm({'frames':torch.from_numpy(a[:,audit.PICK].transpose(2,0,1).copy())})['frames']
  with torch.inference_mode():out=model(t.unsqueeze(0))[0];p=out.softmax(0)
  top=p.topk(5);rr={'label':c['label'],'prediction':vocab[int(top.indices[0])],'top5':[vocab[int(j)] for j in top.indices],'target_probability':float(p[vocab.index(c['label'])])}
  pred.setdefault(v,[]).append(rr);inputs[cid+'_'+v]=t.numpy();logits[cid+'_'+v]=out.numpy()
result={'scope':'Oracle diagnosis only. Neither variant is a deployable pixel-only system. Missing renderer reference is not independently verified absence of hand.',
 'summary':{v:ev.metrics(rr) for v,rr in pred.items()},'predictions':pred}
(O/'mask_control.json').write_text(json.dumps(result,indent=2),encoding='utf-8');np.savez_compressed(O/'mask_inputs.npz',**inputs);np.savez_compressed(O/'mask_logits.npz',**logits)
lines=['','## แยกผลพิกัดมือกับการมีข้อมูลอ้างอิง (oracle control เท่านั้น)','| วิธี | ถูก/30 | Macro-F1 % |','|---|---:|---:|']
for v,m in result['summary'].items():lines.append(f"| {v} | {m['correct']} | {m['f1']:.2f} |")
lines+=['ใช้ reference เพื่อแยกสาเหตุเท่านั้น ห้ามรายงานเป็นคะแนน pixel-only หรือเป็นระบบที่เลือกจุดได้เอง การไม่มีreferenceไม่ได้พิสูจน์ว่าไม่มีมือ']
with (O/'REPORT_TH.md').open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(result['summary']),flush=True)
