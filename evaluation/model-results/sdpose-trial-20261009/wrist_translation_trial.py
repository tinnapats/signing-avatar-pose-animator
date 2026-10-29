
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
ROOT=Path(__file__).resolve().parent;T=ROOT/'wrist_crop_tight20_20261011';O=ROOT/'wrist_translation_20261011';O.mkdir(exist_ok=True)
assert not (O/'protocol.json').exists()
A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit');B=A/'pixels30_20261007';R=B/'rtm_trial_20261008'
read=lambda p:_local_json_loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
protocol={'status':'registered','intervention':'Translate all21 crop hand points by RTMW hand root minus crop root. RTMW wholebody hand roots91,112 (not separate body wrists9,10).',
 'gate':'Existing crop accepted AND RTMW hand mean>=.30 and root confidence>=.30. Otherwise unchanged baseline. No new threshold.',
 'fixed':'RTMW body, confidence masks, shape, scale, rotation, timing, STGCN normalization unchanged. No blending, smoothing, gap fill.',
 'reference':'Only after predictions for evaluation, never translation input','scope':'All30 development clips, not independent test',
 'hashes':{str(p):sha(p) for p in [Path(__file__),T/'results.json',T/'model_inputs.npz',A/'wlasl_stgcn.zip']}}
save(O/'protocol.json',protocol)
sys.path.insert(0,str(A));import run_audit as audit
spec=importlib.util.spec_from_file_location('ev',ROOT.parent/'rtm-eval.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
model,norm,vocab=audit.load_model();manifest=read(B/'capture_manifest.json');old=read(T/'results.json');oldinputs=np.load(T/'model_inputs.npz')
pred={'baseline':[],'translated':[]};inputs={};logits={};raws={};masks={};changed_count=0
for c in manifest:
 cid=c['id'];z=np.load(R/f'{cid}.npz');h=np.load(T/f'{cid}.npz')
 for folder in [R,T]:assert sha(folder/f'{cid}.npz')==read(folder/f'{cid}.json')['sha256']
 assert np.allclose(z['times'],h['times']) and np.allclose(h['times'],[f['time'] for f in c['frames']])
 a=ev.convert(z['rtmw_det'],z['rtmw_det_scores']);f=a.copy();mask=np.zeros((len(a),2),bool)
 for side,b in enumerate([33,54]):
  accept=h['accepted'][:,side];f[accept,b:b+21]=h['points'][accept,side]
 shifted=f.copy()
 for side,(b,src) in enumerate([(33,91),(54,112)]):
  s=z['rtmw_det_scores'][:,src:src+21];valid=h['accepted'][:,side]&(s[:,0]>=.3)&(s.mean(1)>=.3);mask[:,side]=valid
  delta=a[valid,b]-f[valid,b];shifted[valid,b:b+21]+=delta[:,None,:]
  assert np.allclose(shifted[valid,b:b+21]-shifted[valid,b,None],f[valid,b:b+21]-f[valid,b,None],atol=1e-4)
  assert np.allclose(shifted[valid,b],a[valid,b],atol=1e-4)
  assert np.array_equal(shifted[~valid,b:b+21],f[~valid,b:b+21])
 changed_count+=int(mask.sum());masks[cid]=mask;fixed=None
 for v,q in [('baseline',f),('translated',shifted)]:
  tensor=norm({'frames':torch.from_numpy(q[:,audit.PICK].transpose(2,0,1).copy())})['frames']
  if v=='baseline':assert np.array_equal(tensor.numpy(),oldinputs[cid+'_tight_fallback']);fixed=tensor[:,:,:7].clone()
  assert torch.equal(fixed,tensor[:,:,:7]) and torch.isfinite(tensor).all()
  with torch.inference_mode():out=model(tensor.unsqueeze(0))[0];p=out.softmax(0)
  top=p.topk(5);target=vocab.index(c['label']);rr={'id':cid,'label':c['label'],'prediction':vocab[int(top.indices[0])],'top5':[vocab[int(j)] for j in top.indices],'target_probability':float(p[target]),'target_rank':torch.argsort(p,descending=True).tolist().index(target)+1}
  if v=='baseline':
   before=next(r for r in old['predictions']['tight_fallback'] if r['id']==cid);assert before['top5']==rr['top5'] and abs(before['target_probability']-rr['target_probability'])<1e-5
  pred[v].append(rr);inputs[cid+'_'+v]=tensor.numpy();logits[cid+'_'+v]=out.numpy();raws[cid+'_'+v]=q
 print(cid,c['label'],pred['baseline'][-1]['prediction'],'->',pred['translated'][-1]['prediction'],flush=True)
np.savez_compressed(O/'model_inputs.npz',**inputs);np.savez_compressed(O/'logits.npz',**logits);np.savez_compressed(O/'points.npz',**raws);np.savez_compressed(O/'translated_masks.npz',**masks)
# Reference accessed only after all predictions fixed.
refs={c['id']:{f['index']:f for f in c['samples']} for c in read(B/'coordinate_reference.json')};details=[]
for c in manifest:
 cid=c['id']
 for i,f in enumerate(c['frames']):
  ref=refs[cid][f['sourceIndex']];assert ref['pngMatches'] and ref['pixelRatio']==1
  for side,name,b in [(0,'left',33),(1,'right',54)]:
   h=ref['hands'][name]
   if not all(str(j) in h for j in range(21)):continue
   if not np.any(raws[cid+'_baseline'][i,b:b+21]):continue
   g=np.array([h[str(j)] for j in range(21)]);d={'label':c['label'],'frame':i,'side':name,'did_translate':bool(masks[cid][i,side])}
   for v in pred:
    q=raws[cid+'_'+v][i,b:b+21]
    d[v]={'total_sq':float(np.mean(np.sum((q-g)**2,axis=1))),'wrist_sq':float(np.sum((q[0]-g[0])**2)),'shape_sq':float(np.mean(np.sum(((q-q[0])[1:]-(g-g[0])[1:])**2,axis=1)))}
   details.append(d)
def geometry(ds):return {'n':len(ds),**{v:{key+'_rmse_px':float(np.sqrt(np.mean([d[v][key+'_sq'] for d in ds]))) for key in ['total','wrist','shape']} for v in pred}}
changes=[{'label':a['label'],'before':a['prediction'],'after':b['prediction'],'before_correct':a['prediction']==a['label'],'after_correct':b['prediction']==b['label']} for a,b in zip(pred['baseline'],pred['translated']) if a['prediction']!=b['prediction']]
result={'summary':{v:ev.metrics(rr) for v,rr in pred.items()},'predictions':pred,'changed_all_hand_frames':changed_count,
 'geometry_all_reference':geometry(details),'geometry_translated_reference':geometry([d for d in details if d['did_translate']]),'changes':changes,'details':details,
 'baseline_reproduced':True,'shape_preserved':True}
save(O/'results.json',result);protocol['status']='completed';save(O/'protocol.json',protocol)
lines=['# เลื่อนมือ crop ไปยังข้อมือ RTMW','',
 'RTMW anchor คือจุดข้อมือรากของชุดมือ21จุด ดัชนี91/112 ไม่ใช่จุดข้อมือลำตัว9/10 ใช้เฉพาะ cropผ่านเกณฑ์และ RTMWมือผ่าน confidence เดิม0.30 คงรูปนิ้ว ขนาด ทิศทาง ลำตัว และเวลา ไม่ใช้ Renderer reference แก้พิกัด',
 'ตรวจ input baselineตรงไฟล์เดิมทุกจุด และคำทาย/top5/probabilityซ้ำตรงรอบก่อน ทดสอบจริง30คลิปพัฒนา ไม่ใช่ validationอิสระ','',
 '| วิธี | ถูก/30 | Accuracy % | Macro-P % | Macro-R % | Macro-F1 % | Top5 % |','|---|---:|---:|---:|---:|---:|---:|']
for v,m in result['summary'].items():lines.append(f"| {v} | {m['correct']} | {m['accuracy']:.2f} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['top5']:.2f} |")
lines+=['','Geometry เฉพาะเฟรมที่ถูกเลื่อนและมีreferenceครบ:','```json',json.dumps(result['geometry_translated_reference'],indent=2),'```','',
 'คำทายที่เปลี่ยน:','```json',json.dumps(changes,indent=2),'```','Macroเฉลี่ยเฉพาะ30true labels ไม่ใช่2000classes ความตรงของพิกัดและคะแนนรู้จำไม่ใช่ความเข้าใจASL ไม่แก้ระบบหลัก']
(O/'REPORT_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['predictions','details']}),flush=True)
