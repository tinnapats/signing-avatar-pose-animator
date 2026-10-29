import sys,json,hashlib,importlib.util,time,collections
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;V=ROOT.parent;O=ROOT/'frozen_tracking_plan_20261011'
A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit');B=A/'pixels30_20261007';R=B/'rtm_trial_20261008'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
assert read(O/'protocol.json')['status']=='extracted' and not (O/'results.json').exists()
sys.path.insert(0,str(A));import run_audit as audit
spec=importlib.util.spec_from_file_location('ev',V/'rtm-eval.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
model,norm,vocab=audit.load_model();allresults={};inputs={};logits={};georows=[]
for setname,folder,bodyfolder in [('development',B,R),('confirmation',V/'confirm30-crop-20261009',V/'confirm30-crop-20261009/rtm')]:
 manifest=read(folder/'capture_manifest.json');pred={v:[] for v in ['rtmw','baseline','tracked']};cache={};status=collections.Counter();runtime=[];model_ms=[]
 for c in manifest:
  cid=c['id'];h=np.load(O/setname/f'{cid}.npz');z=np.load(bodyfolder/f'{cid}.npz');cache[cid]=(h,z)
  assert sha(O/setname/f'{cid}.npz')==read(O/setname/f'{cid}.json')['sha256']
  assert np.allclose(h['times'],[f['time'] for f in c['frames']])
  status.update(l['status'] for frame in read(O/setname/f'{cid}_logs.json') for l in frame)
  runtime.extend(zip(h['base_ms'].tolist(),h['track_ms'].tolist()))
  raw=ev.convert(z['rtmw_det'],z['rtmw_det_scores']);base=raw.copy()
  for side,b in enumerate([33,54]):
   valid=h['mask'][:,side];base[valid,b:b+21]=h['points'][valid,side]
  track=base.copy()
  for side,b in enumerate([33,54]):
   valid=h['track_mask'][:,side];assert not np.any(valid&h['mask'][:,side]);track[valid,b:b+21]=h['tracked'][valid,side]
  fixed=None
  for name,a in [('rtmw',raw),('baseline',base),('tracked',track)]:
   t=norm({'frames':torch.from_numpy(a[:,audit.PICK].transpose(2,0,1).copy())})['frames']
   if fixed is None:fixed=t[:,:,:7].clone()
   assert torch.equal(fixed,t[:,:,:7]) and torch.isfinite(t).all()
   if setname=='development' and name=='baseline':assert np.array_equal(t.numpy(),np.load(ROOT/'wrist_crop_tight20_20261011/model_inputs.npz')[cid+'_tight_fallback'])
   start=time.perf_counter()
   with torch.inference_mode():out=model(t.unsqueeze(0))[0];p=out.softmax(0)
   model_ms.append(1000*(time.perf_counter()-start));top=p.topk(5);target=vocab.index(c['label'])
   pred[name].append({'id':cid,'label':c['label'],'prediction':vocab[int(top.indices[0])],'top5':[vocab[int(j)] for j in top.indices],'target_probability':float(p[target])})
   inputs[setname+'_'+cid+'_'+name]=t.numpy();logits[setname+'_'+cid+'_'+name]=out.numpy()
  cache[cid]+=(base,track)
  print(setname,cid,c['label'],pred['baseline'][-1]['prediction'],'->',pred['tracked'][-1]['prediction'],flush=True)
 # References never used in tracking/inference.
 refs={c['id']:{f['index']:f for f in c['samples']} for c in read(folder/'coordinate_reference.json')}
 coverage={v:0 for v in ['baseline','tracked']};details=[]
 for c in manifest:
  cid=c['id'];h,z,base,track=cache[cid]
  for side,name,b in [(0,'left',33),(1,'right',54)]:
   for v,q in [('baseline',base),('tracked',track)]:coverage[v]+=int(np.any(q[:,b:b+21]!=0,axis=(1,2)).sum())
   for i,f in enumerate(c['frames']):
    ref=refs[cid][f['sourceIndex']];assert ref['pngMatches'] and ref['pixelRatio']==1;hand=ref['hands'][name]
    if not all(str(j) in hand for j in range(21)) or not np.any(base[i,b:b+21]) or not np.any(track[i,b:b+21]):continue
    g=np.array([hand[str(j)] for j in range(21)]);dd={'label':c['label'],'frame':i,'side':name,'changed':bool(h['track_mask'][i,side])}
    qs={'baseline':base[i,b:b+21],'tracked':track[i,b:b+21]};errors={v:[] for v in qs}
    for chain in [(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)]:
     for j in range(3):
      a,bb,cc=chain[j:j+3];vals={}
      for v,q in {'reference':g,**qs}.items():
       u=q[a]-q[bb];w=q[cc]-q[bb];den=np.linalg.norm(u)*np.linalg.norm(w)
       if den>1e-8:vals[v]=float(np.degrees(np.arccos(np.clip(np.dot(u,w)/den,-1,1))))
      if len(vals)==3:
       for v in qs:errors[v].append(abs(vals[v]-vals['reference']))
    for v,q in qs.items():dd[v]={'total':float(np.mean(np.sum((q-g)**2,axis=1))),'wrist':float(np.sum((q[0]-g[0])**2)),'shape':float(np.mean(np.sum(((q-q[0])[1:]-(g-g[0])[1:])**2,axis=1))),'angles':errors[v]}
    details.append(dd)
 def geo(ds):
  return {'hand_frames':len(ds),**{v:{**{k+'_rmse_px':float(np.sqrt(np.mean([d[v][k] for d in ds]))) if ds else None for k in ['total','wrist','shape']},'angle_mae_deg':float(np.mean(sum([d[v]['angles'] for d in ds],[]))) if ds else None} for v in ['baseline','tracked']}}
 changed=[d for d in details if d['changed']]
 changes=[{'label':a['label'],'before':a['prediction'],'after':b['prediction'],'before_correct':a['prediction']==a['label'],'after_correct':b['prediction']==b['label']} for a,b in zip(pred['baseline'],pred['tracked']) if a['prediction']!=b['prediction']]
 allresults[setname]={'summary':{v:ev.metrics(rr) for v,rr in pred.items()},'predictions':pred,'changes':changes,'coverage':coverage,
  'frames':sum(len(c['frames']) for c in manifest),'reasons':dict(status),'runtime_ms_per_frame':{'baseline_hand':float(np.mean([x[0] for x in runtime])),'tracking_extra':float(np.mean([x[1] for x in runtime]))},
  'geometry':geo(details),'changed_geometry':geo(changed),'changed_better':sum(d['tracked']['total']<d['baseline']['total'] for d in changed),'changed_worse':sum(d['tracked']['total']>d['baseline']['total'] for d in changed),'details':details}
np.savez_compressed(O/'model_inputs.npz',**inputs);np.savez_compressed(O/'logits.npz',**logits);save(O/'results.json',allresults)
p=read(O/'protocol.json');p['status']='completed';p['score_script_sha256']=sha(Path(__file__));save(O/'protocol.json',p)
lines=['# ผลแผนติดตามมือระยะสั้นแบบตรึงเกณฑ์','',
 'Development30 และ confirmation30 ไม่ซ้ำคำกัน แต่ confirmation เคยใช้ในงานทดลองก่อน จึงเป็นชุดแยกของรอบนี้ ไม่ใช่ pristine held-out test. ตรึงสูตรก่อนรันทั้งสองชุด ไม่เลือกผู้ชนะ/threshold จาก label',
 'Tracker LK ย้ายกรอบและอ่านมือใหม่จากภาพเฟรมปัจจุบัน ไม่ค้าง landmark ไม่เกิน3เฟรมหลัง detector seed และไม่ต่ออายุจาก tracking; baseline cropที่ผ่านเกณฑ์ชนะเสมอ',
 'ไม่ smoothing ไม่คุมกระดูก ไม่ย้ายข้อมือ และไม่แก้ production','',
 '| ชุด | วิธี | ถูก/30 | Accuracy % | Macro-P % | Macro-R % | Macro-F1 % | Top5 % |','|---|---|---:|---:|---:|---:|---:|---:|']
for st,r in allresults.items():
 for v,m in r['summary'].items():lines.append(f"| {st} | {v} | {m['correct']} | {m['accuracy']:.2f} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['top5']:.2f} |")
for st,r in allresults.items():
 lines+=['',f'## {st}', '```json',json.dumps({k:v for k,v in r.items() if k not in ['predictions','details','summary']},indent=2),'```']
lines+=['','เวลารายงานเฉพาะขั้นมือ/ติดตาม ไม่รวม RTMW bodyเดิม/การcaptureภาพ และ baseline developmentใช้ timingที่เก็บจากการรันก่อนจึงไม่ใช่ benchmarkเครื่องโหลดเท่ากัน',
 'Geometryใช้เฉพาะhand-frameที่มีreferenceและทั้งสองวิธีมีข้อมูล same-point subset; missing referenceไม่ใช่หลักฐานมือพัก, shapeยังรวมทิศทางขนาด, angleเป็น2D ไม่ใช่3D/ASL correctness',
 'Macroเฉลี่ย30true labelsในแต่ละชุด ไม่ใช่2000classes. เก็บคำที่เปลี่ยน ผลดี/เสีย พิกัดเข้าโมเดล logits และ rejection reasonsครบ']
(O/'REPORT_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({s:{k:v for k,v in r.items() if k not in ['details','predictions']} for s,r in allresults.items()}),flush=True)
