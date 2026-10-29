
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
D=Path(__file__).resolve().parent;O=D/'oracle200-20261011'
A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit');B=A/'pixels30_20261007';R=B/'rtm_trial_20261008'
read=lambda p:_local_json_loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def extract():
 base=module('base200',D/'rtm-avatar.py');cv2,rt,det=base.setup()
 models={}
 for k in ['det','rtmw']:
  m=read(R/(k+'_model.json'));assert sha(Path(m['path']))==m['sha256']
  models[k]=(det if k=='det' else rt)(m['path'],model_input_size=tuple(m['input_size']),backend='onnxruntime',device='cpu')
 folder=O/'rtm';folder.mkdir(exist_ok=True)
 manifest=read(O/'capture_manifest.json');assert len(manifest)==200
 old=read(B/'capture_manifest.json');confirm=D/'confirm30-crop-20261009';cm=read(confirm/'capture_manifest.json')
 for n,c in enumerate(manifest):
  p=folder/(c['id']+'.npz');meta=p.with_suffix('.json')
  if p.exists():assert sha(p)==read(meta)['sha256'];continue
  if n<60:
   previous=old[n] if n<30 else cm[n-30]
   assert c['label']==previous['label'] and c['frames']==previous['frames'] and c['width']==previous['width'] and c['height']==previous['height']
   cache=R/(previous['id']+'.npz') if n<30 else confirm/'rtm'/(previous['id']+'.npz')
   assert sha(cache)==read(cache.with_suffix('.json'))['sha256']
   z=np.load(cache);np.savez_compressed(p,rtmw_det=z['rtmw_det'],rtmw_det_scores=z['rtmw_det_scores'],times=z['times'])
   save(meta,{'sha256':sha(p),'label':c['label'],'cache':str(cache),'cache_sha256':sha(cache),'png_hashes_exact':True})
  else:
   points=[];scores=[];ms=[]
   for f in c['frames']:
    image=O/'frames'/c['id']/f['file'];assert sha(image)==f['sha256'];im=base.image(cv2,image);t=time.perf_counter();boxes=models['det'](im)
    if len(boxes):
     box=max(boxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1]))[:4];pp,ss=models['rtmw'](im,bboxes=[box]);pnt,sc=pp[0],ss[0]
    else:pnt=np.zeros((133,2),np.float32);sc=np.zeros(133,np.float32)
    points.append(pnt);scores.append(sc);ms.append((time.perf_counter()-t)*1000)
   np.savez_compressed(p,rtmw_det=points,rtmw_det_scores=scores,times=[f['time'] for f in c['frames']],ms=ms)
   save(meta,{'sha256':sha(p),'label':c['label'],'mean_ms':float(np.mean(ms))})
  print('EXTRACT',n+1,'/200',c['label'],flush=True)
 save(O/'extraction_complete.json',{'clips':200,'manifest_sha256':sha(O/'capture_manifest.json'),'models':{k:read(R/(k+'_model.json')) for k in models}})
def score():
 import torch
 sys.path.insert(0,str(A));import run_audit as audit
 ev=module('ev200',D/'rtm-eval.py');model,norm,vocab=audit.load_model()
 refs={r['id']:r for r in read(O/'coordinate_reference.json')};manifest=read(O/'capture_manifest.json');assert len(manifest)==200 and len({c['label'] for c in manifest})==200
 variants=['pixels','mask_only','coordinates_only','both_oracle'];pred={v:[] for v in variants};folder=O/'inference';folder.mkdir(exist_ok=True)
 for n,c in enumerate(manifest):
  p=O/'rtm'/(c['id']+'.npz');assert sha(p)==read(p.with_suffix('.json'))['sha256'];z=np.load(p);assert np.allclose(z['times'],[f['time'] for f in c['frames']])
  raw=ev.convert(z['rtmw_det'],z['rtmw_det_scores']);arrays={v:raw.copy() for v in variants}
  rr=refs[c['id']];assert rr['label']==c['label'] and len(rr['samples'])==len(raw)
  for i,r in enumerate(rr['samples']):
   assert r['index']==c['frames'][i]['sourceIndex'] and r['pngMatches'] and r['pixelRatio']==1
   for side,b in [('left',33),('right',54)]:
    h=r['hands'][side];arrays['both_oracle'][i,b:b+21]=0
    for j,point in h.items():arrays['both_oracle'][i,b+int(j)]=point
    if not h:arrays['mask_only'][i,b:b+21]=0
    elif len(h)==21 and np.any(raw[i,b:b+21]):arrays['coordinates_only'][i,b:b+21]=np.array([h[str(j)] for j in range(21)])
  archive={}
  for v,a in arrays.items():
   x=norm({'frames':torch.from_numpy(a[:,audit.PICK].transpose(2,0,1).copy())})['frames'];assert torch.isfinite(x).all()
   with torch.inference_mode():log=model(x.unsqueeze(0))[0];prob=log.softmax(0)
   top=prob.topk(5);pred[v].append({'id':c['id'],'label':c['label'],'prediction':vocab[int(top.indices[0])],'top5':[vocab[int(j)] for j in top.indices],'target_probability':float(prob[vocab.index(c['label'])])});archive[v+'_input']=x.numpy();archive[v+'_logits']=log.numpy()
  np.savez_compressed(folder/(c['id']+'.npz'),**archive);print('SCORE',n+1,'/200',c['label'],flush=True)
 old=D/'sdpose-trial-20261009/stage_inputs_20261011';base=read(old/'results.json')['predictions'];mask=read(old/'mask_control.json')['predictions']
 previous={'pixels':base['avatar_pixels_rtmw'],'both_oracle':base['renderer_hands_rtmw_body_diagnostic'],'mask_only':mask['reference_presence_mask_only'],'coordinates_only':mask['reference_coordinates_keep_detection_presence']}
 for v in variants:
  for new,before in zip(pred[v][:30],previous[v]):assert new['label']==before['label'] and new['top5']==before['top5'] and abs(new['target_probability']-before['target_probability'])<1e-5
 summary={group:{v:ev.metrics(rows[start:end]) for v,rows in pred.items()} for group,start,end in [('original30',0,30),('additional170',30,200),('all200',0,200)]}
 result={'summary':summary,'predictions':pred,'first30_exact_reproduction':True,'scope':'Archived development200, one clip per label. Oracle diagnostics, not pixel-only improvement, not sign-language validity. Macro over true labels of each subset; unrestricted 2000-class predictions. Failed hand detections retained.', 'clips':200,'frames':sum(len(c['frames']) for c in manifest)};save(O/'results.json',result)
 lines=['# ST-GCN: 200 คลิป แยกพิกัดมือและสถานะเฟรม','',result['scope'],'','30 คลิปแรกให้คำทำนาย Top5 และความน่าจะเป็นซ้ำตรงรอบเดิม ทุกวิธีใช้ภาพและเวลาเดียวกัน ไม่แก้ระบบหลัก ไม่ปรับเกณฑ์ตามคะแนน','']
 for group,methods in summary.items():
  lines += ['## '+group,'','| วิธี | ถูก | Accuracy % | Macro-P % | Macro-R % | Macro-F1 % |','|---|---:|---:|---:|---:|---:|']
  for v,m in methods.items():lines.append(f"| {v} | {m['correct']} | {m['accuracy']:.2f} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} |")
  lines.append('')
 lines+=['pixels คือ RTMW ตรวจภาพเอง; mask_only ใช้สถานะ reference ตัดมือ; coordinates_only เปลี่ยนพิกัดเฉพาะมือที่มี reference ครบและมี detection; both_oracle ใช้ทั้งสองส่วน โดยคงลำตัว RTMW ทุกวิธี','การไม่มี reference ไม่ยืนยันว่าไม่มีมือจริง อาจเป็นมือพัก SVG คะแนน oracle ไม่ใช่คะแนนที่ระบบตรวจภาพเองทำได้ ข้อมูลเพิ่มเติมเคยใช้ในงานทดลองอื่น จึงไม่ใช่ pristine holdout']
 (O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(summary),flush=True)
if __name__=='__main__':{'extract':extract,'score':score}[sys.argv[1]]()
