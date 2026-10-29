
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import sys,json,time,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
BASE=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007')
OUT=BASE/'wrist_tracking';sys.path.insert(0,str(BASE.parent))
def read(p):return _local_json_loads(p.read_text(encoding='utf8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False),encoding='utf8')
def detect():
 import numpy as np,cv2,mediapipe as mp
 OUT.mkdir(exist_ok=True)
 protocol=dict(variants=['crop_only','crop_flow3'],rule='Preserve Holistic; fill missing hands only. Full-image pose wrist visibility >= .5. Crop center wrist + .25*(wrist-elbow); side=clip(2.2*forearm,96,256). Static Hands .5; match wrist distance <= .65 forearm; reject proximity <20px to opposite observed wrist. Optical flow only after crop miss, at most3 consecutive frames, all21 forward/back errors <=2px, wrist gate, image bounds. No renderer coordinates or labels for detection.',fps=30)
 save(OUT/'protocol.json',protocol)
 for cap in read(BASE/'capture_manifest.json'):
  target=OUT/(cap['id']+'.json')
  if target.exists():continue
  old=read(BASE/'detected'/(cap['id']+'.json'))['rows'];result=[];prevgray=None;previous={};age={'left':0,'right':0}
  with mp.solutions.hands.Hands(static_image_mode=True,max_num_hands=2,model_complexity=1,min_detection_confidence=.5) as detector:
   for i,(f,d) in enumerate(zip(cap['frames'],old)):
    start=time.perf_counter();image=cv2.imread(str(BASE/'frames'/cap['id']/f['file']),cv2.IMREAD_UNCHANGED)
    if image.shape[2]==4:
     a=image[:,:,3:4]/255.;image=np.rint(image[:,:,:3]*a+255*(1-a)).astype(np.uint8)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);crop={};track={};sources={};rois={}
    for side,wi,ei in [('left',15,13),('right',16,14)]:
     base=np.array(d[side],np.float32)[:,:2]*808 if d[side] else None
     wrist=None;length=None;q=None
     if base is not None:q=base;sources[side]='holistic'
     elif d['pose'] and d['pose'][wi][3]>=.5 and d['pose'][ei][3]>=.5:
      wrist=np.array(d['pose'][wi][:2])*808;elbow=np.array(d['pose'][ei][:2])*808;length=np.linalg.norm(wrist-elbow);size=float(np.clip(2.2*length,96,256));center=wrist+.25*(wrist-elbow)
      x0,y0=np.maximum(0,np.floor(center-size/2)).astype(int);x1,y1=np.minimum(808,np.ceil(center+size/2)).astype(int);rois[side]=[int(x0),int(y0),int(x1),int(y1)]
      if x1>x0 and y1>y0:
       found=detector.process(cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2RGB));candidates=[]
       for hand in found.multi_hand_landmarks or []:
        points=np.array([[p.x*(x1-x0)+x0,p.y*(y1-y0)+y0] for p in hand.landmark],np.float32)
        dist=np.linalg.norm(points[0]-wrist);other='right' if side=='left' else 'left'
        duplicate=bool(d[other]) and np.linalg.norm(points[0]-np.array(d[other][0][:2])*808)<20
        if dist<=.65*length and not duplicate:candidates.append((dist,points))
       if candidates:q=min(candidates,key=lambda x:x[0])[1];sources[side]='crop'
     crop[side]=q.tolist() if q is not None else []
     if q is not None:age[side]=0
     elif wrist is not None and side in previous and prevgray is not None and age[side]<3:
      p=previous[side].astype(np.float32).reshape(-1,1,2);n,st,_=cv2.calcOpticalFlowPyrLK(prevgray,gray,p,None);back,bs,_=cv2.calcOpticalFlowPyrLK(gray,prevgray,n,None)
      if st.all() and bs.all() and np.max(np.linalg.norm(back-p,axis=2))<=2 and np.all(n>=0) and np.all(n<808) and np.linalg.norm(n[0,0]-wrist)<=.65*length:
       q=n[:,0];age[side]+=1;sources[side]='flow'
     track[side]=q.tolist() if q is not None else []
     if q is not None:previous[side]=q
     else:previous.pop(side,None);age[side]=0;sources[side]='missing'
    result.append(dict(frame=i,crop=crop,track=track,sources=sources,rois=rois,elapsed_ms=(time.perf_counter()-start)*1000));prevgray=gray
  save(target,dict(id=cap['id'],label=cap['label'],rows=result));print('completed',cap['label'],len(result),flush=True)
def score():
 import numpy as np,torch,run_audit as audit
 model,norm,vocab=audit.load_model();pred=[];geometry=[];coverage={};references={c['id']:c for c in read(BASE/'coordinate_reference.json')}
 for cap in read(BASE/'capture_manifest.json'):
  old=read(BASE/'detected'/(cap['id']+'.json'))['rows'];new=read(OUT/(cap['id']+'.json'))['rows']
  for variant,key in [('baseline',None),('crop_only','crop'),('crop_flow3','track')]:
   a=np.zeros((len(old),75,2),np.float32);stats=coverage.setdefault(variant,dict(found=0,reference=0,matched=0,source_crop=0,source_flow=0))
   for i,(d,r) in enumerate(zip(old,new)):
    if d['pose']:a[i,:33]=np.array(d['pose'])[:,:2]*808
    for s,offset in [('left',33),('right',54)]:
     points=(np.array(d[s])[:,:2]*808).tolist() if key is None and d[s] else ([] if key is None else r[key][s]);g=references[cap['id']]['samples'][i]['hands'][s]
     if len(g)==21:stats['reference']+=1
     if points:
      a[i,offset:offset+21]=points;stats['found']+=1
      if key and r['sources'][s]=='crop':stats['source_crop']+=1
      if key=='track' and r['sources'][s]=='flow':stats['source_flow']+=1
      if len(g)==21:
       dist=np.linalg.norm(np.array(points)-np.array([g[str(j)] for j in range(21)]),axis=1);stats['matched']+=1
       geometry.append(dict(variant=variant,id=cap['id'],frame=i,side=s,squared=float(np.mean(dist**2)),mean=float(dist.mean())))
   raw=torch.from_numpy(a[:,audit.PICK,:].transpose(2,0,1).copy());x=norm({'frames':raw})['frames'].unsqueeze(0);assert torch.isfinite(x).all()
   with torch.inference_mode():p=model(x)[0].softmax(0)
   top=p.topk(5);pred.append(dict(label=cap['label'],variant=variant,prediction=vocab[int(top.indices[0])],top5=[vocab[int(v)] for v in top.indices],probabilities=top.values.tolist()))
 labels=read(BASE/'protocol.json')['labels'];summ={}
 for v in coverage:
  rr=[r for r in pred if r['variant']==v];pr=[];re=[];f1=[]
  for l in labels:
   tp=sum(r['label']==l and r['prediction']==l for r in rr);fp=sum(r['label']!=l and r['prediction']==l for r in rr);pp=tp/(tp+fp) if tp+fp else 0;pr.append(pp);re.append(tp);f1.append(2*pp*tp/(pp+tp) if pp+tp else 0)
  gg=[r for r in geometry if r['variant']==v];summ[v]=dict(correct=sum(r['label']==r['prediction'] for r in rr),precision=100*np.mean(pr),recall=100*np.mean(re),f1=100*np.mean(f1),rmse=float(np.sqrt(np.mean([r['squared'] for r in gg]))),**coverage[v])
 baseline=read(BASE/'results.json')['predictions'];assert all(r['prediction']==next(b['prediction'] for b in baseline if b['variant']=='pixels_holistic' and b['label']==r['label']) for r in pred if r['variant']=='baseline')
 paired={}
 for v in ['crop_only','crop_flow3']:
  common=[];added=[]
  basekeys={(r['id'],r['frame'],r['side']) for r in geometry if r['variant']=='baseline'}
  for r in geometry:
   if r['variant']==v:(common if (r['id'],r['frame'],r['side']) in basekeys else added).append(r)
  paired[v]=dict(common_count=len(common),common_rmse=float(np.sqrt(np.mean([r['squared'] for r in common]))),added_count=len(added),added_rmse=float(np.sqrt(np.mean([r['squared'] for r in added]))) if added else None)
 for file,h in read(BASE/'protocol.json')['production'].items():assert hashlib.sha256((BASE.parent.parent/file).read_bytes()).hexdigest()==h
 save(OUT/'results.json',dict(summary=summ,paired_geometry=paired,predictions=pred,geometry=geometry,production_unchanged=True));print(json.dumps(dict(summary=summ,paired_geometry=paired),indent=2))
 lines=['# Wrist crop + tracking: 30 words','', 'ภาพเดิม1964เฟรมที่30FPS ใช้ poseจากภาพเพื่อหาข้อมือ ไม่มีreferenceRendererช่วยตรวจจับ; referenceใช้วัดผลหลังตรวจเสร็จเท่านั้น คงโมเดลและnormalizationเดิม ไม่เปลี่ยนproduction','', '| วิธี | ถูก/30 | Macro-P | Macro-R | Macro-F1 | พบมืออ้างอิง | RMSE px |','|---|---:|---:|---:|---:|---:|---:|']
 for v,s in summ.items():lines.append(f"| {v} | {s['correct']}/30 | {s['precision']:.2f}% | {s['recall']:.2f}% | {s['f1']:.2f}% | {s['matched']}/{s['reference']} | {s['rmse']:.2f} |")
 lines+=['','RMSEแต่ละวิธีคำนวณบนมือที่พบจึงมีจำนวนไม่เท่ากัน: ดู paired_geometry ในresults.json ประกอบ ซึ่งแยกมือเดิมกับมือเติมใหม่ ไม่ถือมือที่หายว่าคลาดเคลื่อน0','', 'ตัวติดตามใช้ optical flow สูงสุด3เฟรม ผ่านการตรวจforward/backwardและระยะจากข้อมือ ไม่ใช่การค้างพิกัดเดิม ป้ายข้างใช้poseไม่ใช้handedness ส่วนHolisticเดิมคงไว้ทั้งหมด','', 'เป็น30คำชุดพัฒนาเดิม ไม่ใช่held-out คะแนนไม่ใช่เปอร์เซ็นต์ความเหมือนท่าหรือความเข้าใจของผู้ใช้ ทุกพิกัดและแหล่งที่มาเก็บแยกรายเฟรมใน000.json–029.json']
 (OUT/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf8')
if __name__=='__main__':{'detect':detect,'score':score}[sys.argv[1]]()
