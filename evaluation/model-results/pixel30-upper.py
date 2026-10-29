import sys,json,hashlib,time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');O=B/'upper_crop_20261008'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def detect():
 import cv2,mediapipe as mp
 O.mkdir(exist_ok=True)
 save(O/'detector_protocol.json',dict(crop='One fixed square per clip from image-only baseline upper pose and detected hands,1st/99th percentile bounds +20% margin; side clipped384..808; resize808; white padding. No Renderer in crop or detection. Holistic tracking .5 complexity1. Preserve rest_roi3 detections, fill missing nongated hands only; reject >.65 forearm wrist distance or near opposite existing hand20px.',human='Same-word controls only; lowest video_id official test per plan/please/buy if present, else lowest video_id. No verified SLclean video mapping; no framewise human-avatar error.',score='5frame bounded interpolation same prior rules; camera metadata alignment same. Known development30.'))
 manifest=read(B.parent/'head_pilot_20261005/manifest.json');human=[]
 for word in ['plan','please','buy']:
  candidates=[r for r in manifest if r.get('label',r.get('word'))==word and Path(r['local_video']).exists()];candidates.sort(key=lambda r:(r['metadata']['split']!='test',int(r['metadata']['video_id'])));r=candidates[0];assert sha(Path(r['local_video']))==r['sha256'];human.append(r)
 save(O/'human_selection.json',human)
 for h in human:
  target=O/('human_'+h['label']+'.npz')
  if target.exists():continue
  cap=cv2.VideoCapture(h['local_video']);fps=cap.get(cv2.CAP_PROP_FPS);rows=[]
  with mp.solutions.holistic.Holistic(model_complexity=1,min_detection_confidence=.5,min_tracking_confidence=.5) as detector:
   while True:
    ok,img=cap.read()
    if not ok:break
    hh,ww=img.shape[:2];d=detector.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));a=np.zeros((75,2),np.float32)
    for lm,off in [(d.pose_landmarks,0),(d.left_hand_landmarks,33),(d.right_hand_landmarks,54)]:
     if lm:a[off:off+len(lm.landmark)]=np.array([[p.x*ww,p.y*hh] for p in lm.landmark])
    rows.append(a)
  cap.release();array=np.array(rows);idx=np.minimum(np.rint(np.arange(0,len(rows)/fps,1/30)*fps).astype(int),len(rows)-1);np.savez_compressed(target,points=array[idx],native=array,fps=fps);print('human',h['label'],len(rows),'frames',flush=True)
 for cap in read(B/'capture_manifest.json'):
  target=O/(cap['id']+'.npz')
  if target.exists():continue
  old=read(B/'detected'/(cap['id']+'.json'))['rows'];points=[]
  for d in old:
   if d['pose']:points.extend([[d['pose'][j][0]*808,d['pose'][j][1]*808] for j in [0,2,5,11,12,13,14,15,16,23,24] if d['pose'][j][3]>=.5])
   for s in ['left','right']:
    if d[s]:points.extend((np.array(d[s])[:,:2]*808).tolist())
  if points:
   lo,hi=np.percentile(points,[1,99],axis=0);center=(lo+hi)/2;size=float(np.clip(np.max(hi-lo)*1.2,384,808))
  else:center=np.array([404,404]);size=808.
  scale=808/size;origin=center-size/2;mat=np.array([[scale,0,-origin[0]*scale],[0,scale,-origin[1]*scale]],np.float32);z=np.load(B/'rest_gate_flow'/(cap['id']+'.npz'));base=z['rest_roi3'];data=base.copy();allpoints=[];added=[];start=time.perf_counter()
  with mp.solutions.holistic.Holistic(model_complexity=1,min_detection_confidence=.5,min_tracking_confidence=.5) as detector:
   for i,f in enumerate(cap['frames']):
    im=cv2.imread(str(B/'frames'/cap['id']/f['file']),cv2.IMREAD_UNCHANGED);alpha=im[:,:,3:4]/255;rgb=cv2.cvtColor(np.uint8(np.rint(im[:,:,:3]*alpha+255*(1-alpha))),cv2.COLOR_BGR2RGB);crop=cv2.warpAffine(rgb,mat,(808,808),borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255));d=detector.process(crop);a=np.zeros((75,2),np.float32)
    for lm,off in [(d.pose_landmarks,0),(d.left_hand_landmarks,33),(d.right_hand_landmarks,54)]:
     if lm:a[off:off+len(lm.landmark)]=np.array([[p.x*808/scale+origin[0],p.y*808/scale+origin[1]] for p in lm.landmark])
    allpoints.append(a)
    for si,s,wi,ei,off in [(0,'left',15,13,33),(1,'right',16,14,54)]:
     if z['gate'][i,si] or np.any(base[i,off:off+21]) or not np.any(a[off:off+21]):continue
     pose=old[i]['pose']
     if not pose or min(pose[wi][3],pose[ei][3])<.5:continue
     wrist=np.array(pose[wi][:2])*808;length=np.linalg.norm(wrist-np.array(pose[ei][:2])*808);other=54 if off==33 else 33
     if np.linalg.norm(a[off]-wrist)>.65*length:continue
     if np.any(data[i,other:other+21]) and np.linalg.norm(a[off]-data[i,other])<20:continue
     data[i,off:off+21]=a[off:off+21];added.append(dict(frame=i,side=s))
  np.savez_compressed(target,rest_roi3=data,gate=z['gate'],crop_raw=np.array(allpoints));save(O/(cap['id']+'.json'),dict(label=cap['label'],added=added,scale=scale,origin=origin.tolist(),seconds=time.perf_counter()-start));print(cap['label'],'scale',round(scale,2),'added',len(added),flush=True)
def score():
 # Instantiate the same verified bounded-gap evaluation with this detector's arrays.
 short=Path(__file__).with_name('pixel30-short-gaps.py');s=short.read_text(encoding='utf8');tail="exec(compile(source,str(Path(__file__)), 'exec'))";assert s.count(tail)==1;ns={'__file__':str(short)};exec(compile(s.replace(tail,'pass'),str(short),'exec'),ns);code=ns['source']
 code=code.replace("O=B/'short_gaps_20261008'","O=B/'upper_crop_20261008/evaluation'").replace("np.load(B/'rest_gate_flow'/(c['id']+'.npz'))","np.load(B/'upper_crop_20261008'/(c['id']+'.npz'))")
 code=code.replace("names={'zero_baseline':'ไม่เติม','linear3':'เติมช่องว่างไม่เกิน3เฟรม','linear5':'เติมช่องว่างไม่เกิน5เฟรม','linear10':'เติมช่องว่างไม่เกิน10เฟรม'}","names={'zero_baseline':'ขยายภาพช่วงบน ไม่เติมเพิ่ม','linear5':'ขยายภาพช่วงบนและเติม5เฟรม'}")
 line="  if variant=='zero_baseline':assert row['prediction']==next(r['prediction'] for r in prior if r['label']==c['label'] and r['variant']=='rest_roi3')";assert code.count(line)==1;code=code.replace(line,'  # New crop detections intentionally differ.')
 code=code.replace("source='rest_gate_flow/rest_roi3'","source='upper_crop_20261008/rest_roi3'").replace('ไม่เกิน3,5,10เฟรม','ไม่เกิน5เฟรม').replace('ใช้ฐานrest_roi3','ใช้ฐานrest_roi3พร้อมตรวจภาพขยายช่วงบน')
 exec(compile(code,str(Path(__file__)),'exec'),{'__file__':__file__})
 import torch,run_audit as audit
 model,norm,vocab=audit.load_model();human=[]
 for h in read(O/'human_selection.json'):
  z=np.load(O/('human_'+h['label']+'.npz'));a=z['points'];raw=z['native'];x=norm({'frames':torch.from_numpy(a[:,audit.PICK,:].transpose(2,0,1).copy())})['frames'].unsqueeze(0)
  with torch.inference_mode():p=model(x)[0].softmax(0)
  coverage=100*float(np.any(raw[:,33:54]!=0,axis=(1,2)).sum()+np.any(raw[:,54:]!=0,axis=(1,2)).sum())/(2*len(raw))
  top=p.topk(5);human.append(dict(label=h['label'],video_id=h['metadata']['video_id'],split=h['metadata']['split'],frames30=len(a),native_frames=len(raw),native_fps=float(z['fps']),prediction=vocab[int(top.indices[0])],top5=[vocab[int(i)] for i in top.indices],hand_coverage_pct=coverage))
 old=read(B/'short_gaps_20261008/results.json');new=read(O/'evaluation/results.json');added=sum(len(read(O/(cap['id']+'.json'))['added']) for cap in read(B/'capture_manifest.json'));ref={r['id']:r for r in read(B/'coordinate_reference.json')};errors=[]
 avatar_predictions=read(B/'results.json')['predictions'];caps=read(B/'capture_manifest.json')
 for h in human:
  cap=next(c for c in caps if c['label']==h['label']);a=np.load(B/'arrays'/(cap['id']+'.npz'))['primary'];h['avatar_prediction']=next(r['prediction'] for r in avatar_predictions if r['label']==h['label'] and r['variant']=='pixels_holistic');h['avatar_hand_coverage_pct']=100*float(np.any(a[:,33:54]!=0,axis=(1,2)).sum()+np.any(a[:,54:]!=0,axis=(1,2)).sum())/(2*len(a))
 for cap in read(B/'capture_manifest.json'):
  data=np.load(O/(cap['id']+'.npz'))['rest_roi3']
  for a in read(O/(cap['id']+'.json'))['added']:
   i=a['frame'];s=a['side'];g=ref[cap['id']]['samples'][i]['hands'][s];off=33 if s=='left' else 54
   if len(g)==21:errors.append(float(np.mean(np.sum((data[i,off:off+21]-np.array([g[str(j)] for j in range(21)]))**2,axis=1))))
 result=dict(human_controls=human,before=old['summary']['linear5'],after=new['summary']['linear5'],added_hand_frames=added,added_reference_count=len(errors),added_rmse=float(np.sqrt(np.mean(errors))) if errors else None,scope='Human controls same labels but unverified source-video pairing. Descriptive only; not causal domain gap or framewise comparison.')
 save(O/'comparison.json',result);report=O/'evaluation/REPORT_TH.md';report.write_text(report.read_text(encoding='utf8')+'\n\n## ตรวจวิดีโอและผลขยายภาพ\nวิดีโอมนุษย์3คลิปเลือกtest IDต่ำสุดที่มีในเครื่องของplan/please/buy ไม่มีหลักฐานว่าตรงคลิปSLcleanเดิม จึงเทียบเชิงคำเท่านั้น ห้ามเทียบพิกัดรายเฟรมหรือสรุปช่องว่างคะแนนจากdomainอย่างเดียว\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':{'detect':detect,'score':score}[sys.argv[1]]()
