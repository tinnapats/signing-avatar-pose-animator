
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import sys,json,time,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');O=B/'rotated_roi_20261008'
def read(p):return _local_json_loads(p.read_text(encoding='utf8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def detect():
 import cv2,mediapipe as mp
 O.mkdir(exist_ok=True);save(O/'detector_protocol.json',dict(source='rest_gate_flow/rest_roi3',angles=[0,90,-90],crop_side='clip(2.4*forearm,128,320)',resize=256,pose_visibility=.3,hands_confidence=.5,selection='nearest normalized wrist distance, minus .05*confidence; wrist distance <=.65 forearm; reject closer to opposite detected pose wrist and duplicate existing opposite hand within20px; fill missing non-rest only',scope='All rules fixed before scores. No Renderer reference or target word used.'))
 for cap in read(B/'capture_manifest.json'):
  target=O/(cap['id']+'.npz')
  if target.exists():continue
  z=np.load(B/'rest_gate_flow'/(cap['id']+'.npz'));base=z['rest_roi3'];data=base.copy();gate=z['gate'];raw=read(B/'detected'/(cap['id']+'.json'))['rows'];added=[];attempts=0;begin=time.perf_counter()
  with mp.solutions.hands.Hands(static_image_mode=True,max_num_hands=2,model_complexity=1,min_detection_confidence=.5) as detector:
   for i,d in enumerate(raw):
    missing=[(si,s,wi,ei,off) for si,s,wi,ei,off in [(0,'left',15,13,33),(1,'right',16,14,54)] if not gate[i,si] and not np.any(base[i,off:off+21])]
    if not missing or not d['pose']:continue
    path=B/'frames'/cap['id']/cap['frames'][i]['file'];im=cv2.imread(str(path),cv2.IMREAD_UNCHANGED);a=im[:,:,3:4]/255;rgb=cv2.cvtColor(np.uint8(np.rint(im[:,:,:3]*a+255*(1-a))),cv2.COLOR_BGR2RGB);pose=np.array(d['pose']);xy=pose[:,:2]*808
    for si,s,wi,ei,off in missing:
     if min(pose[wi,3],pose[ei,3])<.3:continue
     wrist=xy[wi];elbow=xy[ei];length=float(np.linalg.norm(wrist-elbow))
     if length<10:continue
     center=wrist+.25*(wrist-elbow);size=float(np.clip(2.4*length,128,320));options=[]
     for angle in [0,90,-90]:
      mat=cv2.getRotationMatrix2D(tuple(center),angle,256/size);mat[:,2]+=np.array([128,128])-center;roi=cv2.warpAffine(rgb,mat,(256,256),borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255));result=detector.process(roi);attempts+=1;inv=cv2.invertAffineTransform(mat)
      for h,handed in zip(result.multi_hand_landmarks or [],result.multi_handedness or []):
       q=np.array([[p.x*256,p.y*256,1] for p in h.landmark])@inv.T;distance=float(np.linalg.norm(q[0]-wrist));otherwi=16 if wi==15 else 15;otheroff=54 if off==33 else 33
       if distance>.65*length or np.any(q<0) or np.any(q>=808):continue
       if pose[otherwi,3]>=.3 and np.linalg.norm(q[0]-xy[otherwi])+4<distance:continue
       if np.any(data[i,otheroff:otheroff+21]) and np.linalg.norm(q[0]-data[i,otheroff])<20:continue
       conf=float(handed.classification[0].score);options.append((distance/length-.05*conf,q,angle,distance,conf))
     if options:
      best=min(options,key=lambda r:r[0]);data[i,off:off+21]=best[1];added.append(dict(frame=i,side=s,angle=best[2],wrist_distance=best[3],handedness_confidence=best[4]))
  for si,off in enumerate([33,54]):
   known=np.any(base[:,off:off+21]!=0,axis=(1,2));assert np.array_equal(data[known,off:off+21],base[known,off:off+21]);assert not data[gate[:,si],off:off+21].any()
  np.savez_compressed(target,rest_roi3=data,gate=gate);save(O/(cap['id']+'.json'),dict(label=cap['label'],added=added,attempts=attempts,seconds=time.perf_counter()-begin));print(cap['label'],'added',len(added),'attempts',attempts,flush=True)
def score():
 short=Path(__file__).with_name('pixel30-short-gaps.py');s=short.read_text(encoding='utf8');tail="exec(compile(source,str(Path(__file__)), 'exec'))";assert s.count(tail)==1;ns={'__file__':str(short)};exec(compile(s.replace(tail,'pass'),str(short),'exec'),ns);code=ns['source']
 code=code.replace("O=B/'short_gaps_20261008'","O=B/'rotated_roi_20261008/evaluation'")
 code=code.replace("np.load(B/'rest_gate_flow'/(c['id']+'.npz'))","np.load(B/'rotated_roi_20261008'/(c['id']+'.npz'))")
 code=code.replace("names={'zero_baseline':'ไม่เติม','linear3':'เติมช่องว่างไม่เกิน3เฟรม','linear5':'เติมช่องว่างไม่เกิน5เฟรม','linear10':'เติมช่องว่างไม่เกิน10เฟรม'}","names={'zero_baseline':'ตรวจหลายมุม ไม่เติมเพิ่ม','linear5':'ตรวจหลายมุมและเติมไม่เกิน5เฟรม'}")
 line="  if variant=='zero_baseline':assert row['prediction']==next(r['prediction'] for r in prior if r['label']==c['label'] and r['variant']=='rest_roi3')";assert code.count(line)==1;code=code.replace(line,'  # New detector baseline, intentionally different from archived predictions.')
 code=code.replace('ใช้ฐานrest_roi3','ใช้ฐานrest_roi3พร้อมตรวจซ้ำROIหลายมุม').replace('ไม่เกิน3,5,10เฟรม','ไม่เกิน5เฟรม')
 code=code.replace("source='rest_gate_flow/rest_roi3'","source='rotated_roi_20261008/rest_roi3'")
 exec(compile(code,str(Path(__file__)),'exec'),{'__file__':__file__})
 report=O/'evaluation/REPORT_TH.md';report.write_text(report.read_text(encoding='utf8')+'\n\nขั้นตรวจใหม่อ่านภาพcropที่0,+90,-90องศา ย่อ/ขยายเป็น256×256 โดยอิงposeข้อมือจากภาพ บันทึกเฉพาะมือที่หายและไม่ถูกคัดพัก จุดเดิมคงไว้ทั้งหมด กติกาดูdetector_protocol.json\n',encoding='utf8')
 old=read(B/'short_gaps_20261008/results.json');new=read(O/'evaluation/results.json');references={c['id']:c for c in read(B/'coordinate_reference.json')};squares=[];total=0;angles={};attempts=0;seconds=0
 for cap in read(B/'capture_manifest.json'):
  details=read(O/(cap['id']+'.json'));a=np.load(O/(cap['id']+'.npz'))['rest_roi3'];total+=len(details['added']);attempts+=details['attempts'];seconds+=details['seconds']
  for added in details['added']:
   angle=str(added['angle']);angles[angle]=angles.get(angle,0)+1;i=added['frame'];side=added['side'];off=33 if side=='left' else 54;g=references[cap['id']]['samples'][i]['hands'][side]
   if len(g)==21:squares.append(float(np.mean(np.sum((a[i,off:off+21]-np.array([g[str(j)] for j in range(21)]))**2,axis=1))))
 comparison=dict(before=old['summary']['linear5'],after=new['summary']['linear5'],new_detection_hand_frames=total,new_detection_reference_count=len(squares),new_detection_rmse=float(np.sqrt(np.mean(squares))) if squares else None,angles=angles,detector_calls=attempts,detector_seconds=seconds)
 save(O/'comparison.json',comparison);print(json.dumps(comparison,indent=2))
 report.write_text(report.read_text(encoding='utf8')+'\n## เทียบรอบเดิมและจุดตรวจใหม่\n```json\n'+json.dumps(comparison,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf8')
if __name__=='__main__':{'detect':detect,'score':score}[sys.argv[1]]()
