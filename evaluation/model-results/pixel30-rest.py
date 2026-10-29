import sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');O=B/'rest_gate_flow';sys.path.insert(0,str(B.parent))
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def prepare():
 import cv2
 O.mkdir(exist_ok=True)
 save(O/'protocol.json',dict(rule='Image-only gate: pose shoulders+hips+wrist visibility>=.5; wrist y >= midhip y - .1 torso length; >=4/5 detected fingertips below wrist. Suppress only3 consecutive positive frames. No Renderer inputs. Offline bidirectional optical flow for bounded gaps <=5 frames; require both endpoints, forward/backward disagreement max<=10px each frame; preserve anchors; reject if any frame rest-gated. No parameter tuning on scores.',variants=['image','rest_gate','rest_roi3','rest_biflow5'],scope='Pilot30 development. Rest rule is a hypothesis, not semantic sign/no-sign truth. Existing ROI3 uses image-only detections.'))
 for c in read(B/'capture_manifest.json'):
  raw=read(B/'detected'/(c['id']+'.json'))['rows'];roi=read(B/'wrist_tracking'/(c['id']+'.json'))['rows'];n=len(raw);image=np.zeros((n,75,2),np.float32);track=image.copy();gate=np.zeros((n,2),bool)
  for i,(d,r) in enumerate(zip(raw,roi)):
   if d['pose']:image[i,:33]=np.array(d['pose'])[:,:2]*808
   for si,(side,wi,offset) in enumerate([('left',15,33),('right',16,54)]):
    if d[side]:image[i,offset:offset+21]=np.array(d[side])[:,:2]*808
    if r['track'][side]:track[i,offset:offset+21]=r['track'][side]
    pose=d['pose'];hand=r['track'][side]
    if pose and hand and all(pose[j][3]>=.5 for j in [11,12,23,24,wi]):
     p=np.array(pose)[:,:2]*808;h=np.array(hand);hip=(p[23]+p[24])/2;shoulder=(p[11]+p[12])/2;torso=np.linalg.norm(hip-shoulder)
     gate[i,si]=p[wi,1]>=hip[1]-.1*torso and sum(h[j,1]>h[0,1] for j in [4,8,12,16,20])>=4
  # Require consecutive evidence; do not classify isolated positives as rest.
  stable=np.zeros_like(gate)
  for si in range(2):
   for i in range(n-2):
    if gate[i:i+3,si].all():stable[i:i+3,si]=True
  track[:,:33]=image[:,:33];gated=image.copy();tracked=track.copy()
  for si,off in enumerate([33,54]):gated[stable[:,si],off:off+21]=0;tracked[stable[:,si],off:off+21]=0
  filled=tracked.copy();gray=[]
  for f in c['frames']:
   im=cv2.imread(str(B/'frames'/c['id']/f['file']),cv2.IMREAD_UNCHANGED);a=im[:,:,3:4]/255;im=np.uint8(np.rint(im[:,:,:3]*a+255*(1-a)));gray.append(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
  added=[]
  def propagate(start,end,p):
   result={start:p};step=1 if end>start else -1
   for t in range(start+step,end+step,step):
    q,st,_=cv2.calcOpticalFlowPyrLK(gray[t-step],gray[t],result[t-step].astype(np.float32).reshape(-1,1,2),None)
    if q is None or not st.all() or not np.isfinite(q).all() or np.any(q<0) or np.any(q>=808):return None
    result[t]=q[:,0]
   return result
  for si,off in enumerate([33,54]):
   observed=np.any(tracked[:,off:off+21]!=0,axis=(1,2));anchors=np.flatnonzero(observed)
   for l,r in zip(anchors[:-1],anchors[1:]):
    if not 1<=r-l-1<=5 or stable[l:r+1,si].any():continue
    f=propagate(int(l),int(r),tracked[l,off:off+21]);b=propagate(int(r),int(l),tracked[r,off:off+21])
    if f is None or b is None or any(np.max(np.linalg.norm(f[t]-b[t],axis=1))>10 for t in range(l,r+1)):continue
    for t in range(l+1,r):w=(t-l)/(r-l);filled[t,off:off+21]=(1-w)*f[t]+w*b[t];added.append([int(t),si])
  np.savez_compressed(O/(c['id']+'.npz'),image=image,rest_gate=gated,rest_roi3=tracked,rest_biflow5=filled,gate=stable)
  save(O/(c['id']+'.json'),dict(label=c['label'],gated_hand_frames=int(stable.sum()),flow_added=added));print('prepared',c['label'],int(stable.sum()),len(added),flush=True)
def score():
 import torch,run_audit as audit
 model,norm,vocab=audit.load_model();refs={c['id']:c for c in read(B/'mixed_reference.json')};rows=[];geoms={};gating=dict(detailed_suppressed=0,fallback_candidate_suppressed=0,detailed_total=0,fallback_candidate_total=0);byclip=[]
 for cap in read(B/'capture_manifest.json'):
  c=refs[cap['id']];z=np.load(O/(c['id']+'.npz'));n=len(c['samples']);mask=np.array([[len(s['hands'][side])==21 for side in ['left','right']] for s in c['samples']]);gate=z['gate'];gating['detailed_suppressed']+=int((mask&gate).sum());gating['fallback_candidate_suppressed']+=int((~mask&gate).sum());gating['detailed_total']+=int(mask.sum());gating['fallback_candidate_total']+=int((~mask).sum())
  byclip.append(dict(label=c['label'],detailed_suppressed=int((mask&gate).sum()),fallback_candidate_suppressed=int((~mask&gate).sum())))
  for variant in ['image','rest_gate','rest_roi3','rest_biflow5']:
   a=z[variant].copy();g=geoms.setdefault(variant,dict(matched=0,squared_sum=0.0,detected_fallback_candidate=0))
   for i,r in enumerate(c['samples']):
    origin,ex,ey=np.array(r['affine'])*r['pixelRatio'];inv=np.linalg.inv(np.stack([ex-origin,ey-origin],axis=1))
    for si,(side,off) in enumerate([('left',33),('right',54)]):
     if np.any(a[i,off:off+21]):
      if mask[i,si]:
       gt=np.array([r['hands'][side][str(j)] for j in range(21)]);g['matched']+=1;g['squared_sum']+=float(np.mean(np.sum((a[i,off:off+21]-gt)**2,axis=1)))
      else:g['detected_fallback_candidate']+=1
    valid=np.any(a[i]!=0,axis=1);a[i,valid]=(a[i,valid]-origin)@inv.T
   x=norm({'frames':torch.from_numpy(a[:,audit.PICK,:].transpose(2,0,1).copy())})['frames'].unsqueeze(0)
   with torch.inference_mode():p=model(x)[0].softmax(0)
   top=p.topk(5);rows.append(dict(label=c['label'],variant=variant,prediction=vocab[int(top.indices[0])],top5=[vocab[int(v)] for v in top.indices]))
 summary={}
 for v,g in geoms.items():
  rr=[r for r in rows if r['variant']==v];ps=[];rs=[];fs=[]
  for l in read(B/'protocol.json')['labels']:
   tp=sum(r['label']==l and r['prediction']==l for r in rr);fp=sum(r['label']!=l and r['prediction']==l for r in rr);p=tp/(tp+fp) if tp+fp else 0;ps.append(p);rs.append(tp);fs.append(2*p*tp/(p+tp) if p+tp else 0)
  summary[v]=dict(correct=sum(r['label']==r['prediction'] for r in rr),precision=100*np.mean(ps),recall=100*np.mean(rs),f1=100*np.mean(fs),matched=g['matched'],rmse=float(np.sqrt(g['squared_sum']/g['matched'])),detected_fallback_candidate=g['detected_fallback_candidate'])
 prior=read(B/'mixed_body_hands/results.json')['predictions'];assert all(r['prediction']==next(p['prediction'] for p in prior if p['label']==r['label'] and p['variant']=='A') for r in rows if r['variant']=='image')
 for f,h in read(B/'protocol.json')['production'].items():assert hashlib.sha256((B.parent.parent/f).read_bytes()).hexdigest()==h
 save(O/'results.json',dict(summary=summary,gating=gating,per_clip_gating=byclip,predictions=rows,production_unchanged=True));print(json.dumps(dict(summary=summary,gating=gating),indent=2))
 lines=['# ตรวจมือพักและรักษาการตรวจจับ: 30 คำ','', 'ตรวจโค้ดพบว่า native SVG hand กลับมาเมื่อ detailedHandAlpha เป็น0 และ pathผ่านconfidence gate ดังนั้นไม่มีมือ21จุดอ้างอิงเป็นเพียงfallback candidate ไม่ใช่หลักฐานว่าSVGแสดงจริงทุกเฟรมหรือผู้ใช้กำลังพัก ห้ามเท่ากับsemantic rest','', 'กฎทดลองใช้พิกัดจากภาพเท่านั้น: มืออยู่ต่ำใกล้สะโพกและอย่างน้อย4ปลายนิ้วอยู่ต่ำกว่าข้อมือ ต่อเนื่อง3เฟรม ส่วนติดตามเติมช่องว่างภายในไม่เกิน5เฟรมโดยoptical flowสองทิศและตรวจความสอดคล้อง ไม่ใช้Rendererช่วยเลือกจุด/ช่วง','', 'เป็นoffline diagnostic ใช้camera metadataย้อนพิกัดเหมือนรอบก่อน ไม่ใช่ระบบimage-onlyแบบend-to-endหรือreal-time ชุดเดิม30คำ ไม่ปรับproduction','', '| วิธี | ถูก/30 | Macro-F1 | พบมือแบบละเอียดอ้างอิง/1872 | RMSE px |','|---|---:|---:|---:|---:|']
 for v,s in summary.items():lines.append(f"| {v} | {s['correct']}/30 | {s['f1']:.2f}% | {s['matched']} | {s['rmse']:.2f} |")
 lines+=['','RMSEวัดเฉพาะมือที่เทียบได้ ซึ่งแต่ละวิธีมีจำนวนไม่เท่ากัน','', '## ผลกฎคัดมือพัก','```json',json.dumps(gating,indent=2),'```','','detailed_suppressedคือมือแบบละเอียดที่กฎลบผิดจากเป้าหมายการรักษามือ ต้องรายงาน ไม่ใช่ประเมินว่าท่าภาษามือผิด รายคำทุกวิธีอยู่results.json และพิกัดทุกเฟรมอยู่npz']
 (O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf8')
if __name__=='__main__':{'prepare':prepare,'score':score}[sys.argv[1]]()
