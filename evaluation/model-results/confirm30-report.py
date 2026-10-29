
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json,math,hashlib
from pathlib import Path
import numpy as np
import cv2
O=Path(__file__).parent/'confirm30-crop-20261009';read=lambda p:_local_json_loads(p.read_text(encoding='utf8'));r=read(O/'results.json');manifest=read(O/'capture_manifest.json');ref={c['id']:c for c in read(O/'coordinate_reference.json')};a=r['predictions']['baseline'];b=r['predictions']['wide_120']
g=[x['label'] for x,y in zip(b,a) if x['prediction']==x['label'] and y['prediction']!=y['label']];l=[x['label'] for x,y in zip(b,a) if x['prediction']!=x['label'] and y['prediction']==y['label']];n=len(g)+len(l);p=min(1,2*sum(math.comb(n,j) for j in range(min(len(g),len(l))+1))/2**n) if n else 1
def wilson(k,n):
 z=1.95996398454;mid=(k/n+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n);return [100*(mid-half),100*(mid+half)]
counts={'baseline':0,'wide_120':0};common={k:[] for k in counts};hashes={};diff=[]
for c in manifest:
 z=np.load(O/'rtm'/(c['id']+'.npz'));v=np.load(O/'model_inputs'/(c['id']+'.npz'));hashes[c['id']]=hashlib.sha256((O/'model_inputs'/(c['id']+'.npz')).read_bytes()).hexdigest()
 assert np.array_equal(z['times'],v['times']);assert len(z['times'])==len(c['frames'])
 for k in counts:counts[k]+=int(z[k+'_mask'].sum())
 for i,s in enumerate(ref[c['id']]['samples']):
  for side,off in [('left',33),('right',54)]:
   h=s['hands'][side]
   if len(h)!=21 or not all(np.any(v[k][i,off:off+21]) for k in counts):continue
   gt=np.array([h[str(j)] for j in range(21)])
   for k in counts:common[k].extend(np.linalg.norm(v[k][i,off:off+21]-gt,axis=1).tolist())
common_stats={k:dict(hand_frames=len(ds)//21,rmse=float(np.sqrt(np.mean(np.square(ds)))),pck10=float(np.mean(np.array(ds)<=10)*100)) for k,ds in common.items()}
stats=dict(gained=g,lost=l,exact_mcnemar_p=p,accuracy_wilson95={k:wilson(r['summary'][k]['correct'],30) for k in counts},accepted=counts,common_geometry=common_stats,input_hashes=hashes)
(O/'confirmation_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf8')
lines=['# ยืนยันกรอบมือกว้างขึ้น20% บน30คลิปใหม่สำหรับการทดลองภาพ','', '9 ตุลาคม 2026 — ไม่แก้ระบบหลัก signing-avatar-github-tinnapat','', '## การตรึงการทดลอง','เลือกคลิปลำดับ31–60ในfrozen200ก่อนดึงจุดและก่อนดูคะแนน คำไม่ซ้ำชุดพัฒนา30คำเดิม ไม่มีคัดคำตามคะแนน','ชุดนี้ใหม่สำหรับการทดลองขนาดกรอบจากภาพ แต่เคยอยู่ในงานประเมินพิกัดก่อนหน้า จึงไม่ใช่ชุดทดสอบที่ไม่เคยใช้ในโครงการ และยังตรวจไม่ได้ว่าทับข้อมูลฝึกST-GCNหรือไม่','เปรียบเทียบกรอบเดิมfactor1.0กับกรอบกว้างขึ้นfactor1.2 โดยตรึงconfidenceมือเฉลี่ยและข้อมือ0.30 โมเดลมือRTMPose-hand256×256 paddingภายใน1.25เหมือนเดิม','anime detector .395, bodyRTMW .3, wrist distance .65forearm, assignmentหนึ่งต่อหนึ่ง stateless, fallbackMediaPipeเมื่อมีมือไม่เช่นนั้นRTMW ไม่เติม/เกลี่ยจุดเพิ่ม','ข้อมูลเข้าโมเดลมาจากภาพPNGเท่านั้น Rendererreferenceเก็บแยกสำหรับประเมินgeometry ไม่ใช้เลือกกรอบ/มือ/คำตอบ','ST-GCNเดิม27จุด2D shoulder normalization ทาย2000คลาส ไม่จำกัดคำตอบ30คำ ไม่มีปรับค่าระหว่างการทดสอบ',f"ภาพ {r['frames']} เฟรม 30FPS เวลาเดิมครบทุกคลิป ผู้ใช้ไม่ได้ขอแก้Rendererจึงไม่แก้production",'', '## ผลรู้จำจริง','| วิธี | ถูก/30 | Accuracy | Macro-P | Macro-R | Macro-F1 | Top5 |','|---|---:|---:|---:|---:|---:|---:|']
for k,s in r['summary'].items():lines.append(f"| {k} | {s['correct']}/30 | {s['accuracy']:.2f}% | {s['precision']:.2f}% | {s['recall']:.2f}% | {s['f1']:.2f}% | {s['top5']:.2f}% |")
lines+=['','Macroเฉลี่ย30คลาสคำจริง zero_division0; Accuracyไม่ใช่เปอร์เซ็นต์ความเหมือนของท่าหรือความเข้าใจของผู้ใช้ASL','', '## เปรียบเทียบแบบจับคู่',f"- กลับมาถูก: {', '.join(g) or 'ไม่มี'}",f"- จากถูกเป็นผิด: {', '.join(l) or 'ไม่มี'}",f'- Exact McNemar p={p:.4f} (ชุดเล็ก exploratory)','ช่วงWilson95%สำหรับaccuracyของแต่ละวิธี ไม่ใช่ช่วงความเชื่อมั่นของผลต่าง:']
for k,ci in stats['accuracy_wilson95'].items():lines.append(f'- {k}: {ci[0]:.2f}–{ci[1]:.2f}%')
lines+=['','## ความตรงของมือ','| วิธี | มือเฟรมมีreferenceและตรวจพบ | RMSE(px) | PCK10เมื่อพบ | PCK10รวมmissing |','|---|---:|---:|---:|---:|']
for k in counts:
 gg=r['geometry'][k];lines.append(f"| {k} | {gg['hand_frames']} | {gg['rmse_px']:.2f} | {gg['pck10_pct']:.2f}% | {gg['pck10_all_reference_pct']:.2f}% |")
lines+=['','PCK10=จุดห่างreferenceไม่เกิน10pxบนภาพ808×808 มีreferenceเฉพาะdetailed hands การไม่มีreferenceไม่พิสูจน์ว่าเป็นท่าพัก','| วิธี | มือเฟรมร่วม | RMSEร่วม | PCK10ร่วม |','|---|---:|---:|---:|']
for k,s in common_stats.items():lines.append(f"| {k} | {s['hand_frames']} | {s['rmse']:.2f} | {s['pck10']:.2f}% |")
delta=r['summary']['wide_120']['correct']-r['summary']['baseline']['correct']
lines+=['','## การตัดสินตามผลยืนยัน',f'ผลต่างจำนวนคำถูกของกรอบกว้างเทียบกรอบเดิม: {delta:+d}/30','แม้คะแนนเพิ่มก็ยังไม่เปลี่ยนproductionจาก30คลิปนี้เพียงชุดเดียว หากคะแนนไม่เพิ่มให้เก็บผลเชิงลบและไม่ปรับค่าเพื่อไล่คะแนนชุดยืนยันนี้','พิจารณาคะแนนรู้จำและgeometryร่วมกัน ไม่ใช้คะแนนทายคำเป็นหลักฐานว่ารายละเอียดนิ้วถูกต้องทั้งหมด','', '## การตรวจสอบ','ตรวจhashต้นฉบับproduction4ไฟล์ตรงชุดพัฒนา ตรวจhashPNGทุกเฟรมก่อนดึงจุด ตรวจmodelONNX SHA256 ใช้timestampเดียวกันทั้ง2วิธี','บันทึกprotocolก่อนการรัน, casesและhash, rawกรอบ+จุด+confidenceทุกcandidate, assignments, พิกัดเข้าST-GCNและคำทายรายคลิปครบ','MediaPipeและRTMรันบางช่วงพร้อมกัน ไม่รายงานเวลาที่ได้เป็นbenchmarkเปรียบเทียบความเร็ว','', '## คำทายรายคลิป','| คำจริง | กรอบเดิม | กว้าง20% |','|---|---|---|']
for x,y in zip(a,b):lines.append(f"| {x['label']} | {x['prediction']} | {y['prediction']} |")
lines+=['','## ข้อสรุปของชุดยืนยันนี้','กรอบเดิมและกรอบกว้าง20%ได้4/30เหมือนกัน (Accuracy13.33%, Macro-P11.67%, Macro-R13.33%, Macro-F1 12.22%, Top5 20%) ไม่มีคำที่กลับมาถูกเพิ่มหรือเสียไป','ดังนั้นคะแนนเพิ่ม7→9/30ในชุดพัฒนาไม่เกิดซ้ำบนชุดยืนยันนี้ ยังไม่มีหลักฐานพอให้เปลี่ยนกรอบเป็นค่าdefault','กรอบกว้างรับมือเพิ่ม2124→2338 hand-slots และPCK10รวมmissing53.12→53.41% แต่RMSEบน2160มือเฟรมเดียวกันแย่ลง27.81→28.07px ผลgeometryจึงไม่ดีขึ้นทุกด้าน','คงกรอบเดิมเป็นbaseline เก็บกรอบกว้างเป็นผลทดลอง ไม่ปรับขนาดหรือconfidenceเพิ่มบนชุดยืนยันนี้','ขั้นถัดไปควรตรวจความสามารถของโมเดลรู้จำบนข้อมูลมนุษย์ที่มีรหัสคลิปชัด และ/หรือความคลาดเคลื่อนรายข้อนิ้ว เพื่อแยกdomain mismatchจากการตรวจจุด ไม่ตีความ4/30ว่าAvatarถูกต้องเพียง13.33%','ข้อจำกัดการแยกข้อมูล: 30คำชุดนี้ไม่ซ้ำชุดพัฒนาภาพ แต่เคยใช้ในงานพิกัดก่อนหน้าและไม่ยืนยันไม่ทับชุดฝึกST-GCN จึงใช้คำว่าprospective confirmationสำหรับcropเท่านั้น ไม่ใช่unseen end-to-end benchmark']
(O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf8')
edges=[(0,1),(1,2),(2,3),(3,4)]+[(0,k) for k in [5,9,13,17]]+[(k+j,k+j+1) for k in [5,9,13,17] for j in range(3)]
selected=[c for c in manifest if c['label'] in set(g+l)][:6] or manifest[:3];panels=[]
for c in selected:
 z=np.load(O/'model_inputs'/(c['id']+'.npz'));i=int(np.argmax(np.linalg.norm(z['baseline'][:,33:]-z['wide_120'][:,33:],axis=2).mean(1)));f=c['frames'][i];im=cv2.imread(str(O/'frames'/c['id']/f['file']),-1);alpha=im[:,:,3:4].astype(np.float32)/255;im=np.rint(im[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8);row=[]
 for k in counts:
  img=im.copy()
  for side,off,col in [('left',33,(255,80,0)),('right',54,(0,40,255))]:
   h=ref[c['id']]['samples'][i]['hands'][side]
   if len(h)==21:
    gt=np.array([h[str(j)] for j in range(21)]).astype(int)
    for j,l in edges:cv2.line(img,tuple(gt[j]),tuple(gt[l]),(0,150,0),2)
   q=z[k][i,off:off+21].astype(int)
   if np.any(q):
    for j,l in edges:cv2.line(img,tuple(q[j]),tuple(q[l]),col,1)
  img=cv2.resize(img,(404,404));band=np.full((65,404,3),255,np.uint8);pred=r['predictions'][k][int(c['id'])]['prediction'];cv2.putText(band,k,(6,20),0,.55,(0,0,0),1);cv2.putText(band,f"{c['label']} f{i} / clip prediction: {pred}",(6,45),0,.47,(0,0,0),1);row.append(np.vstack([band,img]))
 panels.append(np.hstack(row))
cv2.imwrite(str(O/'comparison.png'),np.vstack(panels));print(json.dumps(stats,ensure_ascii=True))
