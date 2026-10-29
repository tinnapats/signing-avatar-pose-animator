import json,math,hashlib
from pathlib import Path
from collections import Counter
import numpy as np,cv2
O=Path(__file__).parent/'optical-hand-track-20261009';B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');read=lambda p:json.loads(p.read_text(encoding='utf8'));r=read(O/'results.json');manifest=read(B/'capture_manifest.json');refs={c['id']:c for c in read(B/'coordinate_reference.json')};counts=Counter();rows=[];ms=[]
for c in manifest:
 z=np.load(O/(c['id']+'.npz'));a=np.load(O/'inputs'/(c['id']+'.npz'));logs=read(O/(c['id']+'_raw.json'));ms.extend(z['ms'].tolist())
 for i,rr in enumerate(logs):
  for log in rr:
   counts[log['status']]+=1
   if log['status']!='accepted_flow':continue
   side=log['side'];off=[33,54][side];h=refs[c['id']]['samples'][i]['hands'][['left','right'][side]];row=dict(id=c['id'],label=c['label'],frame=i,side=side,age=log['age'],box=log['box'],reference=len(h)==21)
   if len(h)==21:
    g=np.array([h[str(j)] for j in range(21)]);row['baseline_mean_error_px']=float(np.linalg.norm(a['baseline'][i,off:off+21]-g,axis=1).mean());row['flow_mean_error_px']=float(np.linalg.norm(a['flow_fallback'][i,off:off+21]-g,axis=1).mean())
   rows.append(row)
old=r['predictions']['baseline'];new=r['predictions']['flow_fallback'];gained=[a['label'] for a,b in zip(new,old) if a['prediction']==a['label'] and b['prediction']!=b['label']];lost=[a['label'] for a,b in zip(new,old) if a['prediction']!=a['label'] and b['prediction']==b['label']];n=len(gained)+len(lost);p=min(1,2*sum(math.comb(n,j) for j in range(min(len(gained),len(lost))+1))/2**n) if n else 1
refrows=[x for x in rows if x['reference']];improved=sum(x['flow_mean_error_px']<x['baseline_mean_error_px']-1e-6 for x in refrows);worse=sum(x['flow_mean_error_px']>x['baseline_mean_error_px']+1e-6 for x in refrows)
diag=dict(status_counts=dict(counts),accepted_hand_frames=len(rows),accepted_with_reference=len(refrows),improved=improved,worsened=worse,added_stage_mean_ms=float(np.mean(ms)),gained=gained,lost=lost,exact_mcnemar_p=p,rows=rows)
(O/'diagnostics.json').write_text(json.dumps(diag,ensure_ascii=False,indent=2),encoding='utf8')
lines=['# Optical-flow hand ROI tracking → RTMPose-hand','', '9 ตุลาคม2026 — signing-avatar-github-tinnapat ชุดพัฒนา30คลิป 1964เฟรม30FPS ไม่แก้ระบบหลัก','', '## วิธี','ตรึงRTMWลำตัว/normalize, ST-GCN2000เดิม, เวลาเดิม, bboxเดิมและconfidence0.30 ไม่เปิดชุดยืนยัน','เมื่อมือanimeเดิมผ่านเกณฑ์ ใช้ผลเดิมและเริ่มกรอบtrackingใหม่ ถ้าไม่ผ่านให้ติดตามกรอบด้วยภาพก่อนหน้า→ปัจจุบัน แล้วรันRTMPose-handใหม่จากภาพปัจจุบัน','Sparse Lucas–Kanade corners60, FBerror<=1.5px, residualจากmediantranslation<=3px, inliers>=6และอย่างน้อย50%; maxgap3เฟรมตั้งแต่seedจากdetectorล่าสุด ไม่ต่ออายุด้วยผลtracking','ติดตามเฉพาะtranslation ไม่เปลี่ยนขนาดกรอบ ไม่มีค้าง/เกลี่ย/เติมlandmarkจากเฟรมก่อน จุดนิ้วทั้งหมดของมือtrackingที่ผ่านเกณฑ์อ่านจากภาพใหม่','RTMPosehandmean/wrist>=.30, RTMWbody>=.30, wristdistance<=.65forearm; ไม่ใช้trackedwristที่ใกล้มือที่ยอมรับอีกข้าง<20px เมื่อไม่ผ่านใช้MP/RTMWfallbackเดิม','ไม่มีRenderer/payload/ฉลากคำในการติดตามหรือเลือกมือ Rendererreferenceใช้หลังscoreเพื่อประเมิน','', '## คะแนนจริง','| วิธี | ถูก/30 | Macro-P | Macro-R | Macro-F1 | Top5 | PCK10รวมmissing | RMSE(px) |','|---|---:|---:|---:|---:|---:|---:|---:|']
for k in ['baseline','flow_fallback']:
 s=r['summary'][k];gg=r['geometry'][k];lines.append(f"| {k} | {s['correct']}/30 | {s['precision']:.2f}% | {s['recall']:.2f}% | {s['f1']:.2f}% | {s['top5']:.2f}% | {gg['pck10_all_reference_pct']:.2f}% | {gg['rmse_px']:.2f} |")
lines+=['','Macroเฉลี่ย30true classes zero_division0 ไม่จำกัดคำตอบ; PCK10ไม่ใช่ความถูกต้องASL',f"คำที่กลับมาถูก: {', '.join(gained) or 'ไม่มี'}; คำที่เสีย: {', '.join(lost) or 'ไม่มี'}; exploratory McNemar p={p:.4f}",'', '## การติดตามช่วยเพิ่มมือจริงแค่ไหน',f'เพิ่มมือจากtracking {len(rows)} hand-frames จาก3928hand-slotsทั้งหมด',f'มีreferenceเทียบได้ {len(refrows)} ช่วง: mean21-point errorดีขึ้น {improved} แย่ลง {worse} ที่เหลือเท่าเดิม',f'เวลาเพิ่มเฉลี่ย {np.mean(ms):.2f}ms/frameเฉพาะtracking+handอ่านใหม่ ไม่รวมdetector/body/recognizerและไม่ใช่benchmarkreal-time', '```json',json.dumps(dict(counts),indent=2),'```','จำนวนมือเพิ่มหมายถึงแทนที่MP/RTMWfallbackในช่องที่animeเดิมไม่ผ่าน ไม่ได้แปลว่าระบบเดิมไม่มีมือในทุกช่องนั้น','', '## ข้อจำกัด','ช่องว่างยาวเกิน3เฟรมหรือเริ่มต้นไม่เคยตรวจผ่าน ไม่อยู่ในขอบเขตการซ่อมของวิธีนี้','featureอาจติดไปกับฉาก/แขนหรือมืออีกข้าง; การหมุนและการบังมืออาจทำให้translationtrackingล้มเหลว','ยังไม่ทดสอบtrackingด้วยวิธีอื่น/หน้าต่างยาวอื่น จึงไม่สรุปว่าtrackingทุกวิธีไม่ได้ผล','ชุดพัฒนาถูกทดลองซ้ำ ผลไม่ใช่held-out; ไม่ปรับเกณฑ์รายคำหรือไล่windowเพิ่มหลังเห็นคะแนน','บันทึกพิกัดและconfidenceจากภาพใหม่, bbox, จำนวนinlier, เหตุผลปฏิเสธ, timestampsและhashครบ; baselineคะแนน/Top1/Top5/targetprobทำซ้ำตรงรอบก่อน; production4ไฟล์ไม่เปลี่ยน','', '## รายคำ','| คำ | baseline | flow_fallback |','|---|---|---|']
for a,b in zip(old,new):lines.append(f"| {a['label']} | {a['prediction']} | {b['prediction']} |")
lines+=['','## ข้อสรุป','คะแนนbaselineและtrackingเท่ากัน7/30 Macro-F1 23.33% Top5 56.67% ไม่มีคำกลับมาถูกเพิ่มหรือเสียไป','trackingเพิ่ม42hand-frames มีreference37: ดีขึ้น19แย่ลง18; PCK10รวมmissing53.42→53.81% แต่RMSE30.75→30.80px ผลไม่ดีขึ้นทุกด้าน','ไม่เปลี่ยนproductionและไม่ขยายtrackingwindowเพื่อไล่คะแนนบนชุดเดิมต่อ เก็บเป็นnegative/mixed resultของtrackingแบบtranslation3framesนี้ ไม่เหมารวมว่าtrackingทุกชนิดไม่ได้ผล','ตามลำดับแผน ขั้นถัดไปคือประเมินความพร้อมSDPose-WholeBodyและทรัพยากรเครื่องก่อนติดตั้ง/ดาวน์โหลดใหญ่ แล้วเริ่มจากภาพย่อยเพื่อตรวจว่าหาจุดบนAvatarได้จริง ไม่เริ่มfine-tuneจากผลนี้ทันที','1370hand-slotsไม่มีseedที่ใช้งานได้ขณะนั้น สะท้อนว่าขอบเขตshort-gap trackingไม่ครอบคลุมช่วงที่ไม่เคยตรวจผ่านหรือหลังstateถูกยกเลิก มิใช่จำนวนมือหายจริงทั้งหมด']
(O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf8')
edges=[(0,1),(1,2),(2,3),(3,4)]+[(0,k) for k in [5,9,13,17]]+[(k+j,k+j+1) for k in [5,9,13,17] for j in range(3)];chosen=sorted(refrows,key=lambda x:x['flow_mean_error_px']-x['baseline_mean_error_px']);examples=chosen[:2]+chosen[-2:] if len(chosen)>4 else chosen;panels=[]
for ex in examples:
 c=next(c for c in manifest if c['id']==ex['id']);i=ex['frame'];side=ex['side'];off=[33,54][side];f=c['frames'][i];im=cv2.imread(str(B/'frames'/c['id']/f['file']),-1);alpha=im[:,:,3:4].astype(np.float32)/255;im=np.rint(im[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8);z=np.load(O/'inputs'/(c['id']+'.npz'));g=np.array([refs[c['id']]['samples'][i]['hands'][['left','right'][side]][str(j)] for j in range(21)]);row=[]
 for k in ['baseline','flow_fallback']:
  img=im.copy()
  if k=='flow_fallback':
   box=np.rint(ex['box']).astype(int);cv2.rectangle(img,tuple(box[:2]),tuple(box[2:]),(160,60,180),2)
  for pts,col in [(g,(0,150,0)),(z[k][i,off:off+21],(0,40,230))]:
   pts=np.rint(pts).astype(int)
   for a,b in edges:cv2.line(img,tuple(pts[a]),tuple(pts[b]),col,2)
  img=cv2.resize(img,(404,404));band=np.full((55,404,3),255,np.uint8);cv2.putText(band,k,(6,20),0,.55,(0,0,0),1);cv2.putText(band,f"{c['label']} f{i} age{ex['age']} | green=ref red=estimate",(6,43),0,.43,(0,0,0),1);row.append(np.vstack([band,img]))
 panels.append(np.hstack(row))
if panels:cv2.imwrite(str(O/'examples.png'),np.vstack(panels))
print(json.dumps({k:v for k,v in diag.items() if k!='rows'},ensure_ascii=True))
