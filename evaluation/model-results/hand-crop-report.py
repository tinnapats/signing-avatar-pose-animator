import sys,importlib.util,json,math
from pathlib import Path
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('trial',Path(__file__).with_name('hand-crop-trial.py'));t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
np=t.np;O=t.O;B=t.B;read=t.read;r=read(O/'results.json');manifest=read(B/'capture_manifest.json');refs={c['id']:c for c in read(B/'coordinate_reference.json')};paired={};counts={k:0 for k in t.FACTORS};dur={k:[] for k in t.FACTORS};matched={k:[] for k in ['baseline']+list(t.FACTORS)}
for k in t.FACTORS:
 rr=r['predictions'][k];base=r['predictions']['baseline'];g=[a['label'] for a,b in zip(rr,base) if a['prediction']==a['label'] and b['prediction']!=b['label']];l=[a['label'] for a,b in zip(rr,base) if a['prediction']!=a['label'] and b['prediction']==b['label']];n=len(g)+len(l);p=min(1,2*sum(math.comb(n,j) for j in range(min(len(g),len(l))+1))/2**n) if n else 1;paired[k]=dict(gained=g,lost=l,exact_mcnemar_p=p)
spec=importlib.util.spec_from_file_location('ae',Path(__file__).with_name('anime-eval.py'));ae=importlib.util.module_from_spec(spec);spec.loader.exec_module(ae)
def arrays(c):
 old=ae.data(c);out={'baseline':old['anime_replace_best']};z=np.load(O/(c['id']+'.npz'))
 for k in t.FACTORS:
  a=old['previous_best'].copy()
  for side,off in enumerate([33,54]):
   mask=z[k+'_mask'][:,side];a[mask,off:off+21]=z[k][mask,side]
  out[k]=a
 return out
for c in manifest:
 z=np.load(O/(c['id']+'.npz'));a=arrays(c)
 for k in t.FACTORS:counts[k]+=int(z[k+'_mask'].sum());dur[k].extend(z[k+'_ms'].tolist())
 for i,ref in enumerate(refs[c['id']]['samples']):
  for side,off in [('left',33),('right',54)]:
   h=ref['hands'][side]
   if len(h)!=21 or not all(np.any(v[i,off:off+21]) for v in a.values()):continue
   g=np.array([h[str(j)] for j in range(21)])
   for k,v in a.items():matched[k].extend(np.linalg.norm(v[i,off:off+21]-g,axis=1).tolist())
common={k:dict(hand_frames=len(v)//21,rmse_px=float(np.sqrt(np.mean(np.square(v)))),pck10_pct=float(100*np.mean(np.array(v)<=10))) for k,v in matched.items()}
t.save(O/'paired.json',paired);t.save(O/'diagnostic_summary.json',dict(accepted=counts,hand_inference_mean_ms={k:float(np.mean(v)) for k,v in dur.items()},common_geometry=common))
lines=['# ทดลองขนาดกรอบตัดมือจากภาพ Avatar','', '9 ตุลาคม 2026 — ข้อมูลเฉพาะ signing-avatar-github-tinnapat; ไม่แก้ระบบหลัก','', '## เงื่อนไข','30คลิปเดิม 1964เฟรม 30FPS เวลาเดิม ภาพ808×808 ใช้กรอบจากanime detectorเดิม เปลี่ยนขนาดกรอบรอบศูนย์กลางเท่านั้น รันRTMPose-handใหม่จริง และST-GCNเดิมทาย2000คำ','- tight_080: ย่อกรอบเหลือ80% ก่อนpaddingภายใน1.25 ทำให้พื้นที่ตัดสุดท้ายมีด้านยาว1.00เท่ากรอบ detector','- baseline: กรอบเดิม พื้นที่ตัดสุดท้าย1.25เท่ากรอบ detector ใช้ผลcachedและตรวจทำซ้ำบนเฟรมแรก','- wide_120: ขยายกรอบเป็น120% พื้นที่ตัดสุดท้าย1.50เท่ากรอบ detector','พื้นที่ตัดเป็นสี่เหลี่ยมจัตุรัสตามด้านยาว และถูกปรับเป็น256×256เหมือนกันทั้งหมด ไม่ใช่เพิ่มความละเอียดโมเดล','ตรึงconfidenceมือเฉลี่ย/ข้อมือ0.30, body0.30, ระยะข้อมือ0.65เท่าท่อนแขน, assignmentหนึ่งต่อหนึ่ง, fallbackMediaPipe/RTMWเหมือนเดิม ไม่ทำsmoothing/เติมเฟรมเพิ่ม','ไม่ใช้Renderer/CSV/payloadสร้างพิกัดเข้าโมเดล Rendererใช้เฉพาะประเมินความคลาดเคลื่อน','', '## คะแนนจริง','| วิธี | ถูก/30 | Macro-P | Macro-R | Macro-F1 | Top5 | PCK10รวมmissing | RMSEเมื่อพบ(px) |','|---|---:|---:|---:|---:|---:|---:|---:|']
for k in ['previous_best','baseline']+list(t.FACTORS):
 s=r['summary'][k];g=r['geometry'][k];lines.append(f"| {k} | {s['correct']}/30 | {s['precision']:.2f}% | {s['recall']:.2f}% | {s['f1']:.2f}% | {s['top5']:.2f}% | {g['pck10_all_reference_pct']:.2f}% | {g['rmse_px']:.2f} |")
lines+=['','Macroเฉลี่ย30true classes zero_division0; PCK10คือจุดห่างreferenceไม่เกิน10px ไม่ใช่ความถูกต้องASL','', '## คำที่ได้–เสียเทียบbaseline']
for k,p in paired.items():lines.append(f"- {k}: ได้ {', '.join(p['gained']) or 'ไม่มี'}; เสีย {', '.join(p['lost']) or 'ไม่มี'}; exploratory exact McNemar p={p['exact_mcnemar_p']:.4f}")
lines+=['','## Geometry เฉพาะมือและเฟรมที่พบร่วมกันทุกขนาดกรอบ','| วิธี | hand-framesร่วม | RMSE(px) | PCK10 |','|---|---:|---:|---:|']
for k,g in common.items():lines.append(f"| {k} | {g['hand_frames']} | {g['rmse_px']:.2f} | {g['pck10_pct']:.2f}% |")
lines+=['','## จำนวนมือที่ผ่านและเวลาเฉพาะขั้นมือ','จำนวนรวมทั้งช่วงไม่มีdetailed reference ไม่ใช่recallของมือทำท่า','| วิธี | ยอมรับ/3928 hand-slots | เวลาเฉลี่ย(ms/frame) |','|---|---:|---:|']
for k in t.FACTORS:lines.append(f'| {k} | {counts[k]} | {np.mean(dur[k]):.2f} |')
lines+=['','เวลาเป็นhand inferenceและassignmentบนCPUเท่านั้น ไม่รวมdetector/body/recognizer ไม่ใช่FPSระบบปลายทาง','', '## ข้อจำกัด','30คลิปนี้ใช้พัฒนาซ้ำ ไม่มีheld-outใหม่ คำละ1คลิป จึงไม่ยืนยันผลทั่วไป และไม่ได้แสดงความเข้าใจของผู้ใช้ASL','baselineคำทายTop1/Top5และtarget probabilityตรวจตรงรอบก่อน ไฟล์production4ไฟล์ตรวจhashตรงเดิม เก็บrawจุด/คะแนนทุกcandidate, boxes, assignments, timestamp, hashesและprotocolไว้ครบ','', '## คำทายรายคลิป','| คำ | baseline | tight_080 | wide_120 |','|---|---|---|---|']
for i,c in enumerate(manifest):lines.append('| '+c['label']+' | '+' | '.join(r['predictions'][k][i]['prediction'] for k in ['baseline']+list(t.FACTORS))+' |')
lines+=['','## ข้อสรุปเฉพาะรอบนี้','กรอบกว้างขึ้น20%ที่confidence0.30ได้9/30 เพิ่มarrestและletter ไม่เสียคำที่เดิมถูก เทียบbaseline7/30; กรอบแคบลง20%ได้8/30 เพิ่มarrestไม่เสียคำเดิม','กรอบกว้างมีPCK10รวมmissing53.56%เทียบ53.42%เดิม แต่RMSE30.79pxเทียบ30.75pxเดิม จึงยังไม่กล่าวว่าพิกัดดีขึ้นทุกด้าน; Top5เท่ากัน56.67%ทั้ง3ขนาด','เทียบรอบconfidence0.40เดิม8/30: กรอบกว้างconfidence0.30ได้9/30 โดยsinceยังถูก แต่เป็นการเปรียบเทียบในdevelopment setเดิม ไม่ใช่ผลยืนยันอิสระ','แนะนำตรึงwide_120+confidence0.30เป็นcandidateหนึ่งตัว แล้วทดสอบกับคลิปใหม่พร้อมbaselineเดิมก่อนปรับproduction ไม่ไล่เพิ่มค่ากรอบเพื่อเลือกคะแนนสูงสุดบน30คำเดิมต่อไป','ผลp=.5จาก2คำที่ได้เพิ่มและ0คำเสียเป็นexploratory ไม่ยืนยันความเหนือกว่า']
(O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf8')
import cv2
edges=[(0,1),(1,2),(2,3),(3,4)]+[(0,k) for k in [5,9,13,17]]+[(k+j,k+j+1) for k in [5,9,13,17] for j in range(3)]
changed=set(sum([p['gained']+p['lost'] for p in paired.values()],[])); selected=[c for c in manifest if c['label'] in changed][:6] or [manifest[1],manifest[2]];panels=[]
for c in selected:
 a=arrays(c);i=int(np.argmax(np.linalg.norm(a['tight_080'][:,33:]-a['wide_120'][:,33:],axis=2).mean(1)));f=c['frames'][i];p=B/'frames'/c['id']/f['file'];assert t.sha(p)==f['sha256'];im=t.ah.base.image(cv2,p);row=[]
 for k in a:
  img=im.copy()
  for side,off,col in [('left',33,(255,80,0)),('right',54,(0,40,255))]:
   h=refs[c['id']]['samples'][i]['hands'][side]
   if len(h)==21:
    g=np.array([h[str(j)] for j in range(21)]).astype(int)
    for j,l in edges:cv2.line(img,tuple(g[j]),tuple(g[l]),(0,150,0),2)
   q=a[k][i,off:off+21].astype(int)
   if np.any(q):
    for j,l in edges:cv2.line(img,tuple(q[j]),tuple(q[l]),col,1)
  img=cv2.resize(img,(404,404));band=np.full((65,404,3),255,np.uint8);pred=r['predictions'][k][int(c['id'])]['prediction'];cv2.putText(band,k,(6,20),0,.55,(0,0,0),1);cv2.putText(band,f"{c['label']} f{i} / clip prediction: {pred}",(6,45),0,.47,(0,0,0),1);row.append(np.vstack([band,img]))
 panels.append(np.hstack(row))
cv2.imwrite(str(O/'comparison.png'),np.vstack(panels))
print(json.dumps(dict(paired=paired,common=common,accepted=counts),ensure_ascii=True))
