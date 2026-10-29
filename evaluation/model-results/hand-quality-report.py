import sys,importlib.util,math,json
from pathlib import Path
sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('q',Path(__file__).with_name('hand-quality-trial.py'));q=importlib.util.module_from_spec(s);s.loader.exec_module(q)
np=q.np;e=q.e;O=q.O;read=e.read;r=read(O/'results.json');manifest=read(e.B/'capture_manifest.json');refs={c['id']:c for c in read(e.B/'coordinate_reference.json')}
paired={};rejects={k:0 for k in ['anchor_guard','shape_guard','combined_guard']};rows=[];wrist={k:[] for k in ['previous_best','baseline','anchor_guard','shape_guard','combined_guard']};relative={k:[] for k in wrist};absolute={k:[] for k in wrist};shoulders=[]
for k in q.NAMES[3:]:
 a=r['predictions'][k];b=r['predictions']['baseline'];g=[x['label'] for x,y in zip(a,b) if x['prediction']==x['label'] and y['prediction']!=y['label']];l=[x['label'] for x,y in zip(a,b) if x['prediction']!=x['label'] and y['prediction']==y['label']];n=len(g)+len(l);p=min(1,2*sum(math.comb(n,j) for j in range(min(len(g),len(l))+1))/2**n) if n else 1;paired[k]=dict(gained=g,lost=l,exact_mcnemar_p=p)
for c in manifest:
 z=np.load(O/'inputs'/(c['id']+'.npz'));body=np.load(q.ae.R/(c['id']+'.npz'));logs=read(O/'inputs'/(c['id']+'.json'))['rules']
 for x in logs:rejects[x['variant']]+=int(x['reject'])
 for i,ref in enumerate(refs[c['id']]['samples']):
  mp=z['mediapipe'][i];rt=z['baseline'][i]
  if all(np.any(p[j]) for p in [mp,rt] for j in [11,12]):
   sm=float(np.linalg.norm(mp[11]-mp[12]));sr=float(np.linalg.norm(rt[11]-rt[12]));
   if sr>0:shoulders.append(sm/sr)
  for side,off,(w,elbow) in zip(['left','right'],[33,54],[(9,7),(10,8)]):
   bp=body['rtmw_det'][i];bs=body['rtmw_det_scores'][i];h=ref['hands'][side]
   for k in wrist:
    p=z[k][i,off:off+21]
    if not np.any(p):continue
    if bs[w]>=.3 and bs[elbow]>=.3:
     wrist[k].append(float(np.linalg.norm(p[0]-bp[w])/max(np.linalg.norm(bp[w]-bp[elbow]),10)))
    if len(h)==21:
     gt=np.array([h[str(j)] for j in range(21)]);absolute[k].extend(np.linalg.norm(p-gt,axis=1).tolist());relative[k].extend(np.linalg.norm((p-p[0])-(gt-gt[0]),axis=1).tolist())
diag=dict(rejections=rejects,shoulder_scale_MP_over_RTMW=dict(n=len(shoulders),median=float(np.median(shoulders)),p05=float(np.percentile(shoulders,5)),p95=float(np.percentile(shoulders,95))),hands={k:dict(wrist_forearm_ratio_p50=float(np.median(v)),wrist_forearm_ratio_p95=float(np.percentile(v,95)),reference_hand_frames=len(absolute[k])//21,absolute_rmse_px=float(np.sqrt(np.mean(np.square(absolute[k])))),wrist_relative_rmse_px=float(np.sqrt(np.mean(np.square(relative[k]))))) for k,v in wrist.items()})
e.save(O/'diagnostics.json',diag);e.save(O/'paired.json',paired)
lines=['# ทดลองเลือกพิกัดมือด้วยคุณภาพ และตรวจมือ–ลำตัว','', '9 ตุลาคม2026 — signing-avatar-github-tinnapat เท่านั้น ไม่แก้production','', '## ขอบเขต','ใช้ชุดพัฒนา30คลิปเดิม1964เฟรม30FPS ไม่อ่านหรือปรับค่าจากชุดยืนยัน30คลิปใหม่ ใช้กรอบเดิมและconfidence0.30 ภาพ/เวลาท่า/โมเดลST-GCNเดิม','กำหนดกฎทุกตัวก่อนดูคะแนน ไม่ปรับเป็นรายคำ ไม่ฝึกโมเดลใหม่ ไม่มีแก้/ตรึง/เกลี่ยตำแหน่งนิ้ว เพียงเลือกทั้งมือระหว่างanimeที่ผ่านเกณฑ์กับfallbackMediaPipe/RTMW','', '## วิธี','- baseline: anime+RTMPose-handเมื่อผ่านเกณฑ์ ไม่ผ่านใช้MediaPipe/RTMWเดิม','- anchor_guard: ไม่ใช้มือanimeเมื่อข้อมือไกลจากข้อมือลำตัวกว่าfallbackเกิน0.15เท่าความยาวท่อนแขน','- shape_guard: เทียบรูปมือกับเฟรมก่อนหน้าแบบหักตำแหน่งข้อมือออก ตรวจmedianการเปลี่ยนความยาวกระดูก2Dและการกระโดดจุดนิ้ว เลือกfallbackเฉพาะcandidateผิดปกติและแย่กว่าfallbackตามเกณฑ์ตรึงในprotocol','- combined_guard: รวมกฎข้อมือและรูปมือ โดยมีประวัติมือที่เลือกของตัวเอง','- mp_body_same_hands: การตรวจแยกสาเหตุ เปลี่ยนเฉพาะลำตัวเป็นMediaPipe มือเหมือนbaselineทุกจุด ไม่ใช่วิธีแก้นิ้ว','รายละเอียดตัวเลขทุกเกณฑ์อยู่ในprotocol.json ไม่มีใช้Rendererreferenceในการเลือกมือ','', '## คะแนนจริง','| วิธี | ถูก/30 | Macro-P | Macro-R | Macro-F1 | Top5 | PCK10รวมmissing | RMSE(px) |','|---|---:|---:|---:|---:|---:|---:|---:|']
for k,ss in r['summary'].items():
 gg=r['geometry'][k];lines.append(f"| {k} | {ss['correct']}/30 | {ss['precision']:.2f}% | {ss['recall']:.2f}% | {ss['f1']:.2f}% | {ss['top5']:.2f}% | {gg['pck10_all_reference_pct']:.2f}% | {gg['rmse_px']:.2f} |")
lines+=['','Macroเฉลี่ย30true classes zero_division0 ทายได้2000คลาส; PCK10เป็นจุดห่างreference<=10px ไม่ใช่ความถูกต้องASL','', '## คำได้–เสียเทียบbaseline']
for k,p in paired.items():lines.append(f"- {k}: ได้ {', '.join(p['gained']) or 'ไม่มี'}; เสีย {', '.join(p['lost']) or 'ไม่มี'}; exploratory McNemar p={p['exact_mcnemar_p']:.4f}")
lines+=['','## การตรวจมือ–ลำตัว','ระยะข้อมือกับRTMWเป็นความสอดคล้องกับตัวตรวจลำตัว ไม่ใช่ground truth: ลดระยะนี้ได้ไม่ได้แปลว่าถูกต้องกว่า หากRTMWข้อมือผิด','Wrist-relative RMSEหักข้อมือออกก่อนเทียบ จึงดูส่วนรูปมือแยกจากตำแหน่งรวม แต่ไม่ได้ชดเชยscale/rotation และใช้2D','| วิธี | wrist/forearm p50 | p95 | absolute RMSE | wrist-relative RMSE |','|---|---:|---:|---:|---:|']
for k,ss in diag['hands'].items():lines.append(f"| {k} | {ss['wrist_forearm_ratio_p50']:.3f} | {ss['wrist_forearm_ratio_p95']:.3f} | {ss['absolute_rmse_px']:.2f} | {ss['wrist_relative_rmse_px']:.2f} |")
lines+=['','```json',json.dumps(diag,indent=2),'```','', '## ข้อจำกัด','การหมุนมือจริงอาจทำให้กระดูก2Dดูสั้นลง กฎshapeจึงอาจปฏิเสธการเคลื่อนไหวจริง ยังไม่ใช่ข้อจำกัดทางกายวิภาค3D','ใช้การเลือกมือจากตัวตรวจหลายตัว ไม่รับรองความเป็นอิสระของความผิดพลาด; มือเปลี่ยนแหล่งที่มาอาจกระโดดได้','ชุดพัฒนาถูกใช้หลายรอบ ไม่ใช่ผลยืนยันอิสระ ไม่เลือกค่าต่อจากผลชุดยืนยันเดิม และไม่สรุปคะแนนรู้จำเป็นความเข้าใจภาษามือของผู้ใช้','baselineตรวจทำซ้ำคำทายและtarget probabilityตรงรอบก่อน productionhashตรวจ4ไฟล์ตรงเดิม','', '## รายคำ','| คำ | '+' | '.join(q.NAMES[2:])+' |','|---|'+'---|'*len(q.NAMES[2:])]
for i,c in enumerate(manifest):lines.append('| '+c['label']+' | '+' | '.join(r['predictions'][k][i]['prediction'] for k in q.NAMES[2:])+' |')
(O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(dict(paired=paired,diagnostics=diag),ensure_ascii=True))
