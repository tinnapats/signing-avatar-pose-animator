
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json,hashlib,math
from pathlib import Path
import numpy as np
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');R=B/'rtm_trial_20261008';O=B/'anime_hand_trial_20261009'
read=lambda p:_local_json_loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
refs=read(B/'coordinate_reference.json');manifest=read(B/'capture_manifest.json');stats=dict(reference_hands=0,anime_box_80pct=0,wrist_crop_80pct=0,assignment_changes=0,raw_boxes=0);dur=[];rows=[]
for c in refs:
 raw=read(O/(c['id']+'_raw.json'));wr=np.load(R/'hand_crop'/(c['id']+'.npz'))['boxes'];z=np.load(O/(c['id']+'.npz'));dur.extend(z['ms'].tolist())
 for i,r in enumerate(c['samples']):
  stats['raw_boxes']+=len(raw[i]['boxes']);stats['assignment_changes']+=raw[i]['assignment']['plain']!=raw[i]['assignment']['tracked']
  for side,hand in enumerate(['left','right']):
   h=r['hands'][hand]
   if len(h)!=21:continue
   g=np.array([h[str(j)] for j in range(21)]);stats['reference_hands']+=1
   def covers(box):
    if not np.any(box):return 0.
    box=np.array(box);center=(box[:2]+box[2:])/2;size=max(box[2:]-box[:2])*1.25;return float(np.mean(np.all((g>=center-size/2)&(g<=center+size/2),axis=1)))
   best=max([covers(b) for b in raw[i]['boxes']]+[0]);wc=covers(wr[i,side]);stats['anime_box_80pct']+=best>=.8;stats['wrist_crop_80pct']+=wc>=.8
   rows.append(dict(id=c['id'],frame=i,side=hand,anime_coverage=best,wrist_coverage=wc,accepted=bool(z['plain_mask'][i,side])))
stats['mean_ms_per_frame']=float(np.mean(dur));save(O/'box_diagnostics.json',dict(summary=stats,rows=rows,rules='Diagnostic only: reference21 points inside square crop with same bbox padding1.25. Anime uses any detected bbox => optimistic coverage upper bound before assignment. Wrist uses corresponding side crop. >=80% points inside threshold fixed before reading final scores. Reference never supplied at inference.'))
r=read(O/'results.json');base={p['label']:p['prediction']==p['label'] for p in r['predictions']['previous_best']};paired={}
for k,rr in r['predictions'].items():
 if k in ['mediapipe','rtmw_det','previous_best','wrist_crop_only']:continue
 g=[p['label'] for p in rr if not base[p['label']] and p['prediction']==p['label']];l=[p['label'] for p in rr if base[p['label']] and p['prediction']!=p['label']];n=len(g)+len(l);p=min(1,2*sum(math.comb(n,j) for j in range(min(len(g),len(l))+1))/2**n) if n else 1;paired[k]=dict(gained=g,lost=l,exact_mcnemar_p=p)
save(O/'paired.json',paired)
lines=['# Anime hand detector + RTMPose-hand: ผลทดลอง30คลิป','', '9 ตุลาคม2026 — signing-avatar-github-tinnapat/recognition_audit','ภาพ Avatar เดิม30คลิป 1964เฟรม 30FPS เวลาเดิม ตรวจhashครบ รันimage-only จุดRendererใช้เฉพาะตรวจgeometry','', '## วิธี','- body: RTMW-detเดิม ยกเว้นbaseline MediaPipe','- previous_best: bodyRTMW + มือMediaPipeเมื่อพบ ไม่พบใช้RTMW (รอบก่อน6/30)','- wrist_crop_only: cropข้อมือเดิม + RTMPose-hand','- anime_only: anime_hand_detection v1.0_s + RTMPose-hand + จับคู่ข้อมือทีละเฟรม','- anime_tracked: เหมือนanime_only เพิ่มประวัติข้อมือที่ยอมรับใน3เฟรมช่วยจับคู่ ไม่มีเติมตำแหน่งเมื่อหาย','- anime_replace_best / anime_tracked_replace_best: ใช้มือanimeที่ผ่านเกณฑ์ ไม่ผ่านใช้previous_best','- mp_anime_rtmw: เก็บมือMPก่อน เมื่อหายใช้animeแบบติดตาม ไม่ผ่านจึงใช้RTMW','', 'ค่าตรึง: detector640×640 RGB; confidence .395 ตามthreshold.json; NMS IoU .7; hand256×256; confidenceเฉลี่ยและข้อมือ>=.3; ระยะข้อมือ<=.65ความยาวท่อนแขน; assignmentหนึ่งต่อหนึ่ง กรอบละมือเดียว เกณฑ์เดียวทุกคำ','', '| วิธี | ถูก/30 | Accuracy | Macro-P | Macro-R | Macro-F1 | Top5 |','|---|---:|---:|---:|---:|---:|---:|']
for v,s in r['summary'].items():lines.append(f"| {v} | {s['correct']}/30 | {s['accuracy']:.2f}% | {s['precision']:.2f}% | {s['recall']:.2f}% | {s['f1']:.2f}% | {s['top5']:.2f}% |")
lines+=['','## ความตรงของมือ','PCK@10px วัดจุดห่างreferenceไม่เกิน10px ที่ภาพ808×808; คอลัมน์allนับreferenceที่ตรวจไม่พบว่าไม่ผ่าน','| วิธี | พบreferenceมือ/1872 | RMSEเมื่อพบ(px) | PCK@10px เมื่อพบ | PCK@10px all |','|---|---:|---:|---:|---:|']
for v,g in r['geometry'].items():lines.append(f"| {v} | {g['hand_frames']} | {g['rmse_px']:.2f} | {g['pck10_pct']:.2f}% | {g['pck10_all_reference_pct']:.2f}% |")
lines+=['','## กรอบมือและการจับคู่','```json',json.dumps(stats,indent=2),'```','การมีจุดอยู่ในcrop>=80% เป็นการตรวจว่ากรอบครอบมือได้ ไม่ได้รับรองข้อนิ้วถูกต้อง ช่องanimeเป็นany-box upper boundยังไม่ผ่านassignment; กรอบข้อมืออาจใหญ่กว่าและมีฉากรบกวนมากกว่า จึงเทียบเป็นbox recall diagnostic ไม่ใช่precision','การตรวจจับมือรวมมีทั้งSVGfallbackและdetailed hand การพบSVGfallbackไม่ถือว่าเพิ่มคุณภาพช่วงทำท่าโดยอัตโนมัติ','', '## การสลับข้าง','```json',json.dumps(r['pairing'],indent=2),'```','วัดเฉพาะเฟรมที่มีreferenceและตรวจพบสองมือ; swappedหมายถึงจับคู่สลับได้errorต่ำกว่าเกิน4px เป็นdiagnostic ไม่ใช้แก้ผลโมเดล','', '## เทียบกับ6/30แบบจับคู่','```json',json.dumps(paired,ensure_ascii=False,indent=2),'```','p-valueเป็นexploratory ไม่ปรับmultiple comparisons ไม่ใช้ยืนยันความเหนือกว่าจากชุดเล็ก','', '## ข้อจำกัดและการทำซ้ำ','30คลิปนี้เป็นชุดพัฒนาที่ทดลองซ้ำ ไม่ใช่held-out; คำละ1คลิป; โมเดลST-GCNทายได้2000คำ Macroเฉลี่ย30true classesแบบzero_division0','คะแนนไม่ใช่ความถูกต้องASLหรือความเหมือนภาพ; คะแนนดีขึ้นอาจเกิดจากพิกัดที่เข้ากับโมเดลมากขึ้น','ไม่มีsmoothingภายนอก; baselineMPมีsmoothingเดิมภายใน detector; รุ่นใหม่confidenceไม่calibrateเท่ากัน','เก็บONNX SHA256, protocol, rawกล่อง+จุด+confidenceทุกเฟรม, assignments, times, predictions, geometry และภาพreview','ตรวจhash production4ไฟล์ตรงเดิม','', 'แหล่งโมเดล: https://huggingface.co/deepghs/anime_hand_detection','อ้างอิงpre/postprocess: https://github.com/deepghs/imgutils/blob/main/imgutils/generic/yolo.py','RTMPose-hand: https://github.com/Tau-J/rtmlib','', '## คำทายรายคลิป','| คำ | '+' | '.join(r['predictions'])+' |','|---|'+'---|'*len(r['predictions'])]
for i,c in enumerate(manifest):lines.append('| '+c['label']+' | '+' | '.join(rr[i]['prediction'] for rr in r['predictions'].values())+' |')
for f,h in read(O/'protocol.json')['production'].items():assert hashlib.sha256((B.parent.parent/f).read_bytes()).hexdigest()==h
lines+=['','## ข้อสรุปจากรอบนี้','วิธี anime_replace_best ได้7/30 เพิ่มจาก6/30 โดยเพิ่ม since และไม่เสียคำที่เคยถูก; แบบติดตามและแบบเก็บMPก่อนให้Top1/Top5เท่ากัน แต่geometryแตกต่างกัน','anime_replace_best มีPCK@10pxรวมmissing53.42% เทียบ47.70%ของprevious_best บนreferenceชุดเดียวกัน; RMSE30.75pxเทียบ31.48px โดยมีreferenceที่ตรวจพบ1870hand-framesเท่ากัน คะแนนยังต่ำและหนึ่งคลิปที่เพิ่มไม่ใช่หลักฐานยืนยันความเหนือกว่าโดยทั่วไป','Animeมือเดี่ยวพบreference473/1872hand-frames แม้จุดที่พบRMSE11.70pxและPCK86.01% แต่coverageจำกัด ทำให้ทายถูกเพียง1/30','การติดตามช่วยassignmentไม่เปลี่ยนเลยทั้ง1964เฟรมภายใต้กฎนี้ ไม่สนับสนุนเพิ่มขั้นนี้ในรูปแบบปัจจุบัน','ตัวกรอบมือครอบ>=80%referenceได้1595/1872 เทียบกรอบข้อมือ1782/1872 แต่โมเดลมือผ่านเกณฑ์บนreferenceมากขึ้น101→473 แสดงว่าการครอบคลุมอย่างเดียวอธิบายผลไม่ได้ ขนาดcrop/ฉากรบกวนและความสามารถอ่านนิ้วมีส่วนที่ต้องแยกตรวจ','เวลาเพิ่ม detector+hand เฉลี่ย237.49ms/frameบนCPUรอบนี้ (~4.21FPS) ไม่เหมาะสรุปว่าใช้งานreal-timeได้; เวลาแอนิเมชันจริงยังเท่าเดิม','แนะนำเก็บ anime_replace_best เป็นcandidateสำหรับการประเมินภาพ และคงระบบหลักเดิม รอบถัดไปตรวจกรณีพบกรอบแต่ไม่ผ่านตัวหาข้อต่อ พร้อมทดลองSDPoseหรือชุดlabelมือเฉพาะAvatarตามทรัพยากร ไม่ลดเกณฑ์รายคำเพื่อไล่คะแนน']
(O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(dict(boxes=stats,paired=paired),ensure_ascii=True))
