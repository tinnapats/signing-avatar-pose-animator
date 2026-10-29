import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont
D=Path(__file__).resolve().parent;O=D/'exp2-bidirectional-20261013';B=D/'oracle200-20261011';P=D/'pixel-only-plan-20261012'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def review():
 mf={c['id']:c for c in read(B/'capture_manifest.json')};chosen=[]
 for path in sorted((O/'events').glob('*.json')):
  for row in read(path):
   if row['accepted']:chosen.append(row);break
  if len(chosen)==6:break
 board=Image.new('RGB',(1200,max(1,len(chosen))*440),'white');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
 for n,r in enumerate(chosen):
  c=mf[r['id']];im=Image.open(B/'frames'/r['id']/c['frames'][r['frame']]['file']).convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im);im=bg.convert('RGB');aa=r['accepted'][0];q=np.array(r['candidates'][aa['candidate']]['points']);tracks=r['tracks'][str(aa['side'])];ctr=(q.min(0)+q.max(0))/2;sz=max(120,float(max(q.max(0)-q.min(0))+50));bb=np.r_[ctr-sz/2,ctr+sz/2].astype(int);sz=bb[2]-bb[0]
  for k in [0,1,2]:
   crop=im.crop(tuple(bb)).resize((380,380));cd=ImageDraw.Draw(crop)
   if k==1:
    for name,color in [('before','blue'),('after','green')]:
     x,y=(np.array(tracks[name]['predicted_wrist'])-bb[:2])*380/sz;cd.ellipse((x-5,y-5,x+5,y+5),outline=color,width=3);cd.text((x+6,y),name,font=font,fill=color)
   if k==2:
    for j,pt in enumerate(q):
     x,y=(pt-bb[:2])*380/sz;cd.ellipse((x-3,y-3,x+3,y+3),fill='red');cd.text((x+3,y),str(j),font=font,fill='blue',stroke_width=1,stroke_fill='white')
   board.paste(crop,(400*k,440*n+45));d.text((400*k+4,440*n+4),[f"{r['id']} f{r['frame']} hand{aa['side']}",'Tracked wrist estimates','Current-image hand detection'][k],font=font,fill='black')
 board.save(O/'review.png');(O/'review-selection.json').write_text(json.dumps(chosen,indent=2),encoding='utf-8');print('review',len(chosen))
def report():
 result=read(O/'results.json');ex=read(O/'extraction.json');reasons=Counter();rejects=Counter();pairs=0;agrees=0;changed=0;selected=0
 for path in sorted((O/'arrays').glob('*.npz')):
  z=np.load(path);old=np.load(P/'arrays'/path.name);expected=old['raw'].copy()
  for s,b in enumerate([33,54]):expected[old['flags'][:,s],b:b+21]=0
  assert np.array_equal(expected,z['baseline']) and np.isfinite(z['bidirectional']).all() and expected.shape==z['bidirectional'].shape
  assert np.array_equal(expected[:,:33],z['bidirectional'][:,:33])
  for s,b in enumerate([33,54]):
   assert np.array_equal(expected[~old['flags'][:,s],b:b+21],z['bidirectional'][~old['flags'][:,s],b:b+21]);changed+=int(np.any(expected[:,b:b+21]!=z['bidirectional'][:,b:b+21],axis=(1,2)).sum())
  for r in read(O/'events'/(path.stem+'.json')):
   for s,t in r['tracks'].items():
    pairs+=1;agrees+=int(t['agrees'])
    for direction in ['before','after']:reasons[t[direction]['reason']]+=1
   for re in r['rejections']:rejects[re['reason']]+=1
   for a in r['accepted']:
    selected+=1;assert np.array_equal(z['bidirectional'][r['frame'],[33,54][a['side']]:[33,54][a['side']]+21],np.array(r['candidates'][a['candidate']]['points'],np.float32))
 assert selected==changed==ex['accepted_hand_frames']
 changes=[dict(label=b['label'],before=a['prediction'],after=b['prediction']) for a,b in zip(result['predictions']['baseline'],result['predictions']['bidirectional']) if a['prediction']!=b['prediction']]
 audit=dict(changed_hand_frames=changed,side_frame_pairs=pairs,agreed_pairs=agrees,tracking_reasons=dict(reasons),candidate_rejections=dict(rejects),prediction_changes=changes,all_invariants_passed=True,final_points_from_current_detection=True);(O/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
 lines=['# Experiment 2 — ติดตามจากภาพก่อนและหลัง','','วันที่13ตุลาคม2026 ทดลองจริงเฉพาะExperiment2 ไม่แก้ระบบหลัก ไม่ใช้CSV พิกัดก่อนRender พิกัดRenderer หรือmaskจากRenderer และไม่ใช้คำตอบจริงเลือกจุด การดึงพิกัดแยกจากการคำนวณคะแนน คำตอบจริงใช้เฉพาะscoringภายหลัง','','## วิธีที่ตรึงก่อนวัดผล','','คงกรอบมือและผลRTMPose-handจากภาพปัจจุบันชุดเดียวกับรอบกรอบอิสระ เปลี่ยนเฉพาะตัวจับคู่ข้าง เลือกseedมือRTMWจากภาพก่อน/หลังไม่เกิน3เฟรมที่ผ่านconfidenceและpixel support จากนั้นติดตามมุมภาพด้วยLKและตรวจforward-backward error ใช้median image flowประมาณตำแหน่งข้อมือ ต้องสำเร็จทั้งสองทิศและตำแหน่งสอดคล้องกันจึงจับคู่กับcandidateในภาพปัจจุบัน','','พิกัดนิ้วสุดท้ายคือผลตรวจมือในภาพปัจจุบัน ไม่ใช่การเฉลี่ย/interpolate/ค้างรูปนิ้ว seedเป็นพิกัดดั้งเดิมจากภาพ ไม่ใช้ผลที่กู้คืนวนกลับเป็นseed มือ/ลำตัวที่ไม่อยู่ในขอบเขตคงเดิม','','## คะแนนST-GCNจริงบน200คลิป','','| วิธี | ถูก/200 | Accuracy | Macro-P | Macro-R | Macro-F1 | Top5 |','|---|---:|---:|---:|---:|---:|---:|']
 for v,m in result['summary'].items():lines.append(f"| {v} | {m['correct']} | {m['accuracy']:.2f}% | {m['precision']:.2f}% | {m['recall']:.2f}% | {m['f1']:.2f}% | {m['top5']:.2f}% |")
 lines+=['',f"เติมจริง {changed} มือ-เฟรม; ทั้งสองทิศสอดคล้อง {agrees}/{pairs} side-frame pairs (รวมทั้งข้างที่คงไว้เพื่อกันจับคู่ผิด)",f"ติดตาม {ex['tracking_attempts']} ทิศทาง ใช้เวลาเพิ่ม {ex['seconds']:.2f} วินาที ไม่รวมขั้นตรวจRTMW/กรอบมือ/RTMPoseที่cacheไว้และST-GCN จึงไม่ใช่end-to-end latency",f"คำที่ถูกเพิ่ม: {result['gains']}",f"คำที่ถูกเดิมแต่ผิดใหม่: {result['losses']}",'','## ความน่าเชื่อถือและข้อจำกัด','','baseline/forwardถูกรันใหม่และตรวจtop5ตรงเดิมกับtarget probability tolerance1e-5 ตรวจbody/unflagged handsไม่เปลี่ยน shape/frame countเดิม ทุกarrayfinite และพิกัดมือสุดท้ายเท่ากับcandidateจากภาพปัจจุบันจริงทุกจุด เก็บinput/logitsครบ','','การใช้ภาพอนาคตเป็นการวิเคราะห์วิดีโอoffline ไม่ใช่ใช้เฉลย แต่ไม่เหมาะกับระบบสดที่ไม่ยอมรอเฟรมอนาคต การติดตามสอดคล้องไม่ได้แปลว่าตรวจนิ้วถูก ความผิดร่วมกันและมือซ้อนยังเกิดได้ ไม่มีground truthหรือPCKที่ตรวจรับรอง','','เลือกเฉพาะ341มือ-เฟรมที่ถูกflagใน303ภาพ ไม่ใช่กู้ทุกมือที่หาย ทุกวิธีใช้คลิป200เดิมที่สำรวจหลายรอบแล้ว ไม่ใช่held-out validation ไม่ได้ใช้reserved30 ไม่ฝึกโมเดล ไม่ปรับเกณฑ์หลังเห็นคะแนน','','หลักฐาน: PROTOCOL.md, extraction.json, results.json, audit.json, events/, arrays/, inference/, review.png']
 (O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(audit,indent=2))
if __name__=='__main__':
 import sys
 {'review':review,'report':report}[sys.argv[1]]()
