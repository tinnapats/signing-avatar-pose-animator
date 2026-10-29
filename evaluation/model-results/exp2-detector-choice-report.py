import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont
D=Path(__file__).resolve().parent;O=D/'exp2-detector-choice-20261013';B=D/'oracle200-20261011';P=D/'pixel-only-plan-20261012'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def review():
 events=read(O/'selection.json')['events'];mf={c['id']:c for c in read(B/'capture_manifest.json')};chosen=[];seen=set()
 for model in ['MP','RT']:
  count=0
  for r in events:
   if r['id'] in seen:continue
   selected=next((s for s in r['selected']['mixed'] if s['model']==model),None)
   if selected:chosen.append((r,selected));seen.add(r['id']);count+=1
   if count==3:break
 board=Image.new('RGB',(1600,max(1,len(chosen))*440),'white');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
 for n,(r,s) in enumerate(chosen):
  cand=r['proposals'][s['proposal']];q=np.array(cand['points']);c=mf[r['id']];im=Image.open(B/'frames'/r['id']/c['frames'][r['frame']]['file']).convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im);im=bg.convert('RGB');ctr=(q.min(0)+q.max(0))/2;sz=max(120,float(max(q.max(0)-q.min(0))+50));bb=np.r_[ctr-sz/2,ctr+sz/2].astype(int);sz=bb[2]-bb[0];rt=next((p for p in r['proposals'] if p['box_index']==cand['box_index'] and p['model']=='RT'),None);mp=next((p for p in r['proposals'] if p['box_index']==cand['box_index'] and p['model']=='MP'),None)
  for k,proposal in enumerate([None,rt,mp,cand]):
   crop=im.crop(tuple(bb)).resize((380,380));cd=ImageDraw.Draw(crop)
   if proposal:
    for j,pt in enumerate(np.array(proposal['points'])):
     x,y=(pt-bb[:2])*380/sz;cd.ellipse((x-3,y-3,x+3,y+3),fill='red');cd.text((x+3,y),str(j),font=font,fill='blue',stroke_width=1,stroke_fill='white')
   board.paste(crop,(400*k,440*n+45));label=[f"{r['id']} f{r['frame']} hand{s['side']}",'RTMPose-hand','MediaPipe'+(' (none)' if mp is None else ''),'Selected: '+s['model']][k];d.text((400*k+4,440*n+4),label,font=font,fill='black')
 board.save(O/'review.png');(O/'review-selection.json').write_text(json.dumps(chosen,indent=2),encoding='utf-8');print('review',len(chosen))
def report():
 result=read(O/'results.json');sel=read(O/'selection.json');rt=read(O/'rt.json');mp=read(O/'mp.json');counts=Counter();choices=Counter();valid=Counter();changed=Counter()
 for e in sel['events']:
  for p in e['proposals']:valid[p['model']]+=int(p['valid'])
  for v,ss in e['selected'].items():
   counts[v]+=len(ss)
   if v=='mixed':
    for s in ss:choices[s['model']]+=1
 for path in sorted((O/'arrays').glob('*.npz')):
  z=np.load(path);old=np.load(P/'arrays'/path.name);base=old['raw'].copy()
  for s,b in enumerate([33,54]):base[old['flags'][:,s],b:b+21]=0
  assert np.array_equal(base,z['baseline'])
  for v in ['RT','MP','mixed']:
   assert np.isfinite(z[v]).all() and z[v].shape==base.shape and np.array_equal(base[:,:33],z[v][:,:33])
   for s,b in enumerate([33,54]):
    assert np.array_equal(base[~old['flags'][:,s],b:b+21],z[v][~old['flags'][:,s],b:b+21]);changed[v]+=int(np.any(base[:,b:b+21]!=z[v][:,b:b+21],axis=(1,2)).sum())
 assert dict(changed)==dict(counts)
 audit=dict(selected_hand_frames=dict(counts),mixed_choices=dict(choices),quality_passed_proposals=dict(valid),mp_hands_returned=sum(len(r['hands']) for r in mp['rows']),all_invariants_passed=True,baseline_reproduced=result['baseline_reproduced']);(O/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
 lines=['# Experiment2 — เปรียบเทียบตัวตรวจมือบนcropเดียวกัน','','13ตุลาคม2026: ทดลองเฉพาะภาพAnimation ไม่ใช้Renderer/CSV/เฉลยเพื่อเลือกพิกัด ไม่ทำExperiment1 ไม่ฝึกโมเดล ไม่แก้production ไม่ใช้reserved30','','## วิธี','','RTMPose-hand และ MediaPipe Hands อ่านPNGcropเดียวกัน497ภาพ ตรวจhashและตรวจpixel preprocessingของRTMPoseเทียบcropครบทุกภาพ แปลงพิกัดกลับภาพต้นฉบับเหมือนกัน คัดconfidenceเฉพาะภายในโมเดล: RTmean/wrist>=.3 ส่วนMPใช้detection threshold.5 ไม่ตีความhandedness scoreเป็นconfidenceของจุดและไม่เทียบconfidenceข้ามโมเดล','','ตัวเลือกRT-only,MP-only,mixedใช้กฎตำแหน่งเดียวกัน:confidenceภายในโมเดลผ่าน,span>=10px,ข้อมือใกล้สีผิวไม่เกิน6pxและอย่างน้อยครึ่งจุดมือใกล้สีผิว,จับคู่มือกับseedภาพก่อนหน้าไม่เกิน3เฟรม คัดความกำกวม/มือซ้ำ เลือกด้วยระยะข้อมือnormalized+.25*สัดส่วนจุดนอกสีผิว ไม่ใช้ชื่อคำหรือคะแนนST-GCN ไม่มีการเฉลี่ยพิกัดข้ามโมเดล','','baselineคือsupport_gate22/200 ไม่ใช่RT-only: RT-onlyรอบนี้ตรวจcropใหม่และใช้กฎเลือกที่ตรึงร่วมกับMPเพื่อให้เทียบกันได้','','## ผลโมเดลจริง','','| วิธี | ถูก/200 | Accuracy | Macro-P | Macro-R | Macro-F1 | Top5 |','|---|---:|---:|---:|---:|---:|---:|']
 for v,m in result['summary'].items():lines.append(f"| {v} | {m['correct']} | {m['accuracy']:.2f}% | {m['precision']:.2f}% | {m['recall']:.2f}% | {m['f1']:.2f}% | {m['top5']:.2f}% |")
 lines+=['',f"เติมผ่านเกณฑ์: {dict(counts)} มือ-เฟรม",f"mixedเลือกโมเดล: {dict(choices)}",f"MediaPipeคืนมือ {audit['mp_hands_returned']} จาก497crop (หนึ่งcropคืนได้มากกว่าหนึ่งมือ)",f"candidateผ่านคุณภาพ: {dict(valid)} ยังไม่ใช่ความถูกต้องที่ตรวจรับรอง",f"เวลาRTรวมเตรียมcrop/บันทึก {rt['seconds']:.2f}s; MPอ่านcrop/ตรวจ {mp['seconds']:.2f}s; selection {sel['seconds']:.2f}s ไม่ใช่benchmarkend-to-endและไม่รวมRTMW/กรอบเดิม/ST-GCN",'','## คำที่เปลี่ยนความถูกต้อง']
 for v,pair in result['pairs'].items():lines.append(f"- {v}: ถูกเพิ่ม {pair['gains']}; ถูกเดิมแต่ผิดใหม่ {pair['losses']}")
 lines+=['','## ตรวจซ้ำและข้อจำกัด','','baselineรันST-GCNซ้ำtop5ตรงเดิมและtarget probabilityต่างไม่เกิน1e-5 ตรวจbody/unflagged handsไม่เปลี่ยน,shapeและจำนวนเฟรมเดิม,ทุกarrayfinite,จำนวนhand-framesที่เปลี่ยนตรงกับบันทึกselection เก็บinput/logitsครบ','','เฉพาะ341มือ-เฟรมที่flagใน303ภาพ ไม่ใช่ทุกมือที่หาย และยังต้องใช้seedที่ตรวจจากภาพได้ จึงไม่ใช่การแก้ปัญหามือทุกชนิด ชุด200เดิมใช้สำรวจหลายรอบ ไม่ใช่held-out ไม่มีground truthที่คนตรวจเพื่อวัดjoint accuracy การเลือกตามสีผิว/ตำแหน่งเป็นheuristicและอาจเลือกรูปนิ้วผิด','','การอ่านเฉลยคำเกิดเฉพาะscoringหลังบันทึกพิกัดแล้ว ไม่มีword routing ไม่จำกัดคำตอบโมเดลจาก2000เป็น200 ไม่ปรับเกณฑ์ตามผลรอบนี้','','หลักฐาน: PROTOCOL.md,rt.json,mp.json,selection.json,results.json,audit.json,crops/,arrays/,inference/,review.png']
 (O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(audit,indent=2))
if __name__=='__main__':
 import sys
 {'review':review,'report':report}[sys.argv[1]]()
