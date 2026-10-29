import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont
D=Path(__file__).resolve().parent;O=D/'exp2-fullhand20-20261014';B=D/'oracle200-20261011';P=D/'pixel-only-plan-20261012'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def review():
 mf={c['id']:c for c in read(B/'capture_manifest.json')};samples=[]
 for cid in read(O/'selection.json')['ids'][:3]:
  T=len(mf[cid]['frames'])
  for frac in [.25,.75]:samples.append((cid,int(round((T-1)*frac))))
 board=Image.new('RGB',(1200,440*len(samples)),'white');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
 for n,(cid,i) in enumerate(samples):
  c=mf[cid];p=B/'frames'/cid/c['frames'][i]['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==c['frames'][i]['sha256'];rgba=Image.open(p).convert('RGBA');bg=Image.new('RGBA',rgba.size,'white');bg.alpha_composite(rgba);im=bg.convert('RGB');z=np.load(O/'arrays'/(cid+'.npz'))
  for k,v in enumerate([None,'raw','fullhand']):
   panel=im.copy();dd=ImageDraw.Draw(panel)
   if v:
    a=z[v][i]
    for b,color in [(33,'blue'),(54,'red')]:
     q=a[b:b+21]
     if not np.any(q):continue
     for root in [1,5,9,13,17]:
      chain=[0,root,root+1,root+2,root+3]
      for u,w in zip(chain,chain[1:]):dd.line([tuple(q[u]),tuple(q[w])],fill=color,width=2)
     for x,y in q:dd.ellipse((x-2,y-2,x+2,y+2),fill=color)
   board.paste(panel.resize((390,390)),(400*k,440*n+45));d.text((400*k+3,440*n+3),[f"{cid} frame{i}",'Raw RTMW','New full-sequence hands'][k],font=font,fill='black')
 board.save(O/'review.png');(O/'review-selection.json').write_text(json.dumps(samples),encoding='utf-8');print('review saved')
def report():
 res=read(O/'results.json');ex=read(O/'extraction.json');selection=read(O/'selection.json');coverage=Counter();diagnostics=Counter();perclip=[]
 for cid in selection['ids']:
  z=np.load(O/'arrays'/(cid+'.npz'));old=np.load(P/'arrays'/(cid+'.npz'));assert np.array_equal(z['raw'],old['raw']);a=z['fullhand'];assert a.shape==z['raw'].shape and np.isfinite(a).all() and np.array_equal(a[:,:33],z['raw'][:,:33]);counts={}
  for v in ['raw','support_gate','fullhand']:
   counts[v]=sum(int(np.any(z[v][:,b:b+21],axis=(1,2)).sum()) for b in [33,54]);coverage[v]+=counts[v]
  events=read(O/'events'/(cid+'.json'));assert len(events)==len(a);selected=0
  for i,r in enumerate(events):
   diagnostics['slots_without_anchor']+=2-len(r['anchors'])
   for ca in r['candidates']:
    if not ca['valid']:diagnostics['quality_rejected']+=1
    elif 'rejection' in ca:diagnostics[ca['rejection']]+=1
   for s in r['selected']:
    b=[33,54][s['side']];assert np.array_equal(a[i,b:b+21],np.array(r['candidates'][s['candidate']]['points'],np.float32));selected+=1
  assert selected==counts['fullhand'];perclip.append(dict(id=cid,frames=len(a),coverage=counts))
 assert coverage['fullhand']==ex['accepted'];audit=dict(coverage=dict(coverage),diagnostics=dict(diagnostics),perclip=perclip,all_invariants_passed=True,final_points_from_current_image_candidates=True);(O/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');den=2*ex['frames']
 lines=['# Experiment2: ตรวจมือใหม่ทุกเฟรมบน20คลิป','','14ตุลาคม2026 — pilotที่กำหนดเกณฑ์ก่อนรัน ไม่ใช้คำเฉลยเลือกคลิป/พิกัด ไม่ใช้RendererหรือCSV ไม่มีการฝึกโมเดลและไม่แก้ระบบหลัก','','## การเลือกและวิธี','','เลือก20clipIDด้วยSHA256ของข้อความคงที่+ID โดยไม่อ่านคะแนนเดิม คัดจากชุด200ที่เคยสำรวจแล้วจึงเป็นdevelopmentไม่ใช่heldout ไม่ใช้reserved30 ชื่อคำใช้เฉพาะตอนscoringหลังบันทึกพิกัดเสร็จ','','ทุกเฟรมใช้anime hand detectorหากรอบจากภาพทั้งภาพ แล้วRTMPose-handหาจุด21จุด กรอบไม่ถูกกำหนดโดยข้อมือRTMW แต่การระบุซ้ายขวายังใช้มือRTMWจากภาพที่ผ่านเกณฑ์คุณภาพในเฟรมปัจจุบันหรือก่อนหน้าไม่เกิน3เฟรมเป็นanchor ถ้าanchorผิดก็ยังมีผลต่อการจับคู่ได้ ถ้าไม่มีanchorหรือcandidateไม่ผ่านปล่อยmissing ไม่มีfallbackพิกัดมือเก่า ไม่มีการค้างหรือinterpolateนิ้ว','','คงลำตัวRTMWเวลาเฟรม30FPSnormalizeและST-GCN27จุดเหมือนเดิม วิธีใหม่แทนมือทั้งสองข้างทุกเฟรม ไม่ใช่เติมเฉพาะ341มือ-เฟรมเดิม','','## ผลโมเดล (ตัวหาร20คลิปเท่านั้น)','','| วิธี | ถูก/20 | Accuracy | Macro-P | Macro-R | Macro-F1 | Top5 |','|---|---:|---:|---:|---:|---:|---:|']
 for v,m in res['summary'].items():lines.append(f"| {v} | {m['correct']} | {m['accuracy']:.2f}% | {m['precision']:.2f}% | {m['recall']:.2f}% | {m['f1']:.2f}% | {m['top5']:.2f}% |")
 lines+=['','ห้ามเทียบเปอร์เซ็นต์20คลิปนี้กับ11%ของ200คลิปโดยตรง เพราะคนละชุด/ตัวหาร ให้ใช้baselineในตารางเดียวกัน','','## ความครอบคลุมและต้นทุน',f"รวม{ex['frames']}เฟรม หรือ{den}มือ-เฟรม ตรวจพบ{ex['boxes']}กรอบ ผ่านคุณภาพ{ex['valid_candidates']}candidate"]
 for v,n in coverage.items():lines.append(f"- {v}: มีพิกัด{n}/{den}มือ-เฟรม ({100*n/den:.2f}%)")
 lines +=[f"เวลาextraction {ex['seconds']:.2f}s รวมอ่าน/ตรวจhashภาพ,detector,handposeและassociation แต่ไม่รวมRTMWcacheเดิม/การrender/ST-GCN ไม่ใช่เวลาตอบสนองระบบจริง",'','มีพิกัดมากไม่ใช่แปลว่าตรงภาพมาก ต้องดูภาพและความผิดพลาดประกอบ','','## เปลี่ยนความถูกต้องรายคำ']
 for v,pair in res['pairs'].items():lines.append(f"- เทียบ{v}: ถูกเพิ่ม{pair['gains']}; ถูกเดิมแต่ผิดใหม่{pair['losses']}")
 lines+=['','## การตรวจซ้ำ','','raw/support_gateรันโมเดลซ้ำtop5ตรงผลเดิมและprobabilityต่างไม่เกิน1e-5 รูปแบบ/จำนวนเฟรม/ลำตัวไม่เปลี่ยน ทุกarrayfinite และมือที่ไม่missingเท่ากับจุดจากcandidateภาพปัจจุบันทุกจุด ไม่มีการคัดลอกanchorเป็นคำตอบ เก็บinputs/logitsครบ','','review.pngเลือกเฟรมที่25%/75%ของ3คลิปแรกตามกฎhashก่อนทราบคะแนน ไม่เลือกเฉพาะคำที่ทายถูก ไม่มีground truthที่คนตรวจ จึงไม่รายงานPCKหรือความถูกต้องนิ้วเป็นเปอร์เซ็นต์','','ข้อจำกัด:ผลบน20คลิปอาจผันผวน การเปลี่ยนทั้งมือรวมผลของdetector,handposeและassociation ถ้าคะแนนลดไม่สามารถโทษขั้นเดียวได้ และการใช้RTMWanchorยังเป็นการพึ่งตัวตรวจภาพเดิม (ไม่ใช่Rendererหรือเฉลย)','','หลักฐาน:PROTOCOL.md,selection.json,extraction.json,results.json,audit.json,events/,arrays/,inference/,review.png']
 (O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(audit,indent=2))
if __name__=='__main__':
 import sys
 {'review':review,'report':report}[sys.argv[1]]()
