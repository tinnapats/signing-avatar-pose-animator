
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json,hashlib
from pathlib import Path
import numpy as np
import cv2
BASE=Path(__file__).parent;O=BASE/'hand-quality-trial-20261009';B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');read=lambda p:_local_json_loads(p.read_text(encoding='utf8'))
refs={c['id']:c for c in read(B/'coordinate_reference.json')};manifest=read(B/'capture_manifest.json');methods=['anchor_guard','shape_guard','combined_guard'];totals={k:dict(rejected=0,reference=0,improved=0,worsened=0,tied=0,without_reference=0) for k in methods};rows=[];examples={}
edges=[(0,1),(1,2),(2,3),(3,4)]+[(0,k) for k in [5,9,13,17]]+[(k+j,k+j+1) for k in [5,9,13,17] for j in range(3)]
for c in manifest:
 z=np.load(O/'inputs'/(c['id']+'.npz'));meta=read(O/'inputs'/(c['id']+'.json'));assert hashlib.sha256((O/'inputs'/(c['id']+'.npz')).read_bytes()).hexdigest()==meta['sha256']
 for log in meta['rules']:
  if not log['reject']:continue
  k=log['variant'];s=totals[k];s['rejected']+=1;i=log['frame'];side=log['side'];off=[33,54][side];h=refs[c['id']]['samples'][i]['hands'][['left','right'][side]]
  if len(h)!=21:s['without_reference']+=1;continue
  gt=np.array([h[str(j)] for j in range(21)]);old=z['baseline'][i,off:off+21];new=z[k][i,off:off+21];before=float(np.linalg.norm(old-gt,axis=1).mean());after=float(np.linalg.norm(new-gt,axis=1).mean());delta=after-before;s['reference']+=1;outcome='worsened' if delta>1e-6 else 'improved' if delta< -1e-6 else 'tied';s[outcome]+=1;row=dict(id=c['id'],label=c['label'],frame=i,side=side,method=k,before_mean_error_px=before,after_mean_error_px=after,delta_px=delta,outcome=outcome);rows.append(row)
  key=k+'_'+outcome
  if key not in examples or abs(delta)>abs(examples[key]['delta_px']):examples[key]=row
for k,s in totals.items():
 rr=[r for r in rows if r['method']==k];s['mean_error_change_on_rejected_px']=float(np.mean([r['delta_px'] for r in rr])) if rr else None
(O/'rejection_audit.json').write_text(json.dumps(dict(summary=totals,rows=rows,rules='Renderer reference used only AFTER scoring for diagnosis. Improvement measured by mean Euclidean21-point error on rejected hand-frames; missing reference is not called correct/incorrect. No rule/threshold updates.'),ensure_ascii=False,indent=2),encoding='utf8')
panels=[]
for key,ex in sorted(examples.items()):
 if ex['outcome']!='worsened':continue
 c=next(c for c in manifest if c['id']==ex['id']);i=ex['frame'];side=ex['side'];f=c['frames'][i];im=cv2.imread(str(B/'frames'/c['id']/f['file']),-1);alpha=im[:,:,3:4].astype(np.float32)/255;im=np.rint(im[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8);z=np.load(O/'inputs'/(c['id']+'.npz'));gt=np.array([refs[c['id']]['samples'][i]['hands'][['left','right'][side]][str(j)] for j in range(21)]);off=[33,54][side]
 # Same ROI in both panels, from union of reference and both predictions for diagnostic viewing only.
 pts=np.vstack([gt,z['baseline'][i,off:off+21],z[ex['method']][i,off:off+21]]);lo=np.maximum(np.floor(pts.min(0)-35).astype(int),0);hi=np.minimum(np.ceil(pts.max(0)+35).astype(int),808);row=[]
 for method in ['baseline',ex['method']]:
  img=im.copy();hand=z[method][i,off:off+21]
  for pts,color in [(gt,(0,160,0)),(hand,(0,40,255))]:
   pp=np.rint(pts).astype(int)
   for a,b in edges:cv2.line(img,tuple(pp[a]),tuple(pp[b]),color,1)
  crop=img[lo[1]:hi[1],lo[0]:hi[0]];scale=min(400/crop.shape[1],340/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale);tile=np.full((400,420,3),255,np.uint8);tile[60:60+crop.shape[0],:crop.shape[1]]=crop;cv2.putText(tile,method,(6,20),0,.5,(0,0,0),1);cv2.putText(tile,f"{c['label']} frame {i} / green=ref red=estimate",(6,43),0,.43,(0,0,0),1);row.append(tile)
 panels.append(np.hstack(row))
if panels:cv2.imwrite(str(O/'rejection_examples.png'),np.vstack(panels))
report=O/'REPORT_TH.md';text=report.read_text(encoding='utf8');marker='\n## ตรวจการปฏิเสธมือหลังวัดคะแนน';text=text.split(marker)[0];lines=[marker,'วัดเฉพาะมือ–เฟรมที่กฎปฏิเสธและมีRendererreferenceครบ21จุด เป็นdiagnosticหลังคะแนน ไม่ใช้ปรับกฎ','| วิธี | ปฏิเสธทั้งหมด | มีreference | ดีขึ้น | แย่ลง | เท่าเดิม | เปลี่ยนmean error(px) |','|---|---:|---:|---:|---:|---:|---:|']
for k,s in totals.items():lines.append(f"| {k} | {s['rejected']} | {s['reference']} | {s['improved']} | {s['worsened']} | {s['tied']} | {s['mean_error_change_on_rejected_px']:.2f} |")
lines+=['','## ข้อสรุปการตัดสิน','shape_guardได้8/30เพิ่มarrest แต่geometryแย่ลง ไม่พอจะนำไปใช้แทนbaseline7/30; anchor_guardและcombined_guardไม่เพิ่มคะแนนและgeometryแย่ลงเช่นกัน','การลดระยะข้อมือถึงRTMWไม่รับรองว่าใกล้พิกัดที่วาดจริงมากขึ้น เพราะRTMWไม่ใช่ground truth จึงไม่ควรดึงมือไปหาข้อมือลำตัวโดยอัตโนมัติ','mp_body_same_handsเหลือ3/30ทั้งที่มือเหมือนbaselineทุกจุด แสดงว่าตัวเลือกลำตัว/shoulder normalizationสัมพันธ์กับคะแนนมาก แต่ยังไม่พิสูจน์ว่าความคลาดเคลื่อนการปรับขนาดเป็นสาเหตุเดียว','คงbaselineเดิม พักกฎanchor/combined ไม่รวมทุกกฎเข้าproduction และยังไม่ส่งshape_guardไปอ้างว่าเหนือกว่าบนชุดยืนยัน','ขั้นถัดไปที่มีเหตุผล: แยกการเปลี่ยนลำตัวออกจากผลของshoulder normalizationในdiagnosticแบบตรึงมือ แล้วพิจารณาฉลากมือAvatarที่ตรวจโดยคนหากจะfine-tune ไม่ไล่กฎ2Dที่อาจปฏิเสธท่าจริงเพิ่ม','ผลทั้งหมดเป็นdevelopment30ซ้ำ ไม่ใช่การวัดความเข้าใจASL']
report.write_text(text+'\n'.join(lines),encoding='utf8');print(json.dumps(totals))
