"""Read-only evidence audit: crop space proxies and score/error association."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;P=ROOT/'crop_stage1_20261011';O=ROOT/'crop_space_confidence_20261011';O.mkdir(exist_ok=True)
assert not (O/'protocol.json').exists()
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');R=B/'rtm_trial_20261008'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol={'status':'registered','rules':'No threshold tuning, no crop/recognition/production changes',
 'space':'Effective original square bbox incl 1.25 padding / tight axis-aligned bounding box of 21 renderer joints. Not skin segmentation or background percentage.',
 'distractions':'Face proxy = any RTMW COCO face points 0:5 with confidence>=.30 in effective crop; other hand = any reference point of opposite hand if all21 available. Proxies, not validated segmentation.',
 'confidence_bins':[0,.1,.2,.3,.5,1.000001], 'error_tolerances_px':[10,20],
 'limitations':'Correlated frames, development data, incomplete reference; confidence not a calibrated probability. Pixel thresholds descriptive only.',
 'hashes':{str(p):sha(p) for p in [Path(__file__),P/'results.json',P/'rejections.json',B/'coordinate_reference.json']}}
(O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
refs={c['id']:{f['index']:f for f in c['samples']} for c in read(B/'coordinate_reference.json')}
manifest={c['id']:c for c in read(B/'capture_manifest.json')};records=[];cache={}
for d in read(P/'rejections.json')['records']:
 if not d['has_crop']:continue
 cid=d['id'];i=d['frame'];side=0 if d['side']=='left' else 1
 if cid not in cache:
  for folder in [R,R/'hand_crop']:assert sha(folder/f'{cid}.npz')==read(folder/f'{cid}.json')['sha256']
  cache[cid]=(np.load(R/f'{cid}.npz'),np.load(R/'hand_crop'/f'{cid}.npz'))
 z,h=cache[cid];ref=refs[cid][manifest[cid]['frames'][i]['sourceIndex']]
 g=np.array([ref['hands'][d['side']][str(j)] for j in range(21)])
 box=h['boxes'][i,side];mid=(box[:2]+box[2:])/2;half=(box[2:]-box[:2])*1.25/2;lo=mid-half;hi=mid+half
 area=float(np.prod(hi-lo));handarea=float(np.prod(np.ptp(g,axis=0)))
 def inside(p):return np.all((p>=lo)&(p<=hi),axis=-1)
 face=z['rtmw_det'][i,:5];facevalid=z['rtmw_det_scores'][i,:5]>=.3
 other=ref['hands']['right' if side==0 else 'left'];hasother=all(str(j) in other for j in range(21))
 otherin=bool(inside(np.array([other[str(j)] for j in range(21)])).any()) if hasother else None
 wh=np.array([manifest[cid]['width'],manifest[cid]['height']]);visible=float(np.prod(np.maximum(0,np.minimum(hi,wh)-np.maximum(lo,0))))
 q=h['points'][i,side];e=np.linalg.norm(q-g,axis=1)
 records.append({**d,'crop_to_hand_bbox_area':area/handarea if handarea>1e-8 else None,
  'hand_bbox_crop_pct':100*handarea/area,'off_image_pct':100*(1-visible/area),
  'face_proxy_available':bool(facevalid.any()),'face_proxy_inside':bool((inside(face)&facevalid).any()),
  'other_hand_inside':otherin,'rmse_px':float(np.sqrt(np.mean(e*e))),
  'pck10_pct':float(np.mean(e<=10)*100),'pck20_pct':float(np.mean(e<=20)*100),
  'joint_confidence':h['scores'][i,side].tolist(),'joint_errors_px':e.tolist()})
def rank(a):
 a=np.asarray(a);order=np.argsort(a);out=np.empty(len(a),float);j=0
 while j<len(a):
  k=j+1
  while k<len(a) and a[order[k]]==a[order[j]]:k+=1
  out[order[j:k]]=(j+k-1)/2;j=k
 return out
def corr(a,b):
 if len(a)<3 or np.std(a)==0 or np.std(b)==0:return None
 return float(np.corrcoef(rank(a),rank(b))[0,1])
def stats(rr):
 vals=[r['crop_to_hand_bbox_area'] for r in rr if r['crop_to_hand_bbox_area'] is not None]
 return {'n':len(rr),'area_ratio_median':float(np.median(vals)) if vals else None,
 'area_ratio_q25_q75':np.quantile(vals,[.25,.75]).tolist() if vals else None,
 'bbox_occupancy_median_pct':float(np.median([r['hand_bbox_crop_pct'] for r in rr])) if rr else None,
 'rmse_px':float(np.sqrt(np.mean([r['rmse_px']**2 for r in rr]))) if rr else None,
 'median_error_px':float(np.median([r['rmse_px'] for r in rr])) if rr else None,
 'pck10_pct':float(np.mean([r['pck10_pct'] for r in rr])) if rr else None,
 'pck20_pct':float(np.mean([r['pck20_pct'] for r in rr])) if rr else None}
def analyze(rr):
 bins=[];edges=protocol['confidence_bins']
 for a,b in zip(edges,edges[1:]):
  ss=[r for r in rr if a<=r['mean_confidence']<b];bins.append({'range':[a,b],**stats(ss)})
 face=[r for r in rr if r['face_proxy_available']];other=[r for r in rr if r['other_hand_inside'] is not None]
 return {'overall':stats(rr),'confidence_bins':bins,
 'spearman_mean_conf_vs_error':corr([r['mean_confidence'] for r in rr],[r['rmse_px'] for r in rr]),
 'spearman_wrist_conf_vs_error':corr([r['wrist_confidence'] for r in rr],[r['rmse_px'] for r in rr]),
 'spearman_area_ratio_vs_error':corr([r['crop_to_hand_bbox_area'] for r in rr if r['crop_to_hand_bbox_area'] is not None],[r['rmse_px'] for r in rr if r['crop_to_hand_bbox_area'] is not None]),
 'face_proxy_inside':stats([r for r in face if r['face_proxy_inside']]),'face_proxy_outside':stats([r for r in face if not r['face_proxy_inside']]),
 'other_hand_inside':stats([r for r in other if r['other_hand_inside']]),'other_hand_outside':stats([r for r in other if not r['other_hand_inside']]),
 'both_conf_pass':stats([r for r in rr if r['mean_confidence']>=.3 and r['wrist_confidence']>=.3]),
 'off_image_frames':sum(r['off_image_pct']>1e-6 for r in rr)}
result={'all_crops':analyze(records),'fully_covered':analyze([r for r in records if r['full_coverage']]),
 'per_clip':{c['label']:analyze([r for r in records if r['id']==cid]) for cid,c in manifest.items()},'records':records}
(O/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
lines=['# พื้นที่ crop สิ่งรบกวน และ confidence','',
 'ตรวจผลดิบเดิม ไม่แก้ bbox/threshold/model/reference ผลหลักใช้เฉพาะกรอบที่ครอบจุดอ้างอิงครบ เพื่อลดการปนกับปัญหามือถูกตัด',
 'อัตราพื้นที่ = พื้นที่ crop รวม padding / พื้นที่กรอบแน่นของจุดมือ 21 จุด ไม่ใช่พื้นที่ผิวมือหรือเปอร์เซ็นต์พื้นหลัง; ท่าหันขอบทำให้กรอบจุดเล็กลงได้',
 'ใบหน้าใช้จุด RTMW เป็น proxy; มืออีกข้างใช้ reference เฉพาะเมื่อมีครบ 21 จุด ไม่มี reference แปลว่าไม่ทราบ ไม่ใช่ไม่มีมือ', '',
 '| Mean confidence | hand-frames | RMSE px | Median error px | PCK10 % | PCK20 % |','|---|---:|---:|---:|---:|---:|']
for r in result['fully_covered']['confidence_bins']:
 if r['n']:lines.append(f"| {r['range'][0]}–{min(r['range'][1],1):g} | {r['n']} | {r['rmse_px']:.2f} | {r['median_error_px']:.2f} | {r['pck10_pct']:.2f} | {r['pck20_pct']:.2f} |")
lines+=['','## สรุปตัวเลข (fully covered)', '', '```json',json.dumps({k:v for k,v in result['fully_covered'].items() if k!='confidence_bins'},indent=2),'```','',
 'Spearman ติดลบสำหรับ confidence/error หมายถึง confidence สูงสัมพันธ์กับ error ต่ำ ไม่ได้ยืนยันเหตุหรือ calibration และเฟรมต่อเนื่องไม่ใช่ตัวอย่างอิสระ',
 'PCK10/20 คือสัดส่วนจุดในระยะ 10/20 พิกเซล ไม่ใช่ความถูกต้อง ASL และไม่ได้ใช้เลือก threshold ใหม่',
 'ความสัมพันธ์กับกรอบกว้างหรือสิ่งรบกวนไม่ใช่ข้อพิสูจน์สาเหตุ เพราะปนกับชนิดท่า ขนาดมือ และคำ ต้องทดลอง bbox แบบตรึงค่าแยกต่างหากก่อนสรุป']
(O/'REPORT_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
protocol['status']='completed';(O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
print(json.dumps(result['fully_covered']),flush=True)
