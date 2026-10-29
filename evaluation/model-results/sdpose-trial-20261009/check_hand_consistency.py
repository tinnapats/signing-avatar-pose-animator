"""Descriptive six-clip diagnosis. No threshold search or model routing."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007')
O=ROOT/'hand_consistency_20261011';O.mkdir(exist_ok=True)
assert not (O/'protocol.json').exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources=[ROOT/'tracking3/results.json',ROOT/'confirmation3/results.json']
rows=sum([_local_json_loads(p.read_text()) for p in sources],[])
protocol={'status':'registered','scope':'All six available SDPose clips, no label-based exclusions',
 'confidence':.30,'features':'Same nearest RTMW body wrist for both hand wrists; own-body-wrist distance divided by shoulder span; hand-wrist separation / body-wrist separation.',
 'eligibility':'Both SDPose hands pass original mean+wrist .30; RTMW shoulders and wrists each >=.30; nonzero shoulder and body-wrist spans.',
 'interpretation':'Same-nearest-anchor is a warning, NOT proof of duplicate detection. Crossing/touching and erroneous RTMW anchors can confound it.',
 'no_new_threshold_search':True,'no_recognition_or_production_change':True,
 'hashes':{str(p):sha(p) for p in [Path(__file__),*sources]}}
(O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
cache={cid:np.load(B/'rtm_trial_20261008'/f'{cid}.npz') for cid in sorted({r['id'] for r in rows})}
features=[]
for r in rows:
 cid=r['id'];i=r['frame'];c=cache[cid]
 assert abs(c['times'][i]-r['time'])<1e-6
 k=np.array(r['keypoints']);s=np.array(r['scores']);bk=c['rtmw_det'][i];bs=c['rtmw_det_scores'][i]
 sw=k[[91,112]];bw=bk[[9,10]]
 scale=float(np.linalg.norm(bk[5]-bk[6]));span=float(np.linalg.norm(bw[0]-bw[1]))
 eligible=bool(all(s[b]>=.3 and s[b:b+21].mean()>=.3 for b in (91,112)) and np.all(bs[[5,6,9,10]]>=.3) and scale>1e-6 and span>1e-6)
 dist=np.linalg.norm(sw[:,None,:]-bw[None,:,:],axis=2)
 nearest=np.argmin(dist,axis=1)
 f={'id':cid,'label':r['label'],'frame':i,'sourceIndex':r['sourceIndex'],'eligible':eligible,
    'same_nearest_anchor':bool(nearest[0]==nearest[1]) if eligible else None,
    'separation_ratio':float(np.linalg.norm(sw[0]-sw[1])/span) if eligible else None,
    'left_own_wrist_shoulder_units':float(dist[0,0]/scale) if eligible else None,
    'right_own_wrist_shoulder_units':float(dist[1,1]/scale) if eligible else None}
 features.append(f)
# Evidence saved before references: neither reference nor true label enters a feature.
(O/'features_before_reference.json').write_text(json.dumps(features,indent=2),encoding='utf-8')
refs={c['id']:{s['index']:s for s in c['samples']} for c in _local_json_loads((B/'coordinate_reference.json').read_text())}
samples=[]
for r,f in zip(rows,features):
 ref=refs[r['id']][r['sourceIndex']];assert ref['pngMatches'] and ref['pixelRatio']==1
 if not f['eligible']:continue
 k=np.array(r['keypoints']);c=cache[r['id']];i=r['frame']
 for side,b in [('left',91),('right',112)]:
  h=ref['hands'][side]
  if not all(str(j) in h for j in range(21)):continue
  target=np.array([h[str(j)] for j in range(21)])
  sd=float(np.sqrt(np.mean(np.sum((k[b:b+21]-target)**2,axis=1))))
  rt=float(np.sqrt(np.mean(np.sum((c['rtmw_det'][i,b:b+21]-target)**2,axis=1))))
  samples.append({**f,'side':side,'sd_rmse_px':sd,'rtmw_raw_rmse_px':rt,'own_wrist_distance':f[side+'_own_wrist_shoulder_units'],
   'rtmw_hand_accepted':bool(c['rtmw_det_scores'][i,b]>=.3 and c['rtmw_det_scores'][i,b:b+21].mean()>=.3)})
def group(ss):
 return {'hand_frames':len(ss),'sd_rmse_px':float(np.sqrt(np.mean([s['sd_rmse_px']**2 for s in ss]))) if ss else None,
 'rtmw_raw_rmse_px':float(np.sqrt(np.mean([s['rtmw_raw_rmse_px']**2 for s in ss]))) if ss else None}
summary={}
for cid in cache:
 ff=[f for f in features if f['id']==cid];eligible=[f for f in ff if f['eligible']];ss=[s for s in samples if s['id']==cid]
 summary[ff[0]['label']]={'frames':len(ff),'eligible':len(eligible),'same_anchor_frames':sum(f['same_nearest_anchor'] for f in eligible),
  'median_separation_ratio':float(np.median([f['separation_ratio'] for f in eligible])) if eligible else None,
  'same_anchor':group([s for s in ss if s['same_nearest_anchor']]),'different_anchor':group([s for s in ss if not s['same_nearest_anchor']])}
result={'clips':summary,'same_anchor':group([s for s in samples if s['same_nearest_anchor']]),'different_anchor':group([s for s in samples if not s['same_nearest_anchor']]),
 'samples':samples,'frames':len(rows),'limitations':'Correlated frames, selected developmental clips, RTMW anchors are predictions not ground truth. Unaccepted anchors excluded. No ASL accuracy claim.'}
(O/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
lines=['# ตรวจความสอดคล้องของมือกับข้อมือ RTMW','',
 'ใช้ทุก 6 คลิปที่มี SDPose รวม 422 เฟรม ไม่ปรับ threshold ไล่คะแนน ไม่ใช้ชื่อคำเลือกโมเดล และไม่เปลี่ยนระบบหลัก',
 'สัญญาณเตือน: ข้อมือ SDPose ทั้งสองใกล้ข้อมือ RTMW ข้างเดียวกันที่สุด ตรวจเฉพาะเฟรมที่ผ่าน confidence เดิม 0.30 และมีแกนอ้างอิงใช้ได้ ไม่ใช่ข้อพิสูจน์ว่าจับมือซ้ำ',
 'บันทึก features ก่อนอ่าน Renderer reference; reference ใช้วัดความคลาดเคลื่อนเท่านั้น','',
 '| คำ | เฟรมทั้งหมด | ผ่านเงื่อนไขตรวจ | ใกล้ข้อมือข้างเดียวกัน | RMSE SDPose กลุ่มเตือน px | RMSE SDPose กลุ่มไม่เตือน px |',
 '|---|---:|---:|---:|---:|---:|']
def fmt(x):return 'N/A' if x is None else f'{x:.2f}'
for l,m in summary.items():lines.append(f"| {l} | {m['frames']} | {m['eligible']} | {m['same_anchor_frames']} | {fmt(m['same_anchor']['sd_rmse_px'])} | {fmt(m['different_anchor']['sd_rmse_px'])} |")
lines+=['','RMSE รวมจุดมือที่มี reference ครบ 21 จุดเท่านั้น แต่จำนวนเฟรมเตือนนับทุกเฟรมที่ผ่านเงื่อนไขตรวจ จึงมีตัวหารต่างกัน รายละเอียด hand-frame อยู่ใน results.json',
 'มือเข้าใกล้หรือไขว้กันอาจทำให้เกิดสัญญาณเตือนทั้งที่ตรวจถูก ข้อมือ RTMW ก็ผิดได้ และการไม่เตือนไม่ได้แปลว่าพิกัดถูก ไม่ควรนำกฎนี้ไปเลือกโมเดลอัตโนมัติก่อนทดสอบเพิ่ม',
 'ยังไม่รัน ST-GCN เพิ่ม เพราะรอบนี้ตรวจสัญญาณคุณภาพ ไม่ได้เปลี่ยน input ของโมเดล']
(O/'REPORT_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
protocol['status']='completed';(O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
print(json.dumps(summary),flush=True)
