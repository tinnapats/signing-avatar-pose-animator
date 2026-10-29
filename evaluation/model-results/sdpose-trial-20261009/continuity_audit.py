"""Trace actual hand input sources and bounded tracking opportunities, no imputation."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;T=ROOT/'wrist_crop_tight20_20261011';O=ROOT/'continuity_20261011';O.mkdir(exist_ok=True)
assert not (O/'protocol.json').exists()
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');R=B/'rtm_trial_20261008'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(B/'capture_manifest.json');refs={c['id']:{f['index']:f for f in c['samples']} for c in read(B/'coordinate_reference.json')}
save=lambda p,x:p.write_text(json.dumps(x,indent=2),encoding='utf-8')
protocol={'status':'registered','input':'Actual tight20 crop fallback pipeline; accepted crop otherwise accepted RTMW otherwise zeros BEFORE normalization',
 'reference':'Complete21point renderer reference marks evaluable detailed-hand frames, NOT verified active signing or absence/rest when missing',
 'tracking_opportunity':'Rejected crop <=3 frames after accepted crop seed, current RTMW wrist/elbow confidence >=.30. No track propagation beyond3; not proof seed correct.',
 'no_changes':'No threshold tuning, no tracking/imputation yet, no production edits', 'hashes':{str(p):sha(p) for p in [Path(__file__),T/'results.json',B/'coordinate_reference.json']}}
save(O/'protocol.json',protocol)
records=[];runs=[];perclip=[]
def intervals(mask):return np.flatnonzero(np.diff(np.r_[False,mask,False])).reshape(-1,2)
for c in manifest:
 cid=c['id'];h=np.load(T/f'{cid}.npz');z=np.load(R/f'{cid}.npz');assert sha(T/f'{cid}.npz')==read(T/f'{cid}.json')['sha256']
 assert np.allclose(h['times'],z['times']) and np.allclose(h['times'],[f['time'] for f in c['frames']])
 for side,name,b,w,e in [(0,'left',91,9,7),(1,'right',112,10,8)]:
  crop=h['accepted'][:,side];ss=z['rtmw_det_scores'][:,b:b+21];rt=(ss[:,0]>=.3)&(ss.mean(1)>=.3)
  zero=~crop&~rt;refmask=[];opportunities=[];last=None
  for i,f in enumerate(c['frames']):
   ref=refs[cid][f['sourceIndex']];assert ref['pngMatches'] and ref['pixelRatio']==1
   hh=ref['hands'][name];available=all(str(j) in hh for j in range(21));refmask.append(available)
   if crop[i]:last=i
   candidate=bool(not crop[i] and last is not None and i-last<=3 and min(z['rtmw_det_scores'][i,w],z['rtmw_det_scores'][i,e])>=.3)
   opportunities.append(candidate)
   r={'id':cid,'label':c['label'],'side':name,'frame':i,'reference':available,'source':'crop' if crop[i] else ('rtmw' if rt[i] else 'zero'), 'tracking_candidate':candidate,'seed':last if candidate else None}
   if available:
    g=np.array([hh[str(j)] for j in range(21)]);q=h['points'][i,side] if crop[i] else z['rtmw_det'][i,b:b+21]
    r['input_rmse_px']=float(np.sqrt(np.mean(np.sum((q-g)**2,axis=1)))) if not zero[i] else None
   records.append(r)
  refmask=np.array(refmask)
  for typ,mask in [('crop_rejected',~crop),('actual_zero',zero)]:
   for start,end in intervals(mask):runs.append({'id':cid,'label':c['label'],'side':name,'type':typ,'start':int(start),'end_exclusive':int(end),'frames':int(end-start),'seconds':float((end-start)/c['fps']),'reference_frames':int(refmask[start:end].sum())})
  perclip.append({'id':cid,'label':c['label'],'side':name,'frames':len(crop),'crop':int(crop.sum()),'rtmw':int((~crop&rt).sum()),'zero':int(zero.sum()),'reference_frames':int(refmask.sum()),'candidate':sum(opportunities)})
def summarize(rr):
 return {'hand_frames':len(rr),'sources':{s:sum(r['source']==s for r in rr) for s in ['crop','rtmw','zero']},'tracking_candidates':sum(r['tracking_candidate'] for r in rr),
 'tracking_candidates_with_zero':sum(r['tracking_candidate'] and r['source']=='zero' for r in rr)}
summary={'all':summarize(records),'with_reference':summarize([r for r in records if r['reference']]),'without_reference':summarize([r for r in records if not r['reference']])}
result={'summary':summary,'per_clip':perclip,'runs':runs,'records':records,
 'longest_zero_run':max([r['frames'] for r in runs if r['type']=='actual_zero'],default=0),
 'longest_crop_rejection':max([r['frames'] for r in runs if r['type']=='crop_rejected'],default=0)}
save(O/'results.json',result);protocol['status']='completed';save(O/'protocol.json',protocol)
lines=['# ความต่อเนื่องของมือที่ส่งเข้า ST-GCN','',
 'ตรวจลำดับแหล่งพิกัดจริงของ tight20+RTMW fallback ก่อน normalization: crop ที่ผ่านเกณฑ์ → RTMW ที่ผ่านเกณฑ์ → zero เมื่อทั้งสองไม่ผ่าน',
 'การไม่มี Renderer reference ไม่ได้ยืนยันว่าเป็นมือพักหรือไม่มีมือ และการมี reference ไม่ได้ยืนยันว่ากำลังทำส่วนสำคัญของคำ',
 '| กลุ่ม | hand-frames | Crop | RTMW | Zero | โอกาส tracking ≤3เฟรม | โอกาสที่เป็น Zero จริง |','|---|---:|---:|---:|---:|---:|---:|']
for name,s in summary.items():lines.append(f"| {name} | {s['hand_frames']} | {s['sources']['crop']} | {s['sources']['rtmw']} | {s['sources']['zero']} | {s['tracking_candidates']} | {s['tracking_candidates_with_zero']} |")
lines+=['',f"ช่วง zero ยาวสุด {result['longest_zero_run']} เฟรม; ช่วง crop ถูกปฏิเสธยาวสุด {result['longest_crop_rejection']} เฟรม",'ตัวนับ tracking เป็นเพียงโอกาสให้ทดลอง ไม่ใช่จำนวนที่ tracking จะสำเร็จ จุดเริ่มผ่าน confidence ไม่ได้แปลว่าถูก',
 'ยังไม่รัน ST-GCN ใหม่เพราะไม่เปลี่ยน input; ผลนี้ไม่ใช่คะแนน ASL หรือความแม่นยำโมเดลเพิ่ม']
(O/'REPORT_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['runs','records','per_clip']}),flush=True)
