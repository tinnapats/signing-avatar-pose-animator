"""Controlled body-anchor substitution. No reference information until assignments saved."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json, hashlib
from pathlib import Path
import numpy as np
import cv2
from track_hands import assign, tests, SETTINGS

ROOT=Path(__file__).resolve().parent
P=ROOT/'tracking3'
O=P/'rtmw_anchor_trial'
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007')
V=['raw','sd_anchor','sd_temporal','rtmw_anchor','rtmw_temporal']
J=[5,6,7,8,9,10]
def stat(e):
    a=np.asarray(e)
    return {'rmse_px':float(np.sqrt(np.mean(a*a))),'pck10_pct':float(np.mean(a<=10)*100),'n_points':len(a)}

tests()
assert _local_json_loads((P/'protocol.json').read_text())['status']=='three_clip_inference_completed'
O.mkdir(exist_ok=True)
assert not (O/'decisions.json').exists(), 'Preserve prior experiment'
rows=_local_json_loads((P/'results.json').read_text())
capture={c['id']:c for c in _local_json_loads((B/'capture_manifest.json').read_text())}
cache={c:np.load(B/'rtm_trial_20261008'/f'{c}.npz') for c in capture if c in ('000','001','002')}
protocol={'settings':SETTINGS,'variants':V,'changed_factor':'Body anchor coordinates AND confidence for COCO joints 5,6,7,8,9,10 from cached RTMW',
    'output_body':'Always original SDPose body; RTMW body is used for assignment only',
    'hand_source':'SDPose; full 21-point blocks with scores are only reordered',
    'references_used_in_assignment':False,'status':'registered_before_assignment',
    'source_hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [ROOT/'track_hands.py',P/'results.json',*[B/'rtm_trial_20261008'/f'{c}.npz' for c in cache]]}}
(O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
pred={v:[] for v in V}; conf={v:[] for v in V}; history={}; decisions=[]
for r in rows:
    cid=r['id']; i=r['frame']; c=cache[cid]; fr=capture[cid]['frames'][i]
    assert fr['sha256']==r['sha256'] and fr['sourceIndex']==r['sourceIndex']
    assert len(c['times'])==len(capture[cid]['frames']) and abs(c['times'][i]-r['time'])<1e-6
    assert np.allclose(np.maximum(c['boxes'][i][:2].astype(int),0),r['box'][:2])
    k=np.array(r['keypoints']); s=np.array(r['scores']); dec={'id':cid,'frame':i,'tag':r['tag']}
    for v in V:
        work=k.copy(); ws=s.copy()
        if v.startswith('rtmw'):
            work[J]=c['rtmw_det'][i,J]; ws[J]=c['rtmw_det_scores'][i,J]
        prior=history.get((cid,v)); prev=prior[1] if prior and i-prior[0]<=3 else None
        if v=='raw': out,sc,info=work,ws,{'swapped':False,'reason':'baseline'}
        else: out,sc,info=assign(work,ws,prev,v.endswith('temporal'))
        out[:91]=k[:91]; sc[:91]=s[:91]
        assert np.array_equal(out[:91],k[:91]) and np.array_equal(sc[:91],s[:91])
        order=np.r_[112:133,91:112] if info['swapped'] else np.arange(91,133)
        assert np.array_equal(out[91:133],k[order]) and np.array_equal(sc[91:133],s[order])
        if all(sc[b]>=.3 and sc[b:b+21].mean()>=.3 for b in (91,112)):
            history[cid,v]=(i,out[[91,112]].copy())
        pred[v].append(out); conf[v].append(sc); dec[v]=info
    decisions.append(dec)
old=np.load(P/'identity_trial/predictions.npz')
for new,previous in [('raw','raw'),('sd_anchor','body_anchor'),('sd_temporal','body_anchor_temporal')]:
    assert np.array_equal(np.array(pred[new]),old[previous]), 'Baseline did not reproduce'
    assert np.array_equal(np.array(conf[new]),old[previous+'_scores'])
np.savez_compressed(O/'predictions.npz',**{v:np.array(pred[v]) for v in V},**{v+'_scores':np.array(conf[v]) for v in V})
(O/'decisions.json').write_text(json.dumps(decisions,indent=2),encoding='utf-8')
protocol.update(status='assignments_saved_before_reference_access',baseline_reproduced_exactly=True,frame_count=len(rows))
(O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
# Only now access renderer references for evaluation; never modify decisions afterwards.
refs={c['id']:{f['index']:f for f in c['samples']} for c in _local_json_loads((B/'coordinate_reference.json').read_text())}
metrics={}; perclip={}; changes=[]
for v in V:
    errors=[]; wrist=[]; byclip={}
    for i,r in enumerate(rows):
        ref=refs[r['id']][r['sourceIndex']]
        assert ref['pngMatches'] and ref['pixelRatio']==1
        for side,b in [('left',91),('right',112)]:
            h=ref['hands'][side]
            if not all(str(j) in h for j in range(21)): continue
            target=np.array([h[str(j)] for j in range(21)])
            e=np.linalg.norm(pred[v][i][b:b+21]-target,axis=1)
            errors.extend(e.tolist()); wrist.append(float(e[0])); byclip.setdefault(r['id'],[]).extend(e.tolist())
            if decisions[i][v]['swapped']:
                before=np.linalg.norm(pred['raw'][i][b:b+21]-target,axis=1)
                changes.append({'variant':v,'tag':r['tag'],'side':side,'before':float(before.mean()),'after':float(e.mean())})
    metrics[v]={**stat(errors),'wrist_rmse_px':stat(wrist)['rmse_px'],
        'reference_hand_frames':len(wrist),'swaps':sum(d[v]['swapped'] for d in decisions),
        'eligible':sum(d[v]['reason']=='cost_comparison' for d in decisions)}
    perclip[v]={cid:stat(e) for cid,e in byclip.items()}
diag={v:{cid:{'eligible':sum(d['id']==cid and d[v]['reason']=='cost_comparison' for d in decisions),
              'swaps':sum(d['id']==cid and d[v]['swapped'] for d in decisions)} for cid in cache} for v in V}
summary={'metrics':metrics,'per_clip':perclip,'changes':changes,'diagnostics':diag,'recognition_run':False}
(O/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['# RTMW body anchors + SDPose hands','',
'ทดลองบน 212 เฟรมของ abdomen, hello, love ที่เวลาเดิม เปลี่ยนเฉพาะพิกัดและ confidence ของไหล่–ศอก–ข้อมือที่ใช้ตัดสินใจเป็น RTMW; จุดมือยังเป็น SDPose เดิมทั้งหมด',
'','ใช้ assign() ตัวเดิมโดยตรง: confidence 0.30, margin 0.15 shoulder units, temporal weight 0.35, history 3 frames, arm ratio 0.2–5.0 ไม่ปรับเกณฑ์ตามผล',
'RTMW ใช้เป็น anchor เท่านั้น ผลลัพธ์ลำตัวคง SDPose เพื่อแยกผลการจับคู่มือ ไม่ใช่ผลรู้จำของ avatar แบบ hybrid ทั้งตัว',
'','ตรวจ frame index/timestamp/image hash ที่ manifest บันทึกไว้ตรงกัน และรัน baseline เดิมซ้ำได้ตรงทุกพิกัด/คะแนน confidence; บันทึก decisions ก่อนเปิดข้อมูลอ้างอิง',
'','## ผล','| วิธี | เฟรมผ่านกฎ | เฟรมสลับมือ | RMSE px ↓ | PCK10 ↑ | Wrist RMSE px ↓ |','|---|---:|---:|---:|---:|---:|']
for v,z in metrics.items(): lines.append(f"| {v} | {z['eligible']} | {z['swaps']} | {z['rmse_px']:.2f} | {z['pck10_pct']:.2f}% | {z['wrist_rmse_px']:.2f} |")
lines+=['',f"ประเมินมือที่มีอ้างอิงครบ 21 จุด {metrics['raw']['reference_hand_frames']} hand-frames รวม {metrics['raw']['n_points']} จุดเท่ากันทุกวิธี ใช้จุดดิบ ไม่กรอง confidence ใน metrics",'',
'## RMSE รายคลิป','| คลิป | raw | RTMW anchor | RTMW + temporal |','|---|---:|---:|---:|']
for cid in cache:
    lines.append('| '+capture[cid]['label']+' | '+' | '.join(f"{perclip[v][cid]['rmse_px']:.2f}" for v in ['raw','rtmw_anchor','rtmw_temporal'])+' |')
for v in ['rtmw_anchor','rtmw_temporal']:
    ch=[c for c in changes if c['variant']==v]
    lines.append(f"\n{v}: ในมือที่สลับและมีอ้างอิง ดีขึ้น {sum(c['after']<c['before']-1e-8 for c in ch)}, แย่ลง {sum(c['after']>c['before']+1e-8 for c in ch)}, รวม {len(ch)} hand-frames")
lines+=['','## ข้อจำกัด',
'- เป็นชุดพัฒนาวิธี 3 คลิปที่เคยเห็นแล้ว ไม่ใช่ชุดยืนยันอิสระ',
'- ไม่มีการรันโมเดลรู้จำคำ จึงไม่มี Precision/Recall/F1 หรือเปอร์เซ็นต์ความเข้าใจ ASL ใหม่',
'- การเปลี่ยนทั้งพิกัดและ confidence ของ body source เป็นการเปรียบเทียบแหล่งตรวจจับ ไม่ได้แยกว่าพิกัดหรือการปรับเทียบ confidence เป็นสาเหตุเดียว',
'- หากไม่มีการสลับใน love ไม่อาจสรุปว่าการติดตามทั่วไปไม่ช่วย เพราะกฎเดิมกำหนดให้ body gate ผ่านก่อน',
'- ไม่มีการเปลี่ยน production หรือปรับข้อมูลต้นฉบับ']
(O/'REPORT_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
