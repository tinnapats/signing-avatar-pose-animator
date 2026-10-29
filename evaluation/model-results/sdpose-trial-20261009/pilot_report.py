
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parent
O=ROOT/'pilot6'
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007')
rows=_local_json_loads((O/'results.json').read_text())
assert _local_json_loads((O/'protocol.json').read_text())['status']=='six_frame_inference_completed'
refs={c['id']:{f['index']:f for f in c['samples']} for c in _local_json_loads((B/'coordinate_reference.json').read_text())}
edges=[(5,7),(7,9),(6,8),(8,10),(5,6)]
for base in (91,112):
    for chain in ((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)):
        edges.extend((base+a,base+b) for a,b in zip(chain,chain[1:]))
def draw(im,k,s,title):
    for a,b in edges:
        if min(s[a],s[b])>=.3: cv2.line(im,tuple(np.rint(k[a]).astype(int)),tuple(np.rint(k[b]).astype(int)),(0,160,0),2)
    for j in list(range(17))+list(range(91,133)):
        if s[j]>=.3: cv2.circle(im,tuple(np.rint(k[j]).astype(int)),3,(0,0,255),-1)
    cv2.rectangle(im,(0,0),(808,35),(255,255,255),-1)
    cv2.putText(im,title,(7,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),2)
    return cv2.resize(im,(404,404))
panels=[]; metrics=[]; swaps=[]; errors={'SDPose':[],'RTMW':[]}
for r in rows:
    im=cv2.imread(str(O/f"{r['tag']}_input.png"))
    rt=np.load(B/'rtm_trial_20261008'/f"{r['id']}.npz")
    variants={'RTMW':(rt['rtmw_det'][r['frame']],rt['rtmw_det_scores'][r['frame']]),
              'SDPose':(np.array(r['keypoints']),np.array(r['scores']))}
    panels.append(np.hstack([draw(im.copy(),k,s,f"{name} {r['label']} f{r['frame']}") for name,(k,s) in variants.items()]))
    ref=refs[r['id']][r['sourceIndex']]
    assert ref['pngMatches'] and ref['pixelRatio']==1
    if all(all(str(i) in ref['hands'][side] for i in range(21)) for side in ('left','right')):
        target_pair=np.array([[ref['hands'][side][str(i)] for i in range(21)] for side in ('left','right')])
        for name,(k,s) in variants.items():
            p=k[91:133].reshape(2,21,2)
            swaps.append({'tag':r['tag'],'method':name,
                'direct_rmse_px':float(np.sqrt(np.mean(np.sum((p-target_pair)**2,axis=2)))),
                'oracle_swapped_rmse_px':float(np.sqrt(np.mean(np.sum((p[::-1]-target_pair)**2,axis=2))))})
    for side,base in [('left',91),('right',112)]:
        hand=ref['hands'][side]
        if not all(str(i) in hand for i in range(21)): continue
        target=np.array([hand[str(i)] for i in range(21)])
        m={'id':r['id'],'label':r['label'],'frame':r['frame'],'hand':side}
        for name,(k,s) in variants.items():
            e=np.linalg.norm(k[base:base+21]-target,axis=1)
            accepted=bool(s[base]>=.3 and s[base:base+21].mean()>=.3)
            errors[name].extend(e.tolist())
            m[name]={'raw_joint_rmse_px':float(np.sqrt(np.mean(e**2))),
                'raw_pck10_pct':float(np.mean(e<=10)*100),'accepted_mean_and_wrist_0_30':accepted,
                'wrist_error_px':float(e[0])}
        metrics.append(m)
cv2.imwrite(str(O/'comparison.png'),np.vstack(panels))
summary={name:{'n_reference_joints':len(e),'raw_joint_rmse_px':float(np.sqrt(np.mean(np.square(e)))) if e else None,
              'raw_pck10_pct':float(np.mean(np.array(e)<=10)*100) if e else None} for name,e in errors.items()}
data={'hand_reference_frames':len(metrics),'geometry_raw_unfiltered':summary,'per_hand':metrics,
      'swap_diagnostic_only_not_corrected':swaps,
      'note':'Six selected frames only. Reference is renderer geometry, not ASL ground truth. Same cached person boxes. No recognition score.'}
(O/'geometry.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
lines=['# SDPose pilot 6 เฟรม','',
'รันน้ำหนักทางการสำเร็จครบ 6 เฟรม: abdomen, hello, love อย่างละเฟรมแรกและกลางคลิป เลือกก่อนเห็นผล ไม่เปลี่ยนระบบหลัก',
'','## เงื่อนไข',
'- อ่านภาพ Avatar จริงเท่านั้น ใช้กรอบคน YOLOX ที่เคยตรวจจากภาพชุดเดียวกัน ไม่มีการส่งพิกัดอ้างอิงเข้าโมเดล',
'- SDPose: input 768×1024, timestep 999, seed 0 ต่อเฟรม, ไม่ flip-test; UNet FP16 เพื่อจำกัด VRAM ส่วน VAE/text/decoder FP32 โหลดทีละส่วน',
'- เป็นการปรับวิธีจัดหน่วยความจำ ไม่ได้ยืนยันว่าเทียบเท่าการรัน FP32 ต้นฉบับทุกตัวเลข',
'- น้ำหนักโหลดผ่านการตรวจ missing/unexpected keys; decoder strict=True; เก็บ raw 133 จุดและ confidence ครบ',
'- ภาพเปรียบเทียบ: RTMW ซ้าย, SDPose ขวา, แสดงจุด confidence >=0.30',
'','## ตรวจพิกัดมือเทียบ Renderer',
f"มีข้อมูลอ้างอิงมือครบ 21 จุดเพียง {len(metrics)} hand-frames จากภาพ 6 เฟรม ค่าต่อไปนี้ใช้จุดดิบครบทุกจุดของมือที่มีอ้างอิง ไม่กรอง confidence จึงไม่ใช่ metric เดียวกับตาราง hybrid ก่อนหน้า",'',
'| วิธี | จุดอ้างอิง | RMSE พิกเซล ↓ | PCK ระยะ 10 พิกเซล ↑ |','|---|---:|---:|---:|']
for name,m in summary.items(): lines.append(f"| {name} | {m['n_reference_joints']} | {m['raw_joint_rmse_px']:.2f} | {m['raw_pck10_pct']:.2f}% |")
lines+=['','## เวลาและข้อจำกัด',f"- VAE เฉลี่ย {np.mean([r['vae_seconds'] for r in rows]):.2f} วินาที/ภาพ; UNet เฉลี่ย {np.mean([r['unet_seconds'] for r in rows]):.2f} วินาที/ภาพ ไม่รวมโหลดโมเดล/decoder/เขียนไฟล์ จึงไม่ใช่เวลา end-to-end",
f"- Peak CUDA allocated ที่วัดในช่วง UNet สูงสุด {max(r['peak_cuda_allocated_bytes'] for r in rows)/1024**3:.2f} GiB ไม่ใช่ peak ตลอดทั้ง pipeline",
'- ไม่ใช่ผล 30 คลิปหรือคะแนน ST-GCN; ภาพนิ่งไม่พอคำนวณความถูกต้องของคำภาษามือ',
'- ไม่มี ground truth ไหล่/ศอกที่ตรวจโดยคน จึงใช้ภาพซ้อนจุดประกอบ ไม่สรุปความถูกต้องเชิงตัวเลขของลำตัว',
'- พิกัดอ้างอิงจาก Renderer ใช้ตรวจเรขาคณิตเท่านั้น ไม่ใช่การยืนยันความหมายโดยผู้รู้ ASL',
'- ขั้นถัดไปควรใช้คลิปเต็มที่ตรึงไว้และเวลาเดิม เพื่อตรวจความต่อเนื่อง ซ้าย–ขวา และการรู้จำ โดยยังไม่แทนระบบหลัก']
lines+=['','## ตรวจการจับคู่ซ้าย–ขวา (วิเคราะห์เท่านั้น)',
'ใช้ข้อมูลอ้างอิงคำนวณกรณีสมมติสลับมือเพื่อแยกสาเหตุ ไม่แก้จุดจริงและไม่ใช้คะแนนกรณีสลับเป็นผลโมเดล',
'| เฟรม | วิธี | RMSE ตาม ID เดิม | RMSE ถ้าสมมติสลับมือ |','|---|---|---:|---:|']
for d in swaps: lines.append(f"| {d['tag']} | {d['method']} | {d['direct_rmse_px']:.2f} | {d['oracle_swapped_rmse_px']:.2f} |")
lines+=['','ข้อสังเกตภาพ: hello กลางคลิป SDPose วางจุดนิ้วบางส่วนใกล้ภาพขึ้น แต่ abdomen กลางคลิปยังเสียรายละเอียดนิ้ว; ท่าไขว้แขน love ต้องตรวจความสัมพันธ์ไหล่–ศอก–ข้อมือและฝั่งมือก่อนนำไปเข้าโมเดลรู้จำ ไม่ควรสรุปผู้ชนะจากภาพ 6 เฟรม']
(O/'REPORT_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(data,indent=2))
