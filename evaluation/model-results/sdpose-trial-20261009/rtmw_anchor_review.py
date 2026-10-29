
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json
from pathlib import Path
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parent
P=ROOT/'tracking3'; O=P/'rtmw_anchor_trial'
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007/rtm_trial_20261008')
rows=_local_json_loads((P/'results.json').read_text()); d=_local_json_loads((O/'decisions.json').read_text())
m=_local_json_loads((O/'metrics.json').read_text()); z=np.load(O/'predictions.npz')
indices=[next(i for i,r in enumerate(rows) if r['id']=='002'),
 next(i for i,x in enumerate(d) if x['rtmw_anchor']['swapped']),
 next(i for i,x in enumerate(d) if x['rtmw_temporal']['swapped'] and not x['rtmw_anchor']['swapped'])]
selection=[{'tag':rows[i]['tag'],'rule':why} for i,why in zip(indices,['first love frame','first anchor swap','first temporal-only swap'])]
panels=[]
for i in indices:
    r=rows[i]; tiles=[]; rt=np.load(B/f"{r['id']}.npz")
    for v in ('raw','rtmw_anchor','rtmw_temporal'):
        im=cv2.imread(str(P/f"{r['tag']}_input.png")); k=z[v][i]; s=z[v+'_scores'][i]
        # Crop fixed upper-body area only for display; input inference was full person bbox.
        for base,color in [(91,(220,90,0)),(112,(0,60,240))]:
            for chain in ((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)):
                for a,b in zip(chain,chain[1:]):
                    if min(s[base+a],s[base+b])>=.3:
                        cv2.line(im,tuple(np.rint(k[base+a]).astype(int)),tuple(np.rint(k[base+b]).astype(int)),color,2)
            cv2.putText(im,'L' if base==91 else 'R',tuple(np.rint(k[base]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
        tile=cv2.resize(im[:440,70:470],(400,440))
        cv2.rectangle(tile,(0,0),(400,55),(255,255,255),-1)
        cv2.putText(tile,f"{v} love f{r['frame']}",(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
        cv2.putText(tile,f"swap={d[i][v]['swapped']}",(5,43),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
        tiles.append(tile)
    panels.append(np.hstack(tiles))
cv2.imwrite(str(O/'comparison.png'),np.vstack(panels))
(O/'image_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
lines=['','## สรุปผลและตรวจภาพ',
f"RMSE รวมลดจาก {m['metrics']['raw']['rmse_px']:.2f} เป็น {m['metrics']['rtmw_temporal']['rmse_px']:.2f} px; ดีขึ้นเฉพาะ love ใน 3 คลิปนี้ abdomen/hello เหมือนเดิม",
'RTMW ทำให้ love ผ่านเกณฑ์จับคู่ 35/53 เฟรม จากเดิม SDPose 0/53 เฟรม; temporal variant สลับ 20/53 เฟรม จึงยังไม่ได้แก้ทั้งคลิป',
'ภาพ comparison.png ใช้ L สีน้ำเงิน/R สีแดง วาดเฉพาะมือ ผลลำตัวยังคง SDPose ทุกวิธี เลือก love เฟรมแรก, เฟรมแรกที่ anchor สลับ, เฟรมแรกที่ temporal สลับเพิ่ม ไม่เลือกจาก error',
'ข้อสรุป: มีหลักฐานว่าแหล่ง body anchor เป็นข้อจำกัดของกฎเดิมในชุดนี้ แต่ยังระบุไม่ได้ว่าเกิดจากความแม่นพิกัดหรือ confidence calibration เพียงอย่างเดียว',
'น่าทดลองยืนยันบนคลิปใหม่โดยตรึงวิธี RTMW anchors + SDPose hands + temporal ไว้ ไม่ควรนำเข้าระบบหลักหรืออ้างคะแนนรู้จำดีขึ้นจากผลนี้เพียงอย่างเดียว']
for v in ('rtmw_anchor','rtmw_temporal'):
    c=[x for x in m['changes'] if x['variant']==v]
    print(v,'better',sum(x['after']<x['before'] for x in c),'worse',sum(x['after']>x['before'] for x in c))
report=O/'REPORT_TH.md'
text=report.read_text(encoding='utf-8')
if '## สรุปผลและตรวจภาพ' not in text: report.write_text(text+'\n'.join(lines)+'\n',encoding='utf-8')
print('Review artifacts saved')
