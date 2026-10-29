
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json,sys,collections
from pathlib import Path
import numpy as np
import cv2
ROOT=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007')
def read(p):return _local_json_loads((ROOT/p).read_text(encoding='utf8'))
refs=read('coordinate_reference.json'); rows=[]; pairing=[]; separate=[]; coverage=collections.Counter()
assert sum(len(c['samples']) for c in refs)==1964
assert all(r['pngMatches'] for c in refs for r in c['samples'])
for c in refs:
    det=read('detected/'+c['id']+'.json')['rows']
    for i,(ref,d) in enumerate(zip(c['samples'],det)):
        hands={s:np.array([ref['hands'][s][str(j)] for j in range(21)]) for s in ['left','right'] if len(ref['hands'][s])==21}
        coverage['frames']+=1;coverage['reference_hands']+=len(hands)
        for s,g in hands.items():
            if not d[s]:coverage['missing_on_reference']+=1;continue
            q=np.array(d[s])[:,:2]*808;dist=np.linalg.norm(q-g,axis=1);palm=float(np.linalg.norm(g[9]-g[0]))
            rows.append(dict(id=c['id'],label=c['label'],frame=i,side=s,rmse=float(np.sqrt(np.mean(dist**2))),mean=float(dist.mean()),wrist=float(dist[0]),palm=palm,nme=float(dist.mean()/max(palm,1)),shape_rmse=float(np.sqrt(np.mean(np.sum(((q-q[0])-(g-g[0]))**2,axis=1)))),within10=int((dist<=10).sum()),within_half_palm=int((dist<=palm*.5).sum())))
        if len(hands)==2 and d['left'] and d['right']:
            l=np.array(d['left'])[:,:2]*808;r=np.array(d['right'])[:,:2]*808
            same=float((np.linalg.norm(l-hands['left'],axis=1).mean()+np.linalg.norm(r-hands['right'],axis=1).mean())/2)
            swap=float((np.linalg.norm(l-hands['right'],axis=1).mean()+np.linalg.norm(r-hands['left'],axis=1).mean())/2)
            pairing.append(dict(id=c['id'],frame=i,same=same,swap=swap,preferred='same' if same+4<swap else 'swapped' if swap+4<same else 'ambiguous'))
        if len(hands)==2:
            for q,h in zip(d['hands'],d['handedness']):
                q=np.array(q)[:,:2]*808
                distances={s:float(np.linalg.norm(q-g,axis=1).mean()) for s,g in hands.items()}
                raw=h['label'].lower();flipped='right' if raw=='left' else 'left';best=min(distances,key=distances.get)
                separate.append(dict(id=c['id'],frame=i,raw=raw,flipped=flipped,best=best,ambiguous=abs(distances['left']-distances['right'])<=4,confidence=h['score'],distances=distances))
def stats(rr):
    if not rr:return {}
    return dict(hand_frames=len(rr),mean_error_px=float(np.mean([r['mean'] for r in rr])),rmse_px=float(np.sqrt(np.mean([r['rmse']**2 for r in rr]))),median_hand_rmse_px=float(np.median([r['rmse'] for r in rr])),p95_hand_rmse_px=float(np.percentile([r['rmse'] for r in rr],95)),median_wrist_px=float(np.median([r['wrist'] for r in rr])),median_shape_rmse_px=float(np.median([r['shape_rmse'] for r in rr])),median_palm_px=float(np.median([r['palm'] for r in rr])),within10_pct=100*sum(r['within10'] for r in rr)/(21*len(rr)),within_half_palm_pct=100*sum(r['within_half_palm'] for r in rr)/(21*len(rr)))
valid=[r for r in separate if not r['ambiguous']]
summary=dict(png_identical=1964,coverage=dict(coverage),coordinates=stats(rows),holistic_pairing=dict(collections.Counter(r['preferred'] for r in pairing)),pairing_frames=len(pairing),same_mean_px=float(np.mean([r['same'] for r in pairing])),swapped_mean_px=float(np.mean([r['swap'] for r in pairing])),hands_separate=dict(total=len(separate),unambiguous=len(valid),raw_matches=sum(r['raw']==r['best'] for r in valid),flipped_matches=sum(r['flipped']==r['best'] for r in valid)),per_clip=[dict(id=c['id'],label=c['label'],**stats([r for r in rows if r['id']==c['id']])) for c in refs])
(ROOT/'coordinate_diagnostics.json').write_text(json.dumps(dict(summary=summary,hand_frames=rows,pairing=pairing,separate_handedness=separate),ensure_ascii=False,indent=2),encoding='utf8')
# Review fixed representative clips, selecting the largest measured error in each; explicitly not random samples.
edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
for cid in ['001','005','029']:
    chosen=max([r for r in rows if r['id']==cid],key=lambda r:r['rmse']);i=chosen['frame'];c=next(c for c in refs if c['id']==cid);d=read('detected/'+cid+'.json')['rows'][i]
    image=cv2.imread(str(ROOT/'frames'/cid/f'{i:04d}.png'),cv2.IMREAD_UNCHANGED)
    if image.shape[2]==4:
        a=image[:,:,3:4]/255.;image=np.uint8(image[:,:,:3]*a+255*(1-a))
    for s in ['left','right']:
        g=c['samples'][i]['hands'][s]
        if len(g)==21:
            pts=np.array([g[str(j)] for j in range(21)])
            for a,b in edges:cv2.line(image,tuple(np.rint(pts[a]).astype(int)),tuple(np.rint(pts[b]).astype(int)),(0,170,0),2)
        if d[s]:
            pts=np.array(d[s])[:,:2]*808
            for a,b in edges:cv2.line(image,tuple(np.rint(pts[a]).astype(int)),tuple(np.rint(pts[b]).astype(int)),(200,0,200),1)
    image=cv2.copyMakeBorder(image,60,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    cv2.putText(image,f"{c['label']} frame {i}; worst detected-hand RMSE {chosen['rmse']:.1f}px",(12,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),1)
    cv2.putText(image,'Green: renderer reference / Magenta: Holistic',(12,49),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),1)
    cv2.imwrite(str(ROOT/f'coordinate_{cid}.png'),image)
lines=['# ตรวจความตรงพิกัดและซ้าย–ขวา: 30 คำ','',
'## สรุปที่อ่านง่าย',
f"- มือแบบละเอียดที่มีจุดอ้างอิง: {coverage['reference_hands']} หน่วยมือ×เฟรม; Holistic พบข้างที่ตรงชื่อ {len(rows)} ({100*len(rows)/coverage['reference_hands']:.2f}%) และไม่พบ {coverage['missing_on_reference']} ({100*coverage['missing_on_reference']/coverage['reference_hands']:.2f}%) ไม่ใช่จำนวนเฟรมภาพ เพราะหนึ่งเฟรมมีได้สองมือ",
f"- เฉพาะมือที่พบและเทียบได้: ระยะผิดเฉลี่ย {summary['coordinates']['mean_error_px']:.2f}px, RMSEรวม {summary['coordinates']['rmse_px']:.2f}px, จุดที่ห่างไม่เกิน10px {summary['coordinates']['within10_pct']:.2f}% บนภาพ808×808",
'- Holistic คู่เดิมใกล้กว่า123/128เฟรม, คู่สลับใกล้กว่า1/128, กำกวม4/128 จึงไม่มีหลักฐานสนับสนุนให้สลับซ้ายขวาทั้งระบบ; เฟรมcommit28ที่คู่สลับดีกว่ามีระยะผิดสูงทั้งสองแบบ (36.12เทียบ31.14px) ยังยืนยันการสลับจริงไม่ได้',
f"- Hands แยก ใน{len(valid)}ตัวอย่างที่ระยะไม่กำกวม: กลับชื่อแบบnon-selfieตรงข้างใกล้กว่า{summary['hands_separate']['flipped_matches']} ({100*summary['hands_separate']['flipped_matches']/len(valid):.2f}%) ส่วนชื่อดิบตรง{summary['hands_separate']['raw_matches']} ({100*summary['hands_separate']['raw_matches']/len(valid):.2f}%) ความใกล้ไม่ใช่ground-truthของdetector แต่ผลนี้ไม่รองรับการเชื่อชื่อข้างอย่างเดียว",
'- ตัวเลขcoverageเดิมที่พบมือมากรวมมือSVGที่อยู่ท่าพักด้วย จึงไม่เท่ากับการตรวจจับมือแบบละเอียดระหว่างทำท่า การตรวจครั้งนี้แยกสองกรณีออกจากกันแล้ว',
'','## แนวทางต่อไป',
'ยังไม่แก้ระบบหลักหรือสลับจุดทั้งระบบ ควรทดลองแยกการตรวจมือในบริเวณข้อมือที่หาได้จากภาพลำตัวและติดตามข้ามเฟรม โดยใช้กฎเดียวทุกคำ พร้อมวัดว่าพิกัดตรงขึ้นและมือที่หายลดลงจริงก่อนทดสอบคะแนนโมเดลอีกครั้ง ต้องไม่ใช้referenceRendererเลือกคำตอบหรือเลือกข้างตอนประเมินวิธีอ่านภาพจริง',
'การตรวจครั้งนี้ยืนยันเฉพาะพิกัดมือในภาพ2มิติและชื่อข้าง ยังไม่ได้วัดความตรงของลำตัวทั้งหมดหรือมุม3มิติ ความต่างโดเมนภาพการ์ตูนกับภาพมนุษย์ยังเป็นสมมติฐาน ไม่ใช่สาเหตุที่พิสูจน์แยกแล้ว',
'','## วิธีตรวจ',
'ใช้ภาพเดิม1964เฟรม จับ Renderer ซ้ำแล้วตรวจ hash ภาพตรงครบ1964/1964 พิกัดอ้างอิงอ่านจากปลายเส้นมือที่วาดจริงและแปลงเป็น Canvas pixel ด้วย camera/view transform ไม่ใช่พิกัด payload ดิบ ไม่เปลี่ยน production และไม่รันคะแนนรู้จำใหม่',
'','## ขอบเขต','เทียบ x/y มือครบ21จุดเฉพาะเฟรมที่มี detailed hand อ้างอิงและ detector พบมือข้างนั้น ไม่ให้มือที่หายกลายเป็น error0 ไม่รวม native SVG hand ซึ่งไม่มี reference21จุด จึงไม่ใช่ความแม่นยำทุกมือทุกเฟรม และไม่ได้ยืนยัน ASL หรือความถูกต้องของ Renderer เอง',
'','## ผลรวม','```json',json.dumps(summary,ensure_ascii=False,indent=2),'```','',
'## นิยาม','- RMSE: รากค่าเฉลี่ยระยะคลาดเคลื่อนกำลังสองของจุด x/y หน่วยพิกเซลบนภาพ808×808',
'- shape RMSE: หักตำแหน่งข้อมือของแต่ละชุดก่อนเทียบ เหลือความต่างรูปมือ โดยไม่หมุนหรือปรับขนาดให้ตรงกัน',
'- within10: สัดส่วนจุดที่ห่างไม่เกิน10px; within_half_palm: ไม่เกินครึ่งระยะ wrist→middle MCP ของ Renderer เป็นเกณฑ์ตรวจทางวิศวกรรม ไม่ใช่เกณฑ์ความถูกต้อง ASL',
'- ตรวจซ้ายขวา: เทียบระยะของคู่ตามชื่อกับคู่สลับชื่อ ในเฟรมที่มี referenceและHolisticครบสองมือ หากต่างไม่เกิน4px จัดเป็นกำกวม เกณฑ์นี้กำหนดก่อนดูผล',
'- Hands แยก: ใช้ referenceทั้งสองมือเลือกข้างที่ใกล้กว่า แล้วตรวจ labelดิบเทียบการกลับlabel เป็น diagnostic ไม่ใช่การป้อนreferenceเข้าโมเดล',
'','## ภาพตรวจ','coordinate_001.png (hello), coordinate_005.png (please), coordinate_029.png (hard): สีเขียวพิกัดRenderer สีม่วงHolistic เลือกเฟรมที่มีRMSEสูงสุดของแต่ละคำเพื่อเห็นปัญหา ไม่ใช่ตัวอย่างสุ่มหรือค่าเฉลี่ย','',
'รายละเอียดรายเฟรมอยู่ใน coordinate_diagnostics.json; ผลนี้เป็นการวินิจฉัย ไม่แก้พิกัดหรือเปลี่ยนผลทำนายเดิม']
(ROOT/'COORDINATE_AUDIT_TH.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps({k:v for k,v in summary.items() if k!='per_clip'},indent=2))
