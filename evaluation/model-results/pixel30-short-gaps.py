"""Reuse verified evaluation, change only frozen gap completion rule and abstention policy."""
from pathlib import Path
source=Path(__file__).with_name('pixel30-impute.py').read_text(encoding='utf8')
def replace(a,b):
 global source
 assert source.count(a)==1,a
 source=source.replace(a,b)
replace("O=B/'missing_imputation_20261008'","O=B/'short_gaps_20261008'")
replace("names={'zero_baseline':'ค่าศูนย์เดิม','nearest':'ใช้ท่าที่พบใกล้ที่สุด','linear':'แทรกพิกัดระหว่างสองท่า','bone_angles':'แทรกทิศกระดูกและความยาว'}","names={'zero_baseline':'ไม่เติม','linear3':'เติมช่องว่างไม่เกิน3เฟรม','linear5':'เติมช่องว่างไม่เกิน5เฟรม','linear10':'เติมช่องว่างไม่เกิน10เฟรม'}")
replace("Edges use nearest anchor. Segments without anchors remain explicitly unresolved. No maximum gap for exploratory comparison; record distance to nearest evidence. Offline uses future frames. Bone interpolation shortest angle in2D, linear lengths, wrist linear.","Only bounded internal gaps with both original anchors within the same non-rest segment; max missing length3/5/10 respectively. No edge extrapolation or holding. Unresolved stays explicit. Strict policy abstains if any unresolved hand-frame remains. Diagnostic model calls still use missing zeros and must not be called no-zero results. Offline uses future frames. Linear interpolation only.")
start=source.index("     if variant=='nearest' or l==r:")
end=source.index('     state[i,side]=3',start)
source=source[:start]+"""     limit=int(variant.removeprefix('linear'))
     if not l<i<r or r-l-1>limit:continue
     t=(i-l)/(r-l);data[i,off:off+21]=(1-t)*base[l,off:off+21]+t*base[r,off:off+21]
"""+source[end:]
replace("top5=[vocab[int(t)] for t in top.indices]);rows.append(row)","top5=[vocab[int(t)] for t in top.indices],unresolved_hand_frames=int((state==0).sum()),strict_status='insufficient_data' if (state==0).any() else 'ready');rows.append(row)")
replace("for f,h in read(B/'protocol.json')['production'].items():", """for v in names:
 rr=[r for r in rows if r['variant']==v];ready=[r for r in rr if r['strict_status']=='ready']
 summary[v]['strict_ready_clips']=len(ready)
 summary[v]['strict_abstained_clips']=30-len(ready)
 summary[v]['strict_correct_total30']=sum(r['label']==r['prediction'] for r in ready)
 summary[v]['strict_accuracy_accepted']=100*sum(r['label']==r['prediction'] for r in ready)/len(ready) if ready else None
for f,h in read(B/'protocol.json')['production'].items():""")
source=source[:source.index("lines=['# ทดลองแทนค่าศูนย์")]+'''
lines=['# เติมมือเฉพาะช่องว่างสั้น และงดทายเมื่อข้อมูลไม่พอ','',
'8 ตุลาคม2026 — 30คำเดิม1964เฟรม30FPS ใช้ฐานrest_roi3 ตรึงลำตัว กฎคัดมือพัก และโมเดลเหมือนรอบก่อน ใช้จุดRendererเพื่อวัดผลเท่านั้น camera metadataใช้จัดระบบพิกัดเหมือนรอบก่อน',
'','## กฎที่ตรึงก่อนทดลอง',
'เติมlinearเฉพาะช่องว่างภายในไม่เกิน3,5,10เฟรม มีจุดมือทั้งก่อนและหลังในช่วงเดียวกัน ห้ามข้ามช่วงคัดมือพัก ห้ามเติมขอบคลิปด้วยท่าค้าง ทุกจุดเดิมคงเดิมและมีstateกำกับ',
'การใช้งานแบบstrict: หากยังมีstate0แม้หนึ่งหน่วยมือ×เฟรมให้แสดงข้อมูลไม่พอและงดคำทำนาย ส่วนตารางdiagnosticด้านล่างยังทดสอบโมเดลด้วยศูนย์ที่เหลือ เพื่อวัดผลการเติมสั้นเท่านั้น ไม่ใช่คะแนนระบบไร้ศูนย์',
'','## ผลdiagnostic (ยังมีmissing zero)','| วิธี | ถูก/30 | Macro-F1 | เติมมือ×เฟรม | ยังขาดมือ×เฟรม | RMSEจุดเติม(px) |','|---|---:|---:|---:|---:|---:|']
for v,s in summary.items():
 error=f"{s['estimated_rmse']:.2f}" if s['estimated_rmse'] is not None else '—'
 lines.append(f"| {names[v]} | {s['correct']}/30 | {s['f1']:.2f}% | {s['estimated']} | {s['unresolved']} | {error} |")
lines+=['','RMSEจุดเติมแต่ละเงื่อนไขใช้จุดคนละชุด ไม่ควรสรุปว่าเงื่อนไขหนึ่งแม่นกว่าเพียงดูตัวเลขนี้','', '## ผลเมื่อไม่ยอมให้missingเข้าทาย','| วิธี | พร้อมทาย/30 | งดทาย/30 | ทายถูกจากทั้งหมด30 |','|---|---:|---:|---:|']
for v,s in summary.items():lines.append(f"| {names[v]} | {s['strict_ready_clips']} | {s['strict_abstained_clips']} | {s['strict_correct_total30']} |")
lines+=['','state2ของมือที่กฎคัดออกยังเป็นmaskศูนย์ที่ตั้งใจไว้; strictตรวจเฉพาะstate0ที่ขาดหลักฐาน ไม่ได้อ้างว่าทุกช่องของtensorเป็นค่าที่ตรวจพบจริง เกณฑ์strictอาจปฏิเสธคลิปที่ใช้มือเดียว จึงเป็นการทดสอบขอบเขตนโยบาย ไม่ใช่เกณฑ์คุณภาพASLที่ยืนยันแล้ว',
'','เป็นการทดลองofflineที่ใช้เฟรมอนาคตบนชุดพัฒนาเดิม30คำ ยังไม่ใช่held-out ไม่เปลี่ยนproduction ผลรายคำและstatusอยู่results.json ทุกnpzเก็บstateและพิกัดก่อนnormalization']
(O/'REPORT_TH.md').write_text('\\n'.join(lines),encoding='utf8')
print(json.dumps(summary,indent=2))
'''
exec(compile(source,str(Path(__file__)), 'exec'))
