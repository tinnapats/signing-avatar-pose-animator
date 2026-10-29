
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np,torch
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');O=B/'mixed_body_hands';O.mkdir(exist_ok=True)
sys.path.insert(0,str(B.parent));import run_audit as audit
read=lambda p:_local_json_loads(p.read_text(encoding='utf8'))
protocol=read(B/'protocol.json');refs=read(B/'mixed_reference.json');oldcaps={c['phrase']:c for c in read(B.parent/'slgcn200_20261002/renderer/rendered.json')};oldresults=read(B/'results.json')['predictions']
model,norm,vocab=audit.load_model();rows=[];checks=[]
names={'A':'ลำตัวจากภาพ + มือจากภาพ','B':'ลำตัวจากภาพ + มือ Renderer','C':'ลำตัว payload + มือจากภาพ','D':'ลำตัว payload + มือ Renderer','pixel_control':'ภาพล้วนเดิม ไม่ย้อนกล้อง'}
for c in refs:
 case=read(B.parent/'slgcn200_20261002/cases'/(c['id']+'.json'));frames=[case['payload']['frames'][i] for i in case['frameIndices']];body=audit.frame_array(frames);pixel=np.load(B/'arrays'/(c['id']+'.npz'))['primary'];converted=np.zeros_like(pixel);render=np.zeros_like(pixel)
 assert len(c['samples'])==len(pixel);maxerror=0
 for i,r in enumerate(c['samples']):
  assert r['pngMatches'];origin,ex,ey=np.array(r['affine'])*r['pixelRatio'];matrix=np.stack([ex-origin,ey-origin],axis=1);inv=np.linalg.inv(matrix)
  mask=np.any(pixel[i]!=0,axis=1);converted[i,mask]=(pixel[i,mask]-origin)@inv.T
  for side,offset in [('left',33),('right',54)]:
   archived=oldcaps[case['phrase']]['samples'][i]['hands'][side];assert set(r['hands'][side])==set(archived)
   for k,p in r['hands'][side].items():
    q=(np.array(p)-origin)@inv.T;render[i,offset+int(k)]=q;maxerror=max(maxerror,float(np.linalg.norm(q-archived[k])))
 assert maxerror<.001,(c['label'],maxerror)
 checks.append(dict(label=c['label'],max_renderer_coordinate_difference=maxerror))
 variants={'A':converted.copy(),'B':converted.copy(),'C':body.copy(),'D':body.copy(),'pixel_control':pixel.copy()}
 variants['B'][:,33:]=render[:,33:];variants['C'][:,33:]=converted[:,33:];variants['D'][:,33:]=render[:,33:]
 for variant,a in variants.items():
  x=norm({'frames':torch.from_numpy(a[:,audit.PICK,:].transpose(2,0,1).copy())})['frames'].unsqueeze(0);assert torch.isfinite(x).all()
  with torch.inference_mode():p=model(x)[0].softmax(0)
  top=p.topk(5);row=dict(label=c['label'],variant=variant,prediction=vocab[int(top.indices[0])],top5=[vocab[int(t)] for t in top.indices],probabilities=top.values.tolist(),target_probability=float(p[vocab.index(c['label'])]));rows.append(row)
  if variant in ['D','pixel_control']:
   old=next(r for r in oldresults if r['label']==c['label'] and r['variant']==('archived_post' if variant=='D' else 'pixels_holistic'));assert row['prediction']==old['prediction'],(c['label'],variant)
 print(c['label'],[(r['variant'],r['prediction']) for r in rows[-5:]],flush=True)
summary={}
for v in names:
 rr=[r for r in rows if r['variant']==v];ps=[];rs=[];fs=[]
 for l in protocol['labels']:
  tp=sum(r['label']==l and r['prediction']==l for r in rr);fp=sum(r['label']!=l and r['prediction']==l for r in rr);p=tp/(tp+fp) if tp+fp else 0;ps.append(p);rs.append(tp);fs.append(2*p*tp/(p+tp) if p+tp else 0)
 summary[v]=dict(correct=sum(r['label']==r['prediction'] for r in rr),accuracy=100*np.mean([r['label']==r['prediction'] for r in rr]),precision=100*np.mean(ps),recall=100*np.mean(rs),f1=100*np.mean(fs),top5=100*np.mean([r['label'] in r['top5'] for r in rr]))
comparisons={}
for a,b in [('A','B'),('A','C'),('B','D'),('C','D'),('pixel_control','A')]:
 by={(r['label'],r['variant']):r['prediction']==r['label'] for r in rows};comparisons[a+'→'+b]=dict(gained=[l for l in protocol['labels'] if not by[l,a] and by[l,b]],lost=[l for l in protocol['labels'] if by[l,a] and not by[l,b]])
for f,h in protocol['production'].items():assert hashlib.sha256((B.parent.parent/f).read_bytes()).hexdigest()==h
result=dict(summary=summary,comparisons=comparisons,predictions=rows,checks=checks,production_unchanged=True,scope='Diagnostic only: actual camera metadata inverse for A-D, not a pure pixels-only detector deployment; zeros retained for missing hands. No training. 30 known development clips. Renderer detailed hand references exclude native SVG hands.');(O/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
lines=['# แยกผลลำตัวและมือ: 30 คำ','',result['scope'],'','ภาพเดิม1964เฟรมที่30FPS โมเดลST-GCNเดิม เลือก27จุดและnormalizationเดิม ย้อนcamera transformรายเฟรมให้ทุกชุด A–Dอยู่ในพิกัดpayloadเดียวกัน เฉพาะจุดที่มีข้อมูล ค่าศูนย์ของจุดหายยังเป็นศูนย์ ไม่ใช้labelช่วยปรับพิกัด','', 'การย้อนกล้องใช้metadataจากระบบ ไม่ใช่การหาcameraจากภาพ Aจึงเป็นชุดวินิจฉัยที่ใช้จุดจากภาพพร้อมmetadata ไม่ใช่pixel-onlyอย่างเคร่งครัด ชุดpixel_controlคงภาพล้วนเดิม','', '| แบบ | ข้อมูล | ถูก/30 | Accuracy | Macro-P | Macro-R | Macro-F1 | Top5 |','|---|---|---:|---:|---:|---:|---:|---:|']
for v,s in summary.items():lines.append(f"| {v} | {names[v]} | {s['correct']}/30 | {s['accuracy']:.2f}% | {s['precision']:.2f}% | {s['recall']:.2f}% | {s['f1']:.2f}% | {s['top5']:.2f}% |")
lines+=['','ตรวจสอบ:ภาพจับซ้ำตรงเดิมครบ1964เฟรม พิกัดมือย้อนกล้องตรงอ้างอิงเดิมภายใน0.001px ชุดDและpixel_controlทายTop1ตรงผลเดิมครบ30คำ ไม่เปลี่ยนproduction','', '## ผลรายคำ','| คำ | A | B | C | D | ภาพล้วนเดิม |','|---|---|---|---|---|---|']
for l in protocol['labels']:lines.append('| '+l+' | '+' | '.join(next(r['prediction'] for r in rows if r['label']==l and r['variant']==v) for v in names)+' |')
lines+=['','## เปลี่ยนส่วนใดแล้วคำใดดีขึ้น/แย่ลง','```json',json.dumps(comparisons,ensure_ascii=False,indent=2),'```','','การเปลี่ยนไปใช้มือRendererเปลี่ยนทั้งค่าพิกัดและรูปแบบข้อมูลหาย และไม่รวมมือSVGท่าพัก จึงยังแยกผลของความแม่นยำตำแหน่งกับmaskไม่ได้ ผลไม่ใช่ความเข้าใจASLของมนุษย์และไม่ใช่held-out ไม่ควรเอาคะแนนแบบผสมไปอ้างว่าอ่านภาพAnimationได้เท่านั้น']
(O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(summary,indent=2))
