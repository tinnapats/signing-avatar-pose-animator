import sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np,torch
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');O=B/'hand_mask_coordinates';O.mkdir(exist_ok=True)
sys.path.insert(0,str(B.parent));import run_audit as audit
read=lambda p:json.loads(p.read_text(encoding='utf8'))
protocol=read(B/'protocol.json');refs=read(B/'mixed_reference.json');prior=read(B/'mixed_body_hands/results.json')['predictions']
names={'I':'มือจากภาพทั้งหมด','I_common':'มือจากภาพ เฉพาะช่วงที่มีทั้งสองแหล่ง','R_common':'มือRenderer เฉพาะช่วงเดียวกับ I_common','R_all':'มือRenderer ทุกช่วงที่มี','replace_common':'แทนพิกัดด้วยRendererเฉพาะช่วงร่วม เก็บมือภาพช่วงอื่น','fill_missing':'เติมRendererเฉพาะช่วงภาพไม่พบมือ เก็บพิกัดภาพเดิม'}
frozen=dict(variants=names,body_strata=['image','payload'],scope='Diagnostic using Renderer mask and coordinates, not deployable pixels-only accuracy. Mask means complete detailed hand21 reference exists, not semantic active-sign ground truth. No per-word tuning.',source_protocol_sha256=hashlib.sha256((B/'protocol.json').read_bytes()).hexdigest())
if (O/'results.json').exists():raise RuntimeError('Do not overwrite completed experiment')
(O/'protocol.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf8')
model,norm,vocab=audit.load_model();rows=[];counts=[]
for c in refs:
 case=read(B.parent/'slgcn200_20261002/cases'/(c['id']+'.json'));body=audit.frame_array([case['payload']['frames'][i] for i in case['frameIndices']]);pixel=np.load(B/'arrays'/(c['id']+'.npz'))['primary'];image=np.zeros_like(pixel);render=np.zeros_like(pixel)
 for i,r in enumerate(c['samples']):
  assert r['pngMatches'];origin,ex,ey=np.array(r['affine'])*r['pixelRatio'];inv=np.linalg.inv(np.stack([ex-origin,ey-origin],axis=1));valid=np.any(pixel[i]!=0,axis=1);image[i,valid]=(pixel[i,valid]-origin)@inv.T
  for side,offset in [('left',33),('right',54)]:
   assert len(r['hands'][side]) in [0,21]
   for k,p in r['hands'][side].items():render[i,offset+int(k)]=(np.array(p)-origin)@inv.T
 hands={n:np.zeros_like(image[:,33:]) for n in names};hands['I']=image[:,33:].copy();hands['R_all']=render[:,33:].copy();hands['replace_common']=hands['I'].copy();hands['fill_missing']=hands['I'].copy()
 count=dict(label=c['label'],both=0,image_only=0,renderer_only=0,neither=0)
 for offset in [0,21]:
  sl=slice(offset,offset+21);im=np.any(image[:,33+offset:54+offset]!=0,axis=(1,2));rm=np.any(render[:,33+offset:54+offset]!=0,axis=(1,2));common=im&rm
  count['both']+=int(common.sum());count['image_only']+=int((im&~rm).sum());count['renderer_only']+=int((~im&rm).sum());count['neither']+=int((~im&~rm).sum())
  hands['I_common'][common,sl]=hands['I'][common,sl];hands['R_common'][common,sl]=hands['R_all'][common,sl];hands['replace_common'][common,sl]=hands['R_all'][common,sl];hands['fill_missing'][~im&rm,sl]=hands['R_all'][~im&rm,sl]
 counts.append(count)
 for bs,bd in [('image',image),('payload',body)]:
  for v,h in hands.items():
   a=bd.copy();a[:,33:]=h;x=norm({'frames':torch.from_numpy(a[:,audit.PICK,:].transpose(2,0,1).copy())})['frames'].unsqueeze(0);assert torch.isfinite(x).all()
   with torch.inference_mode():p=model(x)[0].softmax(0)
   top=p.topk(5);r=dict(label=c['label'],body=bs,variant=v,prediction=vocab[int(top.indices[0])],top5=[vocab[int(k)] for k in top.indices],target_probability=float(p[vocab.index(c['label'])]));rows.append(r)
   if v in ['I','R_all']:
    oldv={('image','I'):'A',('image','R_all'):'B',('payload','I'):'C',('payload','R_all'):'D'}[bs,v];assert r['prediction']==next(z['prediction'] for z in prior if z['label']==c['label'] and z['variant']==oldv)
 print('scored',c['label'],flush=True)
summary={};pairs={}
for bs in ['image','payload']:
 for v in names:
  rr=[r for r in rows if r['body']==bs and r['variant']==v];ps=[];rs=[];fs=[]
  for l in protocol['labels']:
   tp=sum(r['label']==l and r['prediction']==l for r in rr);fp=sum(r['label']!=l and r['prediction']==l for r in rr);p=tp/(tp+fp) if tp+fp else 0;ps.append(p);rs.append(tp);fs.append(2*p*tp/(p+tp) if p+tp else 0)
  summary[bs+'/'+v]=dict(correct=sum(r['label']==r['prediction'] for r in rr),precision=100*np.mean(ps),recall=100*np.mean(rs),f1=100*np.mean(fs),top5=100*np.mean([r['label'] in r['top5'] for r in rr]))
 for a,b in [('I','I_common'),('I_common','R_common'),('R_common','R_all'),('I','replace_common'),('I','fill_missing')]:
  by={(r['label'],r['variant']):r['label']==r['prediction'] for r in rows if r['body']==bs};pairs[bs+'/'+a+'→'+b]=dict(gained=[l for l in protocol['labels'] if not by[l,a] and by[l,b]],lost=[l for l in protocol['labels'] if by[l,a] and not by[l,b]])
for f,h in protocol['production'].items():assert hashlib.sha256((B.parent.parent/f).read_bytes()).hexdigest()==h
total={k:sum(c[k] for c in counts) for k in ['both','image_only','renderer_only','neither']};result=dict(summary=summary,counts=total,per_clip_counts=counts,paired=pairs,predictions=rows,production_unchanged=True)
(O/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
lines=['# แยกพิกัดมือกับช่วงข้อมูลมือ: 30 คำ','',frozen['scope'],'','ใช้ภาพเดิม1964เฟรมที่30FPS โมเดลST-GCNเดิม และย้อนกล้องเหมือนรอบก่อน ทำซ้ำสองชุดลำตัว แต่ภายในแต่ละชุดตรึงลำตัวเหมือนกันทุกวิธี คะแนนอ้างอิงI/R_allตรงรอบก่อนครบ30คำทั้งสองชุด ไม่เปลี่ยนproduction','', '## นิยาม mask','นับมือ×เฟรม ไม่ใช่เฟรมภาพ: มือภาพคือHolisticพบ21จุด; มือRendererคือมีปลายเส้นมือแบบละเอียด21จุด การไม่มีมือRendererไม่ได้แปลว่าภาพไม่มีมือหรือไม่ใช่ช่วงทำท่า เพราะยังอาจมีSVG fallback จึงไม่เรียกmaskนี้ว่าความจริงของช่วงภาษามือ','',str(total),'','## คะแนน','| ลำตัว | วิธีมือ | ถูก/30 | Macro-P | Macro-R | Macro-F1 |','|---|---|---:|---:|---:|---:|']
for key,s in summary.items():bs,v=key.split('/');lines.append(f"| {bs} | {names[v]} | {s['correct']}/30 | {s['precision']:.2f}% | {s['recall']:.2f}% | {s['f1']:.2f}% |")
lines+=['','## การอ่านผล','I→I_common เปลี่ยนแค่ลบข้อมูลมือภาพที่ไม่มีreferenceคู่กัน; I_common→R_common เปลี่ยนพิกัดบนmaskเดียวกัน; R_common→R_all เติมมือRendererในช่วงที่ภาพไม่พบ; คะแนนผลต่างเหล่านี้เป็นแบบมีเงื่อนไข ไม่ใช่ผลบวกอิสระ','', 'replace_common และ fill_missing ใช้ตรวจว่าการแก้ส่วนเดียวโดยเก็บข้อมูลภาพอื่นไว้มีผลอย่างไร ทุกวิธีที่ใช้RendererหรือmaskจากRendererเป็นoracle diagnostic ไม่ใช่ผลอ่านภาพล้วน','', '## รายคำที่ดีขึ้น/แย่ลง','```json',json.dumps(pairs,ensure_ascii=False,indent=2),'```','','ผลรายคำทุกวิธีอยู่results.json เป็นชุดพัฒนาเดิม1คลิปต่อคำ ยังไม่ยืนยันASLหรือheld-out']
(O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(dict(summary=summary,counts=total),indent=2))
