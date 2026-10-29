import sys,json,hashlib,time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit');sys.path.insert(0,str(A))
import run_audit as audit
B=A/'pixels30_20261007';H=A/'human30_preprocess_20261008';O=B/'complexity2_20261008'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def metrics(rows):
 labels=[r['label'] for r in rows];ps=[];rs=[];fs=[]
 for l in labels:
  tp=sum(r['label']==l and r['prediction']==l for r in rows);fp=sum(r['label']!=l and r['prediction']==l for r in rows);p=tp/(tp+fp) if tp+fp else 0;r=tp;ps.append(p);rs.append(r);fs.append(2*p*r/(p+r) if p+r else 0)
 return dict(correct=sum(rs),n=len(rows),accuracy=100*np.mean(rs),precision=100*np.mean(ps),recall=100*np.mean(rs),f1=100*np.mean(fs),top5=100*np.mean([r['label'] in r['top5'] for r in rows]))
def tensor(a,norm):return norm({'frames':torch.from_numpy(a[:,audit.PICK,:].transpose(2,0,1).copy())})['frames']
def result(p,label,vocab):
 p=p.softmax(0);t=p.topk(5);return dict(label=label,prediction=vocab[int(t.indices[0])],top5=[vocab[int(v)] for v in t.indices],target_probability=float(p[vocab.index(label)]))
def human():
 out=H/'padding_diagnostic';out.mkdir(exist_ok=True);assert not (out/'results.json').exists()
 protocol=read(H/'protocol.json');model,norm,vocab=audit.load_model();xs=[];labels=[]
 for c in protocol['cases']:
  p=H/(c['video_id']+'.npz');assert sha(p)==read(H/(c['video_id']+'.json'))['array_sha256'];xs.append(tensor(np.load(p)['c2'],norm));labels.append(c['label'])
 maxT=max(x.shape[1] for x in xs);rows={};logits={}
 # Predefined diagnostic controls, not an accuracy-tuned production change.
 for mode in ['single','zero_pad_to_batch_max','repeat_last_to_batch_max']:
  rr=[];ll=[]
  for label,x in zip(labels,xs):
   if mode=='zero_pad_to_batch_max':x=torch.nn.functional.pad(x,(0,0,0,maxT-x.shape[1]))
   if mode=='repeat_last_to_batch_max':x=torch.cat([x,x[:,-1:,:].expand(-1,maxT-x.shape[1],-1)],dim=1)
   with torch.inference_mode():p=model(x.unsqueeze(0))[0]
   rr.append(result(p,label,vocab));ll.append(p)
  rows[mode]=rr;logits[mode]=torch.stack(ll);print(mode,metrics(rr),flush=True)
 batch=torch.stack([torch.nn.functional.pad(x,(0,0,0,maxT-x.shape[1])) for x in xs])
 with torch.inference_mode():bp=model(batch)
 maxdiff=float((bp-logits['zero_pad_to_batch_max']).abs().max());same=int((bp.argmax(1)==logits['zero_pad_to_batch_max'].argmax(1)).sum())
 old={r['label']:r['prediction'] for r in read(H/'results.json')['predictions'] if r['variant']=='c2_native'}
 assert all(old[r['label']]==r['prediction'] for r in rows['single'])
 extraction=read(H/'extraction.json');diagnostics=[]
 for m in extraction:
  s=m['stats']['c2'];r=next(r for r in rows['single'] if r['label']==m['label']);diagnostics.append(dict(label=m['label'],correct=r['prediction']==m['label'],prediction=r['prediction'],frames=m['frames'],hand_coverage=100*(s['left_frames']+s['right_frames'])/(2*m['frames']),no_hand_pct=100*s['no_hand_frames']/m['frames'],longest_missing_seconds=max(s['longest_missing_left'],s['longest_missing_right'])/m['fps']))
 groups={}
 for correct in [True,False]:
  ds=[r for r in diagnostics if r['correct']==correct];groups[str(correct)]={k:float(np.mean([r[k] for r in ds])) for k in ['hand_coverage','no_hand_pct','longest_missing_seconds','frames']};groups[str(correct)]['n']=len(ds)
 paired={}
 for mode in list(rows)[1:]:
  base={r['label']:r['prediction']==r['label'] for r in rows['single']};paired[mode]=dict(gained=[r['label'] for r in rows[mode] if not base[r['label']] and r['prediction']==r['label']],lost=[r['label'] for r in rows[mode] if base[r['label']] and r['prediction']!=r['label']])
 save(out/'results.json',dict(summary={k:metrics(v) for k,v in rows.items()},predictions=rows,paired=paired,groups=groups,clips=diagnostics,max_frames=maxT,original_lengths=[x.shape[1] for x in xs],batch_check=dict(max_logit_difference=maxdiff,top1_equal=same),protocol_sha256=sha(H/'protocol.json'),limitations='Batch30 padding diagnostic, not original batch64 membership. Repeat-last control is not upstream preprocessing. No changes to source videos/timing. Missing hand detection is not ground truth absence/error.'))
 print(json.dumps(dict(groups=groups,paired=paired,batch_check=[maxdiff,same]),ensure_ascii=True),flush=True)
def gaps():
 out=H/'padding_diagnostic';assert not (out/'gaps.json').exists();model,norm,vocab=audit.load_model();rows={'original':[],'internal_linear5':[]};filled=0;missing=0
 for c in read(H/'protocol.json')['cases']:
  a=np.load(H/(c['video_id']+'.npz'))['c2'];b=a.copy()
  for off in [33,54]:
   valid=np.any(a[:,off:off+21]!=0,axis=(1,2));missing+=int((~valid).sum());i=0
   while i<len(a):
    if valid[i]:i+=1;continue
    start=i
    while i<len(a) and not valid[i]:i+=1
    if start>0 and i<len(a) and i-start<=5:
     for j in range(start,i):
      t=(j-start+1)/(i-start+1);b[j,off:off+21]=(1-t)*a[start-1,off:off+21]+t*a[i,off:off+21];filled+=1
  for mode,x in [('original',a),('internal_linear5',b)]:
   with torch.inference_mode():p=model(tensor(x,norm).unsqueeze(0))[0]
   rows[mode].append(result(p,c['label'],vocab))
 base={r['label']:r['prediction']==r['label'] for r in rows['original']};rr=rows['internal_linear5'];res=dict(summary={k:metrics(v) for k,v in rows.items()},predictions=rows,filled_hand_frames=filled,original_missing_hand_frames=missing,gained=[r['label'] for r in rr if not base[r['label']] and r['prediction']==r['label']],lost=[r['label'] for r in rr if base[r['label']] and r['prediction']!=r['label']],rules='Only internal gaps <=5 native frames bounded by detections; linear xy; no edge hold, no FPS/duration change. Missing may mean legitimate occlusion/offscreen; no ground truth guarantee.')
 save(out/'gaps.json',res);print(json.dumps({k:v for k,v in res.items() if k!='predictions'}),flush=True)
def avatar():
 assert not (O/'results.json').exists();manifest=read(B/'capture_manifest.json');model,norm,vocab=audit.load_model();refs={r['id']:r for r in read(B/'coordinate_reference.json')};rows={'c1':[],'c2':[]};counts={k:dict(frames=0,pose=0,left=0,right=0) for k in rows};errs={k:[] for k in ['c1_common','c2_common','c2_new']};seconds=0
 for c in manifest:
  f=O/(c['id']+'.npz');meta=read(O/(c['id']+'.json'));assert sha(f)==meta['sha256'];z=np.load(f);a2=z['points'];a1=np.load(B/'arrays'/(c['id']+'.npz'))['primary'];assert a1.shape==a2.shape;assert np.allclose(z['times'],[f['time'] for f in c['frames']]);seconds+=meta['seconds']
  for k,a in [('c1',a1),('c2',a2)]:
   with torch.inference_mode():p=model(tensor(a,norm).unsqueeze(0))[0]
   rows[k].append(result(p,c['label'],vocab));counts[k]['frames']+=len(a)
   for part,sl in [('pose',slice(0,33)),('left',slice(33,54)),('right',slice(54,75))]:counts[k][part]+=int(np.any(a[:,sl]!=0,axis=(1,2)).sum())
  assert len(refs[c['id']]['samples'])==len(a2)
  for i,r in enumerate(refs[c['id']]['samples']):
   assert r['index']==c['frames'][i]['sourceIndex'] and r['pngMatches']
   for hand,off in [('left',33),('right',54)]:
    ref=r['hands'].get(hand,{})
    if len(ref)!=21:continue
    target=np.array([ref[str(j)] for j in range(21)]);p1=a1[i,off:off+21];p2=a2[i,off:off+21];v1=np.any(p1);v2=np.any(p2)
    if v1 and v2:errs['c1_common'].extend(np.linalg.norm(p1-target,axis=1).tolist());errs['c2_common'].extend(np.linalg.norm(p2-target,axis=1).tolist())
    elif v2:errs['c2_new'].extend(np.linalg.norm(p2-target,axis=1).tolist())
  print('scored',c['label'],flush=True)
 for f,h in read(O/'protocol.json')['production'].items():assert sha(A.parent/f)==h
 geometry={k:dict(points=len(v),rmse_px=float(np.sqrt(np.mean(np.square(v)))) if v else None,mean_px=float(np.mean(v)) if v else None) for k,v in errs.items()}
 summary={k:metrics(v) for k,v in rows.items()};save(O/'results.json',dict(summary=summary,predictions=rows,coverage=counts,geometry=geometry,c2_seconds=seconds,production_unchanged=True,timing_unchanged=True,frames=sum(len(c['frames']) for c in manifest),fps=30))
 print(json.dumps(dict(summary=summary,coverage=counts,geometry=geometry,c2_seconds=seconds)),flush=True)
def report():
 a=read(O/'results.json');h=read(H/'padding_diagnostic/results.json');g=read(H/'padding_diagnostic/gaps.json')
 lines=['# Complexity 2 บนภาพ Avatar และการตรวจวิดีโอมนุษย์','', '8 ตุลาคม 2026 — ทดลองแยกจากระบบหลัก เฉพาะ signing-avatar-github-tinnapat','', '## 1. Avatar: คงเวลาเดิม','ใช้ PNG สุดท้ายเดิม 30 คำ 1,964 เฟรม 30 FPS ตรวจ SHA-256 ทุกภาพและ timestamp ไม่มีเพิ่มเฟรม ยืดเวลา crop หรือเติมมือ เปลี่ยนเฉพาะ Holistic complexity 1 → 2; เลือก 27 จุด เข้า ST-GCN เดิมแบบทีละคลิป ใช้พิกัดจากภาพล้วน ไม่ใช้พิกัด Renderer/payload แทนจุดเข้าโมเดล','', '| วิธี | ถูก/30 | Accuracy | Macro-P | Macro-R | Macro-F1 | Top-5 |','|---|---:|---:|---:|---:|---:|---:|']
 def table(name,s):return f"| {name} | {s['correct']}/30 | {s['accuracy']:.2f}% | {s['precision']:.2f}% | {s['recall']:.2f}% | {s['f1']:.2f}% | {s['top5']:.2f}% |"
 for k,s in a['summary'].items():lines.append(table(k,s))
 lines+=['','### การตรวจพบจุดและความตรงของมือ','| วิธี | พบลำตัว | พบมือซ้าย | พบมือขวา | hand-frame coverage |','|---|---:|---:|---:|---:|']
 for k,c in a['coverage'].items():lines.append(f"| {k} | {c['pose']}/{c['frames']} | {c['left']} | {c['right']} | {100*(c['left']+c['right'])/(2*c['frames']):.2f}% |")
 lines+=['','พิกัด Renderer ใช้เฉพาะตรวจความคลาดเคลื่อน ไม่ป้อนเข้าโมเดล ตาราง common ใช้จุดที่ทั้งสองวิธีตรวจพบและมี Renderer reference ตรงกัน เพื่อไม่ให้จำนวนจุดต่างกันทำให้เปรียบเทียบผิด','```json',json.dumps(a['geometry'],indent=2),'```',f"เวลา complexity 2 รวมอ่าน/ตรวจ hash ภาพและสกัดจุด {a['c2_seconds']:.1f} วินาที ไม่ใช่เวลาเล่นแอนิเมชันหรือ benchmark ตัวโมเดลล้วน",'', '## 2. วิดีโอมนุษย์: ทำไมได้เพียง 10/30','ใช้ชุดมนุษย์เดิมอีก 30 คำ (คนละชุดกับ Avatar ห้ามเทียบสองกลุ่มเป็นความต่างจาก rendering โดยตรง) ความเร็วไฟล์เดิม 25 FPS 1,496 เฟรม ตรวจ video_id/gloss/test split และ hash แล้ว ไม่ใช่ชุดทดสอบใหม่ที่ไม่เคยใช้สำรวจ','', '| วิธี | ถูก/30 | Accuracy | Macro-P | Macro-R | Macro-F1 | Top-5 |','|---|---:|---:|---:|---:|---:|---:|']
 for k,s in h['summary'].items():lines.append(table(k,s))
 lines.append(table('internal_linear5',g['summary']['internal_linear5']))
 lines+=['',f"Padding ใช้ความยาวสูงสุดของ 30 คลิปนี้ = {h['max_frames']} เฟรม เติมศูนย์หลัง normalization ตาม collate ของ upstream ไม่ใช่การเปลี่ยน FPS/วิดีโอจริง และไม่อ้างว่าเทียบเท่า batch 64 ดั้งเดิมเพราะสมาชิกใน batch ต่างกัน",f"ตรวจ batch inference เทียบการป้อนทีละคลิปที่ pad เหมือนกัน: Top-1 ตรง {h['batch_check']['top1_equal']}/30; max logit difference {h['batch_check']['max_logit_difference']:.8f}",'','ผลแบบจับคู่:','```json',json.dumps(h['paired'],ensure_ascii=False,indent=2),'```',f"เติมช่องว่างภายในไม่เกิน 5 เฟรม โดยมีมือที่ตรวจพบทั้งสองปลาย เติม {g['filled_hand_frames']} จาก {g['original_missing_hand_frames']} hand-frames ที่ไม่มีจุด; ได้คืน {g['gained']} แต่เสีย {g['lost']} จึงไม่สนับสนุนให้เติมอัตโนมัติทุกกรณี",'', '### ตรวจพบมือในกลุ่มทายถูก/ผิด','```json',json.dumps(h['groups'],indent=2),'```','กลุ่มถูก/ผิดมี hand coverage ใกล้กัน จึงยังโยนสาเหตุให้จำนวนมือหายอย่างเดียวไม่ได้ การตรวจพบไม่ได้รับรองว่าพิกัดถูก มือไม่ปรากฏอาจเป็นท่ามือเดียว/บัง/ออกนอกภาพ ไม่ใช่ detector error เสมอไป','', '## ข้อสรุปและข้อจำกัด','- ความไวต่อ FPS ยืนยันจากรอบก่อน: complexity 2 native 10/30 เทียบ resample 30 FPS 5/30 แต่ไม่เปลี่ยนเวลา Avatar เพื่อไล่คะแนน','- Padding มีผลต่อคำทายแต่ยังแก้คะแนนต่ำไม่ได้ทั้งหมด; repeat-last ทำให้แย่ลง','- ยังพิสูจน์สาเหตุหลักของมนุษย์ 10/30 ไม่ได้ ต้องเทียบกับ pose ที่ใช้ประเมิน checkpoint เดิมของ video_id เดียวกัน จึงจะแยก extractor mismatch ออกจากขีดความสามารถของ checkpoint ได้ชัด','- ช่องทางอื่นที่ยังไม่ยืนยัน: คุณภาพวิดีโอ resized/ขอบเขตช่วงท่า/landmark error และความแตกต่างเวอร์ชัน extractor; ห้ามสรุปว่าเป็นสาเหตุแล้ว','- ขั้นถัดไปควรหา official reference pose ของรหัสคลิปเดียวกันและรัน preprocessing/validation ที่ตรง checkpoint ก่อนเพิ่มเทคนิคกู้มืออีกหลายชนิด','- Macro-P/R/F1 เฉลี่ย 30 true classes, zero_division=0; โมเดลทายได้ 2,000 คลาส รวมคำทายนอกชุด ไม่ใช่การบังคับให้เลือก 30 คำ','- ทั้งหมดเป็นผลสำรวจชุดเล็ก หนึ่งคลิปต่อคำ มีการลองหลายวิธีบนชุดเดิม คะแนนไม่ใช่เปอร์เซ็นต์ความเหมือนภาพหรือความเข้าใจ ASL','- ตรวจ hash 4 ไฟล์ production ตรงเดิม ไม่แก้ Renderer/แอนิเมชัน/โมเดลหลัก','', '## แหล่งตรวจ upstream','- https://raw.githubusercontent.com/AI4Bharat/OpenHands/main/openhands/datasets/pipelines/generate_pose.py','- โมดูล upstream_base.py และ checkpoint config ใน recognition_audit ใช้ตรวจ collate/normalization; live upstream ไม่รับรองว่าเป็นเวอร์ชันสร้าง pose ของ checkpoint เดิม','', '## คำทาย Avatar รายคำ','| คำจริง | complexity 1 | complexity 2 |','|---|---|---|']
 for r1,r2 in zip(a['predictions']['c1'],a['predictions']['c2']):lines.append(f"| {r1['label']} | {r1['prediction']} | {r2['prediction']} |")
 (O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf8')
 print('REPORT_TH.md saved',flush=True)
if __name__=='__main__':{'human':human,'avatar':avatar,'gaps':gaps,'report':report}[sys.argv[1]]()
