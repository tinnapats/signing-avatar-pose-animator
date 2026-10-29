import sys,json,hashlib,zipfile,random,time,math
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
HERE=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit');OUT=HERE/'human30_preprocess_20261008';sys.path.insert(0,str(HERE))
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def freeze():
 OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
 metadata=json.loads(zipfile.ZipFile(HERE/'wlasl_metadata.zip').read('splits/asl2000.json'));ids={str(i['video_id']):(r['gloss'],i['split']) for r in metadata for i in r['instances']};pool={};sources={}
 for name in ['head_pilot_20261005','heldout_scale_trial','broad_pair_trial','paired_video']:
  f=HERE/name/'manifest.json';sources[str(f)]=sha(f)
  for r in read(f):
   label=r.get('label',r['word']);vid=r['metadata']['video_id'];p=Path(r['local_video'])
   if label in ['plan','please','buy'] or ids.get(vid)!=(label,'test') or not p.is_file():continue
   item=dict(label=label,video_id=vid,path=str(p),sha256=r['sha256'],split='test',source_manifest=str(f),metadata=r['metadata'])
   if label not in pool or int(vid)<int(pool[label]['video_id']):pool[label]=item
 labels=sorted(pool);assert len(labels)>=30;selected=random.Random(20261008).sample(labels,30);cases=[pool[l] for l in selected]
 for c in cases:assert sha(Path(c['path']))==c['sha256']
 prod=read(HERE/'pixels30_20261007/protocol.json')['production']
 for f,h in prod.items():assert sha(HERE.parent/f)==h
 protocol=dict(cases=cases,seed=20261008,eligible_labels=len(labels),selection='One lowest numeric test-video ID per word from four existing manifests; exclude prior three control words plan/please/buy; sorted eligible labels sampled seed20261008; no selection by scores. Existing exploratory data, not pristine held-out.',variants=['c1_native','c1_30fps','c2_native','c2_30fps'],rules='Decode whole existing resized clip, RGB, no mirror or additional trim, Holistic tracking min detection/tracking .5 default smoothing; complexity1/2 separately reset each clip. Native normalized x/y pose33,left21,right21. Missing0 before upstreamnorm. Native sequence vs nearest resampling to30FPS. Same27points, same shoulderclip norm, same ST-GCN, independent single-clip inference. No hand repair or rest gate.',sources=sources,production=prod,checkpoint_sha256=sha(HERE/'wlasl_stgcn.zip'))
 save(OUT/'protocol.json',protocol);print('frozen',len(cases),'of',len(labels));print(selected)
def extract():
 import cv2,mediapipe as mp
 protocol=read(OUT/'protocol.json');progress=[]
 for num,c in enumerate(protocol['cases']):
  target=OUT/(c['video_id']+'.npz');sidecar=OUT/(c['video_id']+'.json')
  if target.exists() and sidecar.exists():assert sha(target)==read(sidecar)['array_sha256'];progress.append(read(sidecar));continue
  assert sha(Path(c['path']))==c['sha256'];cap=cv2.VideoCapture(c['path']);fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
  while True:
   ok,im=cap.read()
   if not ok:break
   frames.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
  cap.release();assert fps>0 and frames;arrays={};stats={}
  for complexity in [1,2]:
   points=[];visibility=[];start=time.perf_counter()
   with mp.solutions.holistic.Holistic(static_image_mode=False,model_complexity=complexity,min_detection_confidence=.5,min_tracking_confidence=.5) as detector:
    for frame in frames:
     d=detector.process(frame);a=np.zeros((75,2),np.float32);vis=np.zeros(33,np.float32)
     for lm,off in [(d.pose_landmarks,0),(d.left_hand_landmarks,33),(d.right_hand_landmarks,54)]:
      if lm:a[off:off+len(lm.landmark)]=[[p.x,p.y] for p in lm.landmark]
     if d.pose_landmarks:vis[:]=[p.visibility for p in d.pose_landmarks.landmark]
     points.append(a);visibility.append(vis)
   a=np.array(points);arrays['c'+str(complexity)]=a;arrays['visibility_c'+str(complexity)]=np.array(visibility);left=np.any(a[:,33:54]!=0,axis=(1,2));right=np.any(a[:,54:]!=0,axis=(1,2))
   def longest(mask):
    run=best=0
    for v in mask:run=run+1 if v else 0;best=max(best,run)
    return best
   stats['c'+str(complexity)]=dict(pose_frames=int(np.any(a[:,:33]!=0,axis=(1,2)).sum()),left_frames=int(left.sum()),right_frames=int(right.sum()),both_frames=int((left&right).sum()),no_hand_frames=int((~left&~right).sum()),longest_missing_left=int(longest(~left)),longest_missing_right=int(longest(~right)),processing_seconds=time.perf_counter()-start)
   print(num+1,'/30',c['label'],'complexity',complexity,'frames',len(frames),flush=True)
  np.savez_compressed(target,**arrays,fps=fps);info=dict(label=c['label'],video_id=c['video_id'],fps=fps,frames=len(frames),width=frames[0].shape[1],height=frames[0].shape[0],stats=stats,mediapipe=mp.__version__,opencv=cv2.__version__,array_sha256=sha(target));save(sidecar,info);progress.append(info);save(OUT/'progress.json',dict(completed=len(progress),clips=progress))
 save(OUT/'extraction.json',progress)
def score():
 import torch,run_audit as audit
 assert not (OUT/'results.json').exists();protocol=read(OUT/'protocol.json');extraction=read(OUT/'extraction.json');assert len(extraction)==30;model,norm,vocab=audit.load_model();rows=[]
 for c in protocol['cases']:
  meta=read(OUT/(c['video_id']+'.json'));file=OUT/(c['video_id']+'.npz');assert sha(file)==meta['array_sha256'];z=np.load(file);n=len(z['c1']);fps=float(z['fps']);idx=np.minimum(np.rint(np.arange(0,n/fps,1/30)*fps).astype(int),n-1)
  for v in protocol['variants']:
   a=z[v[:2]];a=a[idx] if v.endswith('30fps') else a;x=norm({'frames':torch.from_numpy(a[:,audit.PICK,:].transpose(2,0,1).copy())})['frames'].unsqueeze(0);assert torch.isfinite(x).all();start=time.perf_counter()
   with torch.inference_mode():p=model(x)[0].softmax(0)
   top=p.topk(5);rows.append(dict(label=c['label'],video_id=c['video_id'],variant=v,frames=len(a),prediction=vocab[int(top.indices[0])],top5=[vocab[int(t)] for t in top.indices],probabilities=top.values.tolist(),target_probability=float(p[vocab.index(c['label'])]),inference_ms=(time.perf_counter()-start)*1000))
  print('scored',c['label'],flush=True)
 summary={};labels=[c['label'] for c in protocol['cases']]
 for v in protocol['variants']:
  rr=[r for r in rows if r['variant']==v];ps=[];rs=[];fs=[]
  for l in labels:
   tp=sum(r['label']==l and r['prediction']==l for r in rr);fp=sum(r['label']!=l and r['prediction']==l for r in rr);p=tp/(tp+fp) if tp+fp else 0;ps.append(p);rs.append(tp);fs.append(2*p*tp/(p+tp) if p+tp else 0)
  summary[v]=dict(correct=sum(r['label']==r['prediction'] for r in rr),accuracy=100*np.mean(rs),precision=100*np.mean(ps),recall=100*np.mean(rs),f1=100*np.mean(fs),top5=100*np.mean([r['label'] in r['top5'] for r in rr]))
 paired={}
 for a,b in [('c1_native','c2_native'),('c1_30fps','c2_30fps'),('c1_native','c1_30fps'),('c2_native','c2_30fps')]:
  by={(r['label'],r['variant']):r['prediction']==r['label'] for r in rows};g=[l for l in labels if not by[l,a] and by[l,b]];lost=[l for l in labels if by[l,a] and not by[l,b]];n=len(g)+len(lost);p=min(1.,2*sum(math.comb(n,k) for k in range(min(len(g),len(lost))+1))/2**n) if n else 1.;paired[a+'→'+b]=dict(gained=g,lost=lost,exact_mcnemar_p=p,bonferroni4=min(1.,p*4))
 coverage={}
 for k in ['c1','c2']:
  n=sum(r['frames'] for r in extraction);hands=sum(r['stats'][k]['left_frames']+r['stats'][k]['right_frames'] for r in extraction);seconds=sum(r['stats'][k]['processing_seconds'] for r in extraction)
  coverage[k]=dict(frames=n,hand_frame_coverage_pct=100*hands/(2*n),equal_clip_hand_coverage_pct=float(np.mean([100*(r['stats'][k]['left_frames']+r['stats'][k]['right_frames'])/(2*r['frames']) for r in extraction])),seconds=seconds,ms_per_frame=1000*seconds/n,mean_longest_missing_seconds=float(np.mean([max(r['stats'][k]['longest_missing_left'],r['stats'][k]['longest_missing_right'])/r['fps'] for r in extraction])))
 for f,h in protocol['production'].items():assert sha(HERE.parent/f)==h
 result=dict(summary=summary,paired=paired,coverage=coverage,predictions=rows,production_unchanged=True,protocol_sha256=sha(OUT/'protocol.json'));save(OUT/'results.json',result)
 lines=['# วิดีโอมนุษย์30คลิป: complexity × FPS','', '8 ตุลาคม2026 — 30คำไม่ซ้ำคำละ1คลิป เลือกและตรึงรายชื่อก่อนสกัด/ดูคะแนน จากไฟล์ที่มีอยู่ตรวจhashและvideo_idกับmetadata test; ไม่รวมplan/please/buyที่เพิ่งตรวจ3คลิป ชุดนี้ยังเป็นข้อมูลที่เคยใช้สำรวจบางส่วนและmetadata test ไม่ยืนยันว่าcheckpointไม่เคยใช้ประเมิน','', '| วิธี | ถูก/30 | Accuracy | Macro-P | Macro-R | Macro-F1 | Top5 |','|---|---:|---:|---:|---:|---:|---:|']
 for v,s in summary.items():lines.append(f"| {v} | {s['correct']}/30 | {s['accuracy']:.2f}% | {s['precision']:.2f}% | {s['recall']:.2f}% | {s['f1']:.2f}% | {s['top5']:.2f}% |")
 lines+=['','## ตรวจพบมือและเวลา','```json',json.dumps(coverage,indent=2),'```','coverageเป็นการพบจุด ไม่ใช่ความถูกต้องของจุด; มือเดียว/การบังอาจถูกต้องตามคลิป ไม่มีGTพิกัดมนุษย์ให้วัดRMSE','', '## ผลแบบจับคู่','```json',json.dumps(paired,ensure_ascii=False,indent=2),'```','', '## รายคลิป','| คำ | video_id | nativeFPS | c1 native | c1 30 | c2 native | c2 30 |','|---|---|---:|---|---|---|---|']
 for c in protocol['cases']:
  m=next(r for r in extraction if r['video_id']==c['video_id']);lines.append('| '+c['label']+' | '+c['video_id']+' | '+str(m['fps'])+' | '+' | '.join(next(r['prediction'] for r in rows if r['label']==c['label'] and r['variant']==v) for v in protocol['variants'])+' |')
 lines+=['','ทุกวิธีใช้75จุดnormalizedก่อนเลือก27และCenterAndScaleNormalizeแบบclipเดียวกัน ST-GCNเดิมไม่มีfine-tuning ไม่มีrest gate/เติมมือ และไม่เปลี่ยนproduction Nativeใช้เฟรมทั้งหมดตามเวลาไฟล์ 30FPSใช้nearest resampling จำนวนเฟรมต่างกันโดยตั้งใจตามfactor','', 'Macroเฉลี่ย30true classesแบบzero_division0 โมเดลเลือกได้2000คำ ไม่ซ่อนคำทายนอกชุด ขนาดทดลองเล็กและการเลือกจากคลิปที่มีอยู่ทำให้ไม่ใช่ตัวแทนWLASLทั้งหมด ไม่ใช่ความเข้าใจASLจากผู้ใช้','', 'ผลของcomplexityไม่บอกเฉพาะมือเพราะเปลี่ยนการประเมินposeด้วย เวลาเป็นการวัดบนเครื่องนี้รอบเดียว ไม่ใช่benchmarkความเร็วที่ทำซ้ำ']
 (OUT/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(dict(summary=summary,coverage=coverage,paired=paired),ensure_ascii=True,indent=2))
if __name__=='__main__':{'freeze':freeze,'extract':extract,'score':score}[sys.argv[1]]()
