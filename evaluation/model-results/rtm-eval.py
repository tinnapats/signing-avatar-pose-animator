
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import sys,json,hashlib,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit');B=A/'pixels30_20261007';O=B/'rtm_trial_20261008';sys.path.insert(0,str(A))
read=lambda p:_local_json_loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
VARIANTS=['mediapipe','dw_det','dw_full','rtmw_det','rtmw_full','crop_only','crop_fallback','rtmw_body_mp_hands','mp_body_rtmw_hands']
RESULT_NAME='results.json'
def convert(p,s):
 a=np.zeros((len(p),75,2),np.float32)
 # Anatomical COCO left/right, never image left/right. COCO eyes approximate MP eye centers.
 for dst,src in [(0,0),(2,1),(5,2),(11,5),(12,6),(13,7),(14,8)]:a[:,dst]=np.where((s[:,src]>=.3)[:,None],p[:,src],0)
 for dst,src in [(33,91),(54,112)]:
  valid=(s[:,src:src+21].mean(1)>=.3)&(s[:,src]>=.3)
  a[:,dst:dst+21]=np.where(valid[:,None,None],p[:,src:src+21],0)
 return a
def data(c):
 p=O/(c['id']+'.npz');assert sha(p)==read(O/(c['id']+'.json'))['sha256'];z=np.load(p)
 out={'mediapipe':np.load(B/'arrays'/(c['id']+'.npz'))['primary']}
 for v in ['dw_det','dw_full','rtmw_det','rtmw_full']:out[v]=convert(z[v],z[v+'_scores'])
 hp=O/'hand_crop'/(c['id']+'.npz');assert sha(hp)==read(hp.with_suffix('.json'))['sha256'];h=np.load(hp)
 assert np.array_equal(h['times'],z['times'])
 out['crop_only']=out['rtmw_det'].copy();out['crop_only'][:,33:]=0;out['crop_fallback']=out['rtmw_det'].copy()
 for side,off in enumerate([33,54]):
  valid=h['accepted'][:,side];out['crop_only'][valid,off:off+21]=h['points'][valid,side];out['crop_fallback'][valid,off:off+21]=h['points'][valid,side]
 out['rtmw_body_mp_hands']=out['rtmw_det'].copy();out['rtmw_body_mp_hands'][:,33:]=out['mediapipe'][:,33:]
 out['mp_body_rtmw_hands']=out['mediapipe'].copy();out['mp_body_rtmw_hands'][:,33:]=out['rtmw_det'][:,33:]
 assert all(a.shape==out['mediapipe'].shape for a in out.values());assert np.allclose(z['times'],[f['time'] for f in c['frames']])
 return out
def metrics(rr):
 labels=[r['label'] for r in rr];ps=[];rs=[];fs=[]
 for l in labels:
  tp=sum(r['label']==l and r['prediction']==l for r in rr);fp=sum(r['label']!=l and r['prediction']==l for r in rr);p=tp/(tp+fp) if tp+fp else 0;ps.append(p);rs.append(tp);fs.append(2*p*tp/(p+tp) if p+tp else 0)
 return dict(correct=sum(rs),accuracy=100*np.mean(rs),precision=100*np.mean(ps),recall=100*np.mean(rs),f1=100*np.mean(fs),top5=100*np.mean([r['label'] in r['top5'] for r in rr]))
def evaluate():
 import torch,run_audit as audit
 probe=np.arange(266,dtype=np.float32).reshape(1,133,2);mapped=convert(probe,np.ones((1,133),np.float32));handpick=[0,4,5,8,9,12,13,16,17,20]
 expected=[0,1,2,5,6,7,8]+[91+j for j in handpick]+[112+j for j in handpick]
 assert np.array_equal(mapped[:,audit.PICK],probe[:,expected]);assert not np.any(convert(probe,np.zeros((1,133),np.float32)))
 assert not (O/RESULT_NAME).exists();model,norm,vocab=audit.load_model();manifest=read(B/'capture_manifest.json');ref={c['id']:c for c in read(B/'coordinate_reference.json')};pred={v:[] for v in VARIANTS};errors={v:{} for v in VARIANTS};coverage={v:0 for v in VARIANTS};pairs={v:dict(same=0,swapped=0,ambiguous=0) for v in VARIANTS};temporal={v:[] for v in VARIANTS};totalrefs=0;detected=0
 for c in manifest:
  arrays=data(c);detected+=read(O/(c['id']+'.json'))['detected']
  for v,a in arrays.items():
   x=norm({'frames':torch.from_numpy(a[:,audit.PICK].transpose(2,0,1).copy())})['frames'].unsqueeze(0)
   with torch.inference_mode():p=model(x)[0].softmax(0)
   t=p.topk(5);pred[v].append(dict(label=c['label'],prediction=vocab[int(t.indices[0])],top5=[vocab[int(j)] for j in t.indices],target_probability=float(p[vocab.index(c['label'])])))
   coverage[v]+=sum(int(np.any(a[:,off:off+21]!=0,axis=(1,2)).sum()) for off in [33,54])
  previous={v:{} for v in VARIANTS}
  for i,r in enumerate(ref[c['id']]['samples']):
   assert r['pngMatches'] and r['index']==c['frames'][i]['sourceIndex'];targets={}
   for side,off in [('left',33),('right',54)]:
    h=r['hands'].get(side,{})
    if len(h)!=21:continue
    totalrefs+=1;g=np.array([h[str(j)] for j in range(21)]);targets[side]=g
    for v,a in arrays.items():
     q=a[i,off:off+21]
     if not np.any(q):continue
     err=q-g;errors[v][f'{c["id"]}:{i}:{side}']=np.linalg.norm(err,axis=1).tolist()
     if side in previous[v] and previous[v][side][0]==i-1:temporal[v].extend(np.linalg.norm((err-previous[v][side][1])*30,axis=1).tolist())
     previous[v][side]=(i,err)
   if len(targets)==2:
    for v,a in arrays.items():
     l=a[i,33:54];rr=a[i,54:75]
     if not np.any(l) or not np.any(rr):continue
     same=(np.linalg.norm(l-targets['left'],axis=1).mean()+np.linalg.norm(rr-targets['right'],axis=1).mean())/2
     swapped=(np.linalg.norm(l-targets['right'],axis=1).mean()+np.linalg.norm(rr-targets['left'],axis=1).mean())/2
     pairs[v]['same' if same+4<swapped else 'swapped' if swapped+4<same else 'ambiguous']+=1
  print('scored',c['label'],flush=True)
 def stats(e):
  ds=np.array(list(e.values())).ravel();return dict(hand_frames=len(e),rmse_px=float(np.sqrt(np.mean(ds**2))) if len(ds) else None,pck10_pct=float(100*np.mean(ds<=10)) if len(ds) else None,pck10_all_reference_pct=float(100*np.sum(ds<=10)/(totalrefs*21)))
 geometry={v:stats(e) for v,e in errors.items()};common={}
 for v in VARIANTS[1:]:
  keys=set(errors[v])&set(errors['mediapipe']);common[v]={k:stats({i:errors[k][i] for i in keys}) for k in ['mediapipe',v]}
 for f,h in read(O/'protocol.json')['production'].items():assert sha(A.parent/f)==h
 result=dict(summary={v:metrics(rr) for v,rr in pred.items()},predictions=pred,geometry=geometry,common=common,coverage_hand_frames=coverage,reference_hand_frames=totalrefs,detected_frames=detected,frames=1964,pairing=pairs,temporal_error_velocity_rms_px_s={v:float(np.sqrt(np.mean(np.square(a)))) if a else None for v,a in temporal.items()},production_unchanged=True,timing_unchanged=True)
 save(O/RESULT_NAME,result);print(json.dumps(result['summary']),flush=True);print(json.dumps(geometry),flush=True)
def review():
 import cv2
 (O/'review').mkdir(exist_ok=True)
 edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
 for c in read(B/'capture_manifest.json'):
  if c['id'] not in ['001','002','005','011','016','025']:continue
  if not (O/(c['id']+'.npz')).exists():continue
  if not (O/'hand_crop'/(c['id']+'.npz')).exists():continue
  arrays=data(c);panels=[]
  for i in np.rint(np.linspace(0,len(c['frames'])-1,3)).astype(int):
   im=cv2.imread(str(B/'frames'/c['id']/c['frames'][i]['file']),-1)
   if im.shape[2]==4:
    alpha=im[:,:,3:4].astype(np.float32)/255;im=np.rint(im[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8)
   row=[]
   for v in VARIANTS[:7]:
    img=im.copy();a=arrays[v][i]
    for off,color in [(33,(255,100,0)),(54,(0,60,255))]:
     q=a[off:off+21]
     if not np.any(q):continue
     for j,k in edges:cv2.line(img,tuple(np.rint(q[j]).astype(int)),tuple(np.rint(q[k]).astype(int)),color,2)
     for p in q:cv2.circle(img,tuple(np.rint(p).astype(int)),3,color,-1)
    for j,k in [(11,12),(11,13),(12,14),(13,33),(14,54)]:
     if np.any(a[j]) and np.any(a[k]):cv2.line(img,tuple(np.rint(a[j]).astype(int)),tuple(np.rint(a[k]).astype(int)),(100,170,40),2)
    img=cv2.resize(img,(300,300));img=cv2.copyMakeBorder(img,30,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255));cv2.putText(img,v+' f'+str(i),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1);row.append(img)
   panels.append(np.concatenate(row,axis=1))
  sheet=np.concatenate(panels,axis=0);assert cv2.imwrite(str(O/'review'/(c['id']+'.png')),sheet)
 print('review saved',flush=True)
def report():
 r=read(O/'results.json');manifest=read(B/'capture_manifest.json');timings=np.concatenate([np.load(O/(c['id']+'.npz'))['pose_ms'] for c in manifest]);det_ms=np.concatenate([np.load(O/(c['id']+'.npz'))['det_ms'] for c in manifest]);hand_ms=np.concatenate([np.load(O/'hand_crop'/(c['id']+'.npz'))['ms'] for c in manifest])
 lines=['# เปรียบเทียบ MediaPipe, DWPose, RTMW และ RTMPose-hand บนภาพ Avatar','', '## ขอบเขต','30 คลิปเดิม 1,964 เฟรม ภาพ808×808 30FPS ตรวจ hash ตรงต้นฉบับ ไม่มีแก้ Renderer/เวลา ไม่มี smoothing/tracking/เติมช่องว่าง ไม่มีฝึกโมเดลทายคำใหม่ ใช้ ST-GCN checkpoint เดิมทาย 2,000 คลาส','', 'ชุดนี้เป็นชุดพัฒนาที่เคยดูผลแล้ว ไม่ใช่ held-out test ใหม่ หนึ่งคลิปต่อคำ ผลไม่ยืนยันความถูกต้อง ASL หรือความเข้าใจผู้ใช้','', '## วิธี','- mediapipe: Holistic complexity1 พิกัดภาพล้วนเดิม','- dw: DWPose-m 256×192; rtmw: RTMW-l 256×192','- det: YOLOX-m HumanArt กรอบใหญ่สุด หากไม่พบกรอบคืน missing ไม่แอบใช้กรอบทั้งภาพ','- full: กรอบทั้งภาพ808×808 เป็นการทดลองแยก มีพื้นที่ว่างและสเกลต่างจากกรอบ detector จึงไม่ใช่ผลของ detection อย่างเดียว','- crop_only: ลำตัว RTMW-det; มือ RTMPose-hand256×256 จากบริเวณข้อมือ/ศอกที่อ่านจากภาพ ไม่ผ่านเกณฑ์เป็น missing','- crop_fallback: ใช้ crop เมื่อผ่านเกณฑ์ ไม่ผ่านใช้มือ RTMW เดิม (ไม่ได้เลือกตามคำตอบ)','- rtmw_body_mp_hands / mp_body_rtmw_hands: แยกผลเปลี่ยนลำตัวกับมือ ข้อมูลทั้งสองส่วนยังมาจากภาพล้วน','', '## คะแนนรู้จำคำ','| วิธี | ถูก/30 | Accuracy | Macro-P | Macro-R | Macro-F1 | Top5 |','|---|---:|---:|---:|---:|---:|---:|']
 for v,s in r['summary'].items():lines.append(f"| {v} | {s['correct']}/30 | {s['accuracy']:.2f}% | {s['precision']:.2f}% | {s['recall']:.2f}% | {s['f1']:.2f}% | {s['top5']:.2f}% |")
 lines+=['','Macroเฉลี่ย30 true classes, zero_division0 ไม่จำกัดคำตอบแค่30คำ ไม่มีเลือก confidence threshold ด้วยคะแนนคำทาย','', '## ความตรงของมือ','Renderer reference ใช้เฉพาะการประเมิน ไม่ได้เป็น input/crop/ตัวเลือกรูปมือ รวม21จุดต่อมือ เฉพาะ detailed hand ที่มีreference; SVG fallback ไม่ถูกนับเป็นท่าพักที่ถูกต้องหรือมือผิดโดยอัตโนมัติ',f"มี reference {r['reference_hand_frames']} hand-frames; ตรวจพบตัวละคร {r['detected_frames']}/{r['frames']} เฟรม",'', '| วิธี | มือผ่านเกณฑ์/3928 | มี reference เทียบได้ | RMSE(px) | PCK@10px เฉพาะที่พบ | PCK@10px นับ missing ว่าผิด |','|---|---:|---:|---:|---:|---:|']
 for v,g in r['geometry'].items():lines.append(f"| {v} | {r['coverage_hand_frames'][v]} | {g['hand_frames']} | {g['rmse_px']:.2f} | {g['pck10_pct']:.2f}% | {g['pck10_all_reference_pct']:.2f}% |")
 lines+=['','PCK@10px = สัดส่วนจุดห่างreferenceไม่เกิน10px เกณฑ์ตรึงก่อนดูผล; raw confidence ข้ามโมเดลไม่ใช่ค่าเดียวกันที่calibrateแล้ว จึงรายงานcoverageและerrorร่วมกัน','', '### เปรียบเทียบเฉพาะมือเฟรมเดียวกันกับ MediaPipe','| วิธี | common hand-frames | MediaPipe RMSE | วิธีใหม่ RMSE |','|---|---:|---:|---:|']
 for v,g in r['common'].items():lines.append(f"| {v} | {g[v]['hand_frames']} | {g['mediapipe']['rmse_px']:.2f} | {g[v]['rmse_px']:.2f} |")
 lines+=['','## ซ้าย–ขวาและความต่อเนื่อง','การสลับข้างวัดเฉพาะเฟรมที่มีreferenceทั้ง2มือและตรวจพบทั้ง2มือ; ambiguousหากerrorต่างไม่เกิน4px ไม่ใช้referenceสลับข้างให้โมเดล','```json',json.dumps(r['pairing'],indent=2),'```','Temporal error velocity RMS(px/s): วัดการเปลี่ยนความคลาดเคลื่อนจากreferenceระหว่างเฟรมติดกัน ไม่ใช่ลดการเคลื่อนไหวจริงให้เหลือน้อยที่สุด; maskแต่ละวิธีต่างกันจึงใช้เป็นdiagnostic ไม่จัดอันดับเดี่ยว','```json',json.dumps(r['temporal_error_velocity_rms_px_s'],indent=2),'```','', '## เวลาและการทำซ้ำ',f"CPU ONNX Runtime4threads วัดระหว่างรันบนเครื่องนี้ (มีงานcropรันร่วมบางช่วง จึงไม่ใช่ benchmarkความเร็วแยกเครื่อง): detector mean {det_ms.mean():.1f}ms; pose [dw_det,dw_full,rtmw_det,rtmw_full] {np.round(timings.mean(0),1).tolist()}ms; crop2มือ mean {hand_ms.mean():.1f}ms/frame ไม่รวมโหลดโมเดล",'', 'ตรวจการmappingด้วยข้อมูลสังเคราะห์: COCO nose,left/right eye,shoulder,elbow และมือ91:112/112:133 ->27จุดSTGCN; COCOeyeกับMediaPipeeyeเป็นนิยามใกล้เคียงแต่ไม่เหมือนทุกประการ ต้องระวังdomain mismatch ไม่มีสลับตามซ้ายขวาบนจอ','', 'เก็บ protocol.json, model SHA256, raw133points+confidence, boxes, timestamp, per-clipSHA และ predictionครบ; production hashes4ไฟล์ตรงเดิม','', '## งานถัดไปตามเงื่อนไข','ยังไม่ทำ anime hand detector, SDPose, temporal repair หรือ fine-tune ในผลชุดนี้ ต้องตัดสินจากทั้งgeometryและrecognition ไม่เลือกจากคะแนนคำทายอย่างเดียว หากตัวcropยังไม่ดีให้ตรวจกรอบและรายละเอียดนิ้วบนภาพก่อนฝึกเพิ่ม','', '## แหล่งโมเดล','https://github.com/Tau-J/rtmlib','https://github.com/open-mmlab/mmpose','', '## คำทายรายคลิป','| คำ | '+' | '.join(VARIANTS)+' |','|---|'+'---|'*len(VARIANTS)]
 for i,c in enumerate(manifest):lines.append('| '+c['label']+' | '+' | '.join(r['predictions'][v][i]['prediction'] for v in VARIANTS)+' |')
 lines+=['','หมายเหตุ smoothing: ไม่เพิ่ม temporal smoothing ในรอบนี้ แต่ MediaPipe baseline เดิมเปิด smooth_landmarks ตามค่า default อยู่แล้ว ขณะที่ RTM ใช้ inference รายเฟรม จึงไม่ใช่การแยกสถาปัตยกรรมโดยตรึง temporal processing เหมือนกันทุกส่วน','', 'crop2มือใช้ข้อเสนอกรอบจากข้อมือเป็นหลัก ไม่ได้รัน anime_hand_detection ในรอบนี้; hand-only crop ถูกกรองด้วยความสอดคล้องกับข้อมือ แต่การผ่านเกณฑ์ไม่ได้รับรองว่าข้อนิ้วถูกต้อง']
 (O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf8');print('report saved',flush=True)
if __name__=='__main__':{'review':review,'report':report}.get(sys.argv[1] if len(sys.argv)>1 else '',evaluate)()
