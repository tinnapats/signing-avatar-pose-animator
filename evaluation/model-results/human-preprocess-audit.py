
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import sys,json,hashlib,zipfile
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
B=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007');O=B/'human_preprocess_audit_20261008';sys.path.insert(0,str(B.parent))
read=lambda p:_local_json_loads(p.read_text(encoding='utf8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def extract():
 import cv2,mediapipe as mp
 O.mkdir(exist_ok=True);selection=read(B/'upper_crop_20261008/human_selection.json');metadata=_local_json_loads(zipfile.ZipFile(B.parent/'wlasl_metadata.zip').read('splits/asl2000.json'));idmap={str(i['video_id']):(r['gloss'],i['split']) for r in metadata for i in r['instances']};verified=[]
 for h in selection:
  video=Path(h['local_video']);assert sha(video)==h['sha256'];assert idmap[h['metadata']['video_id']]==(h['label'],h['metadata']['split']);cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);rows=[];dims=[];wrist=[]
  with mp.solutions.holistic.Holistic(static_image_mode=False,model_complexity=2) as detector:
   while True:
    ok,im=cap.read()
    if not ok:break
    height,width=im.shape[:2];dims.append([width,height]);d=detector.process(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));a=np.zeros((75,2),np.float32)
    for lm,off in [(d.pose_landmarks,0),(d.left_hand_landmarks,33),(d.right_hand_landmarks,54)]:
     if lm:a[off:off+len(lm.landmark)]=[[p.x,p.y] for p in lm.landmark]
    rows.append(a)
  cap.release();assert len(rows)==h['decoded_frames'];np.savez_compressed(O/(h['label']+'.npz'),normalized_native=np.array(rows),dimensions=np.array(dims),fps=fps)
  verified.append(dict(label=h['label'],video_id=h['metadata']['video_id'],split=h['metadata']['split'],sha256=sha(video),frames=len(rows),fps=fps,dimensions=dims[0]));print('extracted',h['label'],len(rows),flush=True)
 save(O/'verified_videos.json',verified)
def score():
 import torch,yaml,run_audit as audit
 model,norm,vocab=audit.load_model();trans=audit.module('audit_check_transforms','upstream_pose_transforms.py');selector=trans.PoseSelect(preset='mediapipe_holistic_minimal_27');assert audit.PICK==selector.pose_indexes
 config=yaml.safe_load(zipfile.ZipFile(B.parent/'wlasl_stgcn.zip').read('wlasl/st_gcn/config.yaml'));assert [next(iter(x)) for x in config['data']['valid_pipeline']['transforms']]==['PoseSelect','CenterAndScaleNormalize']
 rows=[];checks=[]
 for h in read(O/'verified_videos.json'):
  z=np.load(O/(h['label']+'.npz'));heavy=z['normalized_native'];old=np.load(B/'upper_crop_20261008'/('human_'+h['label']+'.npz'));native=old['native'];assert len(native)==len(heavy);dims=z['dimensions'];full=native/dims[:,None,:];idx=np.minimum(np.rint(np.arange(0,len(heavy)/h['fps'],1/30)*h['fps']).astype(int),len(heavy)-1)
  variants={'old_c1_30fps_pixel':old['points'],'c1_native_pixel':native,'c1_native_normalized':full,'c2_native_normalized':heavy,'c2_30fps_normalized':heavy[idx]}
  tensorcheck=norm(selector({'frames':torch.from_numpy(native.astype(np.float32).transpose(2,0,1).copy())}))['frames'];manual=norm({'frames':torch.from_numpy(native[:,audit.PICK].transpose(2,0,1).copy())})['frames'];assert torch.equal(tensorcheck,manual)
  normalized=norm({'frames':torch.from_numpy(full.astype(np.float32)[:,audit.PICK].transpose(2,0,1).copy())})['frames'];diff=float(torch.max(torch.abs(normalized-manual)));checks.append(dict(label=h['label'],selection_normalization_exact=True,pixel_vs_normalized_max_tensor_difference=diff,body_hand_order='pose33,left21,right21',handedness='Holistic anatomical sides; no swap or reflection',shape=list(manual.unsqueeze(0).shape)))
  for v,a in variants.items():
   raw=torch.from_numpy(a.astype(np.float32)[:,audit.PICK].transpose(2,0,1).copy());x=norm({'frames':raw})['frames'].unsqueeze(0);assert torch.isfinite(x).all()
   with torch.inference_mode():p=model(x)[0].softmax(0)
   top=p.topk(5);rows.append(dict(label=h['label'],video_id=h['video_id'],variant=v,frames=len(a),prediction=vocab[int(top.indices[0])],top5=[vocab[int(i)] for i in top.indices],target_probability=float(p[vocab.index(h['label'])])))
  print('audited',h['label'],flush=True)
 # Expose missing-value behavior in the exact upstream normalizer.
 missing_probe=torch.zeros((2,2,27));missing_probe[0,:,3]=2;missing_probe[0,:,4]=4;missing_probe[1,:,3:5]=2;missing_after=norm({'frames':missing_probe.clone()})['frames'][:,0,7].tolist()
 graph=config['model']['encoder']['params']['graph_args'];assert [5,7] in graph['inward_edges'] and [6,17] in graph['inward_edges']
 summary={v:dict(correct=sum(r['prediction']==r['label'] for r in rows if r['variant']==v),top5_correct=sum(r['label'] in r['top5'] for r in rows if r['variant']==v)) for v in variants}
 result=dict(summary=summary,predictions=rows,checks=checks,missing_zero_after_normalization=missing_after,hand_local_ids=[p-33 for p in audit.PICK if 33<=p<54],valid_transforms=config['data']['valid_pipeline']['transforms'],checkpoint_sha256=sha(B.parent/'wlasl_stgcn.zip'),scope='3 known IDs, diagnostic only. Live upstream main examined, checkpoint extraction version not established. No model or production edits. Batch64 padding used by upstream config can differ from single-clip inference; not claimed to reproduce paper accuracy.')
 for f,h in read(B/'protocol.json')['production'].items():assert sha(B.parent.parent/f)==h
 save(O/'results.json',result)
 lines=['# ตรวจขั้นเตรียมข้อมูลจากวิดีโอมนุษย์ที่รู้รหัสคลิป','', '8 ตุลาคม2026 — ตรวจไฟล์ด้วยSHA256และจับคู่video_idกับsplits/asl2000.jsonในarchiveที่ใช้สร้างvocabularyจริงครบ3คลิป','', '| คำ | video_id | native FPS | เฟรม |','|---|---|---:|---:|']
 for h in read(O/'verified_videos.json'):lines.append(f"| {h['label']} | {h['video_id']} | {h['fps']} | {h['frames']} |")
 lines+=['','## สิ่งที่ตรงกับโค้ดโมเดล','- RGBจากOpenCV, pose33→มือซ้าย21→มือขวา21, PoseSelect27ตรงทุกindex, tensorNCTV, sortedgloss2000, shoulderclip-normalizationตรงimplementation',
'- มือแต่ละข้างใช้local IDs 0,4,5,8,9,12,13,16,17,20 = wrist,thumb tip และMCP/tipของนิ้วที่เหลือ คำอธิบายก่อนหน้านี้ที่ระบุthumb MCP/IPหรือindex PIPไม่ถูกต้อง',
'- ไม่สลับชื่อซ้ายขวาจากHolistic กราฟเชื่อมศอกซ้ายกับwristซ้าย และขวากับขวา ตรงตามpreset การเทียบโค้ดไม่รับรองdetectorวางจุดถูกทุกเฟรม',
'','## ความต่างที่ตรวจพบ','- generator upstream main ใช้complexity2; รอบทดลองก่อนใช้complexity1 เวอร์ชันMediaPipe/posepickleจริงตอนฝึกcheckpointยังไม่ทราบ',
'- valid configมีPoseSelectและCenterAndScaleNormalize ไม่มีบังคับ30FPS รอบก่อนเพิ่มเฟรม25→30เอง จึงแยกทดสอบnativeFPS',
'- upstreamเก็บnormalized x/y; รอบเราเก็บpixel ในคลิปที่ภาพเป็นสี่เหลี่ยมจัตุรัสการnormalizeด้วยไหล่ทำให้ใกล้เคียงกันมาก ดูnumeric differenceในresults หากภาพไม่จัตุรัสอาจต่าง',
'- ศูนย์มือที่หายก่อนnormalizeถูกเลื่อน/ปรับสเกลด้วย จึงไม่จำเป็นต้องเป็น0ตอนถึงโมเดล เป็นพฤติกรรมของupstream ไม่ใช่maskที่โมเดลอ่านแยกได้',
'','## ผลทดลอง','| เงื่อนไข | plan | please | buy | ถูก/3 | Top5ถูก/3 |','|---|---|---|---|---:|---:|']
 for v,s in summary.items():lines.append('| '+v+' | '+' | '.join(next(r['prediction'] for r in rows if r['label']==l and r['variant']==v) for l in ['plan','please','buy'])+f" | {s['correct']} | {s['top5_correct']} |")
 lines+=['','ไม่มีการเปลี่ยนโมเดล/production การทดสอบ3คลิปไม่เพียงพอเลือกวิธีจากคะแนนและยังไม่ใช่held-outอิสระ คลิปresizedไม่รับรองpixelเหมือนชุดposeที่ใช้ฝึก ไม่ใช้metadataframe_start/endตัดซ้ำเพราะไฟล์อาจตัดมาแล้ว','',
'upstreamvalid batch64มีpaddingถึงความยาวคลิปสูงสุดในbatch ต่างจากinferทีละคลิปของเรา อีกทั้งST-GCNเฉลี่ยตลอดแกนเวลา จึงไม่อ้างว่าได้เงื่อนไขประเมินcheckpointเหมือนต้นฉบับทุกอย่าง',
'','## แหล่งอ้างอิง','- config.yamlภายในwlasl_stgcn.zip','- https://github.com/AI4Bharat/OpenHands/blob/main/openhands/datasets/pipelines/generate_pose.py','- https://github.com/AI4Bharat/OpenHands/blob/main/openhands/datasets/isolated/wlasl.py','- https://github.com/AI4Bharat/OpenHands/blob/main/openhands/datasets/data_readers.py','- upstream_base.py / upstream_pose_transforms.py / upstream_st_gcn.py ที่ใช้โหลดโมเดลจริง']
 (O/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':{'extract':extract,'score':score}[sys.argv[1]]()
