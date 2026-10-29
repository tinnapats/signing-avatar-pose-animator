
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import sys,json,hashlib,zipfile,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;O=ROOT/'stage_inputs_20261011';O.mkdir(exist_ok=True)
assert not (O/'protocol.json').exists()
A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit');B=A/'pixels30_20261007';H=A/'human30_preprocess_20261008'
read=lambda p:_local_json_loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
protocol={'status':'registered','human':'Existing verified WLASL test video IDs, complexity2 native FPS, normalizedxy. Independent control not asserted to be the source of Avatar CSV.',
 'avatar':'First30 actual captured case hashes and frameIndices. Source30 from archived original30 data matched by label and frame count; human-video origin of CSV not established.',
 'render_reference':'Only hands available in image coordinates; use RTMW image body for a hand-only reference diagnostic, NOT full renderer output and NOT a usable pixel-only system.',
 'scope':'No changes to production/model; no score tuning; exploratory archived data',
 'hashes':{str(p):sha(p) for p in [Path(__file__),A/'wlasl_stgcn.zip',B/'protocol.json',H/'protocol.json',A/'repeat_original30_20260930/inputs.json']}}
save(O/'protocol.json',protocol)
sys.path.insert(0,str(A));import run_audit as audit
spec=importlib.util.spec_from_file_location('ev',ROOT.parent/'rtm-eval.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
model,norm,vocab=audit.load_model()
trans=audit.module('audit_stage_trans','upstream_pose_transforms.py');selector=trans.PoseSelect(preset='mediapipe_holistic_minimal_27');assert selector.pose_indexes==audit.PICK
metadata=_local_json_loads(zipfile.ZipFile(A/'wlasl_metadata.zip').read('splits/asl2000.json'));idmap={str(i['video_id']):(r['gloss'],i['split']) for r in metadata for i in r['instances']}
assert vocab==sorted(r['gloss'] for r in metadata)
pred={};arrays={};logits={};trace=[]
def predict(a,label,variant,key):
 a=np.asarray(a,np.float32);x=norm(selector({'frames':torch.from_numpy(a.transpose(2,0,1).copy())}))['frames']
 manual=norm({'frames':torch.from_numpy(a[:,audit.PICK].transpose(2,0,1).copy())})['frames'];assert torch.equal(x,manual) and torch.isfinite(x).all()
 with torch.inference_mode():out=model(x.unsqueeze(0))[0];p=out.softmax(0)
 top=p.topk(5);rr={'label':label,'prediction':vocab[int(top.indices[0])],'top5':[vocab[int(j)] for j in top.indices],'target_probability':float(p[vocab.index(label)]),'frames':len(a)}
 pred.setdefault(variant,[]).append(rr);arrays[key]=x.numpy();logits[key]=out.numpy();return rr
human_old=read(H/'results.json')
for c in read(H/'protocol.json')['cases']:
 vid=str(c['video_id']);assert idmap[vid]==(c['label'],'test');assert sha(Path(c['path']))==c['sha256']
 meta=read(H/f'{vid}.json');assert sha(H/f'{vid}.npz')==meta['array_sha256'];z=np.load(H/f'{vid}.npz')
 rr=predict(z['c2'],c['label'],'human_control_c2_native','human_'+vid)
 before=next(r for r in human_old['predictions'] if r['video_id']==vid and r['variant']=='c2_native');assert rr['top5']==before['top5'] and abs(rr['target_probability']-before['target_probability'])<1e-5
 trace.append({'kind':'human','label':c['label'],'video_id':vid,'fps':float(z['fps']),'frames':len(z['c2']),'metadata_verified':True})
 print('human',c['label'],rr['prediction'],flush=True)
source={c['phrase']:c for c in read(A/'repeat_original30_20260930/inputs.json')}
refs={c['id']:c for c in read(B/'coordinate_reference.json')};manifest={c['id']:c for c in read(B/'capture_manifest.json')}
for d in read(B/'protocol.json')['descriptors']:
 cid=d['id'];label=d['label'];assert sha(Path(d['file']))==d['sha256'];case=read(Path(d['file']));cap=manifest[cid]
 assert case['frameIndices']==[f['sourceIndex'] for f in cap['frames']]
 src=np.asarray(source[label]['source30'],np.float32);payload=audit.frame_array([case['payload']['frames'][i] for i in case['frameIndices']]);assert len(src)==len(payload)==len(cap['frames'])
 z=np.load(B/'rtm_trial_20261008'/f'{cid}.npz');pixels=ev.convert(z['rtmw_det'],z['rtmw_det_scores']);assert np.allclose(z['times'],[f['time'] for f in cap['frames']])
 refmix=pixels.copy()
 for i,r in enumerate(refs[cid]['samples']):
  assert r['index']==case['frameIndices'][i] and r['pngMatches'] and r['pixelRatio']==1
  for side,b in [('left',33),('right',54)]:
   refmix[i,b:b+21]=0
   for j,point in r['hands'][side].items():refmix[i,b+int(j)]=point
 for name,a in [('source30_archive',src),('actual_payload',payload),('renderer_hands_rtmw_body_diagnostic',refmix),('avatar_pixels_rtmw',pixels)]:predict(a,label,name,cid+'_'+name)
 trace.append({'kind':'avatar','id':cid,'label':label,'case_sha256':d['sha256'],'source_archive_sha256':protocol['hashes'][str(A/'repeat_original30_20260930/inputs.json')],
  'frames':len(src),'fps':30,'human_video_id_of_source_csv':None,'source_link':'same word and frame count; exact historical CSV identity not proven by this archive alone','payload_variant':case['variant']})
 print('avatar',label,[(v,pred[v][-1]['prediction']) for v in pred if v!='human_control_c2_native'],flush=True)
summary={v:ev.metrics(rs) for v,rs in pred.items()}
save(O/'results.json',{'summary':summary,'predictions':pred,'trace':trace,'pick27_exact':True,'vocabulary_verified':True,'human_baseline_reproduced':True})
np.savez_compressed(O/'inputs.npz',**arrays);np.savez_compressed(O/'logits.npz',**logits)
protocol['status']='completed';save(O/'protocol.json',protocol)
lines=['# ตรวจข้อมูลเข้า ST-GCN แยกระดับ','',
 'ข้อจำกัดก่อนอ่านคะแนน: human control ไม่ใช่วิดีโอต้นทางที่ยืนยันแล้วของ CSV Avatar จึงห้ามลบคะแนนเพื่อระบุ domain gap แบบจับคู่คลิป',
 'source30_archive ตรงคำและจำนวนเฟรมกับภาพ Avatar แต่ยังไม่มีหลักฐาน video_id ต้นทางของ CSV; actual_payload ผูก hash กับ case ที่ capture PNG จริง',
 'renderer_hands_rtmw_body_diagnostic ใช้มือจาก Renderer reference + ลำตัวที่ตรวจจากภาพด้วย RTMW ไม่ใช่พิกัด renderer ทั้งตัว ไม่ใช่ระบบใช้งานจริง และมีข้อมูลอ้างอิงเข้าเพื่อวินิจฉัยเท่านั้น',
 '| ระดับ | ถูก/30 | Accuracy % | Macro-P % | Macro-R % | Macro-F1 % | Top5 % |','|---|---:|---:|---:|---:|---:|---:|']
for v,m in summary.items():lines.append(f"| {v} | {m['correct']} | {m['accuracy']:.2f} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['top5']:.2f} |")
lines+=['','ยืนยัน PoseSelect27 ตรง upstream, tensorและnormalizationตรง, sorted2000glossตรงmetadata, human videoID/split/filehashตรง และรันhumanผลซ้ำตรงเดิม',
 'ยังไม่ยืนยัน detector/versionที่ใช้ผลิตข้อมูลฝึกcheckpointจริง และความถูกต้องของจุดตรวจจับทุกเฟรม ไม่ตีความผลต่ำว่าโมเดลผิดหรือAvatarผิดฝ่ายเดียว',
 'Macroเฉลี่ยเฉพาะtrue labels30ของแต่ละชุด ไม่ใช่ทั้ง2000classes; ทุกชุดเป็น exploratory archived data ไม่ใช่ pristine test']
(O/'REPORT_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(summary),flush=True)
