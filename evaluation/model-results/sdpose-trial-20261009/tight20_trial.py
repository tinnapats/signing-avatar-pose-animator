
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import sys,json,hashlib,importlib.util,time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
ROOT=Path(__file__).resolve().parent
A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit');B=A/'pixels30_20261007';R=B/'rtm_trial_20261008';H=R/'hand_crop'
O=ROOT/'wrist_crop_tight20_20261011';O.mkdir(exist_ok=True)
read=lambda p:_local_json_loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
manifest=read(B/'capture_manifest.json')
def extract():
 assert not list(O.glob('*.npz')),'Preserve results'
 info=read(R/'hand_model.json');assert sha(Path(info['path']))==info['sha256']
 protocol={'status':'registered','factor':.8,'crop':'Cached wrist-based crop center unchanged, width and height multiplied .8, padding remains1.25. No image clipping.',
 'rules':read(H/'protocol.json'),'model':info,'clips':len(manifest),'frames':sum(len(c['frames']) for c in manifest),
 'development_only':True,'no_reference_input':True,'production_edited':False,
 'hashes':{str(p):sha(p) for p in [Path(__file__),B/'capture_manifest.json',ROOT.parent/'rtm-hand.py']}}
 save(O/'protocol.json',protocol)
 base=load('base',ROOT.parent/'rtm-avatar.py');cv2,rt,_=base.setup();hand=rt(info['path'],model_input_size=(256,256),backend='onnxruntime',device='cpu')
 checked=False
 for c in manifest:
  cid=c['id'];z=np.load(R/f'{cid}.npz');old=np.load(H/f'{cid}.npz')
  assert sha(H/f'{cid}.npz')==read(H/f'{cid}.json')['sha256']
  pts=[];scores=[];boxes=[];accepted=[];ms=[]
  for i,f in enumerate(c['frames']):
   path=B/'frames'/cid/f['file'];assert sha(path)==f['sha256'];im=base.image(cv2,path)
   pp=np.zeros((2,21,2),np.float32);ss=np.zeros((2,21),np.float32);bb=np.zeros((2,4),np.float32);vv=np.zeros(2,bool);dist=np.full(2,np.inf);start=time.perf_counter()
   for side,(w,e) in enumerate([(9,7),(10,8)]):
    if old['boxes'][i,side,2]<=old['boxes'][i,side,0]:continue
    wrist=z['rtmw_det'][i,w];delta=wrist-z['rtmw_det'][i,e];length=float(np.linalg.norm(delta))
    center=wrist+.25*delta;size=np.clip(2.2*length,96,256);box=np.r_[center-size/2,center+size/2]
    assert np.allclose(box,old['boxes'][i,side],atol=1e-4)
    if not checked:
     p,s=hand(im,bboxes=[box]);print('PROBE differences',float(np.max(abs(p[0]-old['points'][i,side]))),float(np.max(abs(s[0]-old['scores'][i,side]))),flush=True);assert np.allclose(p[0],old['points'][i,side],atol=.02) and np.allclose(s[0],old['scores'][i,side],atol=1e-5);checked=True
    mid=(box[:2]+box[2:])/2;half=(box[2:]-box[:2])*.8/2;bbox=np.r_[mid-half,mid+half];bb[side]=bbox
    p,s=hand(im,bboxes=[bbox]);pp[side]=p[0];ss[side]=s[0]
    wrist=z['rtmw_det'][i,w];length=float(np.linalg.norm(wrist-z['rtmw_det'][i,e]));dist[side]=np.linalg.norm(pp[side,0]-wrist)
    vv[side]=ss[side].mean()>=.3 and ss[side,0]>=.3 and dist[side]<=.65*length
   if vv.all() and np.linalg.norm(pp[0,0]-pp[1,0])<20:vv[int(np.argmax(dist))]=False
   pts.append(pp);scores.append(ss);boxes.append(bb);accepted.append(vv);ms.append(1000*(time.perf_counter()-start))
  np.savez_compressed(O/f'{cid}.npz',points=pts,scores=scores,boxes=boxes,accepted=accepted,times=z['times'],ms=ms)
  save(O/f'{cid}.json',{'sha256':sha(O/f'{cid}.npz'),'label':c['label']})
  print('EXTRACT',cid,c['label'],int(np.sum(accepted)),flush=True)
 protocol['status']='extracted';protocol['baseline_probe_reproduced']=checked;save(O/'protocol.json',protocol)
def score():
 import torch
 sys.path.insert(0,str(A));import run_audit as audit
 assert read(O/'protocol.json')['status']=='extracted' and not (O/'results.json').exists()
 ev=load('ev',ROOT.parent/'rtm-eval.py');model,norm,vocab=audit.load_model()
 pred={v:[] for v in ['rtmw','old_fallback','tight_fallback','old_only','tight_only']};inp={};logits={}
 for c in manifest:
  cid=c['id'];z=np.load(R/f'{cid}.npz');raw=ev.convert(z['rtmw_det'],z['rtmw_det_scores']);vs={'rtmw':raw};fixed=None
  for name,folder in [('old',H),('tight',O)]:
   assert sha(folder/f'{cid}.npz')==read(folder/f'{cid}.json')['sha256']
   h=np.load(folder/f'{cid}.npz');assert np.allclose(h['times'],z['times'])
   for kind in ['fallback','only']:
    a=raw.copy()
    if kind=='only':a[:,33:]=0
    for side,b in enumerate([33,54]):
     valid=h['accepted'][:,side];a[valid,b:b+21]=h['points'][valid,side]
    vs[name+'_'+kind]=a
  for v,a in vs.items():
   t=norm({'frames':torch.from_numpy(a[:,audit.PICK].transpose(2,0,1).copy())})['frames']
   if fixed is None:fixed=t[:,:,:7].clone()
   assert torch.equal(fixed,t[:,:,:7]) and torch.isfinite(t).all()
   with torch.inference_mode():out=model(t.unsqueeze(0))[0];p=out.softmax(0)
   top=p.topk(5);tar=vocab.index(c['label']);rr={'id':cid,'label':c['label'],'prediction':vocab[int(top.indices[0])],'top5':[vocab[int(j)] for j in top.indices],'target_probability':float(p[tar])}
   if v in ['rtmw','old_fallback','old_only']:
    key={'rtmw':'rtmw','old_fallback':'crop_fallback','old_only':'crop_only'}[v]
    before=next(r for r in read(ROOT/'crop_stage1_20261011/results.json')['predictions'][key] if r['id']==cid)
    assert rr['top5']==before['top5'] and abs(rr['target_probability']-before['target_probability'])<1e-5
   pred[v].append(rr);inp[cid+'_'+v]=t.numpy();logits[cid+'_'+v]=out.numpy()
  print('SCORE',cid,c['label'],vs.keys(),flush=True)
 np.savez_compressed(O/'model_inputs.npz',**inp);np.savez_compressed(O/'logits.npz',**logits)
 refs={c['id']:{f['index']:f for f in c['samples']} for c in read(B/'coordinate_reference.json')};details=[];counts={'old':0,'tight':0}
 for c in manifest:
  cid=c['id'];hh={n:np.load(p/f'{cid}.npz') for n,p in [('old',H),('tight',O)]}
  for n,h in hh.items():counts[n]+=int(h['accepted'].sum())
  for i,f in enumerate(c['frames']):
   ref=refs[cid][f['sourceIndex']];assert ref['pngMatches'] and ref['pixelRatio']==1
   for side,name in enumerate(['left','right']):
    if not all(str(j) in ref['hands'][name] for j in range(21)):continue
    g=np.array([ref['hands'][name][str(j)] for j in range(21)]);d={'id':cid,'frame':i,'side':name}
    for n,h in hh.items():
     box=h['boxes'][i,side];mid=(box[:2]+box[2:])/2;half=(box[2:]-box[:2])*1.25/2;q=h['points'][i,side]
     d[n]={'full':bool(box[2]>box[0] and np.all((g>=mid-half)&(g<=mid+half))), 'accepted':bool(h['accepted'][i,side]),
      'err2':float(np.mean(np.sum((q-g)**2,axis=1))), 'wrist2':float(np.sum((q[0]-g[0])**2)),
      'shape2':float(np.mean(np.sum(((q-q[0])[1:]-(g-g[0])[1:])**2,axis=1)))}
    details.append(d)
 paired=[d for d in details if all(d[n]['accepted'] and d[n]['full'] for n in ['old','tight'])]
 geometry={n:{'reference_hand_frames':len(details),'full_coverage':sum(d[n]['full'] for d in details),'accepted_reference':sum(d[n]['accepted'] for d in details),'accepted_all':counts[n],
  'paired_n':len(paired),**{key+'_rmse_px':float(np.sqrt(np.mean([d[n][key+'2'] for d in paired]))) if paired else None for key in ['err','wrist','shape']}} for n in ['old','tight']}
 result={'summary':{v:ev.metrics(rr) for v,rr in pred.items()},'predictions':pred,'geometry':geometry,'details':details,
 'groups':{group:{v:ev.metrics([r for r in rr if (int(r['id'])<6)==first]) for v,rr in pred.items()} for group,first in [('six',True),('remaining24',False)]},'baseline_reproduced':True}
 save(O/'results.json',result);protocol=read(O/'protocol.json');protocol['status']='completed';save(O/'protocol.json',protocol)
 lines=['# Wrist crop เล็กลง 20%','', 'ลดกว้างสูง .8 คงศูนย์กลาง padding1.25 confidence.30 และกฎ fallback เดิม ตรวจ cache checksum และ baseline ซ้ำ ไม่แก้ production; 30 คลิปพัฒนาเดิม ไม่ใช่ independent test',
 '| วิธี | ถูก/30 | Accuracy % | Macro-F1 % | Top5 % |','|---|---:|---:|---:|---:|']
 for v,m in result['summary'].items():lines.append(f"| {v} | {m['correct']} | {m['accuracy']:.2f} | {m['f1']:.2f} | {m['top5']:.2f} |")
 lines+=['','## Geometry','```json',json.dumps(geometry,indent=2),'```','Geometry ใช้ Renderer reference หลัง inference เท่านั้น paired นับ hand-frame ที่กรอบทั้งสองครอบครบและทั้งสองวิธียอมรับผล จึงต้องดู coverage และการยอมรับควบคู่ ไม่ใช่ผลบนทุกเฟรม',
 'Macro เฉลี่ย true labels 30 คำ ไม่ใช่ทั้ง 2000 classes; shape ย้ายข้อมือแล้ววัด20จุด ยังรวมทิศทาง/ขนาด ไม่ใช่ความเข้าใจ ASL']
 (O/'REPORT_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(result['summary']),flush=True);print(json.dumps(geometry),flush=True)
if __name__=='__main__':score() if len(sys.argv)>1 and sys.argv[1]=='score' else extract()
