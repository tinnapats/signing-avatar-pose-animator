import json,hashlib,shutil,sqlite3,zipfile,random
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;B=D/'oracle200-20261011';P=D/'pixel-only-plan-20261012';O=D/'independent-check-20261012';A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
assert not (O/'annotations.json').exists(),'Do not overwrite annotation work'
O.mkdir(exist_ok=True);(O/'images').mkdir(exist_ok=True)
manifest=read(B/'capture_manifest.json');byid={c['id']:c for c in manifest};review={'149','125','195','045','122','147','178','057'}
uniform=sorted([c for c in manifest if c['id'] not in review],key=lambda c:hashlib.sha256(('reference-v2-'+c['id']).encode()).hexdigest())[:6]
selection={}
for c in uniform:
 for i in sorted(set([len(c['frames'])//4,3*len(c['frames'])//4])):selection[(c['id'],i)]='uniform'
flagged=[]
for c in manifest:
 z=np.load(P/'arrays'/(c['id']+'.npz'));indices=np.where(z['flags'].any(1))[0]
 if len(indices):flagged.append((c['id'],int(indices[len(indices)//2])))
for key in sorted(flagged,key=lambda p:hashlib.sha256(str(p).encode()).hexdigest())[:6]:selection.setdefault(key,'flagged')
keys=[side+'_'+j for side in ['left','right'] for j in ['shoulder','elbow']]+[f'{side}_hand_{i:02d}' for side in ['left','right'] for i in range(21)]
names=['ไหล่ซ้าย','ศอกซ้าย','ไหล่ขวา','ศอกขวา']+[('มือซ้าย' if s=='left' else 'มือขวา')+f' จุด {i}' for s in ['left','right'] for i in range(21)]
items=[];provenance=[]
for n,((cid,i),stratum) in enumerate(sorted(selection.items(),key=lambda p:hashlib.sha256(str(p[0]).encode()).hexdigest())):
 c=byid[cid];f=c['frames'][i];src=B/'frames'/cid/f['file'];assert sha(src)==f['sha256'];imageid=f'image_{n+1:03d}';dest=O/'images'/(imageid+'.png');shutil.copyfile(src,dest)
 items.append({'id':imageid,'label':imageid,'frame':n,'image':'images/'+dest.name,'sha256':sha(dest),'width':c['width'],'height':c['height'],'strata':['blinded_review'],'annotation_status':'pending','annotator':None,'reviewer':None,'joints':{k:{'x':None,'y':None,'visibility':'unreviewed','uncertainty_px':None,'note':'','review_status':'pending'} for k in keys}})
 provenance.append({'id':imageid,'clip_id':cid,'label':c['label'],'frame':i,'time':f['time'],'stratum':stratum})
seed={'version':2,'status':'DRAFT_NOT_VERIFIED','coordinate_system':'Original image pixels, anatomical left/right. Hand0 wrist; 1-4 thumb,5-8 index,9-12 middle,13-16 ring,17-20 pinky, proximal to tip. No detector/reference prefill.','items':items}
save(O/'annotations.json',seed);save(O/'selection_provenance.json',provenance)
(O/'seed.js').write_text('window.REFERENCE_SEED='+json.dumps(seed,ensure_ascii=False)+';',encoding='utf-8')
old=D/'body-reference-v1-20261009';js=(old/'annotator.js').read_text(encoding='utf-8');lines=js.splitlines();lines[2]='const keys='+json.dumps(keys)+';';lines[3]='const names='+json.dumps(names,ensure_ascii=False)+';';js='\n'.join(lines).replace('avatar-reference-v1-','avatar-hand-reference-v2-').replace('data.items.length*6','data.items.length*keys.length').replace("n<3?'#086eb8'","k.startsWith('left')?'#086eb8'").replace("a.download='body-reference-'","a.download='hand-reference-'" )
(O/'annotator.js').write_text(js,encoding='utf-8')
html=(old/'annotator.html').read_text(encoding='utf-8').replace('ลงจุดอ้างอิงไหล่–ศอก–ข้อมือ','ตรวจข้อต่อมือ 21 จุดต่อข้าง และไหล่–ศอก').replace('ปุ่ม 1–6 เลือกจุด','ปุ่ม 1–6 เลือกหกจุดแรก').replace('.joints{display:grid;','.joints{max-height:260px;overflow:auto;display:grid;').replace('<p>ข้อมือ:','<p>มือจุด 0 = ข้อมือ; 1–4 นิ้วโป้ง; 5–8 นิ้วชี้; 9–12 นิ้วกลาง; 13–16 นิ้วนาง; 17–20 นิ้วก้อย เรียงจากโคนนิ้วถึงปลาย จุดที่ไม่เห็นห้ามเดา</p><p>ข้อมือ:')
(O/'annotator.html').write_text(html,encoding='utf-8')
# Read-only database inventory; exclude labels found in historical experiment manifests/results.
vocab={x['gloss'] for x in json.loads(zipfile.ZipFile(A/'wlasl_metadata.zip').read('splits/asl2000.json'))};seen={c['label'] for c in manifest};sources={};skipped=[]
fields={'label','phrase','actual','gloss','labels_evaluated','labels','original_words','added_words','words','original_labels','added_labels'}
def collect(x,key=''):
 if isinstance(x,str) and key in fields and x in vocab:seen.add(x)
 elif isinstance(x,dict):
  for k,v in x.items():
   if k not in {'pool','excluded','available','vocabulary','top5','prediction','predicted','predictions_by_class'}:collect(v,k)
 elif isinstance(x,list):
  for v in x:collect(v,key)
for root in [A,D]:
 for p in sorted(root.rglob('*.json')):
  if O in p.parents or any(s in p.parts for s in ['node_modules','site-packages','model_cache','frames','cases','inference','arrays','rtm']):continue
  if not any(w in p.name.lower() for w in ['result','protocol','provenance','manifest','diagnosis','inventory']):continue
  if p.stat().st_size>20_000_000:skipped.append(str(p));continue
  try:collect(read(p));sources[str(p)]=sha(p)
  except (ValueError,UnicodeError):skipped.append(str(p))
db=A.parent/'SLclean.sqlite3'
with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as con:
 rows=[{'id':i,'label':l,'relative_path':p,'sha256':h,'frame_count':f} for i,l,p,h,f in con.execute('select id,label,relative_path,sha256,frame_count from clips')]
pool=[r for r in rows if r['label'] in vocab-seen];bylabel={}
for r in sorted(pool,key=lambda r:r['relative_path']):bylabel.setdefault(r['label'],r)
chosen=random.Random(20261012).sample(sorted(bylabel),min(30,len(bylabel)))
save(O/'reserved30.json',{'status':'RESERVED_NOT_SCORED','seed':20261012,'labels':chosen,'clips':[bylabel[l] for l in chosen],'excluded_labels':sorted(seen),'eligible_labels':len(bylabel),'scanned_files':sources,'skipped_files':skipped,'scope':'Absent from parsed known reports, not proof of globally unseen history or model-training independence. No recognition performed; do not tune after opening confirmation scores. Database read-only.','candidate_sha256':sha(D/'pixel-support-trial.py'),'database_size':db.stat().st_size})
print(json.dumps({'images':len(items),'points_per_image':len(keys),'scanned_reports':len(sources),'excluded_labels':len(seen),'eligible':len(bylabel),'reserved':len(chosen),'skipped':len(skipped)}),flush=True)
