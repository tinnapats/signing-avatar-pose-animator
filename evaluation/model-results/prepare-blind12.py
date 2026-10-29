
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json,hashlib,shutil
from pathlib import Path
from PIL import Image,ImageDraw
D=Path(__file__).resolve().parent;S=D/'hand-reference60-20261014';O=D/'hand-blind12-20261014'
read=lambda p:_local_json_loads(p.read_text(encoding='utf-8'))
samples=read(S/'manifest.json')['samples'];draft=read(S/'assistant-draft.json')['records'];pairs=list(zip(samples,draft))
targets=['unclear']*2+['overlap']*4+['bent']*6
def select(i,used,chosen):
 if i==len(targets):return chosen
 pool=sorted((s for s,r in pairs if r['category']==targets[i] and s['clip'] not in used),key=lambda s:hashlib.sha256(('blind12-v1:'+s['id']).encode()).hexdigest())
 for s in pool:
  found=select(i+1,used|{s['clip']},chosen+[s])
  if found:return found
 return None
chosen=select(0,set(),[]);assert chosen and len({s['clip'] for s in chosen})==12
chosen=sorted(chosen,key=lambda s:hashlib.sha256(('blind12-order:'+s['id']).encode()).hexdigest())
O.mkdir(exist_ok=True);manifest=dict(schema='avatar-hand-reference-v1',dataset_id='hand-blind12-20261014',status='UNANNOTATED_NOT_GROUND_TRUTH',samples=chosen)
(O/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(O/'data.js').write_text('window.REFERENCE_DATA='+json.dumps(manifest)+';',encoding='utf-8')
shutil.copyfile(S/'core.js',O/'core.js')
html=(S/'index.html').read_text(encoding='utf-8');app=(S/'app.js').read_text(encoding='utf-8')
# Package the existing annotation interface mechanically; no source-project edits.
html=html.replace('60 ภาพ','12 ภาพ').replace('<div class="bar"><button id="loadDraft">โหลดฉบับร่าง AI เพื่อแก้ไข (ยังไม่ตรวจยืนยัน)</button></div>','').replace('<script src="assistant-draft-data.js"></script>','')
app='\n'.join(line for line in app.splitlines() if not line.startswith("$('loadDraft').onclick="))
app=app.replace('/60','/${data.samples.length}').replace('Math.min(59,index+1)','Math.min(data.samples.length-1,index+1)').replace('hand-reference60-','hand-blind12-')
app=app.replace("key=data.dataset_id+'-draft-v1'","key=data.dataset_id+'-'+window.REVIEWER_SLOT+'-draft-v1'")
app=app.replace("coordinate_system:'original_full_image_pixels'","reviewer_slot:window.REVIEWER_SLOT,coordinate_system:'original_full_image_pixels'")
app=app.replace("const incoming=C.validate(p,data);","if(p.reviewer_slot!==window.REVIEWER_SLOT)throw Error('ไฟล์มาจากผู้ตรวจคนละช่อง');const incoming=C.validate(p,data);")
app=app.replace('hand-blind12-${new Date()', 'hand-blind12-${window.REVIEWER_SLOT}-${new Date()')
for slot in ['A','B']:
 folder=O/('reviewer-'+slot);(folder/'images').mkdir(parents=True,exist_ok=True)
 for s in chosen:
  for field in ['image','full']:shutil.copyfile(S/s[field],folder/s[field])
 (folder/'data.js').write_text('window.REVIEWER_SLOT='+json.dumps(slot)+';window.REFERENCE_DATA='+json.dumps(manifest)+';',encoding='utf-8')
 shutil.copyfile(S/'core.js',folder/'core.js');(folder/'app.js').write_text(app,encoding='utf-8');(folder/'index.html').write_text(html.replace('<h1>','<h1>ผู้ตรวจ '+slot+' — '),encoding='utf-8')
sheet=Image.new('RGB',(1200,900),'white');draw=ImageDraw.Draw(sheet)
for i,s in enumerate(chosen):
 im=Image.open(S/s['image']);im.thumbnail((280,250));x=i%4*300;y=i//4*300;sheet.paste(im,(x,y+30));draw.text((x,y),f"{i+1:02d} {s['id']}",fill='black')
sheet.save(O/'contact-sheet.png')
print('12 distinct clips; no coordinates, AI categories, word labels, or model predictions in reviewer packages')
