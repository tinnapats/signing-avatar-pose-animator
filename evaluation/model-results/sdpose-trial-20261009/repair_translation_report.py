"""Repair reporting-only key collision, without changing predictions or model inputs."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;O=ROOT/'wrist_translation_20261011'
r=_local_json_loads((O/'results.json').read_text());m=np.load(O/'translated_masks.npz')
manifest=_local_json_loads(Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit/pixels30_20261007/capture_manifest.json').read_text());ids={c['label']:c['id'] for c in manifest}
for d in r['details']:d['did_translate']=bool(m[ids[d['label']]][d['frame'],0 if d['side']=='left' else 1])
ds=[d for d in r['details'] if d['did_translate']];assert len(ds)==207
g={'n':len(ds),**{v:{k+'_rmse_px':float(np.sqrt(np.mean([d[v][k+'_sq'] for d in ds]))) for k in ['total','wrist','shape']} for v in ['baseline','translated']}}
old=r['geometry_translated_reference'];r['geometry_translated_reference']=g
(O/'results.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
p=O/'REPORT_TH.md';text=p.read_text(encoding='utf-8');assert json.dumps(old,indent=2) in text
text=text.replace(json.dumps(old,indent=2),json.dumps(g,indent=2))
text+='\n\nข้อสรุป: คะแนนยัง5/30เท่าเดิม การย้ายข้อมือช่วยตำแหน่งข้อมือใน207hand-framesที่ตรวจได้ แต่ทำให้พิกัดมือรวมแย่ลง รูปมือสัมพัทธ์ไม่เปลี่ยนตามที่ออกแบบ จึงไม่แนะนำใช้การเลื่อนนี้เป็นค่าเริ่มต้น\n'
p.write_text(text,encoding='utf-8')
(O/'report_correction.json').write_text(json.dumps({'reason':'Reporting key translated was used for both boolean and metric dict. Reconstructed did_translate from saved masks; no inference change.','affected_subset_n_before':old['n'],'after':len(ds),'corrected_script_sha256':hashlib.sha256((ROOT/'wrist_translation_trial.py').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
print(json.dumps(g))
