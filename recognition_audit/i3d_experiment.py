"""Canvas pixels only; WLASL I3D. No source-pose inference or model tuning."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parent/'i3d_avatar/deps'))
import numpy as np
import torch
from PIL import Image, ImageOps
from step1_evaluate import hashes

HERE=Path(__file__).resolve().parent
OUT=HERE/'i3d_avatar'
CKPT='archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
    assert not (OUT/'protocol.json').exists()
    cases=_local_json_loads((HERE/'inputs.json').read_text());assert len(cases)==30
    splits=_local_json_loads((OUT/'nslt_2000.json').read_text())
    meta=_local_json_loads((HERE/'kaggle_reference/WLASL_v0.3.json').read_bytes())
    labels={}
    for r in meta:
        ids={splits[i['video_id']]['action'][0] for i in r['instances'] if i['video_id'] in splits}
        assert len(ids)==1,(r['gloss'],ids)
        k=ids.pop();assert k not in labels;labels[k]=r['gloss']
    assert set(labels)==set(range(2000))
    assert all(c['phrase'] in labels.values() for c in cases)
    (OUT/'labels.json').write_text(json.dumps([labels[i] for i in range(2000)]))
    protocol=dict(words=[c['phrase'] for c in cases],production_hashes=hashes(),checkpoint=CKPT,
        archive_sha256=digest(OUT/'archived.zip'),inputs_sha256=digest(HERE/'inputs.json'),
        preprocessing='Production canvas only (no UI labels), white alpha composite; 25 FPS nearest original timestamps; resize square 513 to 256 then center crop224; BGR channels to match upstream nslt_dataset_all OpenCV video loader; [-1,1]; full temporal clip; pad last frame to minimum16 if needed; temporal max logits as official test run.',
        metric='One clip per 30 selected labels. Macro P/R/F1 over these target labels, zero_division=0, unrestricted 2000-class predictions. Accuracy and top5 separately. No score-based exclusion.',
        scope='Exploratory production-avatar pilot. No human video/source keypoints passed to model. No training or parameter search. Domain shift and possible training overlap of source motions remain.',
        code_sha256={p.name:digest(p) for p in [Path(__file__),HERE/'i3d_capture.mjs',OUT/'pytorch_i3d.py']})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));print('Frozen 30 production clips, full 2000 labels, no answer text',flush=True)

def run():
    protocol=_local_json_loads((OUT/'protocol.json').read_text());assert hashes()==protocol['production_hashes']
    assert digest(OUT/'archived.zip')==protocol['archive_sha256']
    torch.set_num_threads(4)
    spec=importlib.util.spec_from_file_location('official_i3d',OUT/'pytorch_i3d.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    model=mod.InceptionI3d(2000,in_channels=3)
    with zipfile.ZipFile(OUT/'archived.zip') as z:state=torch.load(io.BytesIO(z.read(CKPT)),map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True);model.eval()
    labels=_local_json_loads((OUT/'labels.json').read_text());captures=_local_json_loads((OUT/'capture_manifest.json').read_text())
    assert [r['phrase'] for r in captures]==protocol['words']
    rows=[]
    for case in captures:
        paths=sorted((OUT/'frames'/case['id']).glob('*.png'));assert len(paths)==case['frames']
        frames=[]
        for p in paths:
            rgba=Image.open(p).convert('RGBA');bg=Image.new('RGBA',rgba.size,'white');bg.alpha_composite(rgba)
            rgb=ImageOps.pad(bg.convert('RGB'),(256,256),method=Image.Resampling.BILINEAR,color='white').crop((16,16,240,240))
            frames.append(np.asarray(rgb)[:,:,::-1].copy())
        while len(frames)<16:frames.append(frames[-1])
        array=np.asarray(frames,dtype=np.float32)/127.5-1
        tensor=torch.from_numpy(array.transpose(3,0,1,2).copy()).unsqueeze(0)
        start=time.perf_counter()
        with torch.inference_mode():logits=model(tensor).max(dim=2).values[0];probs=logits.softmax(0)
        indices=probs.topk(5).indices.tolist()
        row=dict(id=case['id'],label=case['phrase'],prediction=labels[indices[0]],top5=[labels[i] for i in indices],
                 probability=float(probs[indices[0]]),target_probability=float(probs[labels.index(case['phrase'])]),frames=case['frames'],inference_seconds=time.perf_counter()-start,
                 frame_hashes={p.name:digest(p) for p in paths})
        rows.append(row);(OUT/'predictions.json').write_text(json.dumps(rows,indent=2))
        print('predicted',len(rows),case['phrase'],'->',row['prediction'],'top5',row['top5'],flush=True)
    ps=[];rs=[];fs=[];perclass=[]
    for label in protocol['words']:
        tp=sum(r['label']==label==r['prediction'] for r in rows);fp=sum(r['label']!=label and r['prediction']==label for r in rows);fn=sum(r['label']==label and r['prediction']!=label for r in rows)
        p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0;f=2*p*r/(p+r) if p+r else 0
        ps.append(p);rs.append(r);fs.append(f);perclass.append(dict(label=label,tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=f))
    summary=dict(n=len(rows),correct=sum(r['label']==r['prediction'] for r in rows),accuracy=100*np.mean([r['label']==r['prediction'] for r in rows]),
        macro_precision=100*np.mean(ps),macro_recall=100*np.mean(rs),macro_f1=100*np.mean(fs),top5=100*np.mean([r['label'] in r['top5'] for r in rows]),production_unchanged=hashes()==protocol['production_hashes'])
    assert summary['production_unchanged']
    (OUT/'results.json').write_text(json.dumps(dict(summary=summary,perclass=perclass,predictions=rows),indent=2))
    lines=['# I3D: รู้จำภาพแอนิเมชันสุดท้ายของระบบ','',
        'ใช้ Canvas จริงของ production ปัจจุบัน ไม่ใช้ S50A1 ไม่ส่งวิดีโอมนุษย์หรือพิกัด JSON เข้า recognizer ไม่ให้เห็น Current word/Output', '',
        '| จำนวนคลิป | ถูก | Accuracy | Macro Precision | Macro Recall | Macro F1 | Top-5 |','|---:|---:|---:|---:|---:|---:|---:|',
        f"| {len(rows)} | {summary['correct']} | {summary['accuracy']:.2f}% | {summary['macro_precision']:.2f}% | {summary['macro_recall']:.2f}% | {summary['macro_f1']:.2f}% | {summary['top5']:.2f}% |",'',
        'Macro เฉลี่ยเฉพาะ 30 target labels หารศูนย์เป็น0 โมเดลเลือกได้ครบ2000คำ ไม่บังคับเลือกเฉพาะคำทดสอบ ไม่ใช่ macro บน2000 labels',
        'หนึ่งคลิปต่อคำ เป็น pilot ไม่ใช่การรับรองทั่วไป คะแนนนี้คือความสามารถของ I3D ในการอ่านภาพ Avatar ไม่ใช่เปอร์เซ็นต์ความถูกต้อง ASL หรือความเข้าใจมนุษย์',
        'ฝึกจากมนุษย์แต่ทดสอบการ์ตูน มี domain shift ภาพเต็มตัวทำให้มือเล็กหลัง resize ใช้ preprocessing ตรึงไว้ ไม่ลองเปลี่ยนเพื่อเลือกคะแนนสูง',
        'RGB หมายถึงชนิด stream; โค้ด loader ทางการอ่านวิดีโอเป็น OpenCV BGR จึงรักษาลำดับ BGR สำหรับ checkpoint นี้ พร้อมบันทึกวิธีทั้งหมดใน protocol.json',
        'ไม่ฝึกเพิ่ม ไม่แก้ระบบหลัก ไม่แก้ SQLite; ตัวเลขมาจาก predictions.json และ results.json',
        'แหล่ง checkpoint/code: https://github.com/dxli94/WLASL', '']
    (OUT/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
