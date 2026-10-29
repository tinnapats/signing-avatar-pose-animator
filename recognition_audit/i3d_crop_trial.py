"""One frozen pixel-only crop per clip. No score-based crop search."""
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import time
import zipfile
import i3d_experiment as base
import numpy as np
import torch
from PIL import Image, ImageOps
from step1_evaluate import hashes

DATA=base.OUT
OUT=DATA.parent/'i3d_upperbody_trial'

def skin_bounds(rgba):
    a=np.asarray(rgba,dtype=np.float32);r,g,b,alpha=[a[:,:,i] for i in range(4)]
    mask=(alpha>=128)&(r>180)&(r>1.1*g)&(r>1.1*b)
    y,x=np.where(mask)
    return (int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1,int(mask.sum())) if len(x) else None

def image_for_model(path,box):
    rgba=Image.open(path).convert('RGBA');bg=Image.new('RGBA',rgba.size,'white');bg.alpha_composite(rgba)
    x,y,side=box
    crop=Image.new('RGB',(side,side),'white');crop.paste(bg.convert('RGB'),(-x,-y))
    return crop.resize((256,256),Image.Resampling.BILINEAR).crop((16,16,240,240))

def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'Protocol already frozen'
    captures=json.loads((DATA/'capture_manifest.json').read_text());cases=[]
    for c in captures:
        paths=sorted((DATA/'frames'/c['id']).glob('*.png'));assert len(paths)==c['frames']
        boxes=[]
        for p in paths:
            bounds=skin_bounds(Image.open(p).convert('RGBA'))
            if bounds:boxes.append(bounds)
        assert boxes,'No visible skin for '+c['id']
        l=min(b[0] for b in boxes);t=min(b[1] for b in boxes);r=max(b[2] for b in boxes);b=max(b[3] for b in boxes)
        side=int(math.ceil(1.30*max(r-l,b-t)));x=math.floor((l+r-side)/2);y=math.floor((t+b-side)/2)
        # CenterCrop224 from256 still retains every detected skin pixel.
        assert x+side/16<=l and y+side/16<=t and x+15*side/16>=r and y+15*side/16>=b
        cases.append(dict(**c,box=[x,y,side],skin_union=[l,t,r,b],skin_detected_frames=len(boxes),
            magnification=c['width']/side,frame_hashes={p.name:base.digest(p) for p in paths}))
        for index in sorted({0,len(paths)//2,len(paths)-1}):
            image_for_model(paths[index],[x,y,side]).save(OUT/(c['id']+'_'+str(index)+'.png'))
        print('crop frozen',c['id'],c['phrase'],[x,y,side],flush=True)
    protocol=dict(cases=cases,production_hashes=hashes(),checkpoint_sha256=base.digest(DATA/'archived.zip'),
        baseline_results_sha256=base.digest(DATA/'results.json'),code_sha256=base.digest(Path(__file__)),
        method='Union of visible skin-color pixels over all frames (alpha>=128,R>180,R>1.1G,R>1.1B); one square crop per clip, centered on union with side=ceil(1.30*max(width,height)); white pad outside canvas; resize256 center224. All detected skin retained after center crop. No per-frame tracking, landmarks, class-dependent rules or score tuning.',
        pre_score_correction='Unscored first preparation in i3d_crop_trial mistakenly included shoe pixels (e.g. hello RGB133,120,116 at y769). Corrected alpha/R threshold before any crop inference; prior preparation retained, no model scores used.',
        fixed='Same 30 clips, original25fps and frame order, BGR, [-1,1], same2000-class checkpoint, strict weights load, max temporal logits; all else as full-canvas baseline.',
        limitations='Skin heuristic fits this avatar palette; it is not an independently validated hand detector. Preview inspection and pixel containment do not prove full hand visibility or semantic correctness. Same exploratory clips, not a new final test.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))

def run():
    protocol=json.loads((OUT/'protocol.json').read_text());assert base.digest(Path(__file__))==protocol['code_sha256']
    assert hashes()==protocol['production_hashes']
    assert base.digest(DATA/'archived.zip')==protocol['checkpoint_sha256']
    assert base.digest(DATA/'results.json')==protocol['baseline_results_sha256']
    spec=importlib.util.spec_from_file_location('crop_i3d',DATA/'pytorch_i3d.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    torch.set_num_threads(4);model=mod.InceptionI3d(2000,in_channels=3)
    with zipfile.ZipFile(DATA/'archived.zip') as z:state=torch.load(io.BytesIO(z.read(base.CKPT)),map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True);model.eval();labels=json.loads((DATA/'labels.json').read_text());rows=[]
    for c in protocol['cases']:
        paths=sorted((DATA/'frames'/c['id']).glob('*.png'));assert len(paths)==c['frames']
        assert all(base.digest(p)==c['frame_hashes'][p.name] for p in paths)
        frames=[np.asarray(image_for_model(p,c['box']))[:,:,::-1].copy() for p in paths]
        while len(frames)<16:frames.append(frames[-1])
        tensor=torch.from_numpy((np.asarray(frames,dtype=np.float32)/127.5-1).transpose(3,0,1,2).copy()).unsqueeze(0)
        start=time.perf_counter()
        with torch.inference_mode():probs=model(tensor).max(dim=2).values[0].softmax(0)
        ids=probs.topk(5).indices.tolist();row=dict(id=c['id'],label=c['phrase'],prediction=labels[ids[0]],top5=[labels[i] for i in ids],target_probability=float(probs[labels.index(c['phrase'])]),inference_seconds=time.perf_counter()-start)
        rows.append(row);(OUT/'predictions.json').write_text(json.dumps(rows,indent=2));print('predicted',len(rows),c['phrase'],'->',row['prediction'],row['top5'],flush=True)
    ps=[];rs=[];fs=[];details=[]
    for c in protocol['cases']:
        label=c['phrase'];tp=sum(r['label']==label==r['prediction'] for r in rows);fp=sum(r['label']!=label and r['prediction']==label for r in rows);fn=sum(r['label']==label and r['prediction']!=label for r in rows)
        p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0;f=2*p*r/(p+r) if p+r else 0
        ps.append(p);rs.append(r);fs.append(f);details.append(dict(label=label,tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=f))
    summary=dict(n=len(rows),correct=sum(r['label']==r['prediction'] for r in rows),accuracy=100*np.mean([r['label']==r['prediction'] for r in rows]),macro_precision=100*np.mean(ps),macro_recall=100*np.mean(rs),macro_f1=100*np.mean(fs),top5=100*np.mean([r['label'] in r['top5'] for r in rows]))
    assert hashes()==protocol['production_hashes'] and base.digest(DATA/'results.json')==protocol['baseline_results_sha256']
    (OUT/'results.json').write_text(json.dumps(dict(summary=summary,perclass=details,predictions=rows,production_unchanged=True),indent=2))
    old=json.loads((DATA/'results.json').read_text())['summary']
    lines=['# I3D: ภาพเต็มตัวเทียบครอปส่วนบนที่ครอบคลุมมือ','',
        '30 คลิปเดิมจาก Canvas จริง ไม่แก้ Avatar ไม่ทายต้นทาง ไม่ฝึกเพิ่ม กำหนดกรอบทั้งหมดก่อนเห็นคะแนน crop', '',
        '| ภาพ | ถูก/30 | Accuracy | Macro-P | Macro-R | Macro-F1 | Top-5 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,s in [('เต็มตัว (ผลเดิม)',old),('ครอป',summary)]:lines.append(f"| {name} | {s['correct']} | "+' | '.join(f"{s[k]:.2f}%" for k in ['accuracy','macro_precision','macro_recall','macro_f1','top5'])+' |')
    lines+=['', 'กรอบต่อคลิปคงที่ ครอบคลุม union ของพิกเซลสีผิวตลอดคลิป + margin30% ไม่มีการเลือกกรอบรายคำด้วยคะแนน ไม่ยืดภาพ ไม่เปลี่ยนสี/ช่องสี/FPS/การรวม logits',
        'ตรวจทุกเฟรมว่าพิกเซลสีผิวที่ตรวจพบยังอยู่ในภาพหลัง center crop แต่ mask สีไม่ใช่ตัวตรวจมือที่ผ่านการยืนยัน ต้องอ่านข้อจำกัดใน protocol.json',
        'Macro เฉลี่ย30 target labels; zero division=0; โมเดลทายได้ครบ2000คลาส ไม่ตัดคำผิดออก',
        'ผลเป็นการรู้จำภาพ Avatar โดย I3D ไม่ใช่ความถูกต้อง ASL หรือความเข้าใจของคน ยังแยก domain shift กับข้อจำกัดของ checkpoint/preprocessing ไม่ได้ทั้งหมด',
        'กรอบและ hash ภาพเดิมเก็บใน protocol.json ผลรายคำใน predictions.json ไม่แก้ผล full-canvas เดิมหรือ production', '']
    (OUT/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
