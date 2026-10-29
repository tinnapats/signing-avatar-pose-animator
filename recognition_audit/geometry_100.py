"""Paired 2-D hand fidelity at original 30 Hz timestamps; no production edits."""
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from run_audit import pipeline, frame_array
from step1_evaluate import hashes

HERE=Path(__file__).resolve().parent
DATA=HERE/'recognition_100'
OUT=HERE/'geometry_100'
CHAINS=[[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
EDGES=[(a,b) for chain in CHAINS for a,b in zip(chain,chain[1:])]
TRIPLES=[tuple(chain[i:i+3]) for chain in CHAINS for i in range(3)]

def angle(a,b,c):
    u=a-b; v=c-b; d=np.linalg.norm(u)*np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/d,-1,1)))) if d>1e-10 else None

def metric(ref,pred,mask,shoulder):
    valid=mask & np.isfinite(pred).all(axis=1)
    distances=np.linalg.norm(pred[valid]-ref[valid],axis=1)/shoulder
    palm=np.linalg.norm(ref[9]-ref[0])
    shape=[]; bone=[]; angles=[]
    if valid[0] and palm>1e-6:
        ids=np.flatnonzero(valid); ids=ids[ids!=0]
        shape=(np.linalg.norm((pred[ids]-pred[0])-(ref[ids]-ref[0]),axis=1)/palm).tolist()
    for a,b in EDGES:
        if valid[a] and valid[b]:
            length=np.linalg.norm(ref[b]-ref[a])
            if length>max(1e-6,palm*.01):
                bone.append(float(abs(np.linalg.norm(pred[b]-pred[a])-length)/length))
    for a,b,c in TRIPLES:
        if valid[[a,b,c]].all():
            x=angle(ref[a],ref[b],ref[c]); y=angle(pred[a],pred[b],pred[c])
            if x is not None and y is not None:angles.append(abs(x-y))
    return dict(expected=int(mask.sum()),available=int(valid.sum()),distance=distances.tolist(),
                within5=int((distances<=.05).sum()),shape=shape,bone=bone,angle=angles)

def summarize(items):
    expected=sum(i['expected'] for i in items); available=sum(i['available'] for i in items)
    out=dict(expected_points=expected,available_points=available,
        coverage_pct=100*available/expected if expected else None,
        position_pck5_pct=100*sum(i['within5'] for i in items)/expected if expected else None)
    for name in ['distance','shape','bone','angle']:
        values=[v for i in items for v in i[name]]
        out[name+'_n']=len(values)
        out[name+'_mean']=float(np.mean(values)) if values else None
    return out

def tests():
    ref=np.array([[i%5,i//5] for i in range(21)],dtype=float); mask=np.ones(21,dtype=bool)
    same=metric(ref,ref,mask,10)
    assert same['within5']==21 and max(same['bone'])==0 and max(same['angle'])==0
    shifted=metric(ref,ref+10,mask,10)
    assert shifted['within5']==0 and max(shifted['shape'])==0 and max(shifted['angle'])==0
    scaled=metric(ref,ref*2,mask,10)
    assert np.allclose(scaled['bone'],1) and np.allclose(scaled['angle'],0)
    missing=metric(ref,np.full((21,2),np.nan),mask,10)
    assert missing['available']==0 and summarize([missing])['position_pck5_pct']==0

def main():
    tests(); OUT.mkdir(exist_ok=True); before=hashes()
    original_protocol=json.loads((DATA/'protocol.json').read_text())
    assert before==original_protocol['production_hashes']
    cases=json.loads((DATA/'inputs.json').read_text())
    rendered={r['phrase']:r['samples'] for r in json.loads((DATA/'rendered.json').read_text())}
    allrows=[]; pooled={s:[] for s in ['source_to_processed','source_to_rendered','processed_to_rendered']}
    for case in cases:
        word=case['phrase']; state={}; build=pipeline.build_frames; fit=pipeline.fit_frames_to_canvas
        def capture_build(*a,**kw):
            out=build(*a,**kw); state['raw']=copy.deepcopy(out); return out
        def capture_fit(frames,width,height):
            a=np.array([[k['position']['x'],k['position']['y']] for f in frames for k in f['pose']['keypoints']])
            out=fit(frames,width,height)
            b=np.array([[k['position']['x'],k['position']['y']] for f in out for k in f['pose']['keypoints']])
            scale,dx=np.linalg.lstsq(np.column_stack([a[:,0],np.ones(len(a))]),b[:,0],rcond=None)[0]
            offset=np.array([dx,np.mean(b[:,1]-a[:,1]*scale)])
            assert np.max(np.abs(a*scale+offset-b))<1e-6
            state.update(scale=float(scale),offset=offset); return out
        with patch.object(pipeline,'build_frames',capture_build),patch.object(pipeline,'fit_frames_to_canvas',capture_fit):
            payload=pipeline.build_payload(DATA/'csv',[word+'.csv'],'',30,513,513,0,0,2,1.2,2)
        assert payload['frames']==case['payload']['frames'], word
        raw=state['raw']; indices=case['frameIndices']; captures=rendered[word]
        assert len(raw)==len(indices)==len(captures)
        ref=frame_array(raw)*state['scale']+state['offset']
        processed=frame_array([payload['frames'][i] for i in indices])
        items={s:[] for s in pooled}; observed=0; skipped_shoulders=0
        for t,frame in enumerate(raw):
            pose={k['part']:k for k in frame['pose']['keypoints']}
            shoulder=np.linalg.norm(ref[t,11]-ref[t,12])
            if shoulder<=1e-6 or min(pose[s]['score'] for s in ['leftShoulder','rightShoulder'])<=0:
                skipped_shoulders+=1; continue
            for side,offset in [('left',33),('right',54)]:
                h=frame.get('hands',{}).get(side,{})
                if not h.get('observed'):continue
                mask=np.array([k.get('score',0)>0 for k in h['keypoints']]); assert len(mask)==21
                observed+=1; a=ref[t,offset:offset+21]; b=processed[t,offset:offset+21].copy()
                processed_hand=payload['frames'][indices[t]]['hands'][side]
                for k,kp in enumerate(processed_hand['keypoints']):
                    if kp.get('score',0)<=0 or processed_hand.get('score',0)<=0:b[k]=np.nan
                c=np.full((21,2),np.nan)
                for key,point in captures[t]['hands'][side].items():c[int(key)]=point
                items['source_to_processed'].append(metric(a,b,mask,shoulder))
                items['source_to_rendered'].append(metric(a,c,mask,shoulder))
                items['processed_to_rendered'].append(metric(b,c,mask & np.isfinite(b).all(axis=1),shoulder))
        row=dict(word=word,frames=len(raw),observed_hand_frames=observed,skipped_shoulder_frames=skipped_shoulders,
                 global_fit=dict(scale=state['scale'],offset=state['offset'].tolist()),metrics={s:summarize(v) for s,v in items.items()})
        allrows.append(row)
        for s in pooled:pooled[s].extend(items[s])
        print('measured',len(allrows),word,observed,flush=True)
    assert hashes()==before
    result=dict(n_words=len(allrows),rows=allrows,pooled={s:summarize(v) for s,v in pooled.items()},production_unchanged=True,
        inputs_sha256=hashlib.sha256((DATA/'inputs.json').read_bytes()).hexdigest(),
        rendered_sha256=hashlib.sha256((DATA/'rendered.json').read_bytes()).hexdigest())
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    names={'source_to_processed':'ต้นทาง → ก่อน Renderer','source_to_rendered':'ต้นทาง → หลัง Renderer','processed_to_rendered':'ก่อน → หลัง Renderer'}
    lines=['# ความตรงของมือ 2D บน 100 คลิปเดิม','',
        'ใช้ complexity 1 เดิม ไม่มีการปรับ Avatar เปรียบเทียบ ณ เวลา 30 FPS เดียวกัน ใช้ global fit เดียวกับ output และตัดกล้องออกจากพิกัด Renderer ไม่มีการจัดแนวมือแต่ละข้างเพื่อให้คะแนนดีขึ้น','',
        '| คู่เปรียบเทียบ | Coverage | PCK@5% ไหล่ | ระยะคลาดเคลื่อน/ไหล่ | รูปมือคลาดเคลื่อน/ฝ่ามือ | ความยาวกระดูกคลาดเคลื่อน | มุมคลาดเคลื่อน |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s,m in result['pooled'].items():
        lines.append(f"| {names[s]} | {m['coverage_pct']:.2f}% | {m['position_pck5_pct']:.2f}% | {100*m['distance_mean']:.2f}% | {100*m['shape_mean']:.2f}% | {100*m['bone_mean']:.2f}% | {m['angle_mean']:.2f}° |")
    lines+=['','## วิธีอ่าน',
        '- PCK คือสัดส่วนจุดที่ห่างจากจุดอ้างอิงไม่เกิน 5% ของความกว้างไหล่ จุดที่ไม่แสดงนับว่าไม่ผ่าน ไม่ใช่เปอร์เซ็นต์ความถูกต้องภาษามือ',
        '- ค่า error ยิ่งน้อยยิ่งใกล้ข้อมูลอ้างอิง Shape ลบตำแหน่งข้อมือออกเท่านั้น ไม่ปรับขนาดหรือหมุนให้ตรง',
        '- กระดูก: ค่าเฉลี่ย absolute relative length error ของ 20 ท่อนในภาพ 2D ข้ามท่อนอ้างอิงที่สั้นกว่า 1% ฝ่ามือเพื่อลดปัญหาการหารด้วยค่าจิ๋ว ไม่ใช่ความยาวจริง 3D',
        '- มุม: absolute error ของมุมในภาพ 2D จำนวน 15 มุมต่อมือ ไม่ใช่มุมข้อทางกายวิภาค 3D',
        '- ตารางรวมถ่วงตามจำนวนตัวอย่าง ไม่ใช่เฉลี่ย 100 คำเท่ากัน ผลรายคำและจำนวนจุด/ท่อน/มุมอยู่ใน results.json',
        '- ใช้เฉพาะมือ observed ที่ต้นทางและหัวไหล่ใช้ได้ ค่า error คำนวณจุดที่มีอยู่ ส่วน coverage/PCK แสดงผลของจุดหาย',
        '- คู่ก่อน→หลัง Renderer ใช้ mask เฉพาะจุด processed ที่มีอยู่ จึงมี denominator ต่างได้ ต้องดูจำนวนตัวอย่างประกอบ', '',
        '## สิ่งที่ยังสรุปไม่ได้',
        '- ไม่วัดภาพทั้งตัว ผิวมือ การบังกัน สี หรือความโปร่งใส เป็นตำแหน่งเส้นแกนมือที่วาดจริงจาก audit capture',
        '- การเปลี่ยนขนาดมือและยึดข้อมือกับแขนถือเป็นความต่างจากข้อมูล แม้อาจช่วยความสมส่วนทางภาพ',
        '- แหล่งอ้างอิงเป็นจุดที่สกัดจากวิดีโอ ไม่ใช่ ground truth ที่มนุษย์ตรวจ จุดต้นทางอาจผิด',
        '- ไม่วัดเฟรมที่เพิ่มระหว่างเวลาเดิม เพราะไม่มี reference อิสระ และไม่ใช่ผลความเข้าใจของผู้ใช้',
        '- ไม่เปลี่ยน source/renderer/SQLite; rebuild payload ตรงผลเดิมทุกเฟรม; synthetic checks ผ่าน identity/translation/scaling/missing', '']
    (OUT/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result['pooled'],indent=2))

if __name__=='__main__':main()
