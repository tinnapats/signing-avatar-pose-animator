"""Known-video pilot, isolated from production database. No manual clip matching."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import csv
import hashlib
import json
import sys
import urllib.error
import urllib.parse
from pathlib import Path
from inspect_kaggle import fetch, BASE

HERE=Path(__file__).resolve().parent
OUT=HERE/'paired_video'
WORDS=['motivate','continue','earn']

def download():
    metadata=_local_json_loads((HERE/'kaggle_reference/WLASL_v0.3.json').read_bytes())
    by={r['gloss']:r['instances'] for r in metadata}
    (OUT/'videos').mkdir(parents=True,exist_ok=True)
    selected=[]
    for word in WORDS:
        candidates=sorted([r for r in by[word] if r['split']=='test'],key=lambda r:r['video_id'])
        for c in candidates:
            remote='wlasl-complete/videos/'+c['video_id']+'.mp4'
            url=BASE+'/download/sttaseen/wlasl2000-resized/'+urllib.parse.quote(remote,safe='')+'?datasetVersionNumber=1'
            try:raw=fetch(url)
            except urllib.error.HTTPError as e:
                if e.code==404:print('missing',word,c['video_id'],flush=True);continue
                raise
            assert b'ftyp' in raw[:64], 'Expected MP4, not an error page/archive'
            path=OUT/'videos'/(word+'.mp4')
            path.write_bytes(raw)
            selected.append(dict(word=word,metadata=c,source_url=url,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),local_video=str(path)))
            print('downloaded',word,c['video_id'],len(raw),flush=True)
            break
        else:raise RuntimeError('No available test clip: '+word)
    (OUT/'manifest.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')

def extract():
    import cv2
    import mediapipe as mp
    manifest=_local_json_loads((OUT/'manifest.json').read_text())
    (OUT/'csv').mkdir(exist_ok=True)
    for item in manifest:
        cached=OUT/'csv'/(item['word']+'.csv')
        if 'extraction' in item and cached.exists() and hashlib.sha256(cached.read_bytes()).hexdigest()==item.get('csv_sha256'):
            print('cached extraction',item['word'],flush=True)
            continue
        cap=cv2.VideoCapture(item['local_video'])
        assert cap.isOpened()
        fps=cap.get(cv2.CAP_PROP_FPS);assert fps>0
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        index=0;counts=dict(pose=0,left_hand=0,right_hand=0,face=0)
        with (OUT/'csv'/(item['word']+'.csv')).open('w',newline='',encoding='utf-8') as f, mp.solutions.holistic.Holistic(static_image_mode=False,model_complexity=1,smooth_landmarks=True,enable_segmentation=False,refine_face_landmarks=False,min_detection_confidence=.5,min_tracking_confidence=.5) as model:
            writer=csv.writer(f);writer.writerow(['frame','time_sec','part','landmark_id','x','y','z','px','py','visibility'])
            while True:
                ok,image=cap.read()
                if not ok:break
                result=model.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                for part,attr in [('pose','pose_landmarks'),('face','face_landmarks'),('left_hand','left_hand_landmarks'),('right_hand','right_hand_landmarks')]:
                    landmarks=getattr(result,attr)
                    if landmarks:
                        counts[part]+=1
                        for i,p in enumerate(landmarks.landmark):writer.writerow([index,index/fps,part,i,p.x,p.y,p.z,round(p.x*width),round(p.y*height),p.visibility if part=='pose' else ''])
                    elif part=='pose':
                        for i in range(33):writer.writerow([index,index/fps,part,i,0,0,0,0,0,0])
                index+=1
        cap.release();assert index>0
        item['extraction']=dict(fps=fps,width=width,height=height,frames=index,detected_frames=counts,mediapipe=mp.__version__,opencv=cv2.__version__,mirrored=False,recropped=False,notes='Decode entire Kaggle clip; original metadata frame_start/end not reapplied to avoid double trim. Temporal boundaries not independently verified.')
        item['csv_sha256']=hashlib.sha256((OUT/'csv'/(item['word']+'.csv')).read_bytes()).hexdigest()
        print('extracted',item['word'],index,counts,flush=True)
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

def build():
    import numpy as np
    from run_audit import pipeline,csv_array
    from step1_evaluate import hashes
    before=hashes();cases=[]
    manifest=_local_json_loads((OUT/'manifest.json').read_text())
    for item in manifest:
        word=item['word'];path=OUT/'csv'/(word+'.csv')
        payload=pipeline.build_payload(OUT/'csv',[word+'.csv'],'',30,513,513,0,0,2,1.2,2)
        source=csv_array(pipeline.concat_clips([path],513,513,0,target_fps=30))
        indices=list(range(0,len(payload['frames']),2))
        assert len(source)==len(indices)
        cases.append(dict(phrase=word,payload=payload,source30=source.tolist(),frameIndices=indices,video_id=item['metadata']['video_id']))
    assert before==hashes()
    (OUT/'production_before.json').write_text(json.dumps(before))
    (OUT/'inputs.json').write_text(json.dumps(cases),encoding='utf-8')
    (OUT/'review_data.js').write_text('window.reviewCases='+json.dumps([dict(word=c['phrase'],video_id=c['video_id'],frames=len(c['frameIndices']),fps=30) for c in cases])+';',encoding='utf-8')
    lines=['# ทดลองวิดีโอที่รู้รหัสคลิป → พิกัด → Avatar','', 'ชุดใหม่ ไม่ได้อ้างว่าเป็นวิดีโอเดียวกับ SLclean เดิม เลือก test clip ตาม video_id เรียงลำดับที่ดาวน์โหลดได้ ไม่เลือกตามผลโมเดล','', '| คำ | video_id | split ตาม WLASL | ขนาด | FPS | เฟรม | พบมือซ้าย/ขวา (เฟรม) |','|---|---|---|---|---:|---:|---|']
    for i in manifest:
        e=i['extraction'];c=e['detected_frames'];lines.append(f"| {i['word']} | {i['metadata']['video_id']} | {i['metadata']['split']} | {e['width']}×{e['height']} | {e['fps']:.2f} | {e['frames']} | {c['left_hand']}/{c['right_hand']} |")
    lines += ['', 'MediaPipe Holistic 0.10.21; ไม่สะท้อนภาพ ไม่ครอปซ้ำ ประมวลผลทุกเฟรมตาม FPS ไฟล์จริง', 'รหัสคลิปและ hash วิดีโอ/CSV อยู่ใน manifest.json; ไม่เปลี่ยน SQLite และ hash ไฟล์ tracked ของ production เท่าเดิม', '', '## ข้อจำกัด', '- คำว่า test อ้างอิง metadata WLASL ยังไม่รับรองว่าไม่ซ้ำข้อมูลฝึกของ checkpoint ที่ใช้', '- ต้องตรวจขอบเขตท่าและคุณภาพ keypoints ด้วยสายตา; การตรวจพบ landmark ไม่รับรองว่าจุดถูกต้อง', '- 3 คลิปใช้ตรวจเส้นทางข้อมูล ไม่เพียงพอสำหรับรายงานประสิทธิภาพหรือความเข้าใจของผู้ใช้', '- วิดีโอเป็นเวอร์ชัน resized และต้องเคารพเงื่อนไข C-UDA ไม่รวมไฟล์วิดีโอใน Git', '']
    (OUT/'REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8')

def browser_copies():
    import subprocess
    import imageio_ffmpeg
    import cv2
    manifest=_local_json_loads((OUT/'manifest.json').read_text())
    for item in manifest:
        target=OUT/'videos'/(item['word']+'_browser.mp4')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-n','-i',item['local_video'],'-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True,capture_output=True)
        cap=cv2.VideoCapture(str(target));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==item['extraction']['frames'];cap.release()
        item['browser_copy']=dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),note='H.264 viewing copy only; landmarks extracted from original file')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__':{'download':download,'extract':extract,'build':build,'browser_copies':browser_copies}[sys.argv[1]]()
