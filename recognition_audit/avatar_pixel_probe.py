"""Pixels-only feasibility gate, before attempting semantic recognition."""
import json,hashlib,time
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
HERE=Path(__file__).resolve().parent;SRC=HERE/'adjacent_holdout60_20261004';OUT=HERE/'avatar_pixel_probe_20261004'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pack(lm):return [[p.x,p.y,p.z] for p in lm.landmark] if lm else []

def main():
    assert not OUT.exists();OUT.mkdir()
    images=sorted((SRC/'visual_review/frames').glob('*/*.png'));assert len(images)==234
    human=read(HERE/'paired_video/manifest.json')[0];video=Path(human['local_video']);assert sha(video)==human['sha256']
    settings=dict(min_detection_confidence=.5,min_tracking_confidence=.5,model_complexity=1)
    write(OUT/'protocol.json',dict(scope='Pixels only, feasibility pilot on existing fixed 3 boundary windows, not complete signs. No payload/keypoints fed into detector. No label-conditioned tuning or production edit.',images={str(p):sha(p) for p in images},positive_control=dict(video=str(video),sha256=sha(video),frames='first26 decoded'),negative_control='26 white RGB arrays of matching size, no saved edited images',mediapipe=mp.__version__,opencv=cv2.__version__,settings=settings,detectors=['Holistic tracking reset per sequence','Hands static_image_mode=True max_num_hands2'],gate='Engineering feasibility only: both hands detected in >=90% frames per method by at least one detector, plus Holistic pose >=90% for full pixels-to-STGCN path. Passing would still require geometry and deliberate-error sensitivity checks; not semantic accuracy. Fail means no recognition score from missing-keypoint substitutions.'))
    rows=[]
    def sequence(name,frames,variant=None):
        with mp.solutions.holistic.Holistic(static_image_mode=False,smooth_landmarks=True,enable_segmentation=False,refine_face_landmarks=False,**settings) as h, mp.solutions.hands.Hands(static_image_mode=True,max_num_hands=2,**settings) as hands:
            for i,rgb in enumerate(frames):
                start=time.perf_counter();a=h.process(rgb);b=hands.process(rgb)
                rows.append(dict(sequence=name,variant=variant,index=i,holistic=dict(pose=pack(a.pose_landmarks),left=pack(a.left_hand_landmarks),right=pack(a.right_hand_landmarks)),hands=[pack(x) for x in (b.multi_hand_landmarks or [])],handedness=[dict(label=x.classification[0].label,score=x.classification[0].score) for x in (b.multi_handedness or [])],elapsed_ms=(time.perf_counter()-start)*1000))
        print('Processed',name,len(frames),flush=True)
    for folder in sorted({p.parent for p in images}):
        fs=sorted(folder.glob('*.png'));rgbs=[cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB) for p in fs]
        sequence(folder.name,rgbs,folder.name.split('__')[-1])
    cap=cv2.VideoCapture(str(video));frames=[]
    for _ in range(26):
        ok,bgr=cap.read()
        if not ok:break
        frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release();assert len(frames)==26;sequence('human_positive',frames)
    sample=cv2.imread(str(images[0]));sequence('blank_negative',[np.full_like(sample,255) for _ in range(26)])
    def summarize(rr):
        return dict(n=len(rr),pose=sum(bool(r['holistic']['pose']) for r in rr),holistic_any_hand=sum(bool(r['holistic']['left'] or r['holistic']['right']) for r in rr),holistic_both_hands=sum(bool(r['holistic']['left'] and r['holistic']['right']) for r in rr),hands_any=sum(bool(r['hands']) for r in rr),hands_both=sum(len(r['hands'])==2 for r in rr))
    summary={v:summarize([r for r in rows if r['variant']==v]) for v in ['OFF','OLD','NEW']}
    controls={v:summarize([r for r in rows if r['sequence']==v]) for v in ['human_positive','blank_negative']}
    gate=all(s['pose']/s['n']>=.9 and max(s['holistic_both_hands'],s['hands_both'])/s['n']>=.9 for s in summary.values())
    perseq={v:summarize([r for r in rows if r['sequence']==v]) for v in sorted({r['sequence'] for r in rows})}
    for p in images:assert sha(p)==read(OUT/'protocol.json')['images'][str(p)]
    write(OUT/'results.json',dict(summary=summary,controls=controls,per_sequence=perseq,feasibility_gate_passed=gate,rows=rows))
    print(json.dumps(dict(summary=summary,controls=controls,feasibility_gate_passed=gate),indent=2),flush=True)
if __name__=='__main__':main()
