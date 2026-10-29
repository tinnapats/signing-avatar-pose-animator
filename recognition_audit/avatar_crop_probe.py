"""Fixed image-derived wrist crops; renderer coordinates used only for evaluation."""
import json,hashlib,itertools
from pathlib import Path
import cv2,mediapipe as mp
import numpy as np
HERE=Path(__file__).resolve().parent;SRC=HERE/'adjacent_holdout60_20261004';PRE=HERE/'avatar_pixel_probe_20261004';OUT=HERE/'avatar_crop_probe_20261004'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bbox(p):return [min(x[0] for x in p),min(x[1] for x in p),max(x[0] for x in p),max(x[1] for x in p)]
def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/u if u>0 else 0
def main():
    assert not OUT.exists();OUT.mkdir()
    inputs=[PRE/'results.json',SRC/'visual_review/rendered.json',SRC/'review_index.json',Path(__file__)]
    images=sorted((SRC/'visual_review/frames').glob('*/*.png'));hashes={str(p):sha(p) for p in inputs+images}
    write(OUT/'protocol.json',dict(hashes=hashes,method='Use cached image-only Holistic pose wrist15/16 and elbow13/14. Center=wrist+.25*(wrist-elbow). Square side=clip(2.5*forearmLength,96,320) pixels. Pad white outside image, resize256x256 INTER_LINEAR. Hands static, complexity1, max_num_hands2, confidence.5. Select detected wrist nearest crop-center (no GT). No oracle crop/no threshold sweep/no mirrored input.',evaluation='Same234 images. Match left/right independently to visible renderer region IoU>=.1; full-frame baseline matched one-to-one by region for evaluation only. Detailed 21-point errors after camera projection conditional on region match; native has no valid 21-point reference. Report missing separately, never score missing as0 error. Same90% coverage feasibility target as prior pilot, not ASL accuracy.',mediapipe=mp.__version__))
    caps={(c['phrase'],c['variant']):c['samples'] for c in read(SRC/'visual_review/rendered.json')};cases={}
    for d in read(SRC/'review_index.json'):
        c=read(Path(d['file']));cases[c['phrase']]=c
    rows=[];controls=[]
    with mp.solutions.hands.Hands(static_image_mode=True,max_num_hands=2,model_complexity=1,min_detection_confidence=.5,min_tracking_confidence=.5) as detector:
        for num,r in enumerate([r for r in read(PRE/'results.json')['rows'] if r['variant']]):
            png=SRC/'visual_review/frames'/r['sequence']/f"{r['index']:04d}.png";rgb=cv2.cvtColor(cv2.imread(str(png)),cv2.COLOR_BGR2RGB);height,width=rgb.shape[:2]
            name=r['sequence'].rsplit('__',1)[0];s=caps[name,r['variant']][r['index']];p={k['part']:k['position'] for k in cases[name]['payload']['frames'][s['index']]['pose']['keypoints']};anchor=np.array([(p['leftHip'][a]+p['rightHip'][a])/2 for a in ['x','y']])
            def project(x):return ((np.array(x)-anchor)*s['zoom']+np.array([513*.5,513*.82])).tolist()
            gt=[project(s['geometry'][side]['bounds'][:2])+project(s['geometry'][side]['bounds'][2:]) for side in ['left','right']]
            full=[(np.array(x)[:,:2]*[width,height]).tolist() for x in r['hands']]
            choices=[-1]+list(range(len(full)));best=None
            for ids in itertools.product(choices,repeat=2):
                real=[x for x in ids if x>=0]
                if len(set(real))!=len(real):continue
                overlaps=[iou(g,bbox(full[j])) if j>=0 else 0 for g,j in zip(gt,ids)];rank=(sum(x>=.1 for x in overlaps),sum(overlaps))
                if best is None or rank>best[0]:best=(rank,ids)
            for si,side in enumerate(['left','right']):
                pose=r['holistic']['pose'];pred=[];rect=None
                if pose:
                    wrist=np.array(pose[15+si][:2])*[width,height];elbow=np.array(pose[13+si][:2])*[width,height]
                    center=wrist+.25*(wrist-elbow);size=int(round(np.clip(2.5*np.linalg.norm(wrist-elbow),96,320)));origin=np.floor(center-size/2).astype(int);x,y=origin
                    crop=np.full((size,size,3),255,dtype=np.uint8);x0=max(0,x);y0=max(0,y);x1=min(width,x+size);y1=min(height,y+size)
                    if x1>x0 and y1>y0:crop[y0-y:y1-y,x0-x:x1-x]=rgb[y0:y1,x0:x1]
                    out=detector.process(cv2.resize(crop,(256,256),interpolation=cv2.INTER_LINEAR));found=[np.array([[p.x,p.y] for p in lm.landmark])*size+origin for lm in (out.multi_hand_landmarks or [])]
                    if found:pred=min(found,key=lambda a:np.linalg.norm(a[0]-center)).tolist()
                    rect=[int(x),int(y),size]
                state='native' if s['native'][side] and not s['hands'][side] else 'detailed'
                reference={int(k):project(v) for k,v in s['hands'][side].items()}
                for method,points in [('FULL',full[best[1][si]] if best[1][si]>=0 else []),('CROP',pred)]:
                    overlap=iou(gt[si],bbox(points)) if points else 0;errors=[]
                    if overlap>=.1 and reference:
                        diag=np.linalg.norm(np.array(gt[si][2:])-gt[si][:2]);errors=[float(np.linalg.norm(np.array(points[k])-v)/diag) for k,v in sorted(reference.items())]
                    rows.append(dict(sequence=r['sequence'],variant=r['variant'],index=s['index'],sample_index=r['index'],side=side,state=state,method=method,detected=bool(points),iou=overlap,region_match=overlap>=.1,errors=errors,points=points,crop=rect,reference=reference))
            if (num+1)%26==0:print('Processed',num+1,'/234',flush=True)
        for _ in range(26):controls.append(bool(detector.process(np.full((256,256,3),255,dtype=np.uint8)).multi_hand_landmarks))
    summary={}
    for method in ['FULL','CROP']:
        for variant in ['OFF','OLD','NEW']:
            for state in ['native','detailed']:
                rr=[r for r in rows if (r['method'],r['variant'],r['state'])==(method,variant,state)];ee=[e for r in rr for e in r['errors']]
                summary[f'{method}_{variant}_{state}']=dict(n=len(rr),region_matches=sum(r['region_match'] for r in rr),joint_error_samples=len(ee),mean_normalized_joint_error=float(np.mean(ee)) if ee else None,pck_01=float(np.mean(np.array(ee)<=.1)) if ee else None)
    for path,h in hashes.items():assert sha(Path(path))==h
    write(OUT/'results.json',dict(summary=summary,blank_false_detections=sum(controls),rows=rows))
    print(json.dumps(dict(summary=summary,blank_false_detections=sum(controls)),indent=2))
if __name__=='__main__':main()
