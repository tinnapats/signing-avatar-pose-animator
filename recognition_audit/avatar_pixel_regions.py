"""Exploratory spatial coverage, not joint accuracy or semantic evaluation."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json,itertools
import cv2
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'avatar_pixel_probe_20261004';SRC=HERE/'adjacent_holdout60_20261004'
def read(p):return _local_json_loads(p.read_text(encoding='utf-8'))
def iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0]));h=max(0,min(a[3],b[3])-max(a[1],b[1]));inter=w*h
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0
def box(points):return [min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points)]
def main():
    caps={(c['phrase'],c['variant']):c['samples'] for c in read(SRC/'visual_review/rendered.json')};cases={}
    for d in read(SRC/'review_index.json'):
        c=read(Path(d['file']));cases[c['phrase']]=c
    rows=[];summary={}
    for r in read(OUT/'results.json')['rows']:
        if not r['variant']:continue
        name=r['sequence'].rsplit('__',1)[0];s=caps[name,r['variant']][r['index']];pose={k['part']:k['position'] for k in cases[name]['payload']['frames'][s['index']]['pose']['keypoints']}
        anchor=[(pose['leftHip'][a]+pose['rightHip'][a])/2 for a in ['x','y']]
        height,width=cv2.imread(str(SRC/'visual_review/frames'/r['sequence']/'0000.png')).shape[:2]
        gt=[]
        for side in ['left','right']:
            b=s['geometry'][side]['bounds'];b=[((b[0]-anchor[0])*s['zoom']+513*.5)/width,((b[1]-anchor[1])*s['zoom']+513*.82)/height,((b[2]-anchor[0])*s['zoom']+513*.5)/width,((b[3]-anchor[1])*s['zoom']+513*.82)/height]
            gt.append(dict(side=side,bounds=b,state='native' if s['native'][side] and not s['hands'][side] else 'detailed'))
        for detector,points in [('Holistic',[r['holistic'][side] for side in ['left','right'] if r['holistic'][side]]),('Hands',r['hands'])]:
            boxes=[box(p) for p in points];options=[-1]+list(range(len(boxes)));best=None
            for assignment in itertools.product(options,repeat=2):
                used=[x for x in assignment if x>=0]
                if len(set(used))!=len(used):continue
                scores=[iou(g['bounds'],boxes[j]) if j>=0 else 0 for g,j in zip(gt,assignment)]
                rank=(sum(x>=.1 for x in scores),sum(scores))
                if best is None or rank>best[0]:best=(rank,scores)
            for g,score in zip(gt,best[1]):
                key=f"{r['variant']}_{detector}_{g['state']}";tot=summary.setdefault(key,dict(n=0,region_matches=0));tot['n']+=1;tot['region_matches']+=score>=.1
                rows.append(dict(sequence=r['sequence'],index=s['index'],detector=detector,side=g['side'],state=g['state'],iou=score,region_match=score>=.1))
    result=dict(note='Exploratory bounding-box IoU>=0.1, best one-to-one matching without handedness labels. GT is visible strokeBounds projected using frozen camera. Not 21-joint correctness, not precision/recall; threshold added for diagnosis after coverage probe, not model tuning.',summary=summary,rows=rows)
    result['camera_note']='Projection adds fixed513 camera target then divides by actual PNG width/height (808 here). Earlier region_diagnostics.json incorrectly divided by513 and is invalid; preserved, not used.'
    (OUT/'region_diagnostics_camera.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
