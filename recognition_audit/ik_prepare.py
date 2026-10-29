"""Experimental planar two-bone IK; no torch, production files untouched."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import copy,json,math,statistics,hashlib,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'ik_trial';OUT.mkdir(exist_ok=True)
def distance(a,b):return math.hypot(a[0]-b[0],a[1]-b[1])
def point(k):return [k['position']['x'],k['position']['y']]
def solve(s,e,target,l1,l2,previous=None):
    delta=[target[0]-s[0],target[1]-s[1]]; d=math.hypot(*delta)
    u=[delta[0]/d,delta[1]/d] if d>=1e-9 else [1,0]
    # Internal elbow angle 15..165 degrees; clamp target, never stretch bones.
    low=math.sqrt(l1*l1+l2*l2-2*l1*l2*math.cos(math.radians(15)))
    high=math.sqrt(l1*l1+l2*l2-2*l1*l2*math.cos(math.radians(165)))
    reach=max(low,min(high,d))
    wrist=[s[0]+u[0]*reach,s[1]+u[1]*reach]
    along=(l1*l1-l2*l2+reach*reach)/(2*reach)
    height=math.sqrt(max(0,l1*l1-along*along))
    elbows=[[s[0]+u[0]*along-sign*u[1]*height,s[1]+u[1]*along+sign*u[0]*height] for sign in [-1,1]]
    def cost(p):return distance(p,e)**2+(0.6*distance([p[0]-s[0],p[1]-s[1]],previous)**2 if previous else 0)
    elbow=min(elbows,key=cost)
    assert abs(distance(s,elbow)-l1)<1e-6 and abs(distance(elbow,wrist)-l2)<1e-6
    return elbow,wrist,abs(reach-d)>1e-6

for target in [[0,0],[100,0],[8,5],[-20,-10]]:
    solve([0,0],[4,2],target,6,5)
cases=_local_json_loads((HERE/'inputs.json').read_text());stats=[]
for case in cases:
    candidate=copy.deepcopy(case['payload']);frames=candidate['frames'];lengths={}
    for side in ['left','right']:
        upper=[];lower=[]
        for frame in frames:
            pose={k['part']:k for k in frame['pose']['keypoints']}
            keys=[pose[side+n] for n in ['Shoulder','Elbow','Wrist']]
            if all(k['score']>=.2 for k in keys):
                a,b,c=map(point,keys);upper.append(distance(a,b));lower.append(distance(b,c))
        lengths[side]=(statistics.median(upper),statistics.median(lower))
    previous={}; records=[]
    for frame in frames:
        pose={k['part']:k for k in frame['pose']['keypoints']}
        for side in ['left','right']:
            hand=frame.get('hands',{}).get(side,{})
            keys=[pose[side+n] for n in ['Shoulder','Elbow','Wrist']]
            if hand.get('score',0)<.001 or len(hand.get('keypoints',[]))!=21 or not all(k['score']>=.2 for k in keys) or hand['keypoints'][0].get('score',0)<=0:
                previous.pop(side,None);continue
            s,e,old=map(point,keys);target=point(hand['keypoints'][0]);l1,l2=lengths[side]
            elbow,wrist,clamped=solve(s,e,target,l1,l2,previous.get(side))
            previous[side]=[elbow[0]-s[0],elbow[1]-s[1]]
            shift=[wrist[0]-target[0],wrist[1]-target[1]]
            for k in hand['keypoints']:
                k['position']['x']+=shift[0];k['position']['y']+=shift[1]
            for k,p in [(keys[1],elbow),(keys[2],wrist)]:k['position'].update(x=p[0],y=p[1])
            records.append(dict(side=side,clamped=clamped,target_shift=distance(target,wrist),elbow_shift=distance(e,elbow),old_upper_error=abs(distance(s,e)-l1)/l1,old_lower_error=abs(distance(e,old)-l2)/l2,new_upper_error=abs(distance(s,elbow)-l1)/l1,new_lower_error=abs(distance(elbow,wrist)-l2)/l2))
    case['ikPayload']=candidate
    stats.append(dict(phrase=case['phrase'],lengths=lengths,records=records))
files=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
manifest={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files if (ROOT/f).is_file()}
(OUT/'production_before.json').write_text(json.dumps(manifest))
(OUT/'inputs.json').write_text(json.dumps(cases))
(OUT/'arm_metrics.json').write_text(json.dumps(stats))
(OUT/'reference_fit.json').write_bytes((HERE/'step1/reference_fit.json').read_bytes())
assert manifest==_local_json_loads((HERE/'step1/production_before.json').read_text()),'Source changed since reference generation'
print('Prepared',len(cases),'cases; projected 2D lengths fixed per clip, elbow angle 15..165; no production changes.')
