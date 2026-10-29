import sys,json,hashlib,zipfile,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch,yaml
D=Path(__file__).resolve().parent;B=D/'oracle200-20261011';O=D/'experiment3-example-20261013';O.mkdir(exist_ok=True);A=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit');sys.path.insert(0,str(A));import run_audit as audit
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
s=importlib.util.spec_from_file_location('ev',D/'rtm-eval.py');ev=importlib.util.module_from_spec(s);s.loader.exec_module(ev)
cid='001';cap=next(c for c in read(B/'capture_manifest.json') if c['id']==cid);assert cap['label']=='hello';ref=next(c for c in read(B/'coordinate_reference.json') if c['id']==cid)
z=np.load(B/'rtm'/(cid+'.npz'));raw=ev.convert(z['rtmw_det'],z['rtmw_det_scores'])
for i,row in enumerate(ref['samples']):
 assert row['index']==cap['frames'][i]['sourceIndex'] and row['pngMatches']
 for side,off in [('left',33),('right',54)]:
  raw[i,off:off+21]=0
  for j,pt in row['hands'][side].items():raw[i,off+int(j)]=pt
a=raw[:,audit.PICK];valid=np.any(a!=0,axis=2);trans=audit.module('example3_trans','upstream_pose_transforms.py');norm=trans.CenterAndScaleNormalize(reference_points_preset='shoulder_mediapipe_holistic_minimal_27');t=norm({'frames':torch.from_numpy(a.transpose(2,0,1).copy())})['frames'].numpy();saved=np.load(B/'inference'/(cid+'.npz'))['both_oracle_input'];assert np.array_equal(t,saved)
cfg=yaml.safe_load(zipfile.ZipFile(A/'wlasl_stgcn.zip').read('wlasl/st_gcn/config.yaml'));graph=cfg['model']['encoder']['params']['graph_args'];edges=graph['inward_edges'];xy=t.transpose(1,2,0);indices=np.rint(np.linspace(0,len(xy)-1,6)).astype(int).tolist();pred=next(r for r in read(B/'results.json')['predictions']['both_oracle'] if r['id']==cid)
record={'experiment':3,'label':'hello','prediction':pred,'indices':indices,'fps':30,'tensor_shape':[1,*t.shape],'raw_selected':a.tolist(),'normalized':xy.tolist(),'non_placeholder_mask':valid.tolist(),'edges':edges,'tensor_exact_match':True,'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [B/'inference'/(cid+'.npz'),B/'rtm'/(cid+'.npz'),B/'coordinate_reference.json']},'note':'RTMW body7 + Renderer hand10 per side; reference-absent hands zero before normalization. Non-placeholder does not certify anatomical accuracy. Visualization hides placeholders only; original tensor unchanged.'}
(O/'example-data.json').write_text(json.dumps(record),encoding='utf-8')
allpoints=xy[valid];lo=allpoints.min(0);hi=allpoints.max(0);scale=min(220/(hi[0]-lo[0]),330/(hi[1]-lo[1]));mid=(hi+lo)/2
def text(x,y,string,size=20,color='#334b5c',anchor='middle'):return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{string}</text>'
colors=['#8299a7','#5B9BD5','#ED7D31'];svg=[text(900,40,'Experiment 3 — actual ST-GCN input: hello',30),text(900,76,'RTMW body/face + Renderer hand coordinates and reference-presence mask',22)]
for col,i in enumerate(indices):
 x=20+col*295;svg += [text(x+140,116,f'Frame {i} | {i/30:.2f} s',19),f'<rect x="{x}" y="135" width="280" height="440" rx="12" fill="#f3f7fa"/>'];p=(xy[i]-mid)*scale+np.array([x+140,335])
 for j,k in edges:
  if valid[i,j] and valid[i,k]:
   color=colors[2 if max(j,k)>=17 else 1 if max(j,k)>=7 else 0];svg.append(f'<line x1="{p[j,0]}" y1="{p[j,1]}" x2="{p[k,0]}" y2="{p[k,1]}" stroke="{color}" stroke-width="2.4"/>')
 for j in range(27):
  if valid[i,j]:svg.append(f'<circle cx="{p[j,0]}" cy="{p[j,1]}" r="3.4" fill="{colors[0 if j<7 else 1 if j<17 else 2]}"/>')
 svg.append(text(x+140,550,f'{int(valid[i].sum())}/27 non-placeholder points',15))
for j,label in enumerate(['7 RTMW body/face points','10 Renderer left-hand points','10 Renderer right-hand points']):
 x=45+j*580;svg.extend([text(x,623,label,22,anchor='start'),f'<line x1="{x}" x2="{x+350}" y1="642" y2="642" stroke="{colors[j]}" stroke-width="5"/>'])
svg.extend([text(900,685,'Same coordinate scale in all panels. Placeholder points are hidden for display only.',21),text(900,719,'Reference-assisted diagnostic, not pixel-only extraction; no new inference or production changes.',20)])
(O/'experiment3-hello.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="750"><style>text{font-family:Calibri,Arial,sans-serif}</style><rect width="1800" height="750" fill="white"/>'+''.join(svg)+'</svg>',encoding='utf-8')
print(json.dumps({'indices':indices,'counts':[int(valid[i].sum()) for i in indices],'shape':list(t.shape),'prediction':pred['prediction'],'exact':True}))
