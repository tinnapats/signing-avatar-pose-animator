import sys,json,zipfile,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
root=Path('C:/Users/newtu/signing-avatar-github-newton-upload/recognition_audit')
sys.path.insert(0,str(root))
import run_audit as audit
import torch,yaml,numpy as np
out=Path(__file__).parent/'model-input';out.mkdir(exist_ok=True)
source=root/'slgcn200_20261002/cases/001.json'
c=json.loads(source.read_text());assert c['label']=='hello'
frames=[c['payload']['frames'][j] for j in c['frameIndices']]
a=audit.frame_array(frames)[:,audit.PICK,:]
transforms=audit.module('input_example_transforms','upstream_pose_transforms.py')
norm=transforms.CenterAndScaleNormalize(reference_points_preset='shoulder_mediapipe_holistic_minimal_27')
t=norm({'frames':torch.from_numpy(a.transpose(2,0,1).copy())})['frames']
config=yaml.safe_load(zipfile.ZipFile(root/'wlasl_stgcn.zip').read('wlasl/st_gcn/config.yaml'))
graph=config['model']['encoder']['params']['graph_args']
valid=np.any(a!=0,axis=2)
record=dict(label='hello',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    frames=len(a),fps=30,pick=audit.PICK,graph=graph,raw=a.tolist(),normalized=t.permute(1,2,0).tolist(),
    valid=valid.tolist(),tensor_shape=[1,*list(t.shape)],indices=np.rint(np.linspace(0,len(a)-1,6)).astype(int).tolist(),
    note='Actual archived pre-render payload. Same float32 selection and upstream clip-level shoulder normalization as audit. No model inference; no production edits.')
(out/'input-data.json').write_text(json.dumps(record),encoding='utf-8')
print(json.dumps({k:record[k] for k in ['frames','tensor_shape','graph','indices']}))
