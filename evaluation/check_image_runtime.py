"""Read-only smoke check: run migrated RTMW weights on one archived avatar PNG."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'evaluation/model-results'
A = ROOT/'recognition_audit'
spec = importlib.util.spec_from_file_location('rtm_runtime', D/'rtm-avatar.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
cv2, RTMPose, YOLOX = runtime.setup()
meta = json.loads((A/'pixels30_20261007/rtm_trial_20261008/rtmw_model.json').read_text())
old = Path(meta['path'])
parts = old.parts
if 'recognition_audit' not in parts:
    raise RuntimeError(f'Model is outside the migrated audit: {old}')
model_path = A.joinpath(*parts[parts.index('recognition_audit')+1:])
assert runtime.sha(model_path) == meta['sha256'], 'Model hash mismatch'
model = RTMPose(str(model_path), model_input_size=tuple(meta['input_size']),
                backend='onnxruntime', device='cpu')
archive = D/'oracle200-20261011'
case = json.loads((archive/'capture_manifest.json').read_text())[0]
frame = case['frames'][0]
png = archive/'frames'/case['id']/frame['file']
assert runtime.sha(png) == frame['sha256'], 'Image hash mismatch'
image = runtime.image(cv2, png)
points, scores = model(image)
assert points.shape[-2:] == (133, 2), points.shape
print(f'RTMW inference OK: {points.shape}; weights and PNG hashes verified; no archive writes.')
