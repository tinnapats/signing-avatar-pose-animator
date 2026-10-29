import json
import os
import sys
import traceback
import importlib.metadata as metadata
from pathlib import Path

root = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(root / 'mpl-cache'))
sys.path.insert(0, str(root / 'source'))
result = {'python': sys.version, 'executable': sys.executable,
          'inference_run': False, 'status': 'dependency_test'}
try:
    import torch
    result.update(torch=torch.__version__, cuda=torch.cuda.is_available())
    if torch.cuda.is_available():
        result['device'] = torch.cuda.get_device_name(0)
        result['cuda_smoke'] = float((torch.ones(4, device='cuda') * 2).sum().cpu())
    import xtcocotools._mask
    result['xtcocotools_binary_loaded'] = True
    from models.HeatmapHead import get_heatmap_head
    head = get_heatmap_head('wholebody')
    result['head'] = type(head).__name__
    # Synthetic features test only, not trained pose inference.
    head.eval()
    with torch.inference_mode():
        y = head((torch.zeros(1, 640, 8, 6),))
    result['synthetic_head_shape'] = list(y.shape)
    del head, y
    from pipelines.SDPose_D_Pipeline import SDPose_D_Pipeline
    result['status'] = 'imports_and_synthetic_head_passed_weights_not_loaded'
except Exception:
    result['status'] = 'dependency_failed'
    result['error'] = traceback.format_exc()
result['packages'] = {d.metadata['Name']: d.version for d in metadata.distributions()}
(root / 'preflight310.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k != 'packages'}, indent=2))
sys.exit(1 if result['status'] == 'dependency_failed' else 0)
