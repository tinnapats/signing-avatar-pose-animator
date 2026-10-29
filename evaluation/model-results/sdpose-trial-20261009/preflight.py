import json
import sys
import traceback
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path[:0] = [str(root / 'deps'), str(root / 'source')]
result = {'status': 'dependency_test', 'inference_run': False}
try:
    import torch
    result.update(torch=torch.__version__, cuda=torch.cuda.is_available())
    from models.HeatmapHead import get_heatmap_head
    head = get_heatmap_head('wholebody')
    result['head'] = type(head).__name__
    from pipelines.SDPose_D_Pipeline import SDPose_D_Pipeline
    result['status'] = 'imports_passed_weights_not_loaded'
except Exception:
    result['status'] = 'dependency_failed'
    result['error'] = traceback.format_exc()
(root / 'preflight.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
