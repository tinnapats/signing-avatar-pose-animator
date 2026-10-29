"""Read-only smoke tests for consolidated pose-estimation runtimes."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'evaluation/model-results/sdpose-trial-20261009/mpl-cache'))
parser = argparse.ArgumentParser()
parser.add_argument('runtime', choices=['mediapipe', 'sdpose'])
args = parser.parse_args()
assert Path(sys.executable).resolve().is_relative_to(ROOT), sys.executable
assert Path(sys.base_prefix).resolve().is_relative_to(ROOT), sys.base_prefix
if args.runtime == 'mediapipe':
    import cv2
    import mediapipe as mp
    folder = ROOT/'evaluation/model-results/oracle200-20261011'
    case = json.loads((folder/'capture_manifest.json').read_text())[0]
    image = cv2.imread(str(folder/'frames'/case['id']/case['frames'][0]['file']))
    assert image is not None
    with mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=1) as model:
        result = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print('MediaPipe actual image inference OK; pose detected:', result.pose_landmarks is not None)
else:
    folder = ROOT/'evaluation/model-results/sdpose-trial-20261009'
    sys.path.insert(0, str(folder/'source'))
    import torch
    torch.set_num_threads(4)
    import xtcocotools._mask
    from models.HeatmapHead import get_heatmap_head
    from pipelines.SDPose_D_Pipeline import SDPose_D_Pipeline
    head = get_heatmap_head('wholebody').eval()
    with torch.inference_mode():
        result = head((torch.zeros(1,640,8,6),))
    assert tuple(result.shape) == (1,133,16,12), result.shape
    print('SDPose imports and synthetic head OK:', tuple(result.shape), torch.__version__)
