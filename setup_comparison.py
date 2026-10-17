"""Download official MediaPipe model bundles for local comparison."""
from pathlib import Path
from urllib.request import urlretrieve

MODELS = {
    'pose_landmarker_full.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
    'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
}

if __name__ == '__main__':
    folder = Path(__file__).parent / 'comparison_models'
    folder.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        target = folder / name
        if not target.exists():
            temporary = target.with_suffix('.download')
            urlretrieve(url, temporary)
            temporary.replace(target)
        print(target)
