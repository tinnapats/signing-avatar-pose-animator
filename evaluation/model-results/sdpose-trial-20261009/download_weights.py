import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent
REPO = 'teemosliang/SDPose-Wholebody'
REV = '0c8153e39458bc8ae87fb68790c141bf0f03c11b'
PREFIXES = ('decoder/', 'scheduler/', 'text_encoder/', 'tokenizer/', 'unet/', 'vae/')
info = HfApi().model_info(REPO, revision=REV, files_metadata=True)
selected = [f for f in info.siblings if f.rfilename.startswith(PREFIXES)]
manifest = {'repo': REPO, 'revision': info.sha, 'status': 'downloading',
            'files': [{'path': f.rfilename, 'bytes': f.size} for f in selected]}
out = ROOT / 'weights_manifest.json'
out.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('Downloading', sum(f.size for f in selected), 'bytes', flush=True)
snapshot_download(REPO, revision=REV, local_dir=ROOT / 'weights',
                  allow_patterns=[p+'*' for p in PREFIXES], max_workers=3)
for row in manifest['files']:
    path = ROOT / 'weights' / row['path']
    assert path.stat().st_size == row['bytes'], row['path']
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    row['sha256'] = h.hexdigest()
manifest['status'] = 'downloaded_size_checked_and_hashed'
out.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('Weights downloaded and hashed', flush=True)
