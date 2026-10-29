"""Isolated, preregistered head fine-tuning pilot. No production writes."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import collections
import hashlib
import json
import sys
import urllib.error
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'head_pilot_20261005'
COUNTS = {'train': 3, 'val': 1, 'test': 2}

def read(p):
    return _local_json_loads(p.read_text(encoding='utf-8'))

def save(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def freeze():
    inventory = HERE / 'training_data_inventory_20261005/inventory.json'
    data = read(inventory)
    assert not data['split_disagreements']
    labels = sorted(data['summary']['eligible_labels'], key=lambda w: hashlib.sha256(('head-pilot-20261005:' + w).encode()).hexdigest())[:20]
    protocol = dict(labels=labels, seed=20261005, counts=COUNTS,
        candidate_order='ascending video_id within official split; next candidate only on unavailable/invalid/duplicate file; no replacement words',
        inventory_sha256=digest(inventory), candidates={w: data['candidates'][w] for w in labels},
        input='actual detailed hand renderer endpoints + payload body; best frozen no_hold_no_hand_smoothing pipeline; 30 FPS, original duration',
        model='original OpenHands ST-GCN encoder frozen in eval mode; original 2000-way linear head fine-tuned',
        training=dict(optimizer='AdamW', lr=0.0001, weight_decay=0.01, epochs=100, batch='all 60 train clips',
                      selection='lowest validation cross-entropy, including epoch 0; no test access before checkpoint selection',
                      regularization='mean squared distance of weights and bias from pretrained head, coefficient 1.0', augmentation='none'),
        metrics='top1 and macro precision/recall/F1 on the 20 target labels; all 2000 logits retained',
        caveats=['Official clip-disjoint, NOT signer-disjoint.', 'Original checkpoint pretraining provenance not independently proven.',
                 'Pilot changes evaluator, not Avatar. Improvement does not prove Avatar is more correct.',
                 'Only 2 held-out clips per word; cannot generalize to all 2000 words.',
                 'Academic computational use only; do not publish source videos.'])
    OUT.mkdir(exist_ok=True)
    p = OUT / 'protocol.json'
    if p.exists():
        assert read(p) == protocol, 'Frozen protocol differs'
    else:
        save(p, protocol)
    print('FROZEN', labels, 'protocol sha256', digest(p), flush=True)

def download():
    import cv2
    from inspect_kaggle import fetch, BASE
    protocol = read(OUT / 'protocol.json')
    data = read(HERE / 'training_data_inventory_20261005/inventory.json')
    local = {str(r['video_id']): r['path'] for r in data['local_manifest_videos'] if r['exists']}
    (OUT / 'videos').mkdir(exist_ok=True)
    manifest = read(OUT / 'manifest.json') if (OUT / 'manifest.json').exists() else []
    attempts = read(OUT / 'download_attempts.json') if (OUT / 'download_attempts.json').exists() else []
    hashes = {r['sha256']: r['metadata']['video_id'] for r in manifest}
    ids = {r['metadata']['video_id'] for r in manifest}
    unavailable = {r['video_id'] for r in attempts if r['status'] in ['404', 'invalid_video', 'duplicate_bytes']}
    deficits = []
    for word in protocol['labels']:
        for split, required in COUNTS.items():
            n = sum(r['label'] == word and r['metadata']['split'] == split for r in manifest)
            for candidate in protocol['candidates'][word][split]:
                if n >= required: break
                vid = candidate['video_id']
                if vid in ids or vid in unavailable: continue
                dest = OUT / 'videos' / (vid + '.mp4')
                url = BASE + '/download/sttaseen/wlasl2000-resized/' + urllib.parse.quote('wlasl-complete/videos/' + vid + '.mp4', safe='') + '?datasetVersionNumber=1'
                try:
                    raw = Path(local[vid]).read_bytes() if vid in local else (dest.read_bytes() if dest.exists() else fetch(url))
                except urllib.error.HTTPError as e:
                    if e.code != 404: raise
                    attempts.append(dict(label=word, split=split, video_id=vid, status='404'))
                    save(OUT / 'download_attempts.json', attempts)
                    print('missing', word, split, vid, flush=True)
                    continue
                assert b'ftyp' in raw[:64], 'Unexpected non-MP4 response'
                sha = hashlib.sha256(raw).hexdigest()
                if sha in hashes:
                    attempts.append(dict(label=word, split=split, video_id=vid, status='duplicate_bytes', duplicate_of=hashes[sha]))
                    save(OUT / 'download_attempts.json', attempts)
                    continue
                dest.write_bytes(raw)
                cap = cv2.VideoCapture(str(dest))
                fps = cap.get(cv2.CAP_PROP_FPS)
                count = 0
                while cap.isOpened():
                    ok, _ = cap.read()
                    if not ok: break
                    count += 1
                cap.release()
                if fps <= 0 or count < 2:
                    attempts.append(dict(label=word, split=split, video_id=vid, status='invalid_video'))
                    save(OUT / 'download_attempts.json', attempts)
                    continue
                manifest.append(dict(word=vid, label=word, metadata=candidate, local_video=str(dest), source_url=url,
                                     sha256=sha, bytes=len(raw), decoded_frames=count, fps=fps))
                hashes[sha] = vid
                ids.add(vid)
                n += 1
                save(OUT / 'manifest.json', manifest)
                print('accepted', len(manifest), word, split, vid, count, flush=True)
            if n < required: deficits.append(dict(label=word, split=split, available=n, required=required))
    assert len(ids) == len(manifest) == len(hashes)
    signers = {s: {r['metadata']['signer_id'] for r in manifest if r['metadata']['split'] == s} for s in COUNTS}
    summary = dict(protocol_sha256=digest(OUT / 'protocol.json'), clips=len(manifest),
                   splits=dict(collections.Counter(r['metadata']['split'] for r in manifest)), deficits=deficits,
                   duplicate_video_ids=0, duplicate_file_hashes=0,
                   signer_overlap={a+'_'+b: sorted(signers[a] & signers[b]) for a,b in [('train','val'),('train','test'),('val','test')]},
                   ready=len(manifest)==120 and not deficits)
    save(OUT / 'data_check.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

def extract():
    assert read(OUT / 'data_check.json')['ready'], 'Incomplete frozen dataset; do not silently change words or counts'
    import paired_video_trial as paired
    paired.OUT = OUT
    paired.extract()

if __name__ == '__main__':
    {'freeze': freeze, 'download': download, 'extract': extract}[sys.argv[1]]()
