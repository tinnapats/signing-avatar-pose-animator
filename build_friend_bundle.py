"""Build an independent ZIP64 handoff; never upload weights to Git.

Contains tracked code, selected local assets, and a SHA-256 manifest.
Use --speech-model to select a complete Hugging Face snapshot directory.
"""
import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speech-model', type=Path, required=True)
    args = parser.parse_args()
    files = {}
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for name in filter(None, tracked):
        source = ROOT/name
        if source.is_file():
            if source.stat().st_size > 50*1024*1024:
                raise RuntimeError(f'Unexpected large tracked file: {name}')
            files[name] = source
    for name in ('SLclean.sqlite3', 'recognition_audit/wlasl_stgcn.zip',
                 'recognition_audit/wlasl_metadata.zip',
                 'recognition_audit/slgcn_trial_20261002/wlasl_slgcn.zip',
                 'evaluation/model-results/oracle200-20261011/capture_manifest.json'):
        files[name] = ROOT/name
    for folder in ('recognition_audit/slgcn200_20261002/cases',
                   'evaluation/model-results/oracle200-20261011/inference'):
        for source in (ROOT/folder).iterdir():
            if source.is_file() and source.suffix in ('.json', '.npz'):
                files[source.relative_to(ROOT).as_posix()] = source
    for name in ('config.json', 'model.safetensors', 'preprocessor_config.json',
                 'special_tokens_map.json', 'vocab.json'):
        files['models/xlsr53-english/'+name] = args.speech_model/name
    for name, source in files.items():
        if not source.is_file():
            raise FileNotFoundError(f'Missing bundle asset: {name}')
    out = ROOT/'friend_bundle'
    out.mkdir(exist_ok=True)
    target = out/f"signing-avatar-newton-friend-{datetime.now():%Y%m%d-%H%M%S}.zip"
    manifest = {'scope': 'Core app, archived-input Experiment 1-3 reruns, coordinate-fidelity source tools. '
                         'No Python environment, no historical pixel-detection experiment archives.',
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'files': []}
    with zipfile.ZipFile(target, 'x', allowZip64=True) as archive:
        for number, (name, source) in enumerate(sorted(files.items()), 1):
            digest = hashlib.sha256()
            info = zipfile.ZipInfo('signing-avatar-newton/'+name)
            info.compress_type = zipfile.ZIP_STORED
            size = 0
            with source.open('rb') as src, archive.open(info, 'w', force_zip64=True) as dst:
                while chunk := src.read(8*1024*1024):
                    digest.update(chunk)
                    size += len(chunk)
                    dst.write(chunk)
            if size != source.stat().st_size:
                raise RuntimeError(f'Source size changed during packing: {name}')
            manifest['files'].append(dict(path=name, bytes=size, sha256=digest.hexdigest()))
            if size > 10*1024*1024 or number % 100 == 0:
                print(f'{number}/{len(files)} {name} ({size} bytes)', flush=True)
        archive.writestr('signing-avatar-newton/BUNDLE_MANIFEST.json',
                         json.dumps(manifest, indent=2, ensure_ascii=False))
    print('Verifying every ZIP entry...', flush=True)
    with zipfile.ZipFile(target) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f'ZIP CRC failed: {bad}')
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        while chunk := stream.read(8*1024*1024):
            digest.update(chunk)
    target.with_suffix('.sha256.txt').write_text(f'{digest.hexdigest()}  {target.name}\n', encoding='utf-8')
    print(f'BUNDLE: {target}\nBYTES: {target.stat().st_size}\nSHA256: {digest.hexdigest()}', flush=True)


if __name__ == '__main__':
    main()
