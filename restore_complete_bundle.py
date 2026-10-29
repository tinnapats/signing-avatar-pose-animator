"""Reconstruct and verify deduplicated ZIP contents, then rebase this checkout."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def safe_path(root, name):
    path = (root/name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe manifest path: {name}')
    return path


def digest(file):
    with file.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def restore(root):
    manifest_file = root/'BUNDLE_MANIFEST.json'
    marker = root/'.complete_bundle_restored.json'
    manifest_hash = digest(manifest_file)
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    if manifest.get('format') != 'deduplicated-zip-v1':
        raise ValueError('Not a COMPLETE bundle')
    if marker.exists():
        if json.loads(marker.read_text())['manifest_sha256'] != manifest_hash:
            raise ValueError('Existing install belongs to a different bundle. Extract into a new folder.')
        print('Bundle already reconstructed; existing files will not be overwritten.',flush=True)
        return manifest
    verified = set()
    for index,item in enumerate(manifest['files'],1):
        source = safe_path(root,item['stored_as'])
        target = safe_path(root,item['path'])
        if source not in verified:
            if not source.is_file() or source.stat().st_size != item['bytes'] or digest(source) != item['sha256']:
                raise ValueError(f'Archive member missing/corrupted: {source}')
            verified.add(source)
        if target != source:
            if target.exists():
                if target.stat().st_size != item['bytes'] or digest(target) != item['sha256']:
                    raise ValueError(f'Refusing to replace changed file: {target}')
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
                if digest(target) != item['sha256']:
                    raise ValueError(f'Copy verification failed: {target}')
        if index % 3000 == 0:
            print(f'Restored/verified {index}/{len(manifest["files"])}',flush=True)
    marker.write_text(json.dumps(dict(manifest_sha256=manifest_hash,files=len(manifest['files']))),encoding='utf-8')
    print(f'All {len(manifest["files"])} files reconstructed and verified.',flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore-only',action='store_true')
    args=parser.parse_args()
    manifest=restore(ROOT)
    if args.restore_only:
        return
    base=ROOT/'local_runtime/python312/python.exe'
    subprocess.run([str(base),'-B',str(ROOT/'rebase_local_runtime.py'),'--source-root',manifest['source_root']],cwd=ROOT,check=True)
    for env in ('.venv','recognition_audit/rtmpose_env','recognition_audit/paired_video_env','local_runtime/sdpose_env'):
        python=ROOT/env/'Scripts/python.exe'
        subprocess.run([str(python),'-B',str(ROOT/'refresh_local_entrypoints.py')],cwd=ROOT,check=True)
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-B',str(ROOT/'check_friend_setup.py')],cwd=ROOT,check=True)
    print('Setup complete. Open open_signing_avatar.cmd or run_evaluation.cmd.')


if __name__=='__main__':
    main()
