"""Full local ZIP64 handoff with content deduplication and SHA-256 manifest."""
import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP = {'.git','friend_bundle','__pycache__','uv-cache','.pytest_cache'}


def digest(file):
    with file.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    tracked = subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)
    if tracked.strip():
        raise RuntimeError('Commit tracked source changes before creating a versioned bundle')
    files = []
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        for name in sorted(names):
            file = Path(base)/name
            if file.suffix.lower() in ('.pyc','.pyo','.log') or name.endswith('.rebase-tmp'):
                continue
            if file.is_symlink():
                if not file.resolve().is_relative_to(ROOT):
                    raise RuntimeError(f'External symlink requires explicit review: {file}')
            files.append(file)
    out = ROOT/'friend_bundle'
    out.mkdir(exist_ok=True)
    target = out/f'signing-avatar-newton-COMPLETE-{datetime.now():%Y%m%d-%H%M%S}.zip'
    partial = target.with_suffix('.zip.partial')
    manifest = dict(format='deduplicated-zip-v1',source_root=str(ROOT),
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    excluded=sorted(SKIP)+['*.pyc','*.pyo','*.log'],files=[])
    content, inodes = {}, {}
    last = time.monotonic()
    raw = stored = 0
    with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for index,file in enumerate(files,1):
            stat = file.stat()
            inode = (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
            sha = inodes.get(inode)
            if sha is None:
                sha = digest(file)
                inodes[inode] = sha
            name = file.relative_to(ROOT).as_posix()
            # Keep the base interpreter complete so setup_complete.cmd can bootstrap
            # even when no Python is installed on the recipient machine.
            required_bootstrap = name.startswith('local_runtime/python312/') or name in {'setup_complete.cmd','restore_complete_bundle.py','INSTALL_COMPLETE_TH.md'}
            canonical = content.get(sha)
            if canonical is None or required_bootstrap:
                archive.write(file,'signing-avatar-newton/'+name)
                canonical = name
                content.setdefault(sha,name)
                stored += stat.st_size
            raw += stat.st_size
            manifest['files'].append(dict(path=name,bytes=stat.st_size,sha256=sha,stored_as=canonical))
            if file.stat().st_size != stat.st_size or file.stat().st_mtime_ns != stat.st_mtime_ns:
                raise RuntimeError(f'File changed while packaging: {name}')
            if time.monotonic()-last > 15:
                print(f'{index}/{len(files)} files; logical={raw/1e9:.2f} GB; ZIP={partial.stat().st_size/1e9:.2f} GB; {name}',flush=True)
                last = time.monotonic()
            if shutil.disk_usage(out).free < 1024**3:
                raise RuntimeError(f'Less than 1 GiB free; incomplete archive retained at {partial}')
        manifest.update(logical_bytes=raw,unique_stored_bytes=stored)
        archive.writestr('signing-avatar-newton/BUNDLE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,separators=(',',':')))
    print('Verifying all stored contents against SHA-256 manifest...',flush=True)
    verified = set()
    with zipfile.ZipFile(partial) as archive:
        for record in manifest['files']:
            canonical = record['stored_as']
            if canonical in verified:
                continue
            with archive.open('signing-avatar-newton/'+canonical) as stream:
                sha = hashlib.file_digest(stream,'sha256').hexdigest()
            if sha != record['sha256']:
                raise RuntimeError(f'Archive hash mismatch: {canonical}')
            verified.add(canonical)
    partial.rename(target)
    sha = digest(target)
    target.with_suffix('.sha256.txt').write_text(f'{sha}  {target.name}\n',encoding='utf-8')
    info = dict(zip=target.name,sha256=sha,zip_bytes=target.stat().st_size,logical_bytes=raw,
                files=len(files),stored_entries=len(verified),source_commit=manifest['source_commit'],
                verification='Every stored member SHA-256 verified; duplicate records reference these verified members')
    target.with_suffix('.summary.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':
    main()
