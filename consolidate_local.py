"""Local-only consolidation; preserves sources and archives conflicting files.

Large files are NTFS hard links when possible, otherwise copied. Updating a
destination later must use atomic replacement, never overwrite a linked file.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP = {'.git', '__pycache__', 'uv-cache'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def transfer(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in {'.py', '.js', '.mjs', '.json', '.cfg', '.pth', '.cmd', '.bat', '.ps1', '.md', '.txt', '.toml', '.yaml', '.yml'}:
        shutil.copy2(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def merge(src, dst, archive, exclude=()):
    counts = dict(copied=0, same=0, archived=0)
    for base, dirs, names in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP and d not in exclude]
        for name in names:
            source = Path(base)/name
            rel = source.relative_to(src)
            target = dst/rel
            if target.exists():
                if source.stat().st_size == target.stat().st_size and digest(source) == digest(target):
                    counts['same'] += 1
                    continue
                target = archive/rel
                if target.exists():
                    continue
                counts['archived'] += 1
            else:
                counts['copied'] += 1
            transfer(source, target)
    print(str(src), counts, flush=True)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--archive-name', required=True)
    parser.add_argument('--exclude', action='append', default=[])
    args = parser.parse_args()
    dst = args.destination.resolve()
    archive = (ROOT/'local_archive'/args.archive_name).resolve()
    if not dst.is_relative_to(ROOT) or not archive.is_relative_to(ROOT/'local_archive'):
        parser.error('Destinations must remain inside this project')
    src = args.source.resolve()
    if src == dst or dst.is_relative_to(src):
        parser.error('Source and destination must not overlap')
    result = merge(src, dst, archive, args.exclude)
    log = ROOT/'local_archive/consolidation-log.jsonl'
    log.parent.mkdir(exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(source=str(src), destination=str(dst), **result))+'\n')


if __name__ == '__main__':
    main()
