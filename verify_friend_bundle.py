"""Verify extracted handoff files against the packaged manifest; Python stdlib only."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT/'BUNDLE_MANIFEST.json').read_text(encoding='utf-8'))
    errors = []
    for record in manifest['files']:
        file = (ROOT/record['path']).resolve()
        if not file.is_relative_to(ROOT) or not file.is_file():
            errors.append(record['path'])
            continue
        digest = hashlib.sha256()
        with file.open('rb') as stream:
            while chunk := stream.read(8*1024*1024):
                digest.update(chunk)
        if file.stat().st_size != record['bytes'] or digest.hexdigest() != record['sha256']:
            errors.append(record['path'])
    print('FAILED: '+str(errors) if errors else f"Verified {len(manifest['files'])} files.")
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
