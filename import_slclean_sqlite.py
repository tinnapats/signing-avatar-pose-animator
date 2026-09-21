"""Import complete CSV clips into SQLite without modifying the source dataset.

CSV bytes are compressed per clip; zlib.decompress(csv_zlib) restores the exact
original file. Indexed labels allow word/phrase lookup without a directory scan.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import zlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('C:/pro1end/SLclean'))
    parser.add_argument('--output', type=Path, default=Path('C:/pro1end/SLclean.sqlite3'))
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    files = sorted(source.rglob('*.csv'))
    if not files:
        raise SystemExit(f'No CSV files in {source}')
    if output.exists():
        raise SystemExit(f'Refusing to overwrite existing database: {output}')
    temporary = output.with_suffix(output.suffix + '.importing')
    if temporary.exists():
        raise SystemExit(f'Previous import exists: {temporary}')
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with sqlite3.connect(temporary) as db:
        db.executescript('''
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE clips (
                id INTEGER PRIMARY KEY,
                label TEXT NOT NULL,
                relative_path TEXT NOT NULL UNIQUE,
                columns_json TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                frame_count INTEGER NOT NULL,
                original_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                csv_zlib BLOB NOT NULL
            );
            CREATE INDEX clips_label ON clips(label COLLATE NOCASE);
        ''')
        total_rows = 0
        for index, path in enumerate(files, 1):
            raw = path.read_bytes()
            lines = raw.splitlines()
            if not lines:
                raise ValueError(f'Empty CSV: {path}')
            columns = lines[0].decode('utf-8-sig').split(',')
            required = {'frame', 'part', 'landmark_id', 'x', 'y'}
            if not required.issubset(columns) or columns[0] != 'frame':
                raise ValueError(f'Unexpected CSV columns: {path}')
            rows = [line for line in lines[1:] if line.strip()]
            frames = {line.split(b',', 1)[0] for line in rows}
            label = path.stem.lower()
            for suffix in ('_holistic_keypoints', '-holistic_keypoints', '_keypoints', '-keypoints'):
                if label.endswith(suffix):
                    label = label[:-len(suffix)]
                    break
            label = label.replace('_', ' ').replace('-', ' ').strip()
            db.execute('INSERT INTO clips VALUES (?,?,?,?,?,?,?,?,?)', (
                index, label, path.relative_to(source).as_posix(),
                json.dumps(columns), len(rows), len(frames), len(raw),
                hashlib.sha256(raw).hexdigest(), zlib.compress(raw, 1)))
            total_rows += len(rows)
            if index % 100 == 0:
                db.commit()
                print(f'Imported {index}/{len(files)} clips', flush=True)
        db.executemany('INSERT INTO metadata VALUES (?,?)', [
            ('format', 'slclean-csv-zlib-v1'), ('source_directory', str(source)),
            ('clip_count', str(len(files))), ('row_count', str(total_rows)),
            ('created_utc', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())),
        ])
        db.commit()
        print('Verifying every stored clip against its original SHA-256...', flush=True)
        for relative, checksum, blob in db.execute('SELECT relative_path, sha256, csv_zlib FROM clips'):
            if hashlib.sha256(zlib.decompress(blob)).hexdigest() != checksum:
                raise ValueError(f'Checksum mismatch: {relative}')
        result = db.execute('PRAGMA integrity_check').fetchone()[0]
        if result != 'ok':
            raise ValueError(result)
    db.close()
    os.rename(temporary, output)
    print(f'Complete: {len(files)} clips, {total_rows:,} rows; {output}; '
          f'{output.stat().st_size / 1e6:.1f} MB; {time.monotonic() - started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
