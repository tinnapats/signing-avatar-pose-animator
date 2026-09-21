"""Read SLclean SQLite clips without extracting files to disk."""
import io
import sqlite3
import zlib
from contextlib import closing
from functools import lru_cache
from pathlib import Path

import pandas as pd


def _connect(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


@lru_cache(maxsize=4)
def _clip_names(path, modified_ns, size):
    with closing(_connect(path)) as db:
        version = db.execute("SELECT value FROM metadata WHERE key='format'").fetchone()
        if version != ('slclean-csv-zlib-v1',):
            raise ValueError('Unsupported clip database format')
        return tuple(row[0] for row in db.execute('SELECT relative_path FROM clips ORDER BY id'))


def database_clips(path):
    path = Path(path).resolve()
    stat = path.stat()
    names = _clip_names(str(path), stat.st_mtime_ns, stat.st_size)
    # Logical paths retain clip labels used by the player; no CSV is extracted.
    return {name: path / name for name in names}


@lru_cache(maxsize=8)
def _clip_bytes(database, relative_path, modified_ns, size):
    with closing(_connect(database)) as db:
        row = db.execute('SELECT csv_zlib FROM clips WHERE relative_path=?', (relative_path,)).fetchone()
    if row is None:
        raise ValueError(f'Clip not found in database: {relative_path}')
    return zlib.decompress(row[0])


def read_clip_csv(path, **kwargs):
    path = Path(path)
    for parent in path.parents:
        if parent.suffix.lower() in ('.sqlite3', '.sqlite', '.db') and parent.is_file():
            stat = parent.stat()
            raw = _clip_bytes(str(parent.resolve()), path.relative_to(parent).as_posix(),
                              stat.st_mtime_ns, stat.st_size)
            return pd.read_csv(io.BytesIO(raw), **kwargs)
    return pd.read_csv(path, **kwargs)
