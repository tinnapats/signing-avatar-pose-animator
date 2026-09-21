import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

from clip_database import database_clips, read_clip_csv
from export_pose_animator_sequence import discover_clips, select_from_text


class DatabaseTests(unittest.TestCase):
    def test_reads_database_without_source_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / 'clips.sqlite3'
            db = sqlite3.connect(database)
            db.executescript("CREATE TABLE metadata(key TEXT, value TEXT);"
                             "INSERT INTO metadata VALUES('format', 'slclean-csv-zlib-v1');"
                             "CREATE TABLE clips(id INTEGER PRIMARY KEY, relative_path TEXT, csv_zlib BLOB);")
            raw = b'frame,part,landmark_id,x,y\n0,pose,11,0.25,0.5\n'
            for name in ['you.csv', 'go.csv', 'high school.csv']:
                db.execute('INSERT INTO clips(relative_path,csv_zlib) VALUES (?,?)',
                           (name, zlib.compress(raw)))
            db.commit()
            db.close()
            with patch('pathlib.Path.rglob', side_effect=AssertionError('No directory scan')):
                clips = discover_clips(database)
                selected = select_from_text('you go high school', clips)
                self.assertEqual([p.stem for p in selected], ['you', 'go', 'high school'])
                self.assertEqual(read_clip_csv(selected[-1]).iloc[0]['x'], 0.25)
                self.assertEqual(read_clip_csv(selected[-1], usecols=['frame']).columns.tolist(), ['frame'])
            self.assertEqual(list(Path(folder).iterdir()), [database])

    def test_rejects_unsupported_format(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / 'clips.sqlite3'
            db = sqlite3.connect(database)
            db.execute('CREATE TABLE metadata(key TEXT, value TEXT)')
            db.commit()
            db.close()
            with self.assertRaises(ValueError):
                database_clips(database)
