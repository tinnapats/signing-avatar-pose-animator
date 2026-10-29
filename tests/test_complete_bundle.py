import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from restore_complete_bundle import restore, safe_path


class CompleteBundleTests(unittest.TestCase):
    def fixture(self, root):
        data=b'fixture-content'
        (root/'original.txt').write_bytes(data)
        item=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),stored_as='original.txt')
        (root/'BUNDLE_MANIFEST.json').write_text(json.dumps(dict(format='deduplicated-zip-v1',files=[dict(path='original.txt',**item),dict(path='sub/duplicate.txt',**item)])))
        return data

    def test_restore_copies_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data=self.fixture(root)
            restore(root)
            self.assertEqual((root/'sub/duplicate.txt').read_bytes(),data)
            (root/'sub/duplicate.txt').write_text('changed')
            self.assertEqual((root/'original.txt').read_bytes(),data)
            restore(root)
            self.assertEqual((root/'sub/duplicate.txt').read_text(),'changed')

    def test_reject_changed_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root)
            (root/'sub').mkdir()
            (root/'sub/duplicate.txt').write_text('user work')
            with self.assertRaises(ValueError): restore(root)

    def test_reject_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): safe_path(Path(directory),'../outside')

    def test_reject_corrupt_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root)
            (root/'original.txt').write_text('bad')
            with self.assertRaises(ValueError): restore(root)
