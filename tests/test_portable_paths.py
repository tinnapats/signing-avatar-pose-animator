import json
import unittest
from portable_paths import ROOT, loads, relocate_path


class PortablePathsTests(unittest.TestCase):
    def test_old_checkout(self):
        self.assertEqual(relocate_path(r'C:\Users\newtu\signing-avatar-github-tinnapat\SLclean.sqlite3'),
                         (ROOT/'SLclean.sqlite3').as_posix())

    def test_old_model(self):
        old = 'C:/Users/newtu/signing-avatar-pose-animator/.venv/Scripts/python.exe'
        self.assertEqual(relocate_path(old), (ROOT/'.venv/Scripts/python.exe').as_posix())

    def test_visualization_archive(self):
        old = 'C:/Users/newtu/.codex/visualizations/2026/07/27/019fa278-2887-7562-a831-6544ada9c461/model-results/oracle200-20261011/rtm/a.npz'
        self.assertEqual(relocate_path(old), (ROOT/'evaluation/model-results/oracle200-20261011/rtm/a.npz').as_posix())

    def test_no_label_or_number_changes(self):
        data = {'label':'hello','points':[[1.25,2.5]],'url':'https://example.com/model-results/a'}
        self.assertEqual(loads(json.dumps(data)), data)

    def test_original_json_unchanged(self):
        text = '{"path":"C:/Users/newtu/signing-avatar-github-tinnapat/a"}'
        self.assertEqual(loads(text)['path'], (ROOT/'a').as_posix())
        self.assertIn('signing-avatar-github-tinnapat', json.loads(text)['path'])


if __name__ == '__main__':
    unittest.main()
