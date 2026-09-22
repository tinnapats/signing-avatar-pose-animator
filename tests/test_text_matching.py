import unittest
from pathlib import Path

from export_pose_animator_sequence import select_from_text


class TextMatchingTests(unittest.TestCase):
    def setUp(self):
        names = ['you', 'go', 'high', 'school', 'high school', 'i', 'love', "don't"]
        self.clips = {name + '.csv': Path(name + '.csv') for name in names}

    def words(self, text):
        return [path.stem for path in select_from_text(text, self.clips)]

    def test_punctuation_keeps_words_and_longest_phrase(self):
        for text in ['you, go high school!', 'YOU,go high school!!!',
                     '“you” — go (high school).', 'you; go: high school?',
                     'you / go / high school']:
            with self.subTest(text=text):
                self.assertEqual(self.words(text), ['you', 'go', 'high school'])

    def test_exact_phrase_before_individual_words(self):
        self.assertEqual(self.words('“high school!”'), ['high school'])

    def test_plain_text_and_repeated_words(self):
        self.assertEqual(self.words('I love you'), ['i', 'love', 'you'])
        self.assertEqual(self.words('you, you!'), ['you', 'you'])

    def test_apostrophes_within_words(self):
        self.assertEqual(self.words("‘don’t’ go!"), ["don't", 'go'])

    def test_empty_and_punctuation_only(self):
        self.assertEqual(self.words(''), [])
        self.assertEqual(self.words('...!?'), [])
