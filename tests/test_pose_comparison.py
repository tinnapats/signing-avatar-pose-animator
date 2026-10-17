import unittest

from pose_comparison import angle, compare_points, similarity_scores


class ComparisonTests(unittest.TestCase):
    def test_similarity_endpoints(self):
        perfect = similarity_scores({'elbowDegrees': 0, 'wristDistance': 0, 'fingerDegrees': 0})
        self.assertEqual(perfect['overall'], 100)
        opposite = similarity_scores({'elbowDegrees': 180, 'wristDistance': 1, 'fingerDegrees': 90})
        self.assertEqual(opposite['elbowDegrees'], 0)
        self.assertEqual(opposite['wristDistance'], 50)
        self.assertEqual(opposite['fingerDegrees'], 50)

    def test_missing_or_invalid_metrics_do_not_get_overall_score(self):
        for missing in (None, float('nan'), -1):
            result = similarity_scores({'elbowDegrees': 0, 'wristDistance': 0, 'fingerDegrees': missing})
            self.assertIsNone(result['overall'])
            self.assertIsNone(result['fingerDegrees'])

    def setUp(self):
        self.pose = {11: [-1, 0], 12: [1, 0], 13: [-1, 1], 14: [1, 1], 15: [0, 1], 16: [2, 1]}

    def test_translation_and_uniform_scale_do_not_change_metrics(self):
        transformed = {i: [x * 3 + 8, y * 3 - 4] for i, (x, y) in self.pose.items()}
        self.assertEqual(compare_points(self.pose, transformed), {'wristDistance': 0, 'elbowDegrees': 0})

    def test_changed_arm_is_detected(self):
        wrong = dict(self.pose)
        wrong[15] = [-1, 2]
        result = compare_points(self.pose, wrong)
        self.assertGreater(result['elbowDegrees'], 40)
        self.assertGreater(result['wristDistance'], .3)

    def test_missing_detection_is_not_perfect_score(self):
        self.assertEqual(compare_points(self.pose, {}), {'wristDistance': None, 'elbowDegrees': None})

    def test_collapsed_joint_cannot_have_valid_angle(self):
        self.assertIsNone(angle({1: [0, 0], 2: [0, 0], 3: [1, 0]}, (1, 2, 3)))

    def test_swapped_left_right_is_not_automatically_corrected(self):
        swapped = {i: self.pose[j] for i, j in [(11, 12), (12, 11), (13, 14), (14, 13), (15, 16), (16, 15)]}
        self.assertGreater(compare_points(self.pose, swapped)['wristDistance'], .5)
