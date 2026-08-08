import copy
import math
import unittest
from pathlib import Path

import pandas as pd

from export_pose_animator_sequence import (
    apply_hand_gaussian_smoothing,
    build_gaussian_kernel,
    estimate_frame_rate,
    estimate_source_aspect,
    enforce_hand_articulation,
    interpolate_hand_kinematically,
    interpolate_frames,
    map_normalized_coordinates,
    repair_hand_tracks,
    repair_invalid_hand_topology,
    resample_clip_to_fps,
    resolve_selected_paths,
    smooth_weighted_series,
    stabilize_hand_bone_lengths,
    suppress_hand_flip_transitions,
    taper_hand_detection_edges,
)


def _hand(score: float, x: float, y: float = 0.0) -> dict:
    return {
        "score": score,
        "keypoints": [
            {
                "landmarkId": landmark_id,
                "score": score,
                "position": {"x": x, "y": y},
            }
            for landmark_id in range(21)
        ],
    }


def _frame(left_score: float, left_x: float) -> dict:
    return {
        "hands": {
            "left": _hand(left_score, left_x),
            "right": _hand(0.0, 0.0),
        }
    }


def _oriented_frame(orientation: int) -> dict:
    frame = _frame(1.0, 0.0)
    keypoints = frame["hands"]["left"]["keypoints"]
    keypoints[0]["position"] = {"x": 0.0, "y": 0.0}
    keypoints[5]["position"] = {"x": 1.0, "y": 0.0}
    keypoints[17]["position"] = {"x": 0.0, "y": float(orientation)}
    return frame


def _topology_frame() -> dict:
    frame = _frame(1.0, 0.0)
    keypoints = frame["hands"]["left"]["keypoints"]
    keypoints[0]["position"] = {"x": 0.0, "y": 0.0}
    finger_y = {1: -1.5, 5: -1.0, 9: -0.3, 13: 0.3, 17: 1.0}
    for chain_start, y in finger_y.items():
        for offset in range(4):
            landmark_id = chain_start + offset
            keypoints[landmark_id]["position"] = {
                "x": float(offset + 1),
                "y": float(y),
            }
    return frame


class HandTrackRepairTests(unittest.TestCase):
    def test_short_internal_gap_is_interpolated(self) -> None:
        frames = [
            _frame(1.0, 0.0),
            _frame(0.0, 0.0),
            _frame(0.0, 0.0),
            _frame(1.0, 30.0),
        ]

        repair_hand_tracks(frames, max_gap_frames=2, fade_frames=1)

        self.assertAlmostEqual(frames[1]["hands"]["left"]["keypoints"][0]["position"]["x"], 10.0)
        self.assertAlmostEqual(frames[2]["hands"]["left"]["keypoints"][0]["position"]["x"], 20.0)
        self.assertEqual(frames[1]["hands"]["left"]["keypoints"][0]["score"], 1.0)
        self.assertEqual(frames[2]["hands"]["left"]["score"], 1.0)

    def test_long_gap_fades_at_each_detected_edge(self) -> None:
        frames = [_frame(1.0, 0.0)]
        frames.extend(_frame(0.0, -100.0) for _ in range(5))
        frames.append(_frame(1.0, 60.0))

        repair_hand_tracks(frames, max_gap_frames=2, fade_frames=2)

        scores = [frame["hands"]["left"]["keypoints"][0]["score"] for frame in frames]
        self.assertAlmostEqual(scores[1], 2.0 / 3.0)
        self.assertAlmostEqual(scores[2], 1.0 / 3.0)
        self.assertEqual(scores[3], 0.0)
        self.assertAlmostEqual(scores[4], 1.0 / 3.0)
        self.assertAlmostEqual(scores[5], 2.0 / 3.0)
        self.assertEqual(frames[1]["hands"]["left"]["keypoints"][0]["position"]["x"], 0.0)
        self.assertEqual(frames[5]["hands"]["left"]["keypoints"][0]["position"]["x"], 60.0)

    def test_missing_placeholders_do_not_pull_weighted_smoothing(self) -> None:
        kernel = build_gaussian_kernel(radius=1, sigma=1.0)
        smoothed = smooth_weighted_series(
            values=[10.0, 1000.0, 10.0],
            confidences=[1.0, 0.0, 1.0],
            kernel=kernel,
        )
        self.assertAlmostEqual(smoothed[1], 10.0)

    def test_hand_smoothing_preserves_zero_confidence_for_fully_missing_hand(self) -> None:
        frames = [_frame(0.0, 100.0) for _ in range(5)]
        apply_hand_gaussian_smoothing(frames, sigma=0.8, radius=1)
        self.assertTrue(all(frame["hands"]["left"]["score"] == 0.0 for frame in frames))

    def test_detected_run_edges_are_tapered_but_stable_center_is_preserved(self) -> None:
        frames = [_frame(0.0, 0.0)]
        frames.extend(_frame(1.0, float(i)) for i in range(7))
        frames.append(_frame(0.0, 0.0))

        taper_hand_detection_edges(frames, edge_frames=3)

        scores = [frame["hands"]["left"]["score"] for frame in frames]
        self.assertEqual(scores, [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0])

    def test_external_fade_never_exceeds_tapered_detection_edge(self) -> None:
        frames = [_frame(score, 0.0) for score in [0.25, 0.5, 0.75]]
        frames.extend(_frame(1.0, float(i)) for i in range(7))
        frames.extend(_frame(score, 0.0) for score in [0.75, 0.5, 0.25])

        taper_hand_detection_edges(frames, edge_frames=3)

        scores = [frame["hands"]["left"]["score"] for frame in frames]
        self.assertEqual(scores[:7], [0.0625, 0.125, 0.1875, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(scores[-7:], [1.0, 0.75, 0.5, 0.25, 0.1875, 0.125, 0.0625])

    def test_consecutive_palm_flip_skips_first_mirrored_frame(self) -> None:
        frames = [_oriented_frame(-1), _oriented_frame(1), _oriented_frame(1)]

        suppress_hand_flip_transitions(frames, orientation_threshold=0.12)

        scores = [frame["hands"]["left"]["score"] for frame in frames]
        self.assertEqual(scores, [1.0, 0.35, 1.0])
        self.assertEqual(
            frames[1]["hands"]["left"]["keypoints"][17]["position"],
            frames[0]["hands"]["left"]["keypoints"][17]["position"],
        )

    def test_kinematic_flip_uses_depth_instead_of_flat_collapse(self) -> None:
        frames = [_oriented_frame(-1), _oriented_frame(1)]
        for frame in frames:
            frame["pose"] = {"score": 1.0, "keypoints": []}
            frame["face"] = {"faceInViewConfidence": 1.0, "positions": []}

        interpolated = interpolate_frames(
            copy.deepcopy(frames),
            upsample_factor=2,
            step_across_hand_flips=False,
        )
        stepped = interpolate_frames(
            copy.deepcopy(frames),
            upsample_factor=2,
            step_across_hand_flips=True,
        )

        self.assertEqual(
            interpolated[1]["hands"]["left"]["keypoints"][17]["position"]["y"],
            0.0,
        )
        mechanical_position = stepped[1]["hands"]["left"]["keypoints"][17]["position"]
        self.assertAlmostEqual(mechanical_position["y"], 0.0, places=6)
        self.assertGreater(abs(mechanical_position["z"]), 0.99)

    def test_finger_joint_keeps_moving_during_regular_interpolation(self) -> None:
        first = _topology_frame()["hands"]["left"]
        second = copy.deepcopy(first)
        second["keypoints"][8]["position"] = {"x": 3.0, "y": -2.0, "z": 0.5}
        first["keypoints"][8]["position"]["z"] = 0.0

        midpoint = interpolate_hand_kinematically(first, second, 0.5)
        start_tip = first["keypoints"][8]["position"]
        middle_tip = midpoint["keypoints"][8]["position"]
        end_tip = second["keypoints"][8]["position"]

        self.assertNotEqual(middle_tip, start_tip)
        self.assertNotEqual(middle_tip, end_tip)
        parent = midpoint["keypoints"][7]["position"]
        bone_length = math.sqrt(
            (middle_tip["x"] - parent["x"]) ** 2
            + (middle_tip["y"] - parent["y"]) ** 2
            + (middle_tip["z"] - parent["z"]) ** 2
        )
        expected_length = (
            math.dist(
                [first["keypoints"][7]["position"].get(axis, 0.0) for axis in ("x", "y", "z")],
                [start_tip.get(axis, 0.0) for axis in ("x", "y", "z")],
            )
            + math.dist(
                [second["keypoints"][7]["position"].get(axis, 0.0) for axis in ("x", "y", "z")],
                [end_tip[axis] for axis in ("x", "y", "z")],
            )
        ) * 0.5
        self.assertAlmostEqual(bone_length, expected_length)

    def test_kinematic_flip_replaces_old_frozen_hold(self) -> None:
        frames = [_oriented_frame(-1), _oriented_frame(1)]
        for frame in frames:
            frame["pose"] = {"score": 1.0, "keypoints": []}
            frame["face"] = {"faceInViewConfidence": 1.0, "positions": []}
        frames[1]["hands"]["left"]["flipHeld"] = True

        output = interpolate_frames(frames, upsample_factor=2, step_across_hand_flips=True)

        self.assertFalse(output[1]["hands"]["left"]["flipHeld"])
        self.assertFalse(output[1]["hands"]["left"]["observed"])
        self.assertTrue(output[1]["hands"]["left"]["mechanicalFlip"])

    def test_kinematic_flip_rotates_through_depth_without_shortening_bone(self) -> None:
        first = _oriented_frame(-1)["hands"]["left"]
        second = _oriented_frame(1)["hands"]["left"]

        midpoint = interpolate_hand_kinematically(first, second, 0.5)
        wrist = midpoint["keypoints"][0]["position"]
        pinky = midpoint["keypoints"][17]["position"]
        length_3d = math.sqrt(
            (pinky["x"] - wrist["x"]) ** 2
            + (pinky["y"] - wrist["y"]) ** 2
            + (pinky["z"] - wrist["z"]) ** 2
        )

        self.assertAlmostEqual(length_3d, 1.0)
        self.assertGreater(abs(pinky["z"] - wrist["z"]), 0.99)
        self.assertFalse(midpoint["flipHeld"])

    def test_hand_smoothing_does_not_blend_across_flip_step(self) -> None:
        frames = [
            _oriented_frame(-1),
            _oriented_frame(-1),
            _oriented_frame(1),
            _oriented_frame(1),
        ]

        apply_hand_gaussian_smoothing(
            frames,
            sigma=1.0,
            radius=1,
            preserve_flip_steps=True,
        )

        self.assertEqual(
            frames[1]["hands"]["left"]["keypoints"][17]["position"]["y"],
            -1.0,
        )
        self.assertEqual(
            frames[2]["hands"]["left"]["keypoints"][17]["position"]["y"],
            1.0,
        )

    def test_shape_stabilizer_repairs_collapsed_finger_bone(self) -> None:
        frames = [_topology_frame(), _topology_frame(), _topology_frame()]
        middle_keypoints = frames[1]["hands"]["left"]["keypoints"]
        wrist_before = dict(middle_keypoints[0]["position"])
        middle_keypoints[6]["position"] = {
            "x": middle_keypoints[5]["position"]["x"] + 0.02,
            "y": middle_keypoints[5]["position"]["y"],
        }
        before = math.hypot(
            middle_keypoints[6]["position"]["x"] - middle_keypoints[5]["position"]["x"],
            middle_keypoints[6]["position"]["y"] - middle_keypoints[5]["position"]["y"],
        )

        stabilized = stabilize_hand_bone_lengths(frames, window_radius=1)

        after = math.hypot(
            middle_keypoints[6]["position"]["x"] - middle_keypoints[5]["position"]["x"],
            middle_keypoints[6]["position"]["y"] - middle_keypoints[5]["position"]["y"],
        )
        self.assertGreater(stabilized, 0)
        self.assertLess(before, 0.1)
        self.assertGreater(after, 0.5)
        self.assertEqual(middle_keypoints[0]["position"], wrist_before)

    def test_articulation_projection_keeps_fixed_bone_lengths_and_wrist(self) -> None:
        frames = [_topology_frame(), _topology_frame(), _topology_frame()]
        middle = frames[1]["hands"]["left"]["keypoints"]
        wrist_before = dict(middle[0]["position"])
        middle[6]["position"] = {"x": 8.0, "y": -1.0}
        middle[7]["position"] = {"x": -4.0, "y": -1.0}
        middle[8]["position"] = {"x": -5.0, "y": -1.0}

        adjusted = enforce_hand_articulation(frames, max_joint_bend_degrees=90.0)

        self.assertGreater(adjusted, 0)
        self.assertEqual(middle[0]["position"], wrist_before)
        reference = frames[0]["hands"]["left"]["keypoints"]
        for parent_id, child_id in [(5, 6), (6, 7), (7, 8)]:
            reference_length = math.dist(
                tuple(reference[parent_id]["position"].values()),
                tuple(reference[child_id]["position"].values()),
            )
            projected_length = math.dist(
                tuple(middle[parent_id]["position"].values()),
                tuple(middle[child_id]["position"].values()),
            )
            self.assertAlmostEqual(projected_length, reference_length, places=6)

    def test_articulation_projection_limits_impossible_joint_reversal(self) -> None:
        frames = [_topology_frame(), _topology_frame(), _topology_frame()]
        middle = frames[1]["hands"]["left"]["keypoints"]
        middle[6]["position"] = {"x": 2.0, "y": -1.0}
        middle[7]["position"] = {"x": 1.0, "y": -1.0}

        enforce_hand_articulation(frames, max_joint_bend_degrees=90.0)

        first = (
            middle[6]["position"]["x"] - middle[5]["position"]["x"],
            middle[6]["position"]["y"] - middle[5]["position"]["y"],
        )
        second = (
            middle[7]["position"]["x"] - middle[6]["position"]["x"],
            middle[7]["position"]["y"] - middle[6]["position"]["y"],
        )
        cosine = sum(a * b for a, b in zip(first, second)) / (
            math.hypot(*first) * math.hypot(*second)
        )
        bend_degrees = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        self.assertLessEqual(bend_degrees, 90.000001)
    def test_invalid_middle_ring_order_uses_nearest_valid_shape(self) -> None:
        valid_before = _topology_frame()
        invalid = _topology_frame()
        valid_after = _topology_frame()
        invalid_keypoints = invalid["hands"]["left"]["keypoints"]
        for offset in range(4):
            middle_id = 9 + offset
            ring_id = 13 + offset
            middle_position = invalid_keypoints[middle_id]["position"]
            invalid_keypoints[middle_id]["position"] = invalid_keypoints[ring_id]["position"]
            invalid_keypoints[ring_id]["position"] = middle_position

        corrections = repair_invalid_hand_topology([valid_before, invalid, valid_after])

        self.assertEqual(corrections, 1)
        self.assertEqual(
            invalid["hands"]["left"]["keypoints"][9]["position"],
            valid_before["hands"]["left"]["keypoints"][9]["position"],
        )

    def test_topology_repair_does_not_raise_detection_confidence(self) -> None:
        valid = _topology_frame()
        invalid = _topology_frame()
        invalid["hands"]["left"]["score"] = 0.1
        for keypoint in invalid["hands"]["left"]["keypoints"]:
            keypoint["score"] = 0.1
        for offset in range(4):
            middle_id = 9 + offset
            ring_id = 13 + offset
            middle_position = invalid["hands"]["left"]["keypoints"][middle_id]["position"]
            invalid["hands"]["left"]["keypoints"][middle_id]["position"] = (
                invalid["hands"]["left"]["keypoints"][ring_id]["position"]
            )
            invalid["hands"]["left"]["keypoints"][ring_id]["position"] = middle_position

        repair_invalid_hand_topology([valid, invalid])

        self.assertAlmostEqual(invalid["hands"]["left"]["score"], 0.1)
        self.assertTrue(
            all(
                keypoint["score"] == 0.1
                for keypoint in invalid["hands"]["left"]["keypoints"]
            )
        )

    def test_topology_repair_does_not_copy_opposite_orientation(self) -> None:
        same_orientation = _topology_frame()
        invalid_padding = _topology_frame()
        target = _topology_frame()
        opposite_orientation = _topology_frame()
        for keypoint in opposite_orientation["hands"]["left"]["keypoints"]:
            keypoint["position"]["y"] *= -1.0
        for frame in [invalid_padding, target]:
            keypoints = frame["hands"]["left"]["keypoints"]
            for offset in range(4):
                middle_id = 9 + offset
                ring_id = 13 + offset
                middle_position = keypoints[middle_id]["position"]
                keypoints[middle_id]["position"] = keypoints[ring_id]["position"]
                keypoints[ring_id]["position"] = middle_position

        repair_invalid_hand_topology(
            [same_orientation, invalid_padding, target, opposite_orientation]
        )

        self.assertEqual(
            target["hands"]["left"]["keypoints"][9]["position"],
            same_orientation["hands"]["left"]["keypoints"][9]["position"],
        )

    def test_repeated_sign_tokens_are_preserved(self) -> None:
        clip = Path("abdomen.csv")
        clips = {"abdomen.csv": clip}

        selected = resolve_selected_paths(
            clips=clips,
            data_dir=Path("."),
            file_tokens=[],
            text="abdomen abdomen",
        )

        self.assertEqual(selected, [clip, clip])

    def test_normalized_coordinates_preserve_source_aspect_ratio(self) -> None:
        clip = pd.DataFrame([{"x": 0.25, "y": 0.25, "px": 50, "py": 25}])
        aspect = estimate_source_aspect(clip)
        clip = map_normalized_coordinates(clip, width=100, height=100, source_aspect=aspect)
        point = clip.iloc[0]
        self.assertAlmostEqual(aspect, 2.0)
        self.assertAlmostEqual(float(point.x), 25.0)
        self.assertAlmostEqual(float(point.y), 37.5)

    def test_timestamp_resampling_preserves_duration_and_whole_frames(self) -> None:
        clip = pd.DataFrame(
            [
                {
                    "frame": frame,
                    "time_sec": frame / 25.0,
                    "part": "right_hand",
                    "landmark_id": 0,
                    "x": float(frame),
                    "y": 0.0,
                }
                for frame in range(26)
            ]
        )

        self.assertAlmostEqual(estimate_frame_rate(clip), 25.0)
        resampled = resample_clip_to_fps(clip, target_fps=30.0)

        self.assertEqual(resampled["frame"].nunique(), 31)
        self.assertAlmostEqual(float(resampled["time_sec"].max()), 1.0)
        self.assertTrue(set(resampled["x"]).issubset(set(clip["x"])))


if __name__ == "__main__":
    unittest.main()
