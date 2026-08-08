import argparse
import bisect
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

POSE_PART_ORDER: List[Tuple[str, int]] = [
    ("nose", 0),
    ("leftEye", 2),
    ("rightEye", 5),
    ("leftEar", 7),
    ("rightEar", 8),
    ("leftShoulder", 11),
    ("rightShoulder", 12),
    ("leftElbow", 13),
    ("rightElbow", 14),
    ("leftWrist", 15),
    ("rightWrist", 16),
    ("leftHip", 23),
    ("rightHip", 24),
    ("leftKnee", 25),
    ("rightKnee", 26),
    ("leftAnkle", 27),
    ("rightAnkle", 28),
]

HAND_SIDES: List[str] = ["left", "right"]
HAND_PART_NAMES: Dict[str, str] = {
    "left": "left_hand",
    "right": "right_hand",
}
HAND_LANDMARK_IDS: List[int] = list(range(21))
HAND_FINGER_CHAINS: List[List[int]] = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
]
HAND_PALM_MCP_IDS: List[int] = [5, 9, 13, 17]
HAND_SHAPE_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
# A parent-first tree used to reconstruct the hand during temporal interpolation.
# Palm cross-links are intentionally excluded: they are constraints, not bones.
HAND_KINEMATIC_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
DEFAULT_HAND_MAX_GAP_FRAMES = 4
DEFAULT_HAND_FADE_FRAMES = 3
DEFAULT_HAND_FLIP_ORIENTATION_THRESHOLD = 0.12
DEFAULT_HAND_FLIP_HOLD_SCORE = 0.35
DEFAULT_HAND_MAX_JOINT_BEND_DEGREES = 115.0
HAND_ARTICULATION_CHAINS: List[List[int]] = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]

FACE_PART_NAME2INDEX: Dict[str, int] = {
    "topMid": 10,
    "rightTop0": 67,
    "rightTop1": 54,
    "leftTop0": 297,
    "leftTop1": 284,
    "rightJaw0": 21,
    "rightJaw1": 162,
    "rightJaw2": 127,
    "rightJaw3": 234,
    "rightJaw4": 132,
    "rightJaw5": 172,
    "rightJaw6": 150,
    "rightJaw7": 176,
    "jawMid": 152,
    "leftJaw7": 400,
    "leftJaw6": 379,
    "leftJaw5": 397,
    "leftJaw4": 361,
    "leftJaw3": 454,
    "leftJaw2": 356,
    "leftJaw1": 389,
    "leftJaw0": 251,
    "rightBrow0": 46,
    "rightBrow1": 53,
    "rightBrow2": 52,
    "rightBrow3": 65,
    "rightBrow4": 55,
    "leftBrow4": 285,
    "leftBrow3": 295,
    "leftBrow2": 282,
    "leftBrow1": 283,
    "leftBrow0": 276,
    "nose0": 6,
    "nose1": 197,
    "nose2": 195,
    "nose3": 5,
    "rightNose0": 48,
    "rightNose1": 220,
    "nose4": 4,
    "leftNose1": 440,
    "leftNose0": 278,
    "rightEye0": 33,
    "rightEye1": 160,
    "rightEye2": 158,
    "rightEye3": 133,
    "rightEye4": 153,
    "rightEye5": 144,
    "leftEye3": 362,
    "leftEye2": 385,
    "leftEye1": 387,
    "leftEye0": 263,
    "leftEye5": 373,
    "leftEye4": 380,
    "rightMouthCorner": 61,
    "rightUpperLipTop0": 40,
    "rightUpperLipTop1": 37,
    "upperLipTopMid": 0,
    "leftUpperLipTop1": 267,
    "leftUpperLipTop0": 270,
    "leftMouthCorner": 291,
    "leftLowerLipBottom0": 321,
    "leftLowerLipBottom1": 314,
    "lowerLipBottomMid": 17,
    "rightLowerLipBottom1": 84,
    "rightLowerLipBottom0": 91,
    "rightMiddleLip": 78,
    "rightUpperLipBottom1": 81,
    "upperLipBottomMid": 13,
    "leftUpperLipBottom1": 311,
    "leftMiddleLip": 308,
    "leftLowerLipTop0": 402,
    "lowerLipTopMid": 14,
    "rightLowerLipTop0": 178,
}

FACE_PART_ORDER: List[str] = [
    "topMid",
    "rightTop0",
    "rightTop1",
    "leftTop0",
    "leftTop1",
    "rightJaw0",
    "rightJaw1",
    "rightJaw2",
    "rightJaw3",
    "rightJaw4",
    "rightJaw5",
    "rightJaw6",
    "rightJaw7",
    "jawMid",
    "leftJaw7",
    "leftJaw6",
    "leftJaw5",
    "leftJaw4",
    "leftJaw3",
    "leftJaw2",
    "leftJaw1",
    "leftJaw0",
    "rightBrow0",
    "rightBrow1",
    "rightBrow2",
    "rightBrow3",
    "rightBrow4",
    "leftBrow4",
    "leftBrow3",
    "leftBrow2",
    "leftBrow1",
    "leftBrow0",
    "nose0",
    "nose1",
    "nose2",
    "nose3",
    "rightNose0",
    "rightNose1",
    "nose4",
    "leftNose1",
    "leftNose0",
    "rightEye0",
    "rightEye1",
    "rightEye2",
    "rightEye3",
    "rightEye4",
    "rightEye5",
    "leftEye3",
    "leftEye2",
    "leftEye1",
    "leftEye0",
    "leftEye5",
    "leftEye4",
    "rightMouthCorner",
    "rightUpperLipTop0",
    "rightUpperLipTop1",
    "upperLipTopMid",
    "leftUpperLipTop1",
    "leftUpperLipTop0",
    "leftMouthCorner",
    "leftLowerLipBottom0",
    "leftLowerLipBottom1",
    "lowerLipBottomMid",
    "rightLowerLipBottom1",
    "rightLowerLipBottom0",
    "rightMiddleLip",
    "rightUpperLipBottom1",
    "upperLipBottomMid",
    "leftUpperLipBottom1",
    "leftMiddleLip",
    "leftLowerLipTop0",
    "lowerLipTopMid",
    "rightLowerLipTop0",
]

NEEDED_POSE_IDS = {lid for _, lid in POSE_PART_ORDER}
NEEDED_FACE_IDS = {FACE_PART_NAME2INDEX[name] for name in FACE_PART_ORDER}
EXPECTED_COLS = ["frame", "part", "landmark_id", "x", "y"]
OPTIONAL_COORD_COLS = ["px", "py", "time_sec", "z", "visibility"]
SEQUENCE_COLS = EXPECTED_COLS + ["z"]


def normalize_name(name: str) -> str:
    value = Path(name).name
    if "." in value:
        value = value.rsplit(".", 1)[0]
    value = value.lower()
    for suffix in ["_holistic_keypoints", "-holistic_keypoints", "_keypoints", "-keypoints"]:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    value = value.replace("_", " ").replace("-", " ")
    return value.strip()


def discover_clips(data_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not data_dir.exists():
        return out
    for p in data_dir.rglob("*.csv"):
        try:
            key = str(p.relative_to(data_dir)).replace("\\", "/")
        except ValueError:
            key = str(p)
        out[key] = p
    return out


def resolve_file_token(token: str, clips: Dict[str, Path], data_dir: Path) -> Path:
    candidate = Path(token)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    rel_candidate = (data_dir / candidate).resolve()
    if rel_candidate.exists():
        return rel_candidate

    normalized = token.replace("\\", "/")
    if normalized in clips:
        return clips[normalized]

    base = Path(token).name.lower()
    matches = [path for key, path in clips.items() if Path(key).name.lower() == base]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous file token '{token}' (matched multiple basenames)")

    raise ValueError(f"Could not resolve file token '{token}'")


def select_from_text(text: str, clips: Dict[str, Path]) -> List[Path]:
    text = (text or "").strip().lower()
    if not text:
        return []

    normalized_index: Dict[str, List[Path]] = {}
    for key, path in clips.items():
        normalized_index.setdefault(normalize_name(Path(key).name), []).append(path)

    selected: List[Path] = []
    exact = normalized_index.get(text, [])
    if exact:
        selected.append(exact[0])
        return selected

    for token in text.split():
        matches = normalized_index.get(token, [])
        if matches:
            selected.append(matches[0])
    return selected


def estimate_source_aspect(df: pd.DataFrame) -> float:
    """Estimate source pixel aspect from normalized and pixel coordinate pairs."""
    if "px" not in df.columns or "py" not in df.columns:
        return 1.0
    aspect_rows = df[
        df["px"].notna()
        & df["py"].notna()
        & (df["x"].abs() > 0.05)
        & (df["y"].abs() > 0.05)
    ]
    if aspect_rows.empty:
        return 1.0
    source_width = float((aspect_rows["px"] / aspect_rows["x"]).median())
    source_height = float((aspect_rows["py"] / aspect_rows["y"]).median())
    candidate_aspect = source_width / max(source_height, 1e-6)
    if math.isfinite(candidate_aspect) and 0.4 <= candidate_aspect <= 4.0:
        return candidate_aspect
    return 1.0


def map_normalized_coordinates(
    df: pd.DataFrame,
    width: int,
    height: int,
    source_aspect: float,
) -> pd.DataFrame:
    """Map normalized coordinates into a letterboxed canvas without distortion."""
    aspect = max(float(source_aspect), 1e-6)
    display_width = min(float(width), float(height) * aspect)
    display_height = display_width / aspect
    offset_x = (float(width) - display_width) * 0.5
    offset_y = (float(height) - display_height) * 0.5
    df["x"] = offset_x + df["x"] * display_width
    df["y"] = offset_y + df["y"] * display_height
    # MediaPipe z uses roughly the same normalized scale as x. Preserve it so
    # palm turns can be interpolated in 3D before projecting back to Canvas.
    if "z" in df.columns:
        df["z"] = df["z"] * display_width
    return df


def estimate_frame_rate(df: pd.DataFrame) -> float | None:
    """Estimate a clip's native FPS from its per-frame timestamps."""
    if "time_sec" not in df.columns:
        return None
    frame_times = (
        df[["frame", "time_sec"]]
        .dropna()
        .drop_duplicates("frame")
        .sort_values("frame")
    )
    if len(frame_times) < 2:
        return None
    deltas = [
        float(current) - float(previous)
        for previous, current in zip(frame_times["time_sec"], frame_times["time_sec"].iloc[1:])
        if float(current) - float(previous) > 1e-8
    ]
    if not deltas:
        return None
    median_delta = float(statistics.median(deltas))
    return 1.0 / median_delta if median_delta > 1e-8 else None


def resample_clip_to_fps(df: pd.DataFrame, target_fps: float) -> pd.DataFrame:
    """Resample complete source frames by timestamp without morphing hand shapes."""
    fps = max(1.0, float(target_fps))
    if df.empty or "time_sec" not in df.columns:
        return df
    frame_times = (
        df[["frame", "time_sec"]]
        .dropna()
        .drop_duplicates("frame")
        .sort_values("time_sec")
    )
    if len(frame_times) < 2:
        return df
    source_frames = [int(value) for value in frame_times["frame"]]
    source_times = [float(value) for value in frame_times["time_sec"]]
    start_time = source_times[0]
    relative_times = [value - start_time for value in source_times]
    duration = relative_times[-1]
    if duration <= 1e-8:
        return df
    target_count = max(2, int(round(duration * fps)) + 1)
    frame_groups = {int(frame): group for frame, group in df.groupby("frame", sort=False)}
    output_parts: List[pd.DataFrame] = []
    for target_index in range(target_count):
        target_time = min(duration, float(target_index) / fps)
        insertion = bisect.bisect_left(relative_times, target_time)
        if insertion <= 0:
            source_index = 0
        elif insertion >= len(relative_times):
            source_index = len(relative_times) - 1
        else:
            before = insertion - 1
            after = insertion
            source_index = (
                before
                if target_time - relative_times[before] <= relative_times[after] - target_time
                else after
            )
        source_frame = source_frames[source_index]
        sampled = frame_groups[source_frame].copy()
        sampled["frame"] = target_index
        sampled["time_sec"] = start_time + target_time
        output_parts.append(sampled)
    return pd.concat(output_parts, ignore_index=True)


def load_clip(
    path: Path,
    width: int,
    height: int,
    target_fps: float | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda column: column in EXPECTED_COLS + OPTIONAL_COORD_COLS)
    missing_cols = [column for column in EXPECTED_COLS if column not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns {missing_cols}: {path}")
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df["landmark_id"] = pd.to_numeric(df["landmark_id"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    for column in OPTIONAL_COORD_COLS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "z" not in df.columns:
        df["z"] = 0.0
    else:
        df["z"] = df["z"].fillna(0.0)
    df = df.dropna(subset=EXPECTED_COLS)

    df["frame"] = df["frame"].astype("int32")
    df["landmark_id"] = df["landmark_id"].astype("int32")
    df["part"] = df["part"].astype("string")
    df["x"] = df["x"].astype("float32")
    df["y"] = df["y"].astype("float32")

    keep_pose = (df["part"] == "pose") & df["landmark_id"].isin(NEEDED_POSE_IDS)
    keep_face = (df["part"] == "face") & df["landmark_id"].isin(NEEDED_FACE_IDS)
    keep_left_hand = (df["part"] == HAND_PART_NAMES["left"]) & df["landmark_id"].isin(HAND_LANDMARK_IDS)
    keep_right_hand = (df["part"] == HAND_PART_NAMES["right"]) & df["landmark_id"].isin(HAND_LANDMARK_IDS)
    df = df[keep_pose | keep_face | keep_left_hand | keep_right_hand].copy()
    if df.empty:
        return df

    q95_x = float(df["x"].abs().quantile(0.95))
    q95_y = float(df["y"].abs().quantile(0.95))
    if q95_x <= 2.0 and q95_y <= 2.0:
        source_aspect = estimate_source_aspect(df)
        df = map_normalized_coordinates(df, width, height, source_aspect)
    if target_fps is not None:
        df = resample_clip_to_fps(df, target_fps)

    return df[SEQUENCE_COLS]


def concat_clips(
    paths: List[Path],
    width: int,
    height: int,
    pause_frames: int,
    target_fps: float | None = None,
) -> pd.DataFrame:
    seq_parts: List[pd.DataFrame] = []
    frame_offset = 0

    for path in paths:
        clip = load_clip(path, width, height, target_fps=target_fps)
        if clip.empty:
            continue

        clip = clip.sort_values(["frame", "part", "landmark_id"]).reset_index(drop=True)
        clip["frame"] = clip["frame"] - int(clip["frame"].min())
        clip["frame"] = clip["frame"] + frame_offset
        seq_parts.append(clip)

        last_frame = int(clip["frame"].max())
        frame_offset = last_frame + 1

        if pause_frames > 0:
            hold = clip[clip["frame"] == last_frame].copy()
            for offset in range(1, pause_frames + 1):
                h = hold.copy()
                h["frame"] = last_frame + offset
                seq_parts.append(h)
            frame_offset = last_frame + pause_frames + 1

    if not seq_parts:
        return pd.DataFrame(columns=SEQUENCE_COLS)

    combined = pd.concat(seq_parts, ignore_index=True)
    combined = combined.sort_values(["frame", "part", "landmark_id"]).reset_index(drop=True)
    return combined


def _xy_from_lookup(
    lookup: Dict[int, Tuple[float, float]],
    lid: int,
    fallback: Tuple[float, float],
) -> Tuple[float, float, float]:
    if lid in lookup:
        x, y = lookup[lid]
        return float(x), float(y), 1.0
    return float(fallback[0]), float(fallback[1]), 0.0


def _xyz_from_lookup(
    lookup: Dict[int, Tuple[float, float, float]],
    lid: int,
    fallback: Tuple[float, float, float],
) -> Tuple[float, float, float, float]:
    if lid in lookup:
        x, y, z = lookup[lid]
        return float(x), float(y), float(z), 1.0
    return float(fallback[0]), float(fallback[1]), float(fallback[2]), 0.0


def build_frames(seq_df: pd.DataFrame, width: int, height: int, max_frames: int) -> List[Dict]:
    if seq_df.empty:
        return []

    frames: List[Dict] = []
    grouped = seq_df.groupby("frame", sort=True)

    center = (float(width) * 0.5, float(height) * 0.5)
    face_center = (float(width) * 0.5, float(height) * 0.35)

    last_pose: Dict[str, Tuple[float, float]] = {name: center for name, _ in POSE_PART_ORDER}
    last_face: Dict[str, Tuple[float, float]] = {name: face_center for name in FACE_PART_ORDER}
    last_hands: Dict[str, Dict[int, Tuple[float, float, float]]] = {
        side: {} for side in HAND_SIDES
    }

    for frame_idx, frame_df in grouped:
        if max_frames > 0 and len(frames) >= max_frames:
            break

        pose_lookup = {
            int(row.landmark_id): (float(row.x), float(row.y))
            for row in frame_df[frame_df["part"] == "pose"][["landmark_id", "x", "y"]]
            .drop_duplicates("landmark_id")
            .itertuples(index=False)
        }
        face_lookup = {
            int(row.landmark_id): (float(row.x), float(row.y))
            for row in frame_df[frame_df["part"] == "face"][["landmark_id", "x", "y"]]
            .drop_duplicates("landmark_id")
            .itertuples(index=False)
        }
        hand_lookups: Dict[str, Dict[int, Tuple[float, float, float]]] = {}
        for side in HAND_SIDES:
            part_name = HAND_PART_NAMES[side]
            hand_lookups[side] = {
                int(row.landmark_id): (float(row.x), float(row.y), float(row.z))
                for row in frame_df[frame_df["part"] == part_name][["landmark_id", "x", "y", "z"]]
                .drop_duplicates("landmark_id")
                .itertuples(index=False)
            }

        keypoints = []
        pose_scores: List[float] = []
        for part_name, lid in POSE_PART_ORDER:
            fallback = last_pose.get(part_name, center)
            x, y, score = _xy_from_lookup(pose_lookup, lid, fallback)
            if score > 0:
                last_pose[part_name] = (x, y)
            pose_scores.append(score)
            keypoints.append(
                {
                    "part": part_name,
                    "score": score,
                    "position": {"x": x, "y": y},
                }
            )

        face_positions: List[float] = []
        found_face = 0
        nose_fallback = last_pose.get("nose", face_center)
        for part_name in FACE_PART_ORDER:
            lid = FACE_PART_NAME2INDEX[part_name]
            fallback = last_face.get(part_name, nose_fallback)
            x, y, score = _xy_from_lookup(face_lookup, lid, fallback)
            if score > 0:
                last_face[part_name] = (x, y)
                found_face += 1
            face_positions.extend([x, y])

        face_conf = 1.0 if found_face >= int(len(FACE_PART_ORDER) * 0.6) else 0.0
        pose_score = float(sum(pose_scores) / max(len(pose_scores), 1))
        hands: Dict[str, Dict] = {}
        for side in HAND_SIDES:
            pose_wrist_name = "leftWrist" if side == "left" else "rightWrist"
            pose_wrist = last_pose.get(pose_wrist_name, center)
            wrist_fallback = (float(pose_wrist[0]), float(pose_wrist[1]), 0.0)
            hand_lookup = hand_lookups[side]
            hand_keypoints: List[Dict] = []
            hand_scores: List[float] = []
            for landmark_id in HAND_LANDMARK_IDS:
                fallback = last_hands[side].get(landmark_id, wrist_fallback)
                x, y, z, score = _xyz_from_lookup(hand_lookup, landmark_id, fallback)
                if score > 0:
                    last_hands[side][landmark_id] = (x, y, z)
                hand_scores.append(score)
                hand_keypoints.append(
                    {
                        "landmarkId": int(landmark_id),
                        "score": score,
                        "position": {"x": x, "y": y, "z": z},
                    }
                )
            hands[side] = {
                "score": float(sum(hand_scores) / max(len(hand_scores), 1)),
                "keypoints": hand_keypoints,
                "observed": len(hand_lookup) == len(HAND_LANDMARK_IDS),
                "flipHeld": False,
            }

        frames.append(
            {
                "frame": int(frame_idx),
                "pose": {
                    "score": pose_score,
                    "keypoints": keypoints,
                },
                "face": {
                    "faceInViewConfidence": face_conf,
                    "positions": face_positions,
                },
                "hands": hands,
            }
        )

    return frames


def _lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * float(t)


def _position_xyz(keypoint: Dict) -> Tuple[float, float, float]:
    position = keypoint.get("position", {})
    return (
        float(position.get("x", 0.0)),
        float(position.get("y", 0.0)),
        float(position.get("z", 0.0)),
    )


def _vector_length(vector: Tuple[float, float, float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def _unit_vector(vector: Tuple[float, float, float]) -> Tuple[float, float, float] | None:
    length = _vector_length(vector)
    if length <= 1e-8:
        return None
    return tuple(float(value) / length for value in vector)


def _slerp_direction(
    first: Tuple[float, float, float],
    second: Tuple[float, float, float],
    t: float,
) -> Tuple[float, float, float]:
    """Spherically interpolate a bone direction without shrinking it."""
    a = _unit_vector(first)
    b = _unit_vector(second)
    if a is None and b is None:
        return (1.0, 0.0, 0.0)
    if a is None:
        return b  # type: ignore[return-value]
    if b is None:
        return a
    dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
    if dot > 0.9995:
        blended = tuple(_lerp(x, y, t) for x, y in zip(a, b))
        return _unit_vector(blended) or a
    if dot < -0.9995:
        # Antiparallel directions have no unique shortest arc. Choose a stable
        # perpendicular so the bone rotates through depth instead of collapsing.
        axis_seed = (0.0, 0.0, 1.0) if abs(a[2]) < 0.9 else (0.0, 1.0, 0.0)
        seed_projection = sum(a[index] * axis_seed[index] for index in range(3))
        perpendicular = _unit_vector(
            tuple(axis_seed[index] - a[index] * seed_projection for index in range(3))
        ) or (0.0, 1.0, 0.0)
        angle = math.pi * float(t)
        return tuple(
            a[index] * math.cos(angle) + perpendicular[index] * math.sin(angle)
            for index in range(3)
        )
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    weight_a = math.sin((1.0 - float(t)) * theta) / sin_theta
    weight_b = math.sin(float(t) * theta) / sin_theta
    return tuple(a[index] * weight_a + b[index] * weight_b for index in range(3))


def interpolate_hand_kinematically(hand0: Dict, hand1: Dict, t: float) -> Dict:
    """Interpolate joint rotations in 3D while preserving every bone length.

    Linear x/y interpolation makes a turning palm and its fingers collapse at
    the midpoint. Reconstructing the parent-first hand tree from interpolated
    3D directions keeps the fingers articulated and lets only real perspective
    foreshortening appear on the 2D Canvas.
    """
    keypoints0 = hand0.get("keypoints", [])
    keypoints1 = hand1.get("keypoints", [])
    if len(keypoints0) < len(HAND_LANDMARK_IDS) or len(keypoints1) < len(HAND_LANDMARK_IDS):
        return {"score": 0.0, "keypoints": [], "observed": False, "flipHeld": False}

    positions: Dict[int, Tuple[float, float, float]] = {}
    root0 = _position_xyz(keypoints0[0])
    root1 = _position_xyz(keypoints1[0])
    positions[0] = tuple(_lerp(a, b, t) for a, b in zip(root0, root1))

    for parent_id, child_id in HAND_KINEMATIC_EDGES:
        parent0 = _position_xyz(keypoints0[parent_id])
        child0 = _position_xyz(keypoints0[child_id])
        parent1 = _position_xyz(keypoints1[parent_id])
        child1 = _position_xyz(keypoints1[child_id])
        vector0 = tuple(child0[index] - parent0[index] for index in range(3))
        vector1 = tuple(child1[index] - parent1[index] for index in range(3))
        length = _lerp(_vector_length(vector0), _vector_length(vector1), t)
        direction = _slerp_direction(vector0, vector1, t)
        parent = positions[parent_id]
        positions[child_id] = tuple(
            parent[index] + direction[index] * length for index in range(3)
        )

    keypoints: List[Dict] = []
    for landmark_id in HAND_LANDMARK_IDS:
        position = positions.get(landmark_id)
        if position is None:
            first = _position_xyz(keypoints0[landmark_id])
            second = _position_xyz(keypoints1[landmark_id])
            position = tuple(_lerp(a, b, t) for a, b in zip(first, second))
        keypoints.append(
            {
                "landmarkId": landmark_id,
                "score": _lerp(
                    float(keypoints0[landmark_id].get("score", 0.0)),
                    float(keypoints1[landmark_id].get("score", 0.0)),
                    t,
                ),
                "position": {"x": position[0], "y": position[1], "z": position[2]},
            }
        )

    orientation0 = _hand_orientation(hand0)
    orientation1 = _hand_orientation(hand1)
    is_flip = bool(
        orientation0 is not None
        and orientation1 is not None
        and orientation0 * orientation1 < 0.0
    )
    return {
        "score": _lerp(float(hand0.get("score", 0.0)), float(hand1.get("score", 0.0)), t),
        "keypoints": keypoints,
        "observed": False,
        "flipHeld": False,
        "mechanicalFlip": is_flip,
    }


def interpolate_hand_linearly(hand0: Dict, hand1: Dict, t: float) -> Dict:
    """Compatibility path for callers that explicitly disable hand mechanics."""
    keypoints: List[Dict] = []
    for first, second in zip(hand0.get("keypoints", []), hand1.get("keypoints", [])):
        a = _position_xyz(first)
        b = _position_xyz(second)
        position = tuple(_lerp(x, y, t) for x, y in zip(a, b))
        keypoints.append(
            {
                "landmarkId": int(first["landmarkId"]),
                "score": _lerp(float(first.get("score", 0.0)), float(second.get("score", 0.0)), t),
                "position": {"x": position[0], "y": position[1], "z": position[2]},
            }
        )
    return {
        "score": _lerp(float(hand0.get("score", 0.0)), float(hand1.get("score", 0.0)), t),
        "keypoints": keypoints,
        "observed": False,
        "flipHeld": False,
        "mechanicalFlip": False,
    }


def _hand_orientation(hand: Dict) -> float | None:
    keypoints = hand.get("keypoints", []) if hand else []
    if len(keypoints) <= 17:
        return None
    wrist = keypoints[0]
    index_mcp = keypoints[5]
    pinky_mcp = keypoints[17]
    if min(
        float(wrist.get("score", 0.0)),
        float(index_mcp.get("score", 0.0)),
        float(pinky_mcp.get("score", 0.0)),
    ) <= 0.0:
        return None
    wx = float(wrist["position"]["x"])
    wy = float(wrist["position"]["y"])
    ax = float(index_mcp["position"]["x"]) - wx
    ay = float(index_mcp["position"]["y"]) - wy
    bx = float(pinky_mcp["position"]["x"]) - wx
    by = float(pinky_mcp["position"]["y"]) - wy
    denominator = math.hypot(ax, ay) * math.hypot(bx, by)
    if denominator <= 1e-8:
        return None
    return (ax * by - ay * bx) / denominator


def interpolate_frames(
    frames: List[Dict],
    upsample_factor: int,
    step_across_hand_flips: bool = True,
) -> List[Dict]:
    factor = max(1, int(upsample_factor))
    if factor <= 1 or len(frames) < 2:
        for i, frame in enumerate(frames):
            frame["frame"] = i
        return frames

    out: List[Dict] = []
    for i in range(len(frames) - 1):
        f0 = frames[i]
        f1 = frames[i + 1]
        out.append(f0)

        kp0 = f0["pose"]["keypoints"]
        kp1 = f1["pose"]["keypoints"]
        face0 = f0["face"]["positions"]
        face1 = f1["face"]["positions"]
        hands0 = f0.get("hands", {})
        hands1 = f1.get("hands", {})

        for step in range(1, factor):
            t = float(step) / float(factor)
            interp_keypoints: List[Dict] = []
            for a, b in zip(kp0, kp1):
                interp_keypoints.append(
                    {
                        "part": a["part"],
                        "score": _lerp(float(a["score"]), float(b["score"]), t),
                        "position": {
                            "x": _lerp(float(a["position"]["x"]), float(b["position"]["x"]), t),
                            "y": _lerp(float(a["position"]["y"]), float(b["position"]["y"]), t),
                        },
                    }
                )

            interp_face: List[float] = [
                _lerp(float(v0), float(v1), t) for v0, v1 in zip(face0, face1)
            ]
            interp_hands: Dict[str, Dict] = {}
            for side in HAND_SIDES:
                hand0 = hands0.get(side, {"score": 0.0, "keypoints": []})
                hand1 = hands1.get(side, {"score": 0.0, "keypoints": []})
                interp_hands[side] = (
                    interpolate_hand_kinematically(hand0, hand1, t)
                    if step_across_hand_flips
                    else interpolate_hand_linearly(hand0, hand1, t)
                )

            out.append(
                {
                    "frame": 0,
                    "pose": {
                        "score": _lerp(float(f0["pose"]["score"]), float(f1["pose"]["score"]), t),
                        "keypoints": interp_keypoints,
                    },
                    "face": {
                        "faceInViewConfidence": _lerp(
                            float(f0["face"]["faceInViewConfidence"]),
                            float(f1["face"]["faceInViewConfidence"]),
                            t,
                        ),
                        "positions": interp_face,
                    },
                    "hands": interp_hands,
                }
            )

    out.append(frames[-1])
    for i, frame in enumerate(out):
        frame["frame"] = i
    return out


def repair_hand_tracks(
    frames: List[Dict],
    max_gap_frames: int,
    fade_frames: int,
) -> List[Dict]:
    """Repair short hand-detection gaps and soften longer visibility changes.

    Short internal gaps are linearly interpolated. Longer gaps keep a small,
    confidence-weighted hold near each detected run so the rendered hand fades
    instead of disappearing in a single frame.
    """
    if not frames:
        return frames

    max_gap = max(0, int(max_gap_frames))
    fade = max(0, int(fade_frames))

    for side in HAND_SIDES:
        for hand_idx in range(len(HAND_LANDMARK_IDS)):
            keypoints = [frame["hands"][side]["keypoints"][hand_idx] for frame in frames]
            valid_indices = [
                i for i, keypoint in enumerate(keypoints) if float(keypoint.get("score", 0.0)) > 0.0
            ]
            if not valid_indices:
                continue

            def _set_faded(index: int, source_index: int, score: float) -> None:
                if index < 0 or index >= len(keypoints):
                    return
                target = keypoints[index]
                if float(target.get("score", 0.0)) >= score:
                    return
                source = keypoints[source_index]
                target["position"]["x"] = float(source["position"]["x"])
                target["position"]["y"] = float(source["position"]["y"])
                if "z" in source["position"] or "z" in target["position"]:
                    target["position"]["z"] = float(source["position"].get("z", 0.0))
                target["score"] = float(score)

            first = valid_indices[0]
            last = valid_indices[-1]
            for distance in range(1, fade + 1):
                confidence = float(keypoints[first]["score"]) * (1.0 - distance / float(fade + 1))
                _set_faded(first - distance, first, confidence)
                confidence = float(keypoints[last]["score"]) * (1.0 - distance / float(fade + 1))
                _set_faded(last + distance, last, confidence)

            for left, right in zip(valid_indices, valid_indices[1:]):
                gap = right - left - 1
                if gap <= 0:
                    continue

                left_keypoint = keypoints[left]
                right_keypoint = keypoints[right]
                if gap <= max_gap:
                    fill_score = min(
                        float(left_keypoint.get("score", 0.0)),
                        float(right_keypoint.get("score", 0.0)),
                    )
                    for step in range(1, gap + 1):
                        t = float(step) / float(gap + 1)
                        target = keypoints[left + step]
                        target["position"]["x"] = _lerp(
                            float(left_keypoint["position"]["x"]),
                            float(right_keypoint["position"]["x"]),
                            t,
                        )
                        target["position"]["y"] = _lerp(
                            float(left_keypoint["position"]["y"]),
                            float(right_keypoint["position"]["y"]),
                            t,
                        )
                        target["position"]["z"] = _lerp(
                            float(left_keypoint["position"].get("z", 0.0)),
                            float(right_keypoint["position"].get("z", 0.0)),
                            t,
                        )
                        target["score"] = fill_score
                    continue

                for distance in range(1, min(fade, gap) + 1):
                    confidence = float(left_keypoint["score"]) * (
                        1.0 - distance / float(fade + 1)
                    )
                    _set_faded(left + distance, left, confidence)
                    confidence = float(right_keypoint["score"]) * (
                        1.0 - distance / float(fade + 1)
                    )
                    _set_faded(right - distance, right, confidence)

        for frame in frames:
            hand = frame["hands"][side]
            scores = [float(kp.get("score", 0.0)) for kp in hand.get("keypoints", [])]
            hand["score"] = float(sum(scores) / max(len(scores), 1))

    return frames


def taper_hand_detection_edges(frames: List[Dict], edge_frames: int) -> List[Dict]:
    """Down-weight unstable entry/exit frames around detected hand runs."""
    edge = max(0, int(edge_frames))
    if not frames or edge <= 0:
        return frames

    for side in HAND_SIDES:
        strong = [
            i
            for i, frame in enumerate(frames)
            if float(frame["hands"][side].get("score", 0.0)) >= 0.999
        ]
        runs: List[Tuple[int, int]] = []
        if strong:
            run_start = strong[0]
            run_end = strong[0]
            for index in strong[1:]:
                if index == run_end + 1:
                    run_end = index
                else:
                    runs.append((run_start, run_end))
                    run_start = index
                    run_end = index
            runs.append((run_start, run_end))

        for run_start, run_end in runs:
            factors: Dict[int, float] = {}
            if run_start > 0:
                for offset in range(min(edge, run_end - run_start + 1)):
                    factors[run_start + offset] = min(
                        factors.get(run_start + offset, 1.0),
                        float(offset + 1) / float(edge + 1),
                    )
            if run_end < len(frames) - 1:
                for offset in range(min(edge, run_end - run_start + 1)):
                    index = run_end - offset
                    factors[index] = min(
                        factors.get(index, 1.0),
                        float(offset + 1) / float(edge + 1),
                    )

            for index, factor in factors.items():
                hand = frames[index]["hands"][side]
                for keypoint in hand.get("keypoints", []):
                    keypoint["score"] = float(keypoint.get("score", 0.0)) * factor

            if run_start > 0:
                boundary_keypoints = frames[run_start]["hands"][side].get("keypoints", [])
                boundary_scores = [float(kp.get("score", 0.0)) for kp in boundary_keypoints]
                boundary_score = float(sum(boundary_scores) / max(len(boundary_scores), 1))
                for distance in range(1, edge + 1):
                    index = run_start - distance
                    if index < 0:
                        break
                    cap = boundary_score * (1.0 - distance / float(edge + 1))
                    for keypoint in frames[index]["hands"][side].get("keypoints", []):
                        keypoint["score"] = min(float(keypoint.get("score", 0.0)), cap)
            if run_end < len(frames) - 1:
                boundary_keypoints = frames[run_end]["hands"][side].get("keypoints", [])
                boundary_scores = [float(kp.get("score", 0.0)) for kp in boundary_keypoints]
                boundary_score = float(sum(boundary_scores) / max(len(boundary_scores), 1))
                for distance in range(1, edge + 1):
                    index = run_end + distance
                    if index >= len(frames):
                        break
                    cap = boundary_score * (1.0 - distance / float(edge + 1))
                    for keypoint in frames[index]["hands"][side].get("keypoints", []):
                        keypoint["score"] = min(float(keypoint.get("score", 0.0)), cap)

        for frame in frames:
            hand = frame["hands"][side]
            scores = [float(kp.get("score", 0.0)) for kp in hand.get("keypoints", [])]
            hand["score"] = float(sum(scores) / max(len(scores), 1))

    return frames


def suppress_hand_flip_transitions(
    frames: List[Dict],
    orientation_threshold: float = DEFAULT_HAND_FLIP_ORIENTATION_THRESHOLD,
) -> List[Dict]:
    """Hide edge-on palm frames between opposite 2D hand orientations.

    Interpolating through a palm flip collapses and mirrors the finger layout.
    Removing only the transition confidence lets the renderer fade out the old
    orientation and fade in the new one without drawing the invalid midpoint.
    """
    threshold = max(0.01, min(0.95, float(orientation_threshold)))
    for side in HAND_SIDES:
        stable_orientations: List[Tuple[int, float]] = []
        for index, frame in enumerate(frames):
            hand = frame["hands"][side]
            if float(hand.get("score", 0.0)) < 0.18:
                continue
            keypoints = hand.get("keypoints", [])
            if len(keypoints) <= 17:
                continue
            orientation = _hand_orientation(hand)
            if orientation is not None and abs(orientation) >= threshold:
                stable_orientations.append((index, orientation))

        suppress_indices = set()
        for previous, current in zip(stable_orientations, stable_orientations[1:]):
            previous_index, previous_orientation = previous
            current_index, current_orientation = current
            if previous_orientation * current_orientation >= 0:
                continue
            transition = list(range(previous_index + 1, current_index))
            if not transition:
                transition = [current_index]
            suppress_indices.update(transition)

        for index in suppress_indices:
            hand = frames[index]["hands"][side]
            previous_stable = max(
                (stable_index for stable_index, _ in stable_orientations if stable_index < index),
                default=None,
            )
            if previous_stable is None:
                continue
            source_hand = frames[previous_stable]["hands"][side]
            for keypoint, source in zip(
                hand.get("keypoints", []),
                source_hand.get("keypoints", []),
            ):
                keypoint["position"]["x"] = float(source["position"]["x"])
                keypoint["position"]["y"] = float(source["position"]["y"])
                if "z" in source["position"] or "z" in keypoint["position"]:
                    keypoint["position"]["z"] = float(source["position"].get("z", 0.0))
                keypoint["score"] = DEFAULT_HAND_FLIP_HOLD_SCORE
            hand["score"] = DEFAULT_HAND_FLIP_HOLD_SCORE
            hand["flipHeld"] = True
            hand["observed"] = False

    return frames


def count_hand_flip_transitions(
    frames: List[Dict],
    orientation_threshold: float = DEFAULT_HAND_FLIP_ORIENTATION_THRESHOLD,
) -> int:
    """Count stable palm-facing changes without modifying any hand pose."""
    threshold = max(0.01, min(0.95, float(orientation_threshold)))
    transition_count = 0
    for side in HAND_SIDES:
        previous_orientation: float | None = None
        for frame in frames:
            hand = frame["hands"][side]
            if float(hand.get("score", 0.0)) < 0.18:
                continue
            orientation = _hand_orientation(hand)
            if orientation is None or abs(orientation) < threshold:
                continue
            if previous_orientation is not None and previous_orientation * orientation < 0.0:
                transition_count += 1
            previous_orientation = orientation
    return transition_count


def _segments_intersect(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
    d: Tuple[float, float],
) -> bool:
    def _orientation(
        p: Tuple[float, float],
        q: Tuple[float, float],
        r: Tuple[float, float],
    ) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return o1 * o2 < 0.0 and o3 * o4 < 0.0


def has_valid_hand_topology(hand: Dict) -> bool:
    """Validate canonical MCP order and reject crossing finger chains."""
    keypoints = hand.get("keypoints", []) if hand else []
    if len(keypoints) < len(HAND_LANDMARK_IDS):
        return False
    points = {
        int(keypoint["landmarkId"]): (
            float(keypoint["position"]["x"]),
            float(keypoint["position"]["y"]),
        )
        for keypoint in keypoints
        if float(keypoint.get("score", 0.0)) > 0.0
    }
    if any(landmark_id not in points for landmark_id in HAND_LANDMARK_IDS):
        return False

    index_base = points[5]
    pinky_base = points[17]
    axis_x = pinky_base[0] - index_base[0]
    axis_y = pinky_base[1] - index_base[1]
    axis_length_sq = axis_x * axis_x + axis_y * axis_y
    if axis_length_sq <= 1e-8:
        return False
    palm_projections = []
    for landmark_id in HAND_PALM_MCP_IDS:
        point = points[landmark_id]
        projection = (
            (point[0] - index_base[0]) * axis_x
            + (point[1] - index_base[1]) * axis_y
        ) / axis_length_sq
        palm_projections.append(projection)
    if any(
        palm_projections[index] > palm_projections[index + 1] + 1e-5
        for index in range(len(palm_projections) - 1)
    ):
        return False

    finger_segments: List[Tuple[int, Tuple[float, float], Tuple[float, float]]] = []
    for chain_index, chain in enumerate(HAND_FINGER_CHAINS):
        for start_id, end_id in zip(chain, chain[1:]):
            finger_segments.append((chain_index, points[start_id], points[end_id]))
    for segment_index, first in enumerate(finger_segments):
        for second in finger_segments[segment_index + 1 :]:
            if first[0] == second[0]:
                continue
            if _segments_intersect(first[1], first[2], second[1], second[2]):
                return False
    return True


def repair_invalid_hand_topology(frames: List[Dict]) -> int:
    """Replace invalid finger ordering with the nearest valid hand shape."""
    correction_count = 0
    for side in HAND_SIDES:
        candidates = [
            index
            for index, frame in enumerate(frames)
            if float(frame["hands"][side].get("score", 0.0)) >= 0.05
            and has_valid_hand_topology(frame["hands"][side])
        ]
        if not candidates:
            continue
        invalid_indices = [
            index
            for index, frame in enumerate(frames)
            if float(frame["hands"][side].get("score", 0.0)) >= 0.05
            and not has_valid_hand_topology(frame["hands"][side])
        ]
        for index in invalid_indices:
            target_orientation = _hand_orientation(frames[index]["hands"][side])
            compatible_candidates = candidates
            if target_orientation is not None:
                same_orientation = [
                    candidate
                    for candidate in candidates
                    if (
                        _hand_orientation(frames[candidate]["hands"][side]) is not None
                        and target_orientation
                        * float(_hand_orientation(frames[candidate]["hands"][side]))
                        > 0.0
                    )
                ]
                if same_orientation:
                    compatible_candidates = same_orientation
            source_index = min(
                compatible_candidates,
                key=lambda candidate: abs(candidate - index),
            )
            source_hand = frames[source_index]["hands"][side]
            target_hand = frames[index]["hands"][side]
            for target, source in zip(
                target_hand.get("keypoints", []),
                source_hand.get("keypoints", []),
            ):
                target["position"]["x"] = float(source["position"]["x"])
                target["position"]["y"] = float(source["position"]["y"])
                if "z" in source["position"] or "z" in target["position"]:
                    target["position"]["z"] = float(source["position"].get("z", 0.0))
            scores = [
                float(keypoint.get("score", 0.0))
                for keypoint in target_hand.get("keypoints", [])
            ]
            target_hand["score"] = float(sum(scores) / max(len(scores), 1))
            correction_count += 1
    return correction_count


def count_invalid_hand_topologies(frames: List[Dict], minimum_score: float = 0.05) -> int:
    return sum(
        1
        for frame in frames
        for side in HAND_SIDES
        if float(frame["hands"][side].get("score", 0.0)) >= float(minimum_score)
        and not has_valid_hand_topology(frame["hands"][side])
    )


def build_gaussian_kernel(radius: int, sigma: float) -> List[float]:
    r = max(0, int(radius))
    s = float(sigma)
    if r <= 0 or s <= 0:
        return [1.0]

    weights = [math.exp(-0.5 * (float(x) / s) ** 2) for x in range(-r, r + 1)]
    total = sum(weights)
    if total <= 0:
        return [1.0]
    return [w / total for w in weights]


def smooth_series(
    values: List[float],
    kernel: List[float],
    segment_ids: List[int] | None = None,
) -> List[float]:
    if len(values) < 3 or len(kernel) <= 1:
        return [float(v) for v in values]

    radius = len(kernel) // 2
    n = len(values)
    out: List[float] = []
    for i in range(n):
        acc = 0.0
        for k, weight in enumerate(kernel):
            idx = i + k - radius
            if idx < 0:
                idx = 0
            elif idx >= n:
                idx = n - 1
            if segment_ids is not None and segment_ids[idx] != segment_ids[i]:
                continue
            acc += float(values[idx]) * float(weight)
        if segment_ids is None:
            out.append(acc)
            continue
        used_weight = sum(
            float(weight)
            for k, weight in enumerate(kernel)
            if segment_ids[min(max(i + k - radius, 0), n - 1)] == segment_ids[i]
        )
        out.append(acc / used_weight if used_weight > 1e-8 else float(values[i]))
    return out


def apply_gaussian_smoothing(frames: List[Dict], sigma: float, radius: int) -> List[Dict]:
    if len(frames) < 3:
        return frames
    kernel = build_gaussian_kernel(radius, sigma)
    if len(kernel) <= 1:
        return frames

    pose_count = len(POSE_PART_ORDER)
    for pose_idx in range(pose_count):
        xs = [float(frame["pose"]["keypoints"][pose_idx]["position"]["x"]) for frame in frames]
        ys = [float(frame["pose"]["keypoints"][pose_idx]["position"]["y"]) for frame in frames]
        xs_sm = smooth_series(xs, kernel)
        ys_sm = smooth_series(ys, kernel)
        for i, frame in enumerate(frames):
            frame["pose"]["keypoints"][pose_idx]["position"]["x"] = xs_sm[i]
            frame["pose"]["keypoints"][pose_idx]["position"]["y"] = ys_sm[i]

    face_pos_count = len(FACE_PART_ORDER) * 2
    for face_idx in range(face_pos_count):
        series = [float(frame["face"]["positions"][face_idx]) for frame in frames]
        smoothed = smooth_series(series, kernel)
        for i, frame in enumerate(frames):
            frame["face"]["positions"][face_idx] = smoothed[i]

    return frames


def smooth_weighted_series(
    values: List[float],
    confidences: List[float],
    kernel: List[float],
    segment_ids: List[int] | None = None,
) -> List[float]:
    """Smooth a coordinate without letting missing placeholders pull the track."""
    if len(values) < 3 or len(kernel) <= 1:
        return [float(v) for v in values]

    radius = len(kernel) // 2
    n = len(values)
    out: List[float] = []
    for i in range(n):
        weighted_value = 0.0
        total_weight = 0.0
        for k, kernel_weight in enumerate(kernel):
            idx = min(max(i + k - radius, 0), n - 1)
            if segment_ids is not None and segment_ids[idx] != segment_ids[i]:
                continue
            confidence = max(0.0, float(confidences[idx]))
            weight = float(kernel_weight) * confidence
            weighted_value += float(values[idx]) * weight
            total_weight += weight
        out.append(weighted_value / total_weight if total_weight > 1e-8 else float(values[i]))
    return out


def _hand_orientation_segments(frames: List[Dict], side: str) -> List[int]:
    """Split a hand track wherever its stable 2D palm orientation changes sign."""
    segment_ids: List[int] = []
    segment_id = 0
    previous_sign: int | None = None
    for frame in frames:
        hand = frame.get("hands", {}).get(side, {})
        orientation = _hand_orientation(hand)
        if orientation is not None and abs(orientation) > 1e-8:
            current_sign = 1 if orientation > 0 else -1
            if previous_sign is not None and current_sign != previous_sign:
                segment_id += 1
            previous_sign = current_sign
        segment_ids.append(segment_id)
    return segment_ids


def apply_hand_gaussian_smoothing(
    frames: List[Dict],
    sigma: float,
    radius: int,
    preserve_flip_steps: bool = True,
) -> List[Dict]:
    """Apply lighter, confidence-aware smoothing to detailed hand landmarks."""
    if len(frames) < 3:
        return frames
    kernel = build_gaussian_kernel(radius, sigma)
    if len(kernel) <= 1:
        return frames

    for side in HAND_SIDES:
        segment_ids = _hand_orientation_segments(frames, side) if preserve_flip_steps else None
        for hand_idx in range(len(HAND_LANDMARK_IDS)):
            keypoints = [frame["hands"][side]["keypoints"][hand_idx] for frame in frames]
            scores = [float(keypoint.get("score", 0.0)) for keypoint in keypoints]
            xs = [float(keypoint["position"]["x"]) for keypoint in keypoints]
            ys = [float(keypoint["position"]["y"]) for keypoint in keypoints]
            zs = [float(keypoint["position"].get("z", 0.0)) for keypoint in keypoints]
            xs_sm = smooth_weighted_series(xs, scores, kernel, segment_ids=segment_ids)
            ys_sm = smooth_weighted_series(ys, scores, kernel, segment_ids=segment_ids)
            zs_sm = smooth_weighted_series(zs, scores, kernel, segment_ids=segment_ids)
            scores_sm = smooth_series(scores, kernel, segment_ids=segment_ids)
            for i, keypoint in enumerate(keypoints):
                keypoint["position"]["x"] = xs_sm[i]
                keypoint["position"]["y"] = ys_sm[i]
                keypoint["position"]["z"] = zs_sm[i]
                keypoint["score"] = min(1.0, max(0.0, scores_sm[i]))

        for frame in frames:
            hand = frame["hands"][side]
            scores = [float(kp.get("score", 0.0)) for kp in hand.get("keypoints", [])]
            hand["score"] = float(sum(scores) / max(len(scores), 1))

    return frames


def stabilize_hand_bone_lengths(
    frames: List[Dict],
    window_radius: int = 3,
    minimum_ratio: float = 0.72,
    maximum_ratio: float = 1.28,
    iterations: int = 3,
) -> int:
    """Constrain severe local bone-length outliers without freezing joint angles."""
    if not frames:
        return 0

    radius = max(1, int(window_radius))
    min_ratio = max(0.1, min(1.0, float(minimum_ratio)))
    max_ratio = max(1.0, float(maximum_ratio))
    iteration_count = max(1, int(iterations))
    stabilized_frames = set()

    for side in HAND_SIDES:
        keypoint_tracks = [frame["hands"][side]["keypoints"] for frame in frames]
        original_vectors: List[List[Tuple[float, float, float, float] | None]] = []
        for keypoints in keypoint_tracks:
            frame_vectors: List[Tuple[float, float, float, float] | None] = []
            for parent_id, child_id in HAND_SHAPE_CONNECTIONS:
                parent = keypoints[parent_id]
                child = keypoints[child_id]
                if min(float(parent.get("score", 0.0)), float(child.get("score", 0.0))) <= 0.0:
                    frame_vectors.append(None)
                    continue
                dx = float(child["position"]["x"]) - float(parent["position"]["x"])
                dy = float(child["position"]["y"]) - float(parent["position"]["y"])
                dz = float(child["position"].get("z", 0.0)) - float(parent["position"].get("z", 0.0))
                frame_vectors.append((dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz)))
            original_vectors.append(frame_vectors)

        targets: List[List[float | None]] = [
            [None for _ in HAND_SHAPE_CONNECTIONS] for _ in frames
        ]
        fallback_directions: List[List[Tuple[float, float, float] | None]] = [
            [None for _ in HAND_SHAPE_CONNECTIONS] for _ in frames
        ]
        for frame_index in range(len(frames)):
            start = max(0, frame_index - radius)
            end = min(len(frames), frame_index + radius + 1)
            for edge_index in range(len(HAND_SHAPE_CONNECTIONS)):
                candidates: List[Tuple[int, float, float, float, float]] = []
                for candidate_index in range(start, end):
                    vector = original_vectors[candidate_index][edge_index]
                    if vector is None:
                        continue
                    parent_id, child_id = HAND_SHAPE_CONNECTIONS[edge_index]
                    candidate_keypoints = keypoint_tracks[candidate_index]
                    confidence = min(
                        float(candidate_keypoints[parent_id].get("score", 0.0)),
                        float(candidate_keypoints[child_id].get("score", 0.0)),
                    )
                    if confidence < 0.18 or vector[3] <= 1e-8:
                        continue
                    candidates.append((candidate_index, vector[0], vector[1], vector[2], vector[3]))
                current = original_vectors[frame_index][edge_index]
                if current is None or not candidates:
                    continue
                reference_length = float(statistics.median(item[4] for item in candidates))
                current_length = float(current[3])
                lower_bound = reference_length * min_ratio
                upper_bound = reference_length * max_ratio
                targets[frame_index][edge_index] = min(
                    upper_bound,
                    max(lower_bound, current_length),
                )
                direction_source = min(
                    candidates,
                    key=lambda item: (abs(item[4] - reference_length), abs(item[0] - frame_index)),
                )
                direction_length = max(direction_source[4], 1e-8)
                fallback_directions[frame_index][edge_index] = (
                    direction_source[1] / direction_length,
                    direction_source[2] / direction_length,
                    direction_source[3] / direction_length,
                )

        for frame_index, keypoints in enumerate(keypoint_tracks):
            has_depth = any("z" in keypoint.get("position", {}) for keypoint in keypoints)
            wrist_position = {
                "x": float(keypoints[0]["position"]["x"]),
                "y": float(keypoints[0]["position"]["y"]),
                "z": float(keypoints[0]["position"].get("z", 0.0)),
            }
            frame_was_stabilized = False
            for _ in range(iteration_count):
                for edge_index, (parent_id, child_id) in enumerate(HAND_SHAPE_CONNECTIONS):
                    target_length = targets[frame_index][edge_index]
                    if target_length is None:
                        continue
                    parent = keypoints[parent_id]["position"]
                    child = keypoints[child_id]["position"]
                    dx = float(child["x"]) - float(parent["x"])
                    dy = float(child["y"]) - float(parent["y"])
                    dz = float(child.get("z", 0.0)) - float(parent.get("z", 0.0))
                    current_length = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if abs(current_length - target_length) <= max(1e-5, target_length * 1e-3):
                        continue
                    if current_length <= 1e-8:
                        fallback = fallback_directions[frame_index][edge_index]
                        if fallback is None:
                            continue
                        unit_x, unit_y, unit_z = fallback
                    else:
                        unit_x = dx / current_length
                        unit_y = dy / current_length
                        unit_z = dz / current_length
                    correction_x = unit_x * (current_length - target_length)
                    correction_y = unit_y * (current_length - target_length)
                    correction_z = unit_z * (current_length - target_length)
                    if parent_id == 0:
                        child["x"] = float(child["x"]) - correction_x
                        child["y"] = float(child["y"]) - correction_y
                        if has_depth:
                            child["z"] = float(child.get("z", 0.0)) - correction_z
                    else:
                        parent["x"] = float(parent["x"]) + correction_x * 0.5
                        parent["y"] = float(parent["y"]) + correction_y * 0.5
                        child["x"] = float(child["x"]) - correction_x * 0.5
                        child["y"] = float(child["y"]) - correction_y * 0.5
                        if has_depth:
                            parent["z"] = float(parent.get("z", 0.0)) + correction_z * 0.5
                            child["z"] = float(child.get("z", 0.0)) - correction_z * 0.5
                    frame_was_stabilized = True
                keypoints[0]["position"]["x"] = wrist_position["x"]
                keypoints[0]["position"]["y"] = wrist_position["y"]
                if has_depth:
                    keypoints[0]["position"]["z"] = wrist_position["z"]
            if frame_was_stabilized:
                stabilized_frames.add((side, frame_index))

    return len(stabilized_frames)


def enforce_hand_articulation(
    frames: List[Dict],
    max_joint_bend_degrees: float = DEFAULT_HAND_MAX_JOINT_BEND_DEGREES,
    minimum_score: float = 0.05,
) -> int:
    """Rebuild every finger as a fixed-length 3D articulated bone chain.

    Temporal smoothing can subtly stretch individual landmarks even after
    severe outliers have been repaired. This final parent-first projection
    anchors the wrist, uses robust observed-frame bone lengths, and clamps only
    impossible joint reversals. Natural joint rotation remains free.
    """
    if not frames:
        return 0

    maximum_bend = math.radians(
        max(15.0, min(175.0, float(max_joint_bend_degrees)))
    )
    score_threshold = max(0.0, float(minimum_score))
    adjusted_frames = set()

    for side in HAND_SIDES:
        observed_samples: Dict[
            Tuple[int, int], List[Tuple[float, Tuple[float, float, float]]]
        ] = {edge: [] for edge in HAND_KINEMATIC_EDGES}
        all_samples: Dict[
            Tuple[int, int], List[Tuple[float, Tuple[float, float, float]]]
        ] = {edge: [] for edge in HAND_KINEMATIC_EDGES}

        for frame in frames:
            hand = frame.get("hands", {}).get(side, {})
            keypoints = hand.get("keypoints", [])
            if float(hand.get("score", 0.0)) < score_threshold or len(keypoints) < 21:
                continue
            for edge in HAND_KINEMATIC_EDGES:
                parent_id, child_id = edge
                parent = keypoints[parent_id]
                child = keypoints[child_id]
                if min(
                    float(parent.get("score", 0.0)),
                    float(child.get("score", 0.0)),
                ) <= 0.0:
                    continue
                parent_position = _position_xyz(parent)
                child_position = _position_xyz(child)
                vector = tuple(
                    child_position[index] - parent_position[index]
                    for index in range(3)
                )
                length = _vector_length(vector)
                direction = _unit_vector(vector)
                if direction is None or length <= 1e-8:
                    continue
                sample = (length, direction)
                all_samples[edge].append(sample)
                if bool(hand.get("observed", False)):
                    observed_samples[edge].append(sample)

        target_lengths: Dict[Tuple[int, int], float] = {}
        fallback_directions: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        for edge in HAND_KINEMATIC_EDGES:
            samples = observed_samples[edge] or all_samples[edge]
            if not samples:
                continue
            target_length = float(statistics.median(sample[0] for sample in samples))
            target_lengths[edge] = target_length
            fallback_directions[edge] = min(
                samples,
                key=lambda sample: abs(sample[0] - target_length),
            )[1]

        if not target_lengths:
            continue

        for frame_index, frame in enumerate(frames):
            hand = frame.get("hands", {}).get(side, {})
            keypoints = hand.get("keypoints", [])
            if float(hand.get("score", 0.0)) < score_threshold or len(keypoints) < 21:
                continue
            if any(float(keypoint.get("score", 0.0)) <= 0.0 for keypoint in keypoints[:21]):
                continue

            original = [_position_xyz(keypoint) for keypoint in keypoints[:21]]
            reconstructed: Dict[int, Tuple[float, float, float]] = {0: original[0]}
            has_depth = any("z" in keypoint.get("position", {}) for keypoint in keypoints[:21])

            for chain in HAND_ARTICULATION_CHAINS:
                previous_direction: Tuple[float, float, float] | None = None
                for parent_id, child_id in zip(chain, chain[1:]):
                    edge = (parent_id, child_id)
                    raw_vector = tuple(
                        original[child_id][axis] - original[parent_id][axis]
                        for axis in range(3)
                    )
                    direction = (
                        _unit_vector(raw_vector)
                        or fallback_directions.get(edge)
                        or previous_direction
                        or (1.0, 0.0, 0.0)
                    )
                    if previous_direction is not None:
                        dot = max(
                            -1.0,
                            min(
                                1.0,
                                sum(
                                    previous_direction[axis] * direction[axis]
                                    for axis in range(3)
                                ),
                            ),
                        )
                        bend = math.acos(dot)
                        if bend > maximum_bend:
                            if not has_depth and dot < -0.9995:
                                perpendicular = (
                                    -previous_direction[1],
                                    previous_direction[0],
                                    0.0,
                                )
                                direction = (
                                    previous_direction[0] * math.cos(maximum_bend)
                                    + perpendicular[0] * math.sin(maximum_bend),
                                    previous_direction[1] * math.cos(maximum_bend)
                                    + perpendicular[1] * math.sin(maximum_bend),
                                    0.0,
                                )
                            else:
                                direction = _slerp_direction(
                                    previous_direction,
                                    direction,
                                    maximum_bend / bend,
                                )

                    target_length = target_lengths.get(edge, _vector_length(raw_vector))
                    parent_position = reconstructed[parent_id]
                    reconstructed[child_id] = tuple(
                        parent_position[axis] + direction[axis] * target_length
                        for axis in range(3)
                    )
                    previous_direction = direction

            frame_adjusted = False
            for landmark_id, position in reconstructed.items():
                previous = original[landmark_id]
                if _vector_length(
                    tuple(position[axis] - previous[axis] for axis in range(3))
                ) > 1e-4:
                    frame_adjusted = True
                target = keypoints[landmark_id]["position"]
                target["x"] = position[0]
                target["y"] = position[1]
                if has_depth:
                    target["z"] = position[2]
            if frame_adjusted:
                adjusted_frames.add((side, frame_index))

    return len(adjusted_frames)

def fit_frames_to_canvas(frames: List[Dict], width: int, height: int) -> List[Dict]:
    if not frames:
        return frames

    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    def _update_bounds(x: float, y: float) -> None:
        nonlocal min_x, max_x, min_y, max_y
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    for frame in frames:
        for kp in frame["pose"]["keypoints"]:
            p = kp["position"]
            _update_bounds(float(p["x"]), float(p["y"]))
        face_pos = frame["face"]["positions"]
        for i in range(0, len(face_pos), 2):
            _update_bounds(float(face_pos[i]), float(face_pos[i + 1]))
        for side in HAND_SIDES:
            hand = frame.get("hands", {}).get(side)
            if not hand:
                continue
            for kp in hand.get("keypoints", []):
                if float(kp.get("score", 0.0)) <= 0.0:
                    continue
                p = kp["position"]
                _update_bounds(float(p["x"]), float(p["y"]))

    src_w = max(max_x - min_x, 1.0)
    src_h = max(max_y - min_y, 1.0)
    target_w = float(width) * 0.62
    target_h = float(height) * 0.82
    scale = min(target_w / src_w, target_h / src_h)
    scale = max(scale, 1e-6)

    src_cx = (min_x + max_x) * 0.5
    src_cy = (min_y + max_y) * 0.5
    dst_cx = float(width) * 0.5
    dst_cy = float(height) * 0.58

    def _tx(x: float) -> float:
        return dst_cx + (x - src_cx) * scale

    def _ty(y: float) -> float:
        return dst_cy + (y - src_cy) * scale

    for frame in frames:
        for kp in frame["pose"]["keypoints"]:
            p = kp["position"]
            p["x"] = _tx(float(p["x"]))
            p["y"] = _ty(float(p["y"]))
        face_pos = frame["face"]["positions"]
        for i in range(0, len(face_pos), 2):
            face_pos[i] = _tx(float(face_pos[i]))
            face_pos[i + 1] = _ty(float(face_pos[i + 1]))
        for side in HAND_SIDES:
            hand = frame.get("hands", {}).get(side)
            if not hand:
                continue
            for kp in hand.get("keypoints", []):
                p = kp["position"]
                p["x"] = _tx(float(p["x"]))
                p["y"] = _ty(float(p["y"]))
                p["z"] = float(p.get("z", 0.0)) * scale

    return frames


def resolve_selected_paths(
    clips: Dict[str, Path],
    data_dir: Path,
    file_tokens: List[str],
    text: str,
) -> List[Path]:
    selected_paths: List[Path] = []
    for token in file_tokens:
        selected_paths.append(resolve_file_token(token, clips, data_dir))
    if text:
        selected_paths.extend(select_from_text(text, clips))

    # Repeated clips are meaningful in signed phrases, so preserve token order
    # and repetition instead of de-duplicating paths.
    return selected_paths


def build_payload(
    data_dir: Path,
    file_tokens: List[str],
    text: str,
    fps: float,
    width: int,
    height: int,
    pause_frames: int,
    max_frames: int,
    upsample_factor: int,
    gaussian_sigma: float,
    gaussian_radius: int,
    hand_gaussian_sigma: float = 1.2,
    hand_gaussian_radius: int = 2,
    hand_max_gap_frames: int = DEFAULT_HAND_MAX_GAP_FRAMES,
    hand_fade_frames: int = DEFAULT_HAND_FADE_FRAMES,
    skip_hand_flips: bool = True,
    hand_flip_orientation_threshold: float = DEFAULT_HAND_FLIP_ORIENTATION_THRESHOLD,
    repair_hand_topology: bool = False,
    stabilize_hand_bones: bool = True,
    articulate_hand_joints: bool = True,
    hand_max_joint_bend_degrees: float = DEFAULT_HAND_MAX_JOINT_BEND_DEGREES,
) -> Dict:
    clips = discover_clips(data_dir)
    if not clips:
        raise ValueError(f"No CSV clips found under: {data_dir}")

    selected_paths = resolve_selected_paths(clips, data_dir, file_tokens, text)
    if not selected_paths:
        sample = list(clips.keys())[:10]
        sample_text = "\n".join(f"  - {item}" for item in sample)
        raise ValueError(
            "No clips selected. Use --files and/or --text.\n"
            f"Example available keys:\n{sample_text}"
        )

    source_frame_rates: List[float | None] = []
    for path in selected_paths:
        timing_df = pd.read_csv(
            path,
            usecols=lambda column: column in {"frame", "time_sec"},
        )
        source_frame_rates.append(estimate_frame_rate(timing_df))

    seq_df = concat_clips(
        selected_paths,
        width=width,
        height=height,
        pause_frames=pause_frames,
        target_fps=fps,
    )
    if seq_df.empty:
        raise ValueError("Selected clips produced empty sequence after filtering pose/face/hand landmarks.")

    upsample_factor = max(1, int(upsample_factor))
    gaussian_sigma = max(0.0, float(gaussian_sigma))
    gaussian_radius = max(0, int(gaussian_radius))
    hand_gaussian_sigma = max(0.0, float(hand_gaussian_sigma))
    hand_gaussian_radius = max(0, int(hand_gaussian_radius))
    hand_max_gap_frames = max(0, int(hand_max_gap_frames))
    hand_fade_frames = max(0, int(hand_fade_frames))
    hand_flip_orientation_threshold = max(
        0.01,
        min(0.95, float(hand_flip_orientation_threshold)),
    )

    frames = build_frames(seq_df, width=width, height=height, max_frames=max_frames)
    if not frames:
        raise ValueError("No frames produced.")
    frames = repair_hand_tracks(
        frames,
        max_gap_frames=hand_max_gap_frames,
        fade_frames=hand_fade_frames,
    )
    hand_mechanical_flip_transitions = count_hand_flip_transitions(
        frames,
        orientation_threshold=hand_flip_orientation_threshold,
    )
    # The old pipeline copied the previous complete hand pose across these
    # transitions. Kinematic interpolation below now rotates each bone in 3D,
    # so no source or generated frame needs to be frozen.
    hand_flip_hold_frames = 0
    hand_topology_warnings_before = count_invalid_hand_topologies(frames)
    hand_topology_corrections = (
        repair_invalid_hand_topology(frames) if repair_hand_topology else 0
    )
    frames = taper_hand_detection_edges(frames, edge_frames=hand_fade_frames)
    frames = interpolate_frames(
        frames,
        upsample_factor=upsample_factor,
        step_across_hand_flips=skip_hand_flips,
    )
    frames = apply_gaussian_smoothing(frames, sigma=gaussian_sigma, radius=gaussian_radius)
    frames = apply_hand_gaussian_smoothing(
        frames,
        sigma=hand_gaussian_sigma,
        radius=hand_gaussian_radius,
        # Kinematic frames already follow the 3D rotation arc; splitting the
        # smoothing track at the 2D sign change would reintroduce a visible snap.
        preserve_flip_steps=not skip_hand_flips,
    )
    hand_shape_stabilized_frames = (
        stabilize_hand_bone_lengths(frames) if stabilize_hand_bones else 0
    )
    hand_articulation_adjusted_frames = (
        enforce_hand_articulation(
            frames,
            max_joint_bend_degrees=hand_max_joint_bend_degrees,
        )
        if articulate_hand_joints
        else 0
    )
    hand_topology_warnings_after_smoothing = count_invalid_hand_topologies(frames)
    post_smoothing_topology_corrections = (
        repair_invalid_hand_topology(frames)
        if repair_hand_topology and skip_hand_flips
        else 0
    )
    hand_topology_corrections += post_smoothing_topology_corrections
    hand_topology_warnings_final = count_invalid_hand_topologies(frames)
    if max_frames > 0 and len(frames) > max_frames:
        frames = frames[:max_frames]
        for i, frame in enumerate(frames):
            frame["frame"] = i
    frames = fit_frames_to_canvas(frames, width=width, height=height)

    return {
        "meta": {
            "fps": float(fps) * float(upsample_factor),
            "canvasWidth": int(width),
            "canvasHeight": int(height),
            "frameCount": int(len(frames)),
            "normalizedForCanvas": True,
            "upsampleFactor": int(upsample_factor),
            "gaussianSigma": float(gaussian_sigma),
            "gaussianRadius": int(gaussian_radius),
            "handGaussianSigma": float(hand_gaussian_sigma),
            "handGaussianRadius": int(hand_gaussian_radius),
            "handTuningProfile": (
                "smooth-s12-r2-a115"
                if abs(hand_gaussian_sigma - 1.2) < 1e-9
                and hand_gaussian_radius == 2
                and abs(hand_max_joint_bend_degrees - 115.0) < 1e-9
                else "custom"
            ),
            "handMaxGapFrames": int(hand_max_gap_frames),
            "handFadeFrames": int(hand_fade_frames),
            "skipHandFlips": bool(skip_hand_flips),
            "handFlipOrientationThreshold": float(hand_flip_orientation_threshold),
            "handFlipHoldFrames": int(hand_flip_hold_frames),
            "handFlipMode": "3d-kinematic" if skip_hand_flips else "linear",
            "handMechanicalFlipTransitions": int(hand_mechanical_flip_transitions),
            "handTopologyCorrections": int(hand_topology_corrections),
            "handPostSmoothingTopologyCorrections": int(post_smoothing_topology_corrections),
            "repairHandTopology": bool(repair_hand_topology),
            "handTopologyWarningsBefore": int(hand_topology_warnings_before),
            "handTopologyWarningsAfterSmoothing": int(hand_topology_warnings_after_smoothing),
            "handTopologyWarningsFinal": int(hand_topology_warnings_final),
            "handShapeStabilizedFrames": int(hand_shape_stabilized_frames),
            "stabilizeHandBones": bool(stabilize_hand_bones),
            "articulateHandJoints": bool(articulate_hand_joints),
            "handArticulationMode": (
                "fixed-length-3d-joints" if articulate_hand_joints else "disabled"
            ),
            "handMaxJointBendDegrees": float(hand_max_joint_bend_degrees),
            "handArticulationAdjustedFrames": int(hand_articulation_adjusted_frames),
            "sourceFps": [
                round(float(value), 6) if value is not None else None
                for value in source_frame_rates
            ],
            "sourceClips": [str(p) for p in selected_paths],
            "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        },
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export CSV keypoints into pose-animator dataset_player JSON format."
    )
    parser.add_argument("--data-dir", type=str, default="SLclean", help="Root folder to search CSV clips.")
    parser.add_argument("--files", nargs="*", default=[], help="CSV files (absolute, relative, key, or basename).")
    parser.add_argument("--text", type=str, default="", help="Optional phrase to match clip names.")
    parser.add_argument("--fps", type=float, default=30.0, help="Target playback FPS in JSON metadata.")
    parser.add_argument("--width", type=int, default=513, help="Canvas width used by pose-animator.")
    parser.add_argument("--height", type=int, default=513, help="Canvas height used by pose-animator.")
    parser.add_argument("--pause-frames", type=int, default=0, help="Hold frames between clips.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit output frames (0 = no limit).")
    parser.add_argument(
        "--upsample-factor",
        type=int,
        default=2,
        help="Linear interpolation factor for temporal upsampling (1 = disabled).",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=1.2,
        help="Gaussian sigma for temporal smoothing (<=0 = disabled).",
    )
    parser.add_argument(
        "--gaussian-radius",
        type=int,
        default=2,
        help="Pose/face Gaussian kernel radius (0 = disabled).",
    )
    parser.add_argument(
        "--hand-gaussian-sigma",
        type=float,
        default=1.2,
        help="Confidence-aware hand smoothing sigma (<=0 = disabled).",
    )
    parser.add_argument(
        "--hand-gaussian-radius",
        type=int,
        default=2,
        help="Hand smoothing kernel radius (0 = disabled).",
    )
    parser.add_argument(
        "--hand-max-gap-frames",
        type=int,
        default=DEFAULT_HAND_MAX_GAP_FRAMES,
        help="Interpolate internal hand gaps up to this many source frames.",
    )
    parser.add_argument(
        "--hand-fade-frames",
        type=int,
        default=DEFAULT_HAND_FADE_FRAMES,
        help="Confidence fade length around longer missing-hand runs.",
    )
    parser.add_argument(
        "--keep-hand-flips",
        dest="skip_hand_flips",
        action="store_false",
        help="Use legacy flat linear interpolation instead of 3D kinematic hand rotation.",
    )
    parser.set_defaults(skip_hand_flips=True)
    parser.add_argument(
        "--hand-flip-orientation-threshold",
        type=float,
        default=DEFAULT_HAND_FLIP_ORIENTATION_THRESHOLD,
        help="Minimum normalized palm orientation treated as stable.",
    )
    parser.add_argument(
        "--repair-hand-topology",
        action="store_true",
        help="Opt in to replacing invalid 2D hand topology; validation-only is the default.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pose-animator/resources/data/sequence.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    try:
        payload = build_payload(
            data_dir=data_dir,
            file_tokens=list(args.files),
            text=str(args.text or ""),
            fps=float(args.fps),
            width=int(args.width),
            height=int(args.height),
            pause_frames=int(args.pause_frames),
            max_frames=int(args.max_frames),
            upsample_factor=int(args.upsample_factor),
            gaussian_sigma=float(args.gaussian_sigma),
            gaussian_radius=int(args.gaussian_radius),
            hand_gaussian_sigma=float(args.hand_gaussian_sigma),
            hand_gaussian_radius=int(args.hand_gaussian_radius),
            hand_max_gap_frames=int(args.hand_max_gap_frames),
            hand_fade_frames=int(args.hand_fade_frames),
            skip_hand_flips=bool(args.skip_hand_flips),
            hand_flip_orientation_threshold=float(args.hand_flip_orientation_threshold),
            repair_hand_topology=bool(args.repair_hand_topology),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {int(payload['meta']['frameCount'])} frames -> {out_path}")


if __name__ == "__main__":
    main()
