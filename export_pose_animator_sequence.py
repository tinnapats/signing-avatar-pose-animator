import argparse
import json
import math
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


def load_clip(path: Path, width: int, height: int) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=EXPECTED_COLS)
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df["landmark_id"] = pd.to_numeric(df["landmark_id"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
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
        df["x"] = df["x"] * float(width)
        df["y"] = df["y"] * float(height)

    return df[EXPECTED_COLS]


def concat_clips(paths: List[Path], width: int, height: int, pause_frames: int) -> pd.DataFrame:
    seq_parts: List[pd.DataFrame] = []
    frame_offset = 0

    for path in paths:
        clip = load_clip(path, width, height)
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
        return pd.DataFrame(columns=EXPECTED_COLS)

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


def build_frames(seq_df: pd.DataFrame, width: int, height: int, max_frames: int) -> List[Dict]:
    if seq_df.empty:
        return []

    frames: List[Dict] = []
    grouped = seq_df.groupby("frame", sort=True)

    center = (float(width) * 0.5, float(height) * 0.5)
    face_center = (float(width) * 0.5, float(height) * 0.35)

    last_pose: Dict[str, Tuple[float, float]] = {name: center for name, _ in POSE_PART_ORDER}
    last_face: Dict[str, Tuple[float, float]] = {name: face_center for name in FACE_PART_ORDER}
    last_hands: Dict[str, Dict[int, Tuple[float, float]]] = {
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
        hand_lookups: Dict[str, Dict[int, Tuple[float, float]]] = {}
        for side in HAND_SIDES:
            part_name = HAND_PART_NAMES[side]
            hand_lookups[side] = {
                int(row.landmark_id): (float(row.x), float(row.y))
                for row in frame_df[frame_df["part"] == part_name][["landmark_id", "x", "y"]]
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
            wrist_fallback = last_pose.get(pose_wrist_name, center)
            hand_lookup = hand_lookups[side]
            hand_keypoints: List[Dict] = []
            hand_scores: List[float] = []
            for landmark_id in HAND_LANDMARK_IDS:
                fallback = last_hands[side].get(landmark_id, wrist_fallback)
                x, y, score = _xy_from_lookup(hand_lookup, landmark_id, fallback)
                if score > 0:
                    last_hands[side][landmark_id] = (x, y)
                hand_scores.append(score)
                hand_keypoints.append(
                    {
                        "landmarkId": int(landmark_id),
                        "score": score,
                        "position": {"x": x, "y": y},
                    }
                )
            hands[side] = {
                "score": float(sum(hand_scores) / max(len(hand_scores), 1)),
                "keypoints": hand_keypoints,
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


def interpolate_frames(frames: List[Dict], upsample_factor: int) -> List[Dict]:
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
                hand_keypoints: List[Dict] = []
                for a, b in zip(hand0.get("keypoints", []), hand1.get("keypoints", [])):
                    hand_keypoints.append(
                        {
                            "landmarkId": int(a["landmarkId"]),
                            "score": _lerp(float(a["score"]), float(b["score"]), t),
                            "position": {
                                "x": _lerp(float(a["position"]["x"]), float(b["position"]["x"]), t),
                                "y": _lerp(float(a["position"]["y"]), float(b["position"]["y"]), t),
                            },
                        }
                    )
                interp_hands[side] = {
                    "score": _lerp(float(hand0.get("score", 0.0)), float(hand1.get("score", 0.0)), t),
                    "keypoints": hand_keypoints,
                }

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


def smooth_series(values: List[float], kernel: List[float]) -> List[float]:
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
            acc += float(values[idx]) * float(weight)
        out.append(acc)
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

    for side in HAND_SIDES:
        hand_count = len(HAND_LANDMARK_IDS)
        for hand_idx in range(hand_count):
            xs = [float(frame["hands"][side]["keypoints"][hand_idx]["position"]["x"]) for frame in frames]
            ys = [float(frame["hands"][side]["keypoints"][hand_idx]["position"]["y"]) for frame in frames]
            xs_sm = smooth_series(xs, kernel)
            ys_sm = smooth_series(ys, kernel)
            for i, frame in enumerate(frames):
                frame["hands"][side]["keypoints"][hand_idx]["position"]["x"] = xs_sm[i]
                frame["hands"][side]["keypoints"][hand_idx]["position"]["y"] = ys_sm[i]

    return frames


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

    # Keep first occurrence order.
    seen = set()
    deduped: List[Path] = []
    for path in selected_paths:
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


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

    seq_df = concat_clips(selected_paths, width=width, height=height, pause_frames=pause_frames)
    if seq_df.empty:
        raise ValueError("Selected clips produced empty sequence after filtering pose/face/hand landmarks.")

    upsample_factor = max(1, int(upsample_factor))
    gaussian_sigma = max(0.0, float(gaussian_sigma))
    gaussian_radius = max(0, int(gaussian_radius))

    frames = build_frames(seq_df, width=width, height=height, max_frames=max_frames)
    if not frames:
        raise ValueError("No frames produced.")
    frames = interpolate_frames(frames, upsample_factor=upsample_factor)
    frames = apply_gaussian_smoothing(frames, sigma=gaussian_sigma, radius=gaussian_radius)
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
        help="Gaussian kernel radius for temporal smoothing (0 = disabled).",
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
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {int(payload['meta']['frameCount'])} frames -> {out_path}")


if __name__ == "__main__":
    main()
