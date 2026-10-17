"""Experimental, image-based comparison against unmodified dataset coordinates."""
import base64
import io
import math
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from clip_database import read_clip_csv
from export_pose_animator_sequence import discover_clips, estimate_source_aspect

MODEL_DIR = Path(__file__).parent / 'comparison_models'
_lock = threading.Lock()
_models = None
POSE_IDS = (11, 12, 13, 14, 15, 16)


def models():
    global _models
    if _models is None:
        import mediapipe as mp
        vision = mp.tasks.vision
        for name in ('pose_landmarker_full.task', 'hand_landmarker.task'):
            if not (MODEL_DIR / name).is_file():
                raise ValueError('Comparison models missing. Run setup_comparison.py first.')
        pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_DIR / 'pose_landmarker_full.task'))))
        try:
            hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_DIR / 'hand_landmarker.task')),
                num_hands=2))
        except Exception:
            pose.close()
            raise
        _models = (mp, pose, hand)
    return _models


def angle(points, ids):
    if not all(i in points for i in ids):
        return None
    a, b, c = (np.array(points[i], dtype=float) for i in ids)
    u, v = a - b, c - b
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    if norm < 1e-10:
        return None
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / norm, -1, 1))))


def body_normalized(points):
    if 11 not in points or 12 not in points:
        return {}
    left, right = np.array(points[11]), np.array(points[12])
    scale = np.linalg.norm(left - right)
    if scale < 1e-8:
        return {}
    center = (left + right) / 2
    return {i: (np.array(p) - center) / scale for i, p in points.items()}


def compare_points(reference, detected):
    a, b = body_normalized(reference), body_normalized(detected)
    wrists = [float(np.linalg.norm(a[i] - b[i])) for i in (15, 16) if i in a and i in b]
    elbows = []
    for ids in ((11, 13, 15), (12, 14, 16)):
        x, y = angle(reference, ids), angle(detected, ids)
        if x is not None and y is not None:
            elbows.append(abs(x - y))
    return {'wristDistance': mean(wrists), 'elbowDegrees': mean(elbows)}


def mean(values):
    values = [float(x) for x in values if x is not None and math.isfinite(x)]
    return round(sum(values) / len(values), 4) if values else None


def similarity_scores(metrics):
    """Heuristic similarities, not calibrated probabilities of a correct sign."""
    scores = {}
    for key in ('elbowDegrees', 'wristDistance', 'fingerDegrees'):
        value = metrics.get(key)
        if value is None or not math.isfinite(value) or value < 0:
            scores[key] = None
        elif key == 'wristDistance':
            scores[key] = 100.0 / (1.0 + value)
        else:
            scores[key] = 100.0 * max(0.0, 1.0 - value / 180.0)
    scores['overall'] = (sum(scores.values()) / 3
                         if all(v is not None for v in scores.values()) else None)
    return {k: round(v, 2) if v is not None else None for k, v in scores.items()}


def raw_points(group, part, aspect):
    points = {}
    for row in group[group.part == part].itertuples():
        x, y = float(row.x), float(row.y)
        visibility = getattr(row, 'visibility', float('nan'))
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if part == 'pose' and math.isfinite(visibility) and visibility < .5:
            continue
        points[int(row.landmark_id)] = [x * aspect, y]
    return points


def detect(image):
    mp, pose, hand = models()
    w, h = image.size
    result = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image)))
    points = {}
    if result.pose_landmarks:
        for i, p in enumerate(result.pose_landmarks[0]):
            if p.visibility >= .5 and p.presence >= .5 and 0 <= p.x <= 1 and 0 <= p.y <= 1:
                points[i] = [p.x * w / h, p.y]
    hands = hand.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image)))
    detected_hands = {}
    # Assign hands by proximity to confidently detected anatomical wrists.
    # Do not choose a side based on which gives a better comparison score.
    for landmarks in hands.hand_landmarks:
        pts = {i: [p.x * w / h, p.y] for i, p in enumerate(landmarks)}
        choices = [(np.linalg.norm(np.array(pts[0]) - points[wid]), side)
                   for side, wid in [('left', 15), ('right', 16)] if wid in points]
        choices.sort()
        if choices and choices[0][0] < .15 and (len(choices) == 1 or choices[1][0] - choices[0][0] > .03):
            side = choices[0][1]
            if side not in detected_hands:
                detected_hands[side] = pts
    return points, detected_hands, len(hands.hand_landmarks)


def compare_request(data_dir, request):
    if not isinstance(request, dict):
        raise ValueError('Expected a JSON object.')
    samples = request.get('samples')
    if not isinstance(samples, list) or not 2 <= len(samples) <= 24:
        raise ValueError('Provide 2–24 sampled frames.')
    clips = discover_clips(data_dir)
    paths = [p for p in clips.values() if str(p) == request.get('clip')]
    if len(paths) != 1:
        raise ValueError('Clip not found in the configured dataset. Generate again.')
    df = read_clip_csv(paths[0])  # No repair, smoothing or generated coordinates.
    aspect = estimate_source_aspect(df)
    groups = list(df.groupby('frame', sort=True))
    if not groups:
        raise ValueError('Empty source clip.')
    times = np.array([float(g.time_sec.iloc[0]) if 'time_sec' in g else float(i)
                      for i, (_, g) in enumerate(groups)])
    if not np.isfinite(times).all() or times[-1] <= times[0]:
        times = np.arange(len(groups), dtype=float)
    rows = []
    with _lock:
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError('Invalid sample.')
            progress = float(sample['progress'])
            if not math.isfinite(progress) or not 0 <= progress <= 1:
                raise ValueError('Invalid sample progress.')
            encoded = sample.get('image', '')
            if not isinstance(encoded, str) or not encoded.startswith('data:image/png;base64,'):
                raise ValueError('Expected a PNG canvas image.')
            raw = base64.b64decode(encoded.split(',', 1)[1], validate=True)
            if len(raw) > 2_000_000:
                raise ValueError('Image too large.')
            image = Image.open(io.BytesIO(raw))
            if image.width > 1024 or image.height > 1024:
                raise ValueError('Image dimensions too large.')
            image = image.convert('RGB')
            source_index = int(np.argmin(abs(times - (times[0] + progress * (times[-1] - times[0])))))
            frame_id, group = groups[source_index]
            reference = raw_points(group, 'pose', aspect)
            detected, hands, hand_count = detect(image)
            metrics = compare_points(reference, detected)
            finger_errors = []
            source_hands, paired_hands = 0, 0
            for side in ('left', 'right'):
                ref = raw_points(group, side + '_hand', aspect)
                if ref:
                    source_hands += 1
                if ref and side in hands:
                    paired_hands += 1
                    for start in (1, 5, 9, 13, 17):
                        for ids in ((start, start + 1, start + 2), (start + 1, start + 2, start + 3)):
                            a, b = angle(ref, ids), angle(hands[side], ids)
                            if a is not None and b is not None:
                                finger_errors.append(abs(a - b))
            # Reference and detected overlays in shoulder-normalized coordinates.
            overlay = lambda pts: {str(i): p.tolist() for i, p in body_normalized(pts).items() if i in POSE_IDS}
            rows.append({'sourceFrame': int(frame_id), 'outputFrame': int(sample['frame']),
                         'progress': progress, **metrics, 'fingerDegrees': mean(finger_errors),
                         'detectedBodyPoints': sum(i in detected for i in POSE_IDS),
                         'sourceBodyPoints': sum(i in reference for i in POSE_IDS),
                         'sourceHands': source_hands, 'detectedHands': hand_count, 'pairedHands': paired_hands,
                         'reference': overlay(reference), 'detected': overlay(detected)})
    summary = {key: mean([r[key] for r in rows]) for key in ('wristDistance', 'elbowDegrees', 'fingerDegrees')}
    summary.update({key + 'Frames': sum(r[key] is not None for r in rows)
                    for key in ('wristDistance', 'elbowDegrees', 'fingerDegrees')})
    for row in rows:
        row['similarity'] = similarity_scores(row)
    summary['similarity'] = {
        key: mean([r['similarity'][key] for r in rows])
        for key in ('elbowDegrees', 'wristDistance', 'fingerDegrees', 'overall')}
    summary['overallFrames'] = sum(r['similarity']['overall'] is not None for r in rows)
    summary['similarityFormula'] = {
        'angles': '100 * max(0, 1 - absolute_angle_error / 180)',
        'wrist': '100 / (1 + shoulder_normalized_wrist_distance)',
        'overall': 'Equal mean of three component scores, only on frames with all three available; then mean across those frames.',
    }
    return {'ok': True, 'clip': paths[0].stem, 'sampleCount': len(rows), 'summary': summary, 'frames': rows,
            'models': ['MediaPipe Pose Landmarker Full', 'MediaPipe Hand Landmarker'],
            'alignment': 'Nearest raw timestamp at equal clip progress (approximate; no DTW)',
            'verdict': 'Experimental measurements only; not a sign-language correctness verdict.'}
