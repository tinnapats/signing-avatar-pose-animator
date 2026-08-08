import argparse
import csv
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from export_pose_animator_sequence import (
    DEFAULT_HAND_MAX_JOINT_BEND_DEGREES,
    HAND_ARTICULATION_CHAINS,
    HAND_KINEMATIC_EDGES,
    build_payload,
    has_valid_hand_topology,
)

VISIBLE_HAND_SCORE = 0.18
MOTION_THRESHOLD_PALM_RATIO = 0.002
JOINT_VIOLATION_DEGREES = DEFAULT_HAND_MAX_JOINT_BEND_DEGREES

VARIANTS: List[Dict] = [
    {
        "name": "full",
        "description": "Full hybrid pipeline (3D SLERP + repair + smoothing + bone stabilizer + fixed-length joints)",
        "overrides": {},
    },
    {
        "name": "no_3d_slerp",
        "description": "Replace 3D kinematic hand interpolation with flat linear interpolation",
        "overrides": {"skip_hand_flips": False},
    },
    {
        "name": "no_gap_repair",
        "description": "Disable internal hand-gap repair and detection-edge fading",
        "overrides": {"hand_max_gap_frames": 0, "hand_fade_frames": 0},
    },
    {
        "name": "no_hand_smoothing",
        "description": "Disable confidence-aware Gaussian hand smoothing",
        "overrides": {"hand_gaussian_sigma": 0.0, "hand_gaussian_radius": 0},
    },
    {
        "name": "no_bone_stabilizer",
        "description": "Disable severe bone-length outlier stabilization",
        "overrides": {"stabilize_hand_bones": False},
    },
    {
        "name": "no_fixed_length",
        "description": "Disable final fixed-length articulated-chain projection",
        "overrides": {"articulate_hand_joints": False},
    },
    {
        "name": "no_joint_limit",
        "description": "Keep fixed bone lengths but relax joint bend limit to 175 degrees",
        "overrides": {"hand_max_joint_bend_degrees": 175.0},
    },
    {
        "name": "raw_hand_baseline",
        "description": "Linear hand interpolation without gap repair, hand smoothing, bone stabilization, or articulation",
        "overrides": {
            "skip_hand_flips": False,
            "hand_max_gap_frames": 0,
            "hand_fade_frames": 0,
            "hand_gaussian_sigma": 0.0,
            "hand_gaussian_radius": 0,
            "stabilize_hand_bones": False,
            "articulate_hand_joints": False,
        },
    },
]


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, float(fraction))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def point_xyz(keypoint: Dict) -> Tuple[float, float, float]:
    position = keypoint.get("position", {})
    return (
        float(position.get("x", 0.0)),
        float(position.get("y", 0.0)),
        float(position.get("z", 0.0)),
    )


def vector(first: Tuple[float, float, float], second: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return tuple(second[index] - first[index] for index in range(3))


def vector_length(value: Tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in value))


def palm_scale(keypoints: Sequence[Dict]) -> float:
    if len(keypoints) < 21:
        return 1.0
    wrist = point_xyz(keypoints[0])
    lengths = [
        vector_length(vector(wrist, point_xyz(keypoints[landmark_id])))
        for landmark_id in (5, 9, 13, 17)
    ]
    valid = [length for length in lengths if length > 1e-8]
    return float(statistics.median(valid)) if valid else 1.0


def relative_shape(keypoints: Sequence[Dict]) -> List[Tuple[float, float, float]]:
    wrist = point_xyz(keypoints[0])
    return [vector(wrist, point_xyz(keypoint)) for keypoint in keypoints[:21]]


def joint_angle_degrees(
    first: Tuple[float, float, float],
    second: Tuple[float, float, float],
) -> float | None:
    first_length = vector_length(first)
    second_length = vector_length(second)
    if first_length <= 1e-8 or second_length <= 1e-8:
        return None
    cosine = sum(a * b for a, b in zip(first, second)) / (first_length * second_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def measure_payload(payload: Dict) -> Dict[str, float | int]:
    frames = payload.get("frames", [])
    total_hand_slots = max(1, len(frames) * 2)
    visible_hand_frames = 0
    topology_invalid = 0
    bone_tracks: Dict[Tuple[str, int, int], List[float]] = {}
    joint_angles: List[float] = []
    motion_steps: List[float] = []
    moving_transitions = 0
    visible_transitions = 0
    jerk_values: List[float] = []

    previous: Dict[str, Tuple[int, List[Tuple[float, float, float]], float]] = {}
    histories: Dict[str, List[Tuple[int, List[Tuple[float, float, float]], float]]] = {
        "left": [],
        "right": [],
    }

    for frame_index, frame in enumerate(frames):
        for side in ("left", "right"):
            hand = frame.get("hands", {}).get(side, {})
            keypoints = hand.get("keypoints", [])
            visible = float(hand.get("score", 0.0)) >= VISIBLE_HAND_SCORE and len(keypoints) >= 21
            if not visible:
                previous.pop(side, None)
                histories[side] = []
                continue

            visible_hand_frames += 1
            if not has_valid_hand_topology(hand):
                topology_invalid += 1

            scale = max(palm_scale(keypoints), 1e-8)
            shape = relative_shape(keypoints)

            for parent_id, child_id in HAND_KINEMATIC_EDGES:
                length = vector_length(
                    vector(point_xyz(keypoints[parent_id]), point_xyz(keypoints[child_id]))
                )
                if length > 1e-8:
                    bone_tracks.setdefault((side, parent_id, child_id), []).append(length)

            for chain in HAND_ARTICULATION_CHAINS:
                for parent_id, joint_id, child_id in zip(chain, chain[1:], chain[2:]):
                    first = vector(point_xyz(keypoints[parent_id]), point_xyz(keypoints[joint_id]))
                    second = vector(point_xyz(keypoints[joint_id]), point_xyz(keypoints[child_id]))
                    angle = joint_angle_degrees(first, second)
                    if angle is not None:
                        joint_angles.append(angle)

            previous_entry = previous.get(side)
            if previous_entry is not None and previous_entry[0] == frame_index - 1:
                previous_shape = previous_entry[1]
                step = max(
                    vector_length(vector(previous_shape[index], shape[index]))
                    for index in range(1, min(len(shape), len(previous_shape)))
                ) / scale
                motion_steps.append(step)
                visible_transitions += 1
                if step > MOTION_THRESHOLD_PALM_RATIO:
                    moving_transitions += 1
            previous[side] = (frame_index, shape, scale)

            history = histories[side]
            history.append((frame_index, shape, scale))
            if len(history) > 4:
                history.pop(0)
            if len(history) == 4 and all(
                history[index][0] == history[0][0] + index for index in range(4)
            ):
                local_jerks = []
                for landmark_id in range(1, 21):
                    p0 = history[0][1][landmark_id]
                    p1 = history[1][1][landmark_id]
                    p2 = history[2][1][landmark_id]
                    p3 = history[3][1][landmark_id]
                    third_difference = tuple(
                        p3[axis] - 3.0 * p2[axis] + 3.0 * p1[axis] - p0[axis]
                        for axis in range(3)
                    )
                    local_jerks.append(vector_length(third_difference) / scale)
                jerk_values.append(float(statistics.mean(local_jerks)))

    robust_spans: List[float] = []
    coefficients_of_variation: List[float] = []
    for lengths in bone_tracks.values():
        if len(lengths) < 2:
            continue
        median_length = float(statistics.median(lengths))
        mean_length = float(statistics.mean(lengths))
        if median_length > 1e-8:
            robust_spans.append(
                (percentile(lengths, 0.95) - percentile(lengths, 0.05)) / median_length
            )
        if mean_length > 1e-8:
            coefficients_of_variation.append(float(statistics.pstdev(lengths)) / mean_length)

    joint_violations = sum(
        angle > JOINT_VIOLATION_DEGREES + 1e-6 for angle in joint_angles
    )
    return {
        "frame_count": len(frames),
        "visible_hand_frames": visible_hand_frames,
        "hand_coverage": visible_hand_frames / total_hand_slots,
        "bone_span_p95_max": max(robust_spans or [0.0]),
        "bone_cv_mean": float(statistics.mean(coefficients_of_variation)) if coefficients_of_variation else 0.0,
        "max_joint_bend_degrees": max(joint_angles or [0.0]),
        "joint_sample_count": len(joint_angles),
        "joint_violation_count": joint_violations,
        "joint_violation_rate": joint_violations / max(1, len(joint_angles)),
        "topology_sample_count": visible_hand_frames,
        "topology_invalid_count": topology_invalid,
        "topology_error_rate": topology_invalid / max(1, visible_hand_frames),
        "motion_transition_count": visible_transitions,
        "moving_transition_count": moving_transitions,
        "motion_preservation_rate": moving_transitions / max(1, visible_transitions),
        "motion_step_p95": percentile(motion_steps, 0.95),
        "jerk_sample_count": len(jerk_values),
        "temporal_jerk_mean": float(statistics.mean(jerk_values)) if jerk_values else 0.0,
    }


def deviation_from_full(payload: Dict, full_payload: Dict) -> float:
    values: List[float] = []
    frames = payload.get("frames", [])
    full_frames = full_payload.get("frames", [])
    for frame, full_frame in zip(frames, full_frames):
        for side in ("left", "right"):
            hand = frame.get("hands", {}).get(side, {})
            full_hand = full_frame.get("hands", {}).get(side, {})
            keypoints = hand.get("keypoints", [])
            full_keypoints = full_hand.get("keypoints", [])
            if (
                float(hand.get("score", 0.0)) < VISIBLE_HAND_SCORE
                or float(full_hand.get("score", 0.0)) < VISIBLE_HAND_SCORE
                or len(keypoints) < 21
                or len(full_keypoints) < 21
            ):
                continue
            scale = max(palm_scale(full_keypoints), 1e-8)
            shape = relative_shape(keypoints)
            full_shape = relative_shape(full_keypoints)
            values.extend(
                vector_length(vector(full_shape[index], shape[index])) / scale
                for index in range(1, 21)
            )
    return float(statistics.mean(values)) if values else 0.0


def weighted_ratio(rows: Iterable[Dict], numerator: str, denominator: str) -> float:
    rows = list(rows)
    total_denominator = sum(int(row[denominator]) for row in rows)
    return (
        sum(int(row[numerator]) for row in rows) / total_denominator
        if total_denominator > 0
        else 0.0
    )


def summarize(rows: List[Dict], variant_names: Sequence[str]) -> List[Dict]:
    summaries: List[Dict] = []
    for variant_name in variant_names:
        subset = [row for row in rows if row["variant"] == variant_name]
        if not subset:
            continue
        summaries.append(
            {
                "variant": variant_name,
                "description": subset[0]["description"],
                "words": len(subset),
                "hand_coverage": float(statistics.mean(row["hand_coverage"] for row in subset)),
                "bone_span_p95_max": max(row["bone_span_p95_max"] for row in subset),
                "bone_cv_mean": float(statistics.mean(row["bone_cv_mean"] for row in subset)),
                "max_joint_bend_degrees": max(row["max_joint_bend_degrees"] for row in subset),
                "joint_violation_rate": weighted_ratio(
                    subset, "joint_violation_count", "joint_sample_count"
                ),
                "topology_error_rate": weighted_ratio(
                    subset, "topology_invalid_count", "topology_sample_count"
                ),
                "motion_preservation_rate": weighted_ratio(
                    subset, "moving_transition_count", "motion_transition_count"
                ),
                "motion_step_p95": float(statistics.mean(row["motion_step_p95"] for row in subset)),
                "temporal_jerk_mean": float(statistics.mean(row["temporal_jerk_mean"] for row in subset)),
                "deviation_from_full": float(statistics.mean(row["deviation_from_full"] for row in subset)),
                "elapsed_seconds": sum(row["elapsed_seconds"] for row in subset),
            }
        )
    return summaries


def percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def make_markdown(
    words: Sequence[str],
    summaries: Sequence[Dict],
    generated_at: str,
    total_seconds: float,
) -> str:
    by_name = {row["variant"]: row for row in summaries}
    lines = [
        "# Hand Pipeline Ablation Study",
        "",
        f"- Generated: `{generated_at}`",
        f"- Signs: `{', '.join(words)}`",
        f"- Runtime: `{total_seconds:.2f} seconds`",
        f"- Visible-hand threshold: `{VISIBLE_HAND_SCORE}`",
        f"- Joint violation threshold: `{JOINT_VIOLATION_DEGREES:.1f} degrees`",
        "",
        "## Summary",
        "",
        "| Variant | Coverage | Bone span (max) ↓ | Max bend ↓ | Joint violations ↓ | Topology errors ↓ | Motion retained ↑ | Shape jerk ↓ | Deviation from full ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {variant} | {coverage} | {bone_span} | {bend:.2f}° | {violations} | {topology} | {motion} | {jerk:.6f} | {deviation:.6f} |".format(
                variant=row["variant"],
                coverage=percent(row["hand_coverage"]),
                bone_span=percent(row["bone_span_p95_max"]),
                bend=row["max_joint_bend_degrees"],
                violations=percent(row["joint_violation_rate"]),
                topology=percent(row["topology_error_rate"]),
                motion=percent(row["motion_preservation_rate"]),
                jerk=row["temporal_jerk_mean"],
                deviation=row["deviation_from_full"],
            )
        )

    lines.extend(["", "## Component effects", ""])
    full = by_name.get("full")
    if full:
        lines.append(
            f"- Full system: bone span `{percent(full['bone_span_p95_max'])}`, "
            f"joint violations `{percent(full['joint_violation_rate'])}`, "
            f"motion retained `{percent(full['motion_preservation_rate'])}`."
        )
    for name, label in (
        ("no_3d_slerp", "3D SLERP"),
        ("no_gap_repair", "gap repair"),
        ("no_hand_smoothing", "hand smoothing"),
        ("no_bone_stabilizer", "bone stabilizer"),
        ("no_fixed_length", "fixed-length articulation"),
        ("no_joint_limit", "joint limit"),
        ("raw_hand_baseline", "raw baseline"),
    ):
        row = by_name.get(name)
        if row:
            lines.append(
                f"- Without {label}: bone span `{percent(row['bone_span_p95_max'])}`, "
                f"max bend `{row['max_joint_bend_degrees']:.2f}°`, "
                f"motion `{percent(row['motion_preservation_rate'])}`, "
                f"jerk `{row['temporal_jerk_mean']:.6f}`."
            )

    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- Bone span uses the robust `(P95 - P05) / median` length of each 3D kinematic bone; the table reports the worst bone across signs.",
            "- Shape jerk is the third temporal difference of wrist-relative landmarks, normalized by palm size.",
            "- Topology errors are 2D validation warnings; some may be real perspective overlap rather than anatomical failure.",
            "- This pilot covers three signs. Run the same script with more dataset words before reporting a final statistical conclusion.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run component ablations for the signing-avatar hand pipeline.")
    parser.add_argument("--data-dir", default=r"C:\งาน\project_1\project_1\SLclean\SLclean")
    parser.add_argument("--words", nargs="+", default=["abdomen", "hello", "love"])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[variant["name"] for variant in VARIANTS],
        choices=[variant["name"] for variant in VARIANTS],
    )
    parser.add_argument("--output-dir", default="ablation_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_variants = [
        variant for variant in VARIANTS if variant["name"] in set(args.variants)
    ]
    if not any(variant["name"] == "full" for variant in selected_variants):
        selected_variants.insert(0, next(variant for variant in VARIANTS if variant["name"] == "full"))

    common = {
        "data_dir": data_dir,
        "file_tokens": [],
        "fps": 30.0,
        "width": 513,
        "height": 513,
        "pause_frames": 0,
        "max_frames": 0,
        "upsample_factor": 2,
        "gaussian_sigma": 1.2,
        "gaussian_radius": 2,
        "hand_gaussian_sigma": 0.8,
        "hand_gaussian_radius": 1,
        "hand_max_gap_frames": 4,
        "hand_fade_frames": 3,
        "skip_hand_flips": True,
        "hand_flip_orientation_threshold": 0.12,
        "repair_hand_topology": False,
        "stabilize_hand_bones": True,
        "articulate_hand_joints": True,
        "hand_max_joint_bend_degrees": DEFAULT_HAND_MAX_JOINT_BEND_DEGREES,
    }

    started = time.perf_counter()
    payloads: Dict[Tuple[str, str], Dict] = {}
    rows: List[Dict] = []

    for word in args.words:
        for variant in selected_variants:
            configuration = dict(common)
            configuration.update(variant["overrides"])
            configuration["text"] = word
            print(f"[{word}] {variant['name']}...", flush=True)
            variant_started = time.perf_counter()
            payload = build_payload(**configuration)
            elapsed = time.perf_counter() - variant_started
            payloads[(word, variant["name"])] = payload
            metrics = measure_payload(payload)
            rows.append(
                {
                    "word": word,
                    "variant": variant["name"],
                    "description": variant["description"],
                    "elapsed_seconds": elapsed,
                    **metrics,
                }
            )

    for row in rows:
        payload = payloads[(row["word"], row["variant"])]
        full_payload = payloads[(row["word"], "full")]
        row["deviation_from_full"] = deviation_from_full(payload, full_payload)

    summaries = summarize(rows, [variant["name"] for variant in selected_variants])
    generated_at = datetime.now(timezone.utc).isoformat()
    total_seconds = time.perf_counter() - started

    csv_path = output_dir / "hand_ablation_per_sign.csv"
    json_path = output_dir / "hand_ablation_results.json"
    markdown_path = output_dir / "hand_ablation_report.md"

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "generatedAtUtc": generated_at,
        "words": list(args.words),
        "variants": selected_variants,
        "thresholds": {
            "visibleHandScore": VISIBLE_HAND_SCORE,
            "jointViolationDegrees": JOINT_VIOLATION_DEGREES,
            "motionThresholdPalmRatio": MOTION_THRESHOLD_PALM_RATIO,
        },
        "perSign": rows,
        "summary": summaries,
        "runtimeSeconds": total_seconds,
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(
        make_markdown(args.words, summaries, generated_at, total_seconds),
        encoding="utf-8",
    )

    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Report: {markdown_path}")


if __name__ == "__main__":
    main()