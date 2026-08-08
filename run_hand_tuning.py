import argparse
import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from export_pose_animator_sequence import build_payload
from run_hand_ablation import deviation_from_full, measure_payload, percent, summarize


CANDIDATES: List[Dict] = [
    {"name": "default_s08_r1_a115", "sigma": 0.8, "radius": 1, "angle": 115.0},
    {"name": "light_s06_r1_a115", "sigma": 0.6, "radius": 1, "angle": 115.0},
    {"name": "smooth_s10_r1_a115", "sigma": 1.0, "radius": 1, "angle": 115.0},
    {"name": "smooth_s10_r2_a115", "sigma": 1.0, "radius": 2, "angle": 115.0},
    {"name": "smooth_s12_r2_a115", "sigma": 1.2, "radius": 2, "angle": 115.0},
    {"name": "joint_s08_r1_a110", "sigma": 0.8, "radius": 1, "angle": 110.0},
    {"name": "joint_s08_r1_a105", "sigma": 0.8, "radius": 1, "angle": 105.0},
    {"name": "balanced_s10_r2_a110", "sigma": 1.0, "radius": 2, "angle": 110.0},
    {"name": "smooth_s12_r2_a110", "sigma": 1.2, "radius": 2, "angle": 110.0},
]


def candidate_description(candidate: Dict) -> str:
    return (
        f"hand sigma={candidate['sigma']}, radius={candidate['radius']}, "
        f"joint limit={candidate['angle']} degrees"
    )


def rank_candidates(summaries: Sequence[Dict], default_name: str) -> List[Dict]:
    by_name = {row["variant"]: row for row in summaries}
    default = by_name[default_name]
    ranked: List[Dict] = []
    for row in summaries:
        jerk_ratio = row["temporal_jerk_mean"] / max(default["temporal_jerk_mean"], 1e-9)
        topology_ratio = row["topology_error_rate"] / max(default["topology_error_rate"], 1e-9)
        motion_loss = max(0.0, default["motion_preservation_rate"] - row["motion_preservation_rate"])
        coverage_loss = max(0.0, default["hand_coverage"] - row["hand_coverage"])
        constraint_penalty = (
            10.0 * row["joint_violation_rate"]
            + 5.0 * row["bone_span_p95_max"]
        )
        score = (
            0.55 * jerk_ratio
            + 0.20 * topology_ratio
            + 12.0 * motion_loss
            + 6.0 * coverage_loss
            + constraint_penalty
        )
        ranked.append({**row, "balanced_score": score})
    return sorted(ranked, key=lambda row: (row["balanced_score"], row["temporal_jerk_mean"]))


def make_report(words: Sequence[str], ranked: Sequence[Dict], runtime: float) -> str:
    lines = [
        "# Hand Hybrid Parameter Tuning",
        "",
        f"- Signs: `{', '.join(words)}`",
        f"- Configurations: `{len(ranked)}`",
        f"- Total runs: `{len(words) * len(ranked)}`",
        f"- Runtime: `{runtime:.2f} seconds`",
        "",
        "The balanced score prioritizes low wrist-relative shape jerk, then topology, while penalizing lost motion, lost coverage, joint violations, and bone-length variation.",
        "",
        "| Rank | Configuration | Coverage | Bone span ↓ | Max bend ↓ | Joint violations ↓ | Topology ↓ | Motion ↑ | Shape jerk ↓ | Score ↓ |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(ranked, 1):
        lines.append(
            f"| {index} | {row['variant']} | {percent(row['hand_coverage'])} | "
            f"{percent(row['bone_span_p95_max'])} | {row['max_joint_bend_degrees']:.2f}° | "
            f"{percent(row['joint_violation_rate'])} | {percent(row['topology_error_rate'])} | "
            f"{percent(row['motion_preservation_rate'])} | {row['temporal_jerk_mean']:.6f} | "
            f"{row['balanced_score']:.4f} |"
        )
    winner = ranked[0]
    lines.extend(
        [
            "",
            "## Automatic recommendation",
            "",
            f"`{winner['variant']}` is the best metric-balanced candidate in this sweep.",
            "Confirm the winner and the default visually on palm-flip and finger-crossing frames before changing production defaults.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune hybrid hand smoothing and joint limits.")
    parser.add_argument("--data-dir", default=r"C:\งาน\project_1\project_1\SLclean\SLclean")
    parser.add_argument(
        "--words",
        nargs="+",
        default=[
            "abdomen", "hello", "love", "accept", "across",
            "airplane", "alphabet", "angry", "animal", "answer",
        ],
    )
    parser.add_argument("--output-dir", default="tuning_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "data_dir": Path(args.data_dir).resolve(),
        "file_tokens": [],
        "fps": 30.0,
        "width": 513,
        "height": 513,
        "pause_frames": 0,
        "max_frames": 0,
        "upsample_factor": 2,
        "gaussian_sigma": 1.2,
        "gaussian_radius": 2,
        "hand_max_gap_frames": 4,
        "hand_fade_frames": 3,
        "skip_hand_flips": True,
        "hand_flip_orientation_threshold": 0.12,
        "repair_hand_topology": False,
        "stabilize_hand_bones": True,
        "articulate_hand_joints": True,
    }
    started = time.perf_counter()
    rows: List[Dict] = []
    payloads: Dict[Tuple[str, str], Dict] = {}
    for word in args.words:
        for candidate in CANDIDATES:
            print(f"[{word}] {candidate['name']}...", flush=True)
            run_started = time.perf_counter()
            payload = build_payload(
                **common,
                text=word,
                hand_gaussian_sigma=candidate["sigma"],
                hand_gaussian_radius=candidate["radius"],
                hand_max_joint_bend_degrees=candidate["angle"],
            )
            payloads[(word, candidate["name"])] = payload
            rows.append(
                {
                    "word": word,
                    "variant": candidate["name"],
                    "description": candidate_description(candidate),
                    "elapsed_seconds": time.perf_counter() - run_started,
                    **measure_payload(payload),
                }
            )
    default_name = CANDIDATES[0]["name"]
    for row in rows:
        row["deviation_from_full"] = deviation_from_full(
            payloads[(row["word"], row["variant"])],
            payloads[(row["word"], default_name)],
        )
    summaries = summarize(rows, [candidate["name"] for candidate in CANDIDATES])
    ranked = rank_candidates(summaries, default_name)
    runtime = time.perf_counter() - started

    csv_path = output_dir / "hand_tuning_per_sign.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "words": args.words,
        "candidates": CANDIDATES,
        "perSign": rows,
        "rankedSummary": ranked,
        "runtimeSeconds": runtime,
    }
    (output_dir / "hand_tuning_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "hand_tuning_report.md").write_text(
        make_report(args.words, ranked, runtime), encoding="utf-8"
    )
    print(f"Winner: {ranked[0]['variant']}")
    print(f"Report: {output_dir / 'hand_tuning_report.md'}")


if __name__ == "__main__":
    main()
