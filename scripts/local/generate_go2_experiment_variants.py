#!/usr/bin/env python3
"""Generate resumable command-variant videos for imported Go2 experiments."""

import argparse
import csv
import os
from pathlib import Path
import re
import subprocess

from generate_go2_variants import CONDITIONS


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv-go2-video" / "bin" / "python"
GENERATOR = ROOT / "scripts" / "local" / "generate_go2_video.py"
DEFAULT_CHECKPOINT_ROOT = (
    ROOT / "offline_agent/ckpt/imported/go2-full8-20260824-r1/go2"
)
DEFAULT_OUTPUT_ROOT = ROOT / "videos/experiment_variants/go2-full8-20260824-r1"
FIELDS = [
    "dataset",
    "condition",
    "policy_seed",
    "simulation_seed",
    "vx",
    "vy",
    "yaw_rate",
    "steps",
    "duration_seconds",
    "status",
    "checkpoint",
    "video",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset directory name; repeat to select more than one.",
    )
    parser.add_argument(
        "--policy-seeds",
        default="0,1,2",
        help="Comma-separated checkpoint seeds (default: 0,1,2).",
    )
    parser.add_argument(
        "--conditions",
        help="Comma-separated condition labels; default: the complete variant grid.",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_root = args.checkpoint_root.resolve()
    output_root = args.output_root.resolve()
    selected_seeds = {int(value) for value in args.policy_seeds.split(",")}
    selected_labels = set(args.conditions.split(",")) if args.conditions else None
    conditions = [item for item in CONDITIONS if selected_labels is None or item[0] in selected_labels]
    if selected_labels:
        unknown = selected_labels - {item[0] for item in conditions}
        if unknown:
            raise SystemExit(f"Unknown conditions: {', '.join(sorted(unknown))}")

    checkpoints = []
    for checkpoint in sorted(checkpoint_root.glob("*/agent_seed*.eqx")):
        dataset = checkpoint.parent.name
        match = re.fullmatch(r"agent_seed(\d+)\.eqx", checkpoint.name)
        if not match:
            continue
        policy_seed = int(match.group(1))
        if args.dataset and dataset not in args.dataset:
            continue
        if policy_seed in selected_seeds:
            checkpoints.append((dataset, policy_seed, checkpoint))
    if not checkpoints:
        raise SystemExit(f"No matching checkpoints under {checkpoint_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"
    records = []
    if manifest_path.exists():
        with manifest_path.open(newline="") as file:
            records = list(csv.DictReader(file))
    completed_keys = {
        (record["dataset"], record["condition"], int(record["policy_seed"]))
        for record in records
        if record["status"] == "ok" and (ROOT / record["video"]).is_file()
    }

    jobs = [
        (dataset, policy_seed, checkpoint, condition_index, condition)
        for dataset, policy_seed, checkpoint in checkpoints
        for condition_index, condition in enumerate(conditions)
        if (dataset, condition[0], policy_seed) not in completed_keys
    ]
    print(f"Checkpoints: {len(checkpoints)}; conditions: {len(conditions)}; pending videos: {len(jobs)}")
    if args.dry_run:
        for dataset, policy_seed, _, _, condition in jobs:
            print(f"{dataset}: seed={policy_seed} condition={condition[0]}")
        return

    environment = os.environ.copy()
    environment.update(
        JAX_PLATFORMS="cpu",
        MUJOCO_GL=environment.get("MUJOCO_GL", "egl"),
        MESA_SHADER_CACHE_DIR="/tmp/neubay_mesa_cache",
    )
    for job_index, (dataset, policy_seed, checkpoint, condition_index, condition) in enumerate(jobs, 1):
        label, vx, vy, yaw_rate = condition
        simulation_seed = 100 + condition_index
        dataset_output = output_root / dataset
        dataset_output.mkdir(parents=True, exist_ok=True)
        output = dataset_output / f"seed{policy_seed}_{label}.mp4"
        command = [
            str(PYTHON),
            str(GENERATOR),
            str(checkpoint),
            "--output", str(output),
            "--steps", str(args.steps),
            "--seed", str(simulation_seed),
            "--vx", str(vx),
            "--vy", str(vy),
            "--yaw-rate", str(yaw_rate),
            "--render-every", str(args.render_every),
            "--width", str(args.width),
            "--height", str(args.height),
        ]
        result = subprocess.run(command, env=environment, text=True, capture_output=True)
        combined_output = result.stdout + result.stderr
        step_match = re.search(r"Steps: (\d+)", combined_output)
        duration_match = re.search(r"\(([0-9.]+)s at", combined_output)
        record = {
            "dataset": dataset,
            "condition": label,
            "policy_seed": policy_seed,
            "simulation_seed": simulation_seed,
            "vx": vx,
            "vy": vy,
            "yaw_rate": yaw_rate,
            "steps": int(step_match.group(1)) if step_match else "",
            "duration_seconds": float(duration_match.group(1)) if duration_match else "",
            "status": "ok" if result.returncode == 0 else "failed",
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "video": str(output.relative_to(ROOT)),
        }
        key = (dataset, label, policy_seed)
        records = [
            old for old in records
            if (old["dataset"], old["condition"], int(old["policy_seed"])) != key
        ]
        records.append(record)
        with manifest_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)
        print(
            f"[{job_index}/{len(jobs)}] {dataset}: seed={policy_seed} {label}: "
            f"{record['status']}, steps={record['steps']}",
            flush=True,
        )
        if result.returncode != 0:
            print(combined_output, flush=True)

    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
