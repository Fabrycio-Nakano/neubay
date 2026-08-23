#!/usr/bin/env python3
"""Generate a matched grid of Go2 videos for policy seeds 0 and 2."""

import csv
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv-go2-video" / "bin" / "python"
GENERATOR = ROOT / "scripts" / "local" / "generate_go2_video.py"
CHECKPOINT_ROOT = (
    ROOT
    / "offline_agent/ckpt/go2/Go2JoystickFlatTerrain-direction-expert-v1"
)
OUTPUT_ROOT = ROOT / "videos" / "variants"

# (label, forward velocity, lateral velocity, yaw rate)
CONDITIONS = [
    ("stand", 0.00, 0.00, 0.00),
    ("fwd010", 0.10, 0.00, 0.00),
    ("fwd025", 0.25, 0.00, 0.00),
    ("fwd050", 0.50, 0.00, 0.00),
    ("fwd075", 0.75, 0.00, 0.00),
    ("back010", -0.10, 0.00, 0.00),
    ("back025", -0.25, 0.00, 0.00),
    ("left010", 0.00, 0.10, 0.00),
    ("left025", 0.00, 0.25, 0.00),
    ("right010", 0.00, -0.10, 0.00),
    ("right025", 0.00, -0.25, 0.00),
    ("yawleft025", 0.00, 0.00, 0.25),
    ("yawleft050", 0.00, 0.00, 0.50),
    ("yawright025", 0.00, 0.00, -0.25),
    ("yawright050", 0.00, 0.00, -0.50),
    ("fwd025_left010", 0.25, 0.10, 0.00),
    ("fwd025_right010", 0.25, -0.10, 0.00),
    ("fwd025_yawleft025", 0.25, 0.00, 0.25),
    ("fwd025_yawright025", 0.25, 0.00, -0.25),
    ("fwd050_yawleft050", 0.50, 0.00, 0.50),
    # Extended sweep: 50 additional command configurations.
    ("back075", -0.75, 0.00, 0.00),
    ("back050", -0.50, 0.00, 0.00),
    ("back040", -0.40, 0.00, 0.00),
    ("back030", -0.30, 0.00, 0.00),
    ("back020", -0.20, 0.00, 0.00),
    ("fwd005", 0.05, 0.00, 0.00),
    ("fwd015", 0.15, 0.00, 0.00),
    ("fwd020", 0.20, 0.00, 0.00),
    ("fwd030", 0.30, 0.00, 0.00),
    ("fwd040", 0.40, 0.00, 0.00),
    ("fwd060", 0.60, 0.00, 0.00),
    ("fwd100", 1.00, 0.00, 0.00),
    ("right050", 0.00, -0.50, 0.00),
    ("right040", 0.00, -0.40, 0.00),
    ("right030", 0.00, -0.30, 0.00),
    ("right020", 0.00, -0.20, 0.00),
    ("right015", 0.00, -0.15, 0.00),
    ("right005", 0.00, -0.05, 0.00),
    ("left005", 0.00, 0.05, 0.00),
    ("left015", 0.00, 0.15, 0.00),
    ("left020", 0.00, 0.20, 0.00),
    ("left030", 0.00, 0.30, 0.00),
    ("left040", 0.00, 0.40, 0.00),
    ("left050", 0.00, 0.50, 0.00),
    ("yawright100", 0.00, 0.00, -1.00),
    ("yawright075", 0.00, 0.00, -0.75),
    ("yawright040", 0.00, 0.00, -0.40),
    ("yawright015", 0.00, 0.00, -0.15),
    ("yawleft015", 0.00, 0.00, 0.15),
    ("yawleft040", 0.00, 0.00, 0.40),
    ("yawleft075", 0.00, 0.00, 0.75),
    ("yawleft100", 0.00, 0.00, 1.00),
    ("fwd010_left010", 0.10, 0.10, 0.00),
    ("fwd010_right010", 0.10, -0.10, 0.00),
    ("fwd050_left020", 0.50, 0.20, 0.00),
    ("fwd050_right020", 0.50, -0.20, 0.00),
    ("back025_left010", -0.25, 0.10, 0.00),
    ("back025_right010", -0.25, -0.10, 0.00),
    ("fwd010_yawleft015", 0.10, 0.00, 0.15),
    ("fwd010_yawright015", 0.10, 0.00, -0.15),
    ("fwd050_yawleft025", 0.50, 0.00, 0.25),
    ("fwd050_yawright025", 0.50, 0.00, -0.25),
    ("back025_yawleft025", -0.25, 0.00, 0.25),
    ("back025_yawright025", -0.25, 0.00, -0.25),
    ("fwd025_left010_yawleft025", 0.25, 0.10, 0.25),
    ("fwd025_left010_yawright025", 0.25, 0.10, -0.25),
    ("fwd025_right010_yawleft025", 0.25, -0.10, 0.25),
    ("fwd025_right010_yawright025", 0.25, -0.10, -0.25),
    ("fwd075_left025_yawleft050", 0.75, 0.25, 0.50),
    ("fwd075_right025_yawright050", 0.75, -0.25, -0.50),
]


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / "manifest.csv"
    environment = os.environ.copy()
    environment.update(
        JAX_PLATFORMS="cpu",
        MUJOCO_GL="egl",
        MESA_SHADER_CACHE_DIR="/tmp/neubay_mesa_cache",
    )
    records = []
    if manifest_path.exists():
        with manifest_path.open(newline="") as file:
            records = list(csv.DictReader(file))
    completed_keys = {
        (record["condition"], int(record["policy_seed"]))
        for record in records
        if record["status"] == "ok" and (ROOT / record["video"]).is_file()
    }
    total = len(CONDITIONS) * 2
    completed = 0
    for condition_index, (label, vx, vy, yaw_rate) in enumerate(CONDITIONS):
        simulation_seed = 100 + condition_index
        for policy_seed in (0, 2):
            if (label, policy_seed) in completed_keys:
                continue
            checkpoint = CHECKPOINT_ROOT / f"agent_seed{policy_seed}.eqx"
            output = OUTPUT_ROOT / f"seed{policy_seed}_{label}.mp4"
            command = [
                str(PYTHON),
                str(GENERATOR),
                str(checkpoint),
                "--output", str(output),
                "--steps", "500",
                "--seed", str(simulation_seed),
                "--vx", str(vx),
                "--vy", str(vy),
                "--yaw-rate", str(yaw_rate),
                "--render-every", "2",
                "--width", "640",
                "--height", "480",
            ]
            result = subprocess.run(
                command, env=environment, text=True, capture_output=True
            )
            combined_output = result.stdout + result.stderr
            step_match = re.search(r"Steps: (\d+)", combined_output)
            duration_match = re.search(r"\(([0-9.]+)s at", combined_output)
            new_record = {
                    "condition": label,
                    "policy_seed": policy_seed,
                    "simulation_seed": simulation_seed,
                    "vx": vx,
                    "vy": vy,
                    "yaw_rate": yaw_rate,
                    "steps": int(step_match.group(1)) if step_match else "",
                    "duration_seconds": (
                        float(duration_match.group(1)) if duration_match else ""
                    ),
                    "status": "ok" if result.returncode == 0 else "failed",
                    "video": str(output.relative_to(ROOT)),
                }
            records = [
                record for record in records
                if not (
                    record["condition"] == label
                    and int(record["policy_seed"]) == policy_seed
                )
            ]
            records.append(new_record)
            completed += 1
            print(
                f"[{completed:02d}/{total}] seed={policy_seed} {label}: "
                f"{records[-1]['status']}, steps={records[-1]['steps']}",
                flush=True,
            )
            if result.returncode != 0:
                print(combined_output, file=sys.stderr, flush=True)

            with manifest_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)

    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
