#!/usr/bin/env bash
# Repeat the three longest 4090 conditions for OVX policy seeds 0, 1 and 2.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mode="${1:-run}"
[[ "$mode" == run || "$mode" == --dry-run ]] || { echo 'Usage: bash scripts/local/generate_forward_medium_ovx.sh [--dry-run]' >&2; exit 2; }
python=.venv-go2-video/bin/python
checkpoints=offline_agent/ckpt/imported/go2-ordenado-20260930T145042/go2/Go2JoystickFlatTerrain-forward-medium-v0
output=videos/forward_medium
export JAX_PLATFORMS=cpu MUJOCO_GL=egl MESA_SHADER_CACHE_DIR=/tmp/neubay_mesa_cache
for seed in 0 1 2; do
  [[ -s "$checkpoints/agent_seed${seed}.eqx" ]] || { echo "Missing checkpoint: seed $seed" >&2; exit 1; }
done
if [[ "$mode" == run ]]; then
  mkdir -p "$output"
  exec 9>"$output/.ovx.lock"
  flock -n 9 || { echo 'An OVX video batch is already running.' >&2; exit 1; }
fi
for seed in 0 1 2; do
  for vx in 1.0 1.5 2.0; do
    name="ovx_go2-ordenado-20260930T145042_seed${seed}_vx${vx}_sim0"
    video="$output/$name.mp4"
    log="$output/$name.log"
    command=("$python" scripts/local/generate_go2_video.py
      "$checkpoints/agent_seed${seed}.eqx"
      --output "$output/$name.partial.mp4"
      --steps 5000 --seed 0 --vx "$vx" --vy 0 --yaw-rate 0
      --render-every 2 --width 640 --height 480 --camera track)
    if [[ "$mode" == --dry-run ]]; then
      printf '%q ' "${command[@]}"; printf '\n'
      continue
    fi
    if [[ -s "$video" && -f "$log" ]] && grep -q '^Steps: ' "$log" && ffprobe -v error "$video" >/dev/null 2>&1; then
      echo "Already completed: $video"
      continue
    fi
    [[ ! -e "$video" ]] || { echo "Existing video without validated log: $video" >&2; exit 1; }
    "${command[@]}" 2>&1 | tee "$log"
    grep -q '^Steps: ' "$log"
    ffprobe -v error "$output/$name.partial.mp4" >/dev/null
    mv "$output/$name.partial.mp4" "$video"
    printf 'Final video: %s\n' "$video" | tee -a "$log"
  done
done
