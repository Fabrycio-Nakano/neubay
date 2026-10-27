#!/usr/bin/env bash
# Reuse the local Isaac image as a Python runtime, without modifying it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MODE="${1:-check}"
IMAGE=ambiente-rl-isaaclab:2.3.0
DEPS="${PWD}/.deps-go2-4090"
mkdir -p "$DEPS" .docker-home .pip-tmp logs
BASE=(docker run --rm --pull=never --user 0:0
  --read-only --tmpfs /tmp:rw,size=512m
  --mount "type=bind,source=${PWD},target=/workspace"
  --workdir /workspace --env HOME=/workspace/.docker-home
  --env "NEUBAY_HOST_UID=$(id -u)" --env "NEUBAY_HOST_GID=$(id -g)"
  --env PYTHONDONTWRITEBYTECODE=1)
if [[ "$MODE" == repair-logging ]]; then
  "${BASE[@]}" --env TMPDIR=/workspace/.pip-tmp \
    --entrypoint /bin/bash "$IMAGE" \
    scripts/local/docker_4090_entrypoint.sh /isaac-sim/python.sh \
    -m pip install --no-cache-dir --ignore-installed --upgrade --no-deps \
    --target /workspace/.deps-go2-4090 \
    pandas==2.3.0 pytz==2025.2 python-dateutil==2.9.0.post0 tzdata==2025.2
  exit 0
fi
if [[ "$MODE" == install ]]; then
  FREE_KIB=$(df -Pk . | awk 'END {print $4}')
  (( FREE_KIB >= 8 * 1024 * 1024 )) || { echo 'Installation needs at least 8 GiB free as a preliminary space check.' >&2; exit 1; }
  [[ ! -f "$DEPS/.complete" ]] || { echo 'Dependencies already installed; use check.'; exit 0; }
  "${BASE[@]}" --env TMPDIR=/workspace/.pip-tmp \
    --entrypoint /bin/bash "$IMAGE" \
    scripts/local/docker_4090_entrypoint.sh /isaac-sim/python.sh \
    -m pip install --no-cache-dir --ignore-installed --upgrade \
    --target /workspace/.deps-go2-4090 \
    -r /workspace/scripts/local/requirements-go2-4090.txt
  touch "$DEPS/.complete"
  du -sh "$DEPS"
  df -h .
  exit 0
fi
[[ -f "$DEPS/.complete" ]] || { echo 'Run install first.' >&2; exit 1; }
# Do not source Isaac's Python setup: its bundled packages/CUDA paths can
# override the project's private dependencies. Keep only interpreter/driver libs.
exec "${BASE[@]}" --gpus 'device=0' \
  --env PYTHON=/isaac-sim/kit/python/bin/python3 \
  --env PYTHONPATH=/workspace/.deps-go2-4090:/workspace \
  --env LD_LIBRARY_PATH=/isaac-sim/kit/python/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
  --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --env "SEED=${SEED:-0}" --env "WANDB_MODE=${WANDB_MODE:-offline}" \
  --env "DATASET_NAME=${DATASET_NAME:-Go2JoystickFlatTerrain-forward-medium-v0}" \
  --entrypoint /bin/bash "$IMAGE" \
  scripts/local/docker_4090_entrypoint.sh \
  bash scripts/local/run_forward_medium_4090.sh "$MODE"
