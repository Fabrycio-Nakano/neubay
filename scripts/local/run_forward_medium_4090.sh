#!/usr/bin/env bash
# Run inside a compatible NEUBAY environment (native or container).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MODE="${1:-check}"
SEED="${SEED:-0}"
PYTHON="${PYTHON:-python3}"
DATASET="${DATASET_NAME:-Go2JoystickFlatTerrain-forward-medium-v0}"
DATA="datasets/${DATASET}/data/main_data.hdf5"
SHA=$(awk -F '\t' -v name="$DATASET" '$4 == name {print $5}' scripts/ovx/go2_datasets.tsv)
[[ "$SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "Dataset not found in manifest: $DATASET" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
# Keep logs locally until the user configures W&B authentication.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${PWD}/wandb"
mkdir -p logs "$WANDB_DIR"

case "$MODE" in
  check|smoke|world|agent|full) ;;
  *) echo 'Usage: bash scripts/local/run_forward_medium_4090.sh check|smoke|world|agent|full' >&2; exit 2 ;;
esac

"$PYTHON" - <<'PY'
import jax
import pandas
from offline_world.cont_ensemble import LearnedContEnv
from experience.agent_buffer import TapeBuffer
import offline_cont
devices = jax.devices()
print('JAX:', jax.__version__, 'Devices:', devices)
print('Pandas:', pandas.__version__, pandas.__file__)
if not any(device.platform == 'gpu' for device in devices):
    raise SystemExit('No GPU detected by JAX; training was not started.')
PY
[[ "$MODE" == check ]] && exit 0
printf '%s  %s\n' "$SHA" "$DATA" | sha256sum -c -

# Serialize launches from this checkout, including different seeds.
exec 9>logs/forward_medium.lock
flock -n 9 || { echo 'Another forward-medium run is active in this checkout.' >&2; exit 1; }
FREE_KIB=$(df -Pk . | awk 'END {print $4}')
(( FREE_KIB >= 3 * 1024 * 1024 )) || { echo 'Less than 3 GiB free; free space before training.' >&2; exit 1; }

WORLD_ROOT=offline_world/ckpt/4090/go2
AGENT_ROOT=offline_agent/ckpt/4090
WORLD_EXTRA=()
AGENT_EXTRA=()
if [[ "$MODE" == smoke ]]; then
  # Use separate paths; these small models are never used by full training.
  WORLD_ROOT=offline_world/ckpt/4090-smoke/go2
  AGENT_ROOT=offline_agent/ckpt/4090-smoke
  WORLD_EXTRA=(ensemble.total_size=2 ensemble.hidden_size=32 ensemble.total_epochs=1 +ensemble.max_samples=10000)
  AGENT_EXTRA=(collect.ensemble_size=2 train.grad_steps=200 train.buffer_size=20000 collect.parallel_size=10 collect.max_rollout_len=5 eval.times=2)
fi
COMMON=(--config-name=base "dataset_name=$DATASET" "dataset_path=$DATA" "dataset_sha256=$SHA" "seed=$SEED" "ensemble.save_dir=$WORLD_ROOT" "wandb_group=$DATASET")
STAMP=$(date -u +%Y%m%dT%H%M%S)-$$

world() {
  local checkpoint="$WORLD_ROOT/$DATASET/ensemble_seed${SEED}.eqx"
  [[ ! -e "$checkpoint" ]] || { echo "Checkpoint already exists: $checkpoint. Use agent mode or another SEED." >&2; return 1; }
  "$PYTHON" offline_world/cont_ensemble.py --config-path=../configs/go2 \
    "${COMMON[@]}" wandb_project=world_models_go2 wandb_job_type=world-model-training \
    "${WORLD_EXTRA[@]}" 2>&1 | tee "logs/world_${DATASET}_${MODE}_seed${SEED}_${STAMP}.log"
}
agent() {
  [[ -f "$WORLD_ROOT/$DATASET/ensemble_seed${SEED}.eqx" ]] || { echo 'Matching world model checkpoint missing.' >&2; return 1; }
  [[ ! -e "$AGENT_ROOT/go2/$DATASET/agent_seed${SEED}.eqx" ]] || { echo 'Agent checkpoint already exists; use another SEED.' >&2; return 1; }
  "$PYTHON" offline_cont.py --config-path=configs/go2 "${COMMON[@]}" \
    task=go2_joystick "save_agent_root=$AGENT_ROOT" wandb_project=agents_go2 \
    "exp_name=NEUBAY-${DATASET}-4090-${MODE}-S${SEED}" \
    "wandb_run_id=4090-${DATASET}-${MODE}-S${SEED}-${STAMP}" \
    "${AGENT_EXTRA[@]}" 2>&1 | tee "logs/agent_${DATASET}_${MODE}_seed${SEED}_${STAMP}.log"
}
case "$MODE" in
  world) world ;;
  agent) agent ;;
  full|smoke) world; agent ;;
esac
