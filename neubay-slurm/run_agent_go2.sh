#!/bin/bash
# =============================================================================
# NEUBAY — Go2 Joystick agent training on pre-trained world models.
# Seeds 0 and 2 match the checkpoints validated by the representation probes.
#
# Full run:
#   sbatch neubay-slurm/run_agent_go2.sh
# Smoke test:
#   sbatch --array=0 --time=01:00:00 --export=ALL,SMOKE_TEST=true \
#     neubay-slurm/run_agent_go2.sh
# =============================================================================

#SBATCH --job-name=neubay_go2_agent
#SBATCH --array=0,2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=/raid/%u/neubay/logs/agent_go2_%A_%a.out
#SBATCH --error=/raid/%u/neubay/logs/agent_go2_%A_%a.err

set -euo pipefail

SEED="${SLURM_ARRAY_TASK_ID}"
RAID_BASE="/raid/${USER}/neubay"
REPO_DIR="${RAID_BASE}"
CONTAINER="${RAID_BASE}/neubay.sif"
WANDB_DIR="${RAID_BASE}/wandb"
LOG_DIR="${RAID_BASE}/logs"
CACHE_DIR="${RAID_BASE}/.cache"
CONTAINER_HOME="/home/${USER}"
WRITABLE_MUJOCO_PY="${CACHE_DIR}/mujoco_py_pkg"
DATASET="${REPO_DIR}/datasets/Go2JoystickFlatTerrain-direction-expert-v1/data/main_data.hdf5"
CHECKPOINT="${REPO_DIR}/offline_world/ckpt/wm_trained/go2/Go2JoystickFlatTerrain-direction-expert-v1/ensemble_seed${SEED}.eqx"

mkdir -p "${LOG_DIR}" "${WANDB_DIR}" "${CACHE_DIR}/home"
test -f "${CONTAINER}" || { echo "[ERROR] Container not found: ${CONTAINER}"; exit 1; }
test -f "${DATASET}" || { echo "[ERROR] Dataset not found: ${DATASET}"; exit 1; }
test -f "${CHECKPOINT}" || { echo "[ERROR] Checkpoint not found: ${CHECKPOINT}"; exit 1; }

if [ -f "/home/${USER}/.netrc" ]; then
    cp "/home/${USER}/.netrc" "${CACHE_DIR}/home/.netrc"
fi
if [ -d "/home/${USER}/.config/wandb" ]; then
    mkdir -p "${CACHE_DIR}/home/.config"
    cp -r "/home/${USER}/.config/wandb" "${CACHE_DIR}/home/.config/"
fi
if [ -f "${REPO_DIR}/wandb.env" ]; then
    source "${REPO_DIR}/wandb.env"
fi

export APPTAINERENV_LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
export APPTAINERENV_JAX_CUDA_P2P_DISABLE=1
export APPTAINERENV_NCCL_P2P_DISABLE=1
export APPTAINERENV_LD_LIBRARY_PATH="/.singularity.d/libs:/usr/lib/x86_64-linux-gnu:${CONTAINER_HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export APPTAINERENV_MUJOCO_GL="osmesa"
export APPTAINERENV_PYOPENGL_PLATFORM="osmesa"

EXTRA_OVERRIDES=""
if [ "${SMOKE_TEST:-false}" = "true" ]; then
    EXTRA_OVERRIDES="train.grad_steps=2000 eval.times=2 train.buffer_size=200000 collect.parallel_size=100 collect.max_rollout_len=10"
fi

echo "============================================"
echo "Job:        ${SLURM_JOB_ID} (${SLURM_ARRAY_JOB_ID}[${SEED}])"
echo "Dataset:    ${DATASET}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Smoke test: ${SMOKE_TEST:-false}"
echo "============================================"

apptainer exec \
    --nv \
    --no-home \
    --bind "${REPO_DIR}:/workspace" \
    --bind "${CACHE_DIR}/home:${CONTAINER_HOME}" \
    --bind "${WRITABLE_MUJOCO_PY}:/opt/neubay/lib/python3.10/site-packages/mujoco_py" \
    --env HOME="${CONTAINER_HOME}" \
    "${CONTAINER}" \
    bash -c "
        set -e
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export WANDB_DIR=${WANDB_DIR}
        export PYTHONPATH=/workspace:\${PYTHONPATH:-}
        cd /workspace
        python offline_cont.py \
            --config-path=configs/go2 \
            --config-name=base \
            task=go2_joystick \
            seed=${SEED} \
            dataset_path=${DATASET} \
            ${EXTRA_OVERRIDES}
    "
