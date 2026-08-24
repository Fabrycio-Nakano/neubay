#!/bin/bash
# =============================================================================
# NEUBAY — Go2 Joystick agent training on pre-trained world models.
# O dataset é informado no primeiro argumento; o padrão preserva o experimento
# expert original. Use --array=0-2 no retreino medium.
#
# Full run:
#   sbatch scripts/ovx/run_agent_go2.sh
# Medium, seeds 0, 1 e 2:
#   sbatch --array=0-2 scripts/ovx/run_agent_go2.sh \
#     Go2JoystickFlatTerrain-direction-medium-replay-v0
# Smoke test:
#   sbatch --array=0 --time=01:00:00 --export=ALL,SMOKE_TEST=true \
#     scripts/ovx/run_agent_go2.sh
# =============================================================================

#SBATCH --job-name=neubay_go2_agent
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=/raid/%u/neubay/logs/agent_go2_%A_%a.out
#SBATCH --error=/raid/%u/neubay/logs/agent_go2_%A_%a.err

set -euo pipefail

DATASET_NAME="${1:-Go2JoystickFlatTerrain-direction-expert-v1}"
RUN_BATCH_ID="${RUN_BATCH_ID:-manual}"
if [ -z "${DATASET_FAMILY:-}" ]; then
    case "${DATASET_NAME}" in
        *-forward-*) DATASET_FAMILY="forward" ;;
        *) DATASET_FAMILY="direction" ;;
    esac
fi

SEED="${SLURM_ARRAY_TASK_ID}"
RAID_BASE="/raid/${USER}/neubay"
REPO_DIR="${RAID_BASE}"
CONTAINER="${RAID_BASE}/neubay.sif"
WANDB_DIR="${RAID_BASE}/wandb"
LOG_DIR="${RAID_BASE}/logs"
CACHE_DIR="${RAID_BASE}/.cache"
CONTAINER_HOME="/home/${USER}"
WRITABLE_MUJOCO_PY="${CACHE_DIR}/mujoco_py_pkg"
DATASET="${REPO_DIR}/datasets/${DATASET_NAME}/data/main_data.hdf5"
WORLD_MODEL_SAVE_DIR="${WORLD_MODEL_SAVE_DIR:-offline_world/ckpt/wm_trained/go2}"
AGENT_SAVE_ROOT="${AGENT_SAVE_ROOT:-offline_agent/ckpt}"
CHECKPOINT="${REPO_DIR}/${WORLD_MODEL_SAVE_DIR}/${DATASET_NAME}/ensemble_seed${SEED}.eqx"

mkdir -p "${LOG_DIR}" "${WANDB_DIR}" "${CACHE_DIR}/home"
test -f "${CONTAINER}" || { echo "[ERROR] Container not found: ${CONTAINER}"; exit 1; }
test -f "${DATASET}" || { echo "[ERROR] Dataset not found: ${DATASET}"; exit 1; }
test -f "${CHECKPOINT}" || { echo "[ERROR] Checkpoint not found: ${CHECKPOINT}"; exit 1; }
DATASET_SHA256="$(sha256sum "${DATASET}" | cut -d' ' -f1)"
if [ -n "${EXPECTED_DATASET_SHA256:-}" ] && [ "${DATASET_SHA256}" != "${EXPECTED_DATASET_SHA256}" ]; then
    echo "[ERROR] Dataset SHA-256 diverge do manifesto" >&2
    exit 1
fi

if [ -f "/home/${USER}/.netrc" ]; then
    cp "/home/${USER}/.netrc" "${CACHE_DIR}/home/.netrc"
fi
if [ -d "/home/${USER}/.config/wandb" ]; then
    mkdir -p "${CACHE_DIR}/home/.config"
    cp -r "/home/${USER}/.config/wandb" "${CACHE_DIR}/home/.config/"
fi
source "${REPO_DIR}/scripts/ovx/load_wandb_env.sh"

export APPTAINERENV_LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
export APPTAINERENV_JAX_CUDA_P2P_DISABLE=1
export APPTAINERENV_NCCL_P2P_DISABLE=1
export APPTAINERENV_LD_LIBRARY_PATH="/.singularity.d/libs:/usr/lib/x86_64-linux-gnu:${CONTAINER_HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export APPTAINERENV_MUJOCO_GL="osmesa"
export APPTAINERENV_PYOPENGL_PLATFORM="osmesa"

EXTRA_OVERRIDES=""
RUN_KIND="full"
WANDB_JOB_TYPE="world-model-agent-training"
if [ "${SMOKE_TEST:-false}" = "true" ]; then
    AGENT_SAVE_ROOT="offline_agent/ckpt/smoke"
    EXTRA_OVERRIDES="train.grad_steps=2000 eval.times=2 train.buffer_size=200000 collect.parallel_size=100 collect.max_rollout_len=10"
    RUN_KIND="smoke"
    WANDB_JOB_TYPE="smoke-test"
fi

if [ "${RUN_KIND}" = "smoke" ]; then
    RUN_NAME="SMOKE-NEUBAY-${DATASET_NAME}-S${SEED}-J${SLURM_ARRAY_JOB_ID}"
else
    RUN_NAME="NEUBAY-${DATASET_NAME}-S${SEED}"
fi
RUN_ID="${RUN_NAME}-J${SLURM_ARRAY_JOB_ID}"
AGENT_PATH="${REPO_DIR}/${AGENT_SAVE_ROOT}/go2/${DATASET_NAME}/agent_seed${SEED}.eqx"
if [ "${RUN_KIND}" != "smoke" ] && [ -e "${AGENT_PATH}" ] && [ "${ALLOW_OVERWRITE:-false}" != "true" ]; then
    echo "[ERROR] Agente já existe: ${AGENT_PATH}" >&2
    echo "Use outro RUN_BATCH_ID ou ALLOW_OVERWRITE=true conscientemente." >&2
    exit 1
fi

echo "============================================"
echo "Job:        ${SLURM_JOB_ID} (${SLURM_ARRAY_JOB_ID}[${SEED}])"
echo "Dataset:    ${DATASET}"
echo "SHA256:     ${DATASET_SHA256}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Agent path:  ${AGENT_PATH}"
echo "Batch:       ${RUN_BATCH_ID}"
echo "Smoke test: ${SMOKE_TEST:-false}"
echo "W&B project: agents_go2"
echo "W&B group:   ${DATASET_NAME}"
echo "W&B run:     ${RUN_NAME}"
echo "W&B run ID:  ${RUN_ID}"
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
            save_agent_root=${AGENT_SAVE_ROOT} \
            dataset_name=${DATASET_NAME} \
            dataset_sha256=${DATASET_SHA256} \
            ensemble.save_dir=${WORLD_MODEL_SAVE_DIR} \
            wandb_project=agents_go2 \
            wandb_group=${DATASET_NAME} \
            dataset_path=${DATASET} \
            exp_name=${RUN_NAME} \
            wandb_run_id=${RUN_ID} \
            wandb_job_type=${WANDB_JOB_TYPE} \
            wandb_tags=[go2,${DATASET_FAMILY},world-model-agent,${RUN_KIND},${RUN_BATCH_ID}] \
            ${EXTRA_OVERRIDES}
    "
