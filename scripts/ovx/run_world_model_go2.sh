#!/bin/bash
# =============================================================================
# NEUBAY — treino do world model Go2 com proveniência explícita do dataset.
#
# Medium, seeds 0, 1 e 2:
#   sbatch --array=0-2 scripts/ovx/run_world_model_go2.sh \
#     Go2JoystickFlatTerrain-direction-medium-replay-v0
# =============================================================================

#SBATCH --job-name=neubay_go2_world
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --output=/raid/%u/neubay/logs/world_go2_%A_%a.out
#SBATCH --error=/raid/%u/neubay/logs/world_go2_%A_%a.err

set -euo pipefail

DATASET_NAME="${1:?Informe o dataset Go2}"
TOTAL_EPOCHS="${2:-}"
SEED="${SLURM_ARRAY_TASK_ID}"

REPO_DIR="/raid/${USER}/neubay"
CONTAINER="${REPO_DIR}/neubay.sif"
DATASET="${REPO_DIR}/datasets/${DATASET_NAME}/data/main_data.hdf5"
LOG_DIR="${REPO_DIR}/logs"
CACHE_DIR="${REPO_DIR}/.cache"
CONTAINER_HOME="/home/${USER}"
WRITABLE_MUJOCO_PY="${CACHE_DIR}/mujoco_py_pkg"

mkdir -p "${LOG_DIR}" "${CACHE_DIR}/home"
test -f "${CONTAINER}" || { echo "[ERROR] Container não encontrado: ${CONTAINER}"; exit 1; }
test -f "${DATASET}" || { echo "[ERROR] Dataset não encontrado: ${DATASET}"; exit 1; }

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

DATASET_SHA256="$(sha256sum "${DATASET}" | cut -d' ' -f1)"
CHECKPOINT_DIR="${REPO_DIR}/offline_world/ckpt/wm_trained/go2/${DATASET_NAME}"

echo "============================================"
echo "Job:            ${SLURM_JOB_ID} (${SLURM_ARRAY_JOB_ID}[${SEED}])"
echo "Dataset name:   ${DATASET_NAME}"
echo "Dataset path:   ${DATASET}"
echo "Dataset SHA256: ${DATASET_SHA256}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Epochs:         ${TOTAL_EPOCHS:-'(padrão do config)'}"
echo "============================================"

EPOCHS_ARG=""
if [ -n "${TOTAL_EPOCHS}" ]; then
    EPOCHS_ARG="ensemble.total_epochs=${TOTAL_EPOCHS}"
fi

export APPTAINERENV_LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
export APPTAINERENV_JAX_CUDA_P2P_DISABLE=1
export APPTAINERENV_NCCL_P2P_DISABLE=1
export APPTAINERENV_LD_LIBRARY_PATH="/.singularity.d/libs:/usr/lib/x86_64-linux-gnu:${CONTAINER_HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export APPTAINERENV_MUJOCO_GL="osmesa"
export APPTAINERENV_PYOPENGL_PLATFORM="osmesa"

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
        export PYTHONPATH=/workspace:\${PYTHONPATH:-}
        cd /workspace
        python offline_world/cont_ensemble.py \
            --config-path=../configs/go2 \
            --config-name=base \
            dataset_name=${DATASET_NAME} \
            dataset_path=${DATASET} \
            ensemble.save_dir=offline_world/ckpt/wm_trained/go2 \
            seed=${SEED} \
            ${EPOCHS_ARG}
    "
