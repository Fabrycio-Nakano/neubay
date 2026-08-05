#!/bin/bash
#SBATCH --job-name=NEUBAY_probes
#SBATCH --partition=ovx01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/probes-%j.out
#SBATCH --error=logs/probes-%j.err

set -euo pipefail

# Ajuste este caminho se o projeto estiver em outro diretorio na OVX.
PROJECT_DIR="${PROJECT_DIR:-/raid/user_fabrycioalmada/neubay}"
SIF_PATH="${SIF_PATH:-${PROJECT_DIR}/neubay.sif}"
CACHE_DIR="${PROJECT_DIR}/.cache"
CONTAINER_HOME="/home/user_fabrycioalmada"
WRITABLE_MUJOCO_PY="${CACHE_DIR}/mujoco_py_pkg"

# Parametros do experimento. Voce pode sobrescrever na chamada do sbatch:
#   DATASET_NAME=hopper-medium-v2 DOMAIN=d4rl_loco sbatch papers/submit_linear_probes_ovx.slurm
DOMAIN="${DOMAIN:-d4rl_loco}"
DATASET_NAME="${DATASET_NAME:-hopper-medium-v2}"
MODEL_SEED="${MODEL_SEED:-0}"
SAMPLE_SIZE="${SAMPLE_SIZE:-10000}"
AGGREGATE="${AGGREGATE:-member}"
MEMBER_INDEX="${MEMBER_INDEX:-0}"
SKIP_MLP="${SKIP_MLP:-false}"
RANDOM_MODEL_IF_MISSING="${RANDOM_MODEL_IF_MISSING:-false}"
HDF5_DATASET="${HDF5_DATASET:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
PROBE_SEEDS="${PROBE_SEEDS:-0:1:2}"
MLP_MAX_EPOCHS="${MLP_MAX_EPOCHS:-500}"
MLP_PATIENCE="${MLP_PATIENCE:-30}"
RUN_TAG="${RUN_TAG:-scaled_mlp_v2}"

EXTRA_ARGS=""
if [ "${SKIP_MLP}" = "true" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --skip-mlp"
fi
if [ "${RANDOM_MODEL_IF_MISSING}" = "true" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --random-model-if-missing"
fi
if [ -n "${HDF5_DATASET}" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --hdf5-dataset '${HDF5_DATASET}'"
fi
if [ -n "${CHECKPOINT_DIR}" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --checkpoint-dir '${CHECKPOINT_DIR}'"
fi
if [ -n "${CHECKPOINT_PATH}" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --checkpoint-path '${CHECKPOINT_PATH}'"
fi

mkdir -p logs "${CACHE_DIR}/home" "${PROJECT_DIR}/papers/probe_outputs"

export APPTAINERENV_D4RL_SUPPRESS_IMPORT_ERROR=1
export APPTAINERENV_PYTHONWARNINGS="ignore"
export APPTAINERENV_LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="0"
export APPTAINERENV_JAX_CUDA_P2P_DISABLE=1
export APPTAINERENV_NCCL_P2P_DISABLE=1
export APPTAINERENV_LD_LIBRARY_PATH="/.singularity.d/libs:/usr/lib/x86_64-linux-gnu:${CONTAINER_HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export APPTAINERENV_MUJOCO_GL="osmesa"
export APPTAINERENV_PYOPENGL_PLATFORM="osmesa"
export APPTAINERENV_XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE="false"

echo "[INFO] Project: ${PROJECT_DIR}"
echo "[INFO] Domain: ${DOMAIN}"
echo "[INFO] Dataset: ${DATASET_NAME}"
echo "[INFO] Seed: ${MODEL_SEED}"
echo "[INFO] Sample size: ${SAMPLE_SIZE}"
echo "[INFO] Aggregate: ${AGGREGATE}"
echo "[INFO] Skip MLP: ${SKIP_MLP}"
echo "[INFO] Random model if missing: ${RANDOM_MODEL_IF_MISSING}"
echo "[INFO] HDF5 dataset: ${HDF5_DATASET:-'(D4RL/NEORL mode)'}"
echo "[INFO] Checkpoint dir: ${CHECKPOINT_DIR:-'(default)'}"
echo "[INFO] Checkpoint path: ${CHECKPOINT_PATH:-'(auto)'}"
echo "[INFO] Probe seeds: ${PROBE_SEEDS}"
echo "[INFO] MLP max epochs: ${MLP_MAX_EPOCHS}"
echo "[INFO] MLP patience: ${MLP_PATIENCE}"
echo "[INFO] Run tag: ${RUN_TAG}"

JOB_STARTED_SECONDS="$(date +%s)"
apptainer exec --nv --no-home \
  --bind "${PROJECT_DIR}:/workspace" \
  --bind "${CACHE_DIR}/home:${CONTAINER_HOME}" \
  --bind "${WRITABLE_MUJOCO_PY}:/opt/neubay/lib/python3.10/site-packages/mujoco_py" \
  --env HOME="${CONTAINER_HOME}" \
  "${SIF_PATH}" \
  bash -c "cd /workspace && python papers/run_linear_probes_neubay.py \
    --domain '${DOMAIN}' \
    --dataset-name '${DATASET_NAME}' \
    --model-seed '${MODEL_SEED}' \
    --sample-size '${SAMPLE_SIZE}' \
    --aggregate '${AGGREGATE}' \
    --member-index '${MEMBER_INDEX}' \
    --probe-seeds '${PROBE_SEEDS}' \
    --mlp-max-epochs '${MLP_MAX_EPOCHS}' \
    --mlp-patience '${MLP_PATIENCE}' \
    --run-tag '${RUN_TAG}' \
    ${EXTRA_ARGS}"
JOB_FINISHED_SECONDS="$(date +%s)"
JOB_ELAPSED_SECONDS=$((JOB_FINISHED_SECONDS - JOB_STARTED_SECONDS))
printf '[TIME] SLURM payload elapsed: %02d:%02d:%02d (%d seconds)\n' \
  $((JOB_ELAPSED_SECONDS / 3600)) \
  $(((JOB_ELAPSED_SECONDS % 3600) / 60)) \
  $((JOB_ELAPSED_SECONDS % 60)) \
  "${JOB_ELAPSED_SECONDS}"
