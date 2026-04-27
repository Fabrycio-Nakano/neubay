#!/bin/bash
#SBATCH --job-name=neubay
#SBATCH --partition=ovx01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/raid/user_fabrycioalmada/neubay/logs/%x-%j.out
#SBATCH --error=/raid/user_fabrycioalmada/neubay/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")}"
ENV_PREFIX="${ENV_PREFIX:-/raid/user_fabrycioalmada/envs/neubay}"

TASK="${TASK:-Hopper_v3_low}"
SEED="${SEED:-0}"

WANDB_DIR="${WANDB_DIR:-${PROJECT_DIR}/wandb}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"

echo "[INFO] PROJECT_DIR=${PROJECT_DIR}"
echo "[INFO] ENV_PREFIX=${ENV_PREFIX}"
echo "[INFO] TASK=${TASK}"
echo "[INFO] SEED=${SEED}"

mkdir -p "${LOG_DIR}" "${WANDB_DIR}" "${OUTPUT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

export PYTHONPATH="${PROJECT_DIR}:$PYTHONPATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR="${WANDB_DIR}"

cd "${PROJECT_DIR}"

python offline_cont.py \
  --config-path=configs/neorl \
  --config-name=base \
  task="${TASK}" \
  seed="${SEED}"
