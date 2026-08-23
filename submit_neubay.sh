#!/bin/bash
#SBATCH --job-name=NEUBAY_repro
#SBATCH --partition=ovx01
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=logs/neubay-%A_%a.out
#SBATCH --error=logs/neubay-%A_%a.err

set -euo pipefail

########################
# CONFIGURAÇÕES DE CAMINHO
########################
PROJECT_DIR="/raid/user_fabrycioalmada/neubay"
SIF_PATH="${PROJECT_DIR}/neubay.sif"
CACHE_DIR="${PROJECT_DIR}/.cache"
CONTAINER_HOME="/home/user_fabrycioalmada"
WRITABLE_MUJOCO_PY="${CACHE_DIR}/mujoco_py_pkg"

# Parâmetros
DATASET_NAME="${DATASET_NAME:-Hopper-v3-high}"
CONFIG_PATH="configs/neorl"
SEED="${SLURM_ARRAY_TASK_ID}"
RUN_NAME="NEUBAY_${DATASET_NAME}_S${SEED}"
ALGO="NEUBAY"
LAMBDA="5.0"

# Silencia avisos de importação do D4RL e deprecations do Python
export APPTAINERENV_D4RL_SUPPRESS_IMPORT_ERROR=1
export APPTAINERENV_PYTHONWARNINGS="ignore"

# O nome "Limpo" para o WandB
CLEAN_NAME="${ALGO}-${DATASET_NAME}-L${LAMBDA}-S${SEED}"

# O novo "Ambiente" (Projeto) no WandB
NEW_PROJECT="neubay-official-results"

mkdir -p logs "${CACHE_DIR}/home"

###########################################################
# FIX DE GPU E COMPATIBILIDADE (O que resolve o cuInit)
###########################################################
export APPTAINERENV_LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="0" 
export APPTAINERENV_JAX_CUDA_P2P_DISABLE=1
export APPTAINERENV_NCCL_P2P_DISABLE=1

export APPTAINERENV_LD_LIBRARY_PATH="/.singularity.d/libs:/usr/lib/x86_64-linux-gnu:${CONTAINER_HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export APPTAINERENV_MUJOCO_GL="osmesa"
export APPTAINERENV_PYOPENGL_PLATFORM="osmesa"
export APPTAINERENV_XLA_PYTHON_CLIENT_ALLOCATOR="platform"

# Carrega as credenciais locais sem incluí-las no repositório.
REPO_DIR="${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/ovx/load_wandb_env.sh"

export APPTAINERENV_WANDB_PROJECT="${NEW_PROJECT}"
export APPTAINERENV_WANDB_NAME="${CLEAN_NAME}"
export APPTAINERENV_WANDB_RUN_ID="${CLEAN_NAME}-${SLURM_ARRAY_JOB_ID}" # Garante que cada run seja única

echo "[INFO] Iniciando Experimento no Ambiente: ${NEW_PROJECT}"
echo "[INFO] Nome da Run: ${CLEAN_NAME}"

########################
# EXECUÇÃO
########################
apptainer exec --nv --no-home \
  --bind "${PROJECT_DIR}:/workspace" \
  --bind "${CACHE_DIR}/home:${CONTAINER_HOME}" \
  --bind "${WRITABLE_MUJOCO_PY}:/opt/neubay/lib/python3.10/site-packages/mujoco_py" \
  --env HOME="${CONTAINER_HOME}" \
  "${SIF_PATH}" \
  bash -c "cd /workspace && python offline_cont.py \
    --config-path=${CONFIG_PATH} \
    --config-name=base \
    task=Hopper_v3_high \
    dataset_name=${DATASET_NAME} \
    seed=${SEED} \
    +exp_name=${CLEAN_NAME} \
    +wandb_project=${NEW_PROJECT}"
