#!/bin/bash
# =============================================================================
# NEUBAY — World Model Training (offline_world/cont_ensemble.py)
# Necessário APENAS se não usar os checkpoints pré-treinados.
# Dispara 3 seeds em paralelo via job array.
#
# Uso:
#   sbatch run_world_model.sh <domínio> <dataset_name> [total_epochs]
# =============================================================================

#SBATCH --job-name=neubay_world
#SBATCH --array=0-2               # 3 seeds em paralelo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6         # treinamento do world model é CPU-light
#SBATCH --gres=gpu:1
#SBATCH --mem=24G                 # dataset D4RL + ensemble 128 modelos
#SBATCH --time=20:00:00           # world model pode demorar (até 2400 epochs)
#SBATCH --output=/raid/%u/neubay/logs/world_%x_%A_%a.out
#SBATCH --error=/raid/%u/neubay/logs/world_%x_%A_%a.err

# --------------------------------------------------------------------------- #
# Argumentos
# --------------------------------------------------------------------------- #
DOMAIN=${1:?"Informe o domínio: d4rl_loco | neorl | adroit | antmaze"}
DATASET=${2:?"Informe o dataset_name (e.g. hopper-random-v2)"}
TOTAL_EPOCHS=${3:-""}            # opcional
SEED=$SLURM_ARRAY_TASK_ID

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
RAID_BASE="/raid/${USER}/neubay"
REPO_DIR="${RAID_BASE}"
CONTAINER="${RAID_BASE}/neubay.sif"
LOG_DIR="${RAID_BASE}/logs"
CACHE_DIR="${RAID_BASE}/.cache"
CONTAINER_HOME="/home/${USER}"
WRITABLE_MUJOCO_PY="${CACHE_DIR}/mujoco_py_pkg"

mkdir -p "${LOG_DIR}" "${CACHE_DIR}/home"

# --------------------------------------------------------------------------- #
# Sincroniza credenciais do WandB da home real para a home do contêiner
# --------------------------------------------------------------------------- #
if [ -f "/home/${USER}/.netrc" ]; then
    cp "/home/${USER}/.netrc" "${CACHE_DIR}/home/.netrc"
fi
if [ -d "/home/${USER}/.config/wandb" ]; then
    mkdir -p "${CACHE_DIR}/home/.config"
    cp -r "/home/${USER}/.config/wandb" "${CACHE_DIR}/home/.config/"
fi

# Carrega as credenciais locais sem incluí-las no repositório.
source "${REPO_DIR}/scripts/ovx/load_wandb_env.sh"

echo "============================================"
echo "Job:      ${SLURM_JOB_ID} (array ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}])"
echo "Nó:       $(hostname)"
echo "Domínio:  ${DOMAIN}"
echo "Dataset:  ${DATASET}"
echo "Epochs:   ${TOTAL_EPOCHS:-'(padrão do config)'}"
echo "Seed:     ${SEED}"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# --------------------------------------------------------------------------- #
# Monta argumento de epochs (opcional)
# --------------------------------------------------------------------------- #
EPOCHS_ARG=""
if [ -n "${TOTAL_EPOCHS}" ]; then
    EPOCHS_ARG="ensemble.total_epochs=${TOTAL_EPOCHS}"
fi

# --------------------------------------------------------------------------- #
# Configurações de GPU e MuJoCo para a OVX (Estilo submit_neubay.sh)
# --------------------------------------------------------------------------- #
export APPTAINERENV_LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
export APPTAINERENV_JAX_CUDA_P2P_DISABLE=1
export APPTAINERENV_NCCL_P2P_DISABLE=1
export APPTAINERENV_LD_LIBRARY_PATH="/.singularity.d/libs:/usr/lib/x86_64-linux-gnu:${CONTAINER_HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export APPTAINERENV_MUJOCO_GL="osmesa"
export APPTAINERENV_PYOPENGL_PLATFORM="osmesa"

# --------------------------------------------------------------------------- #
# Execução via Apptainer (Estilo submit_neubay.sh)
# --------------------------------------------------------------------------- #
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

        export PYTHONPATH=/workspace:\${PYTHONPATH}
        cd /workspace

        python offline_world/cont_ensemble.py \
            --config-path=../configs/${DOMAIN} \
            --config-name=base \
            dataset_name=${DATASET} \
            ${EPOCHS_ARG} \
            seed=${SEED}
    "
