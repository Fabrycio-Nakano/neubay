#!/bin/bash
# =============================================================================
# NEUBAY — Agent Training (offline_cont.py)
# Usa checkpoints de world model já pré-treinados.
# Dispara 3 seeds em paralelo via job array (seeds 0, 1, 2).
#
# Uso:
#   sbatch run_agent.sh <domínio> <tarefa>
# =============================================================================

#SBATCH --job-name=neubay_agent
#SBATCH --array=0               # 3 seeds em paralelo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # ~8 CPUs
#SBATCH --gres=gpu:1              # 1 GPU por seed
#SBATCH --mem=32G                 # buffer + modelo + overhead
#SBATCH --time=06:00:00           # agent training: ~2-4h por seed
#SBATCH --output=/raid/%u/neubay/logs/agent_%x_%A_%a.out
#SBATCH --error=/raid/%u/neubay/logs/agent_%x_%A_%a.err

# --------------------------------------------------------------------------- #
# Argumentos
# --------------------------------------------------------------------------- #
DOMAIN=${1:?"Informe o domínio: d4rl_loco | neorl | adroit | antmaze"}
TASK=${2:?"Informe a tarefa (e.g. halfcheetah_medium_expert)"}
SEED=$SLURM_ARRAY_TASK_ID   # 0, 1 ou 2

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
RAID_BASE="/raid/${USER}/neubay"
REPO_DIR="${RAID_BASE}"
CONTAINER="${RAID_BASE}/neubay.sif"
WANDB_DIR="${RAID_BASE}/wandb"
LOG_DIR="${RAID_BASE}/logs"
CACHE_DIR="${RAID_BASE}/.cache"
CONTAINER_HOME="/home/${USER}"
WRITABLE_MUJOCO_PY="${CACHE_DIR}/mujoco_py_pkg"

mkdir -p "${LOG_DIR}" "${WANDB_DIR}" "${CACHE_DIR}/home"

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
echo "Tarefa:   ${TASK}"
echo "Seed:     ${SEED}"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

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

        # Evita que um único processo aloque toda a VRAM (essencial com array jobs)
        export XLA_PYTHON_CLIENT_PREALLOCATE=false

        # wandb salva runs no raid
        export WANDB_DIR=${WANDB_DIR}

        export PYTHONPATH=/workspace:\${PYTHONPATH}
        cd /workspace

        python offline_cont.py \
            --config-path=configs/${DOMAIN} \
            --config-name=base \
            task=${TASK} \
            seed=${SEED}
    "
