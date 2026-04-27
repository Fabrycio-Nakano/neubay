#!/bin/bash
# =============================================================================
# NEUBAY — World Model Training (offline_world/cont_ensemble.py)
# Necessário APENAS se não usar os checkpoints pré-treinados.
# Dispara 3 seeds em paralelo via job array.
#
# Uso:
#   sbatch run_world_model.sh <domínio> <dataset_name> [total_epochs]
#
# Exemplos:
#   sbatch run_world_model.sh d4rl_loco hopper-random-v2 1200
#   sbatch run_world_model.sh d4rl_loco halfcheetah-medium-replay-v2
#   sbatch run_world_model.sh d4rl_loco walker2d-medium-v2 1200
#   sbatch run_world_model.sh d4rl_loco halfcheetah-medium-expert-v2 600
#   sbatch run_world_model.sh neorl Hopper-v3-low 1200
#   sbatch run_world_model.sh adroit pen-human-v1
#   sbatch run_world_model.sh adroit pen-cloned-v1 2400
#   sbatch run_world_model.sh adroit hammer-cloned-v1 1200
#   sbatch run_world_model.sh antmaze antmaze-umaze-v2 1200
#
# Nota: se total_epochs não for informado, o valor do config base será usado.
# =============================================================================

#SBATCH --job-name=neubay_world
#SBATCH --array=0-2               # 3 seeds em paralelo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6         # treinamento do world model é CPU-light
#SBATCH --gres=gpu:1
#SBATCH --mem=24G                 # dataset D4RL + ensemble 128 modelos
#SBATCH --time=12:00:00           # world model pode demorar (até 2400 epochs)
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
# Caminhos — NADA na home
# --------------------------------------------------------------------------- #
RAID_BASE="/raid/${USER}/neubay"
REPO_DIR="${RAID_BASE}/neubay-main"
CONTAINER="${RAID_BASE}/neubay.sif"
LOG_DIR="${RAID_BASE}/logs"

mkdir -p "${LOG_DIR}"

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
# Execução via Apptainer
# --------------------------------------------------------------------------- #
apptainer exec \
    --nv \
    --bind "${RAID_BASE}:${RAID_BASE}" \
    --bind "/raid/${USER}:/raid/${USER}" \
    "${CONTAINER}" \
    bash -c "
        set -e

        export XLA_PYTHON_CLIENT_PREALLOCATE=false

        export PYTHONPATH=${REPO_DIR}:\${PYTHONPATH}
        cd ${REPO_DIR}

        python offline_world/cont_ensemble.py \
            --config-path=../configs/${DOMAIN} \
            --config-name=base \
            dataset_name=${DATASET} \
            ${EPOCHS_ARG} \
            seed=${SEED}
    "

echo "Job finalizado com código: $?"
