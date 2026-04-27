#!/bin/bash
# =============================================================================
# NEUBAY — Agent Training (offline_cont.py)
# Usa checkpoints de world model já pré-treinados.
# Dispara 3 seeds em paralelo via job array (seeds 0, 1, 2).
#
# Uso:
#   sbatch run_agent.sh <domínio> <tarefa>
#
# Exemplos:
#   sbatch run_agent.sh d4rl_loco halfcheetah_medium_expert
#   sbatch run_agent.sh neorl Hopper_v3_low
#   sbatch run_agent.sh adroit pen_cloned
#   sbatch run_agent.sh antmaze umaze
#
# Domínios disponíveis e tarefas (configs/<domínio>/task/<tarefa>.yaml):
#   d4rl_loco : halfcheetah_medium, halfcheetah_medium_expert,
#               halfcheetah_medium_replay, halfcheetah_random,
#               hopper_medium, hopper_medium_expert, hopper_medium_replay,
#               hopper_random, walker2d_medium, walker2d_medium_expert,
#               walker2d_medium_replay, walker2d_random
#   neorl     : HalfCheetah_v3_{low,medium,high}, Hopper_v3_{low,medium,high},
#               Walker2d_v3_{low,medium,high}
#   adroit    : pen_human, pen_cloned, hammer_cloned
#   antmaze   : umaze, umaze_diverse, medium_diverse, medium_play
# =============================================================================

#SBATCH --job-name=neubay_agent
#SBATCH --array=0-2               # 3 seeds em paralelo (seeds 0, 1, 2)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # ~8 CPUs: JAX + 100 envs paralelos (leves)
#SBATCH --gres=gpu:1              # 1 GPU por seed; JAX ocupa 1 device por processo
#SBATCH --mem=32G                 # buffer de 40M steps (float32) + modelo + overhead
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
# Caminhos — NADA na home
# --------------------------------------------------------------------------- #
RAID_BASE="/raid/${USER}/neubay"
REPO_DIR="${RAID_BASE}/neubay-main"
CONTAINER="${RAID_BASE}/neubay.sif"
WANDB_DIR="${RAID_BASE}/wandb"
LOG_DIR="${RAID_BASE}/logs"

mkdir -p "${LOG_DIR}" "${WANDB_DIR}"

echo "============================================"
echo "Job:      ${SLURM_JOB_ID} (array ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}])"
echo "Nó:       $(hostname)"
echo "Domínio:  ${DOMAIN}"
echo "Tarefa:   ${TASK}"
echo "Seed:     ${SEED}"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

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

        # Evita que um único processo aloque toda a VRAM (essencial com array jobs)
        export XLA_PYTHON_CLIENT_PREALLOCATE=false

        # wandb salva runs no raid
        export WANDB_DIR=${WANDB_DIR}

        export PYTHONPATH=${REPO_DIR}:\${PYTHONPATH}
        cd ${REPO_DIR}

        python offline_cont.py \
            --config-path=configs/${DOMAIN} \
            --config-name=base \
            task=${TASK} \
            seed=${SEED}
    "

echo "Job finalizado com código: $?"
