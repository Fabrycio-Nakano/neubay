#SBATCH --job-name=neubay_world
#SBATCH --array=0               # 3 seeds em paralelo (0, 1 e 2)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6         # treinamento do world model é CPU-light
#SBATCH --gres=gpu:1              # solicita 1 GPU por seed
#SBATCH --mem=24G                 # dataset D4RL + ensemble 128 modelos
#SBATCH --time=12:00:00           # tempo máximo de execução
#SBATCH --output=/raid/%u/neubay/logs/world_%x_%A_%a.out
#SBATCH --error=/raid/%u/neubay/logs/world_%x_%A_%a.err

# --------------------------------------------------------------------------- #
# Argumentos passados na chamada do sbatch
# --------------------------------------------------------------------------- #
DOMAIN=${1:?"Informe o domínio: d4rl_loco | neorl | adroit | antmaze"}
DATASET=${2:?"Informe o dataset_name (e.g. hopper-random-v2)"}
TOTAL_EPOCHS=${3:-""}            # opcional
SEED=$SLURM_ARRAY_TASK_ID

# --------------------------------------------------------------------------- #
# Caminhos no diretório /raid
# --------------------------------------------------------------------------- #
RAID_BASE="/raid/${USER}/neubay"
REPO_DIR="${RAID_BASE}/neubay-main"
CONTAINER="${RAID_BASE}/neubay.sif"
LOG_DIR="${RAID_BASE}/logs"
mkdir -p "${LOG_DIR}"

echo "============================================"
echo "Job:      ${SLURM_JOB_ID} (array ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}])"
echo "Nó:       \$(hostname)"
echo "Domínio:  \${DOMAIN}"
echo "Dataset:  \${DATASET}"
echo "Epochs:   \${TOTAL_EPOCHS:-'(padrão do config)'}"
echo "Seed:     \${SEED}"
echo "GPU:      \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"
# --------------------------------------------------------------------------- #
# Monta argumento de epochs (opcional)
# --------------------------------------------------------------------------- #
EPOCHS_ARG=""
if [ -n "\${TOTAL_EPOCHS}" ]; then
    EPOCHS_ARG="ensemble.total_epochs=\${TOTAL_EPOCHS}"
fi
# --------------------------------------------------------------------------- #
# Execução via Apptainer
# --------------------------------------------------------------------------- #
apptainer exec \
    --nv \
    --bind "\${RAID_BASE}:\${RAID_BASE}" \
    --bind "/raid/\${USER}:/raid/\${USER}" \
    "\${CONTAINER}" \
    bash -c "
        set -e
        # Evita prealocação excessiva de memória do JAX por GPU
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export PYTHONPATH=\${REPO_DIR}:\\\${PYTHONPATH}
        cd \${REPO_DIR}
        python offline_world/cont_ensemble.py \
            --config-path=../configs/\${DOMAIN} \
            --config-name=base \
            dataset_name=\${DATASET} \
            \${EPOCHS_ARG} \
            seed=\${SEED}
    "
echo "Job finalizado com código: \$?"
