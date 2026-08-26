#!/bin/bash
# Um item por combinacao dataset/seed: 8 datasets x 3 seeds = 24 itens.
# O sufixo %4 e o unico throttle do lote, portanto limita o total a 4 GPUs.

#SBATCH --job-name=go2_full_global
#SBATCH --array=0-23%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=60:00:00
#SBATCH --output=/raid/%u/neubay/logs/go2_global_%A_%a.out
#SBATCH --error=/raid/%u/neubay/logs/go2_global_%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/raid/${USER}/neubay}"
MANIFEST="${REPO_DIR}/scripts/ovx/go2_datasets.tsv"
GLOBAL_TASK_ID="${SLURM_ARRAY_TASK_ID:?Este script deve ser executado como array Slurm}"
SEEDS_PER_DATASET=3

dataset_index=$((GLOBAL_TASK_ID / SEEDS_PER_DATASET))
seed=$((GLOBAL_TASK_ID % SEEDS_PER_DATASET))
manifest_line=$((dataset_index + 2)) # pula o cabecalho

IFS=$'\t' read -r family variant hf_path dataset_name expected_sha expected_size \
    < <(sed -n "${manifest_line}p" "${MANIFEST}")

test -n "${dataset_name:-}" || {
    echo "[ERROR] Nenhum dataset para o item ${GLOBAL_TASK_ID}" >&2
    exit 1
}

export RUN_BATCH_ID="${RUN_BATCH_ID:-go2-global-${SLURM_ARRAY_JOB_ID}}"
export WORLD_MODEL_SAVE_DIR="${WORLD_MODEL_SAVE_DIR:-offline_world/ckpt/experiments/${RUN_BATCH_ID}/go2}"
export AGENT_SAVE_ROOT="${AGENT_SAVE_ROOT:-offline_agent/ckpt/experiments/${RUN_BATCH_ID}}"
export DATASET_FAMILY="${family}"
export EXPECTED_DATASET_SHA256="${expected_sha}"

# Os scripts existentes usam SLURM_ARRAY_TASK_ID como seed. Neste array global,
# convertemos o indice 0-23 na seed local 0-2 antes de chama-los.
export SLURM_ARRAY_TASK_ID="${seed}"

echo "Global task: ${GLOBAL_TASK_ID}/23"
echo "Dataset:     ${dataset_name}"
echo "Seed:        ${seed}"
echo "Batch:       ${RUN_BATCH_ID}"

# A mesma alocacao/GPU executa as duas etapas em sequencia. O agente so inicia
# se o world model correspondente terminar com sucesso.
bash "${REPO_DIR}/scripts/ovx/run_world_model_go2.sh" "${dataset_name}"
bash "${REPO_DIR}/scripts/ovx/run_agent_go2.sh" "${dataset_name}"
