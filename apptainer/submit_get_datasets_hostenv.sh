#!/bin/bash
#SBATCH --job-name=neubay-data
#SBATCH --partition=ovx01
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/raid/user_fabrycioalmada/neubay/logs/%x-%j.out
#SBATCH --error=/raid/user_fabrycioalmada/neubay/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")}"
ENV_PREFIX="${ENV_PREFIX:-/raid/user_fabrycioalmada/envs/neubay}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"

mkdir -p "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

export PYTHONPATH="${PROJECT_DIR}:$PYTHONPATH"

cd "${PROJECT_DIR}"
python get_all_datasets.py
