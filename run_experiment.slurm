#!/bin/bash
#SBATCH --job-name=neubay_exp
#SBATCH --partition=ovx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/raid/user_fabrycioalmada/neubay/logs/exp_%j.out

# Defina o diretório base
BASE_DIR="/raid/user_fabrycioalmada/neubay"
SIF_IMAGE="${BASE_DIR}/neubay.sif"

# Executa o experimento usando o container
# Substitua os argumentos conforme o dataset que deseja testar (Ex: Hopper-v3-low)
apptainer exec --nv \
    --bind ${BASE_DIR}:${BASE_DIR} \
    ${SIF_IMAGE} \
    python ${BASE_DIR}/offline_cont.py \
    --config-path=configs/neorl \
    --config-name=base \
    dataset_name=Hopper-v3-low \
    seed=0
