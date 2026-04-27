#!/bin/bash
#SBATCH --job-name=neubay_build
#SBATCH --output=/raid/user_fabrycioalmada/neubay/logs/build_%j.out
#SBATCH --error=/raid/user_fabrycioalmada/neubay/logs/build_%j.err
#SBATCH --partition=ovx # Verifique o nome correto da partição da sua OVX
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

cd /raid/user_fabrycioalmada/neubay/

# Garante que o arquivo de requirements está acessível para o container durante o build
cp requirements.yml /tmp/requirements.yml

apptainer build --fakeroot --bind /tmp/requirements.yml:/requirements.yml \
    neubay.sif neubay.def
