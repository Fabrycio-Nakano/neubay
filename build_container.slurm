#!/bin/bash
#SBATCH --job-name=neubay_build
#SBATCH --output=/raid/user_fabrycioalmada/neubay/logs/build_%j.out
#SBATCH --error=/raid/user_fabrycioalmada/neubay/logs/build_%j.err
#SBATCH --partition=ovx01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

cd /raid/user_fabrycioalmada/neubay/

# Remove a imagem antiga se existir para evitar conflitos de build parcial
rm -f neubay.sif

apptainer build --fakeroot neubay.sif neubay.def
