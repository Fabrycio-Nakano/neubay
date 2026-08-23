#!/bin/bash
# Carrega credenciais locais do W&B sem armazená-las no repositório.

WANDB_ENV_FILE="${WANDB_ENV_FILE:-${REPO_DIR:?REPO_DIR não definido}/wandb.env}"

if [ ! -f "${WANDB_ENV_FILE}" ]; then
    echo "[ERROR] Credenciais W&B não encontradas: ${WANDB_ENV_FILE}" >&2
    echo "Copie wandb.env.example para wandb.env e adicione um token válido." >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "${WANDB_ENV_FILE}"

# Também aceita os nomes usuais e os converte para o formato do Apptainer.
if [ -z "${APPTAINERENV_WANDB_API_KEY:-}" ] && [ -n "${WANDB_API_KEY:-}" ]; then
    export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
fi
if [ -z "${APPTAINERENV_WANDB_ENTITY:-}" ] && [ -n "${WANDB_ENTITY:-}" ]; then
    export APPTAINERENV_WANDB_ENTITY="${WANDB_ENTITY}"
fi

if [ -z "${APPTAINERENV_WANDB_API_KEY:-}" ]; then
    echo "[ERROR] WANDB_API_KEY não foi definida em ${WANDB_ENV_FILE}." >&2
    return 1 2>/dev/null || exit 1
fi

export APPTAINERENV_WANDB_API_KEY
export APPTAINERENV_WANDB_ENTITY
