#!/bin/bash
# Um array de três seeds; cada item executa as duas condições na mesma GPU.
# DRY_RUN=true bash scripts/ovx/submit_hammer_comparison.sh
# SMOKE_TEST=true SEEDS=0 bash scripts/ovx/submit_hammer_comparison.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-/raid/${USER}/neubay}"
PARTITION="${PARTITION:-ovx01}"
SEEDS="${SEEDS:-0-2}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
SMOKE_TEST="${SMOKE_TEST:-false}"
DRY_RUN="${DRY_RUN:-false}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1200}"
PRETRAINED_WM_DIR="${PRETRAINED_WM_DIR:-offline_world/ckpt/pretrained/adroit}"
RUN_KIND=full
[[ "$SMOKE_TEST" != true ]] || RUN_KIND=smoke
RUN_BATCH_ID="${RUN_BATCH_ID:-hammer-${RUN_KIND}-$(TZ=America/Sao_Paulo date +%Y%m%dT%H%M%S)}"
for value in "$DRY_RUN" "$SMOKE_TEST"; do
    [[ "$value" == true || "$value" == false ]] || { echo '[ERROR] Use true ou false.' >&2; exit 1; }
done
[[ "$RUN_BATCH_ID" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "$SEEDS" =~ ^[0-9]+(-[0-9]+)?$ ]] || { echo '[ERROR] SEEDS deve ser uma seed ou intervalo, como 0-2.' >&2; exit 1; }
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ && "$TOTAL_EPOCHS" =~ ^[1-9][0-9]*$ ]]
case "$PRETRAINED_WM_DIR" in
    /*|*..*|*','*|*' '* ) echo '[ERROR] Caminho de WM inválido.' >&2; exit 1 ;;
esac
cd "$REPO_DIR"
test -f neubay.sif
test -f wandb.env
test -d .cache/mujoco_py_pkg
test -f scripts/ovx/run_hammer_comparison.sh
first_seed="${SEEDS%-*}"
last_seed="${SEEDS#*-}"
(( 10#$first_seed <= 10#$last_seed ))
for (( seed=10#$first_seed; seed<=10#$last_seed; seed++ )); do
    checkpoint="${PRETRAINED_WM_DIR}/hammer-cloned-v1/ensemble_seed${seed}.eqx"
    test -s "$checkpoint" || { echo "[ERROR] Checkpoint ausente: $checkpoint" >&2; exit 1; }
done
git_commit="$(git rev-parse HEAD)"
test -z "$(git status --porcelain --untracked-files=no)" || {
    echo '[ERROR] Há alterações rastreadas sem commit. Use código commitado.' >&2; exit 1;
}
# Inclui scripts novos na exigência de código commitado.
git ls-files --error-unmatch scripts/ovx/run_hammer_comparison.sh \
    scripts/ovx/submit_hammer_comparison.sh >/dev/null
TIME_LIMIT=72:00:00
[[ "$SMOKE_TEST" != true ]] || TIME_LIMIT=02:00:00
COMMAND=(sbatch --parsable "--partition=${PARTITION}" "--array=${SEEDS}%${MAX_PARALLEL}"
    "--time=${TIME_LIMIT}" "--chdir=${REPO_DIR}"
    "--export=ALL,REPO_DIR=${REPO_DIR},RUN_BATCH_ID=${RUN_BATCH_ID},SMOKE_TEST=${SMOKE_TEST},TOTAL_EPOCHS=${TOTAL_EPOCHS},PRETRAINED_WM_DIR=${PRETRAINED_WM_DIR},EXPECTED_GIT_COMMIT=${git_commit}"
    "${REPO_DIR}/scripts/ovx/run_hammer_comparison.sh")
echo "Lote: ${RUN_BATCH_ID} | seeds: ${SEEDS} | GPUs simultâneas: até ${MAX_PARALLEL}"
echo 'Por seed: WM novo -> agente scratch -> agente pretrained'
if [[ "$DRY_RUN" == true ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n[DRY RUN] Nenhum job submetido.\n'
    exit 0
fi
command -v sbatch >/dev/null
PIPELINE_DIR="logs/pipelines/${RUN_BATCH_ID}"
mkdir -p logs logs/pipelines
# mkdir sem -p reserva o identificador e impede duas submissões do mesmo lote.
mkdir "$PIPELINE_DIR" || { echo '[ERROR] Lote já existe; escolha outro RUN_BATCH_ID.' >&2; exit 1; }
printf 'RUN_BATCH_ID=%s\nGIT_COMMIT=%s\nSEEDS=%s\nSMOKE_TEST=%s\nTOTAL_EPOCHS=%s\nPRETRAINED_WM_DIR=%s\n' \
    "$RUN_BATCH_ID" "$git_commit" "$SEEDS" "$SMOKE_TEST" "$TOTAL_EPOCHS" "$PRETRAINED_WM_DIR" \
    > "${PIPELINE_DIR}/batch.env"
job_raw="$("${COMMAND[@]}")"
job_id="${job_raw%%;*}"
printf 'job_id\tseeds\tmode\n%s\t%s\t%s\n' "$job_id" "$SEEDS" "$RUN_KIND" > "${PIPELINE_DIR}/jobs.tsv"
echo "Submetido: ${job_id}. Logs: logs/hammer_compare_${job_id}_<seed>.{out,err}"
echo "Registro: ${PIPELINE_DIR}/jobs.tsv"
