#!/bin/bash
# Submete os oito datasets Go2: world models 0-2 e agentes dependentes 0-2.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/raid/${USER}/neubay}"
MANIFEST="${REPO_DIR}/scripts/ovx/go2_datasets.tsv"
PARTITION="${PARTITION:-ovx01}"
SEEDS="${SEEDS:-0-2}"
VERIFY_SHA256="${VERIFY_SHA256:-true}"
DRY_RUN="${DRY_RUN:-false}"
RUN_BATCH_ID="${RUN_BATCH_ID:-go2-full8-$(date +%Y%m%dT%H%M%S)}"
WORLD_MODEL_SAVE_DIR="offline_world/ckpt/experiments/${RUN_BATCH_ID}/go2"
AGENT_SAVE_ROOT="offline_agent/ckpt/experiments/${RUN_BATCH_ID}"
PIPELINE_DIR="${REPO_DIR}/logs/pipelines/${RUN_BATCH_ID}"
if [ "${DRY_RUN}" = "true" ]; then
    PLAN_FILE="${PIPELINE_DIR}/jobs.dryrun.tsv"
else
    PLAN_FILE="${PIPELINE_DIR}/jobs.tsv"
fi
SUBMISSION_LOG="${PIPELINE_DIR}/submission.log"

if [ "${DRY_RUN}" != "true" ]; then
    command -v sbatch >/dev/null || { echo "[ERROR] sbatch não encontrado" >&2; exit 1; }
fi
test -f "${MANIFEST}" || { echo "[ERROR] Manifesto ausente: ${MANIFEST}" >&2; exit 1; }
mkdir -p "${PIPELINE_DIR}"
if [ "${DRY_RUN}" != "true" ] && [ -e "${PLAN_FILE}" ]; then
    echo "[ERROR] Lote já submetido ou reservado: ${PLAN_FILE}" >&2
    echo "Escolha outro RUN_BATCH_ID para evitar submissão duplicada." >&2
    exit 1
fi

echo "Validando datasets antes de submeter qualquer job..."
while IFS=$'\t' read -r family variant hf_path local_name expected_sha expected_size; do
    [ -n "${family}" ] || continue
    [[ "${family}" == \#* ]] && continue
    dataset_file="${REPO_DIR}/datasets/${local_name}/data/main_data.hdf5"
    test -f "${dataset_file}" || { echo "[ERROR] Dataset ausente: ${dataset_file}" >&2; exit 1; }
    actual_size="$(stat -c '%s' "${dataset_file}")"
    test "${actual_size}" = "${expected_size}" || { echo "[ERROR] Tamanho inválido: ${local_name}" >&2; exit 1; }
    if [ "${VERIFY_SHA256}" = "true" ]; then
        actual_sha="$(sha256sum "${dataset_file}" | cut -d' ' -f1)"
        test "${actual_sha}" = "${expected_sha}" || { echo "[ERROR] SHA-256 inválido: ${local_name}" >&2; exit 1; }
    fi
    echo "[OK] ${local_name}"
done < "${MANIFEST}"

git_commit="$(git -C "${REPO_DIR}" rev-parse HEAD)" || {
    echo "[ERROR] REPO_DIR não é um repositório Git válido" >&2
    exit 1
}
if [ -n "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=no)" ] && [ "${ALLOW_DIRTY_CODE:-false}" != "true" ]; then
    echo "[ERROR] Há alterações rastreadas não commitadas em ${REPO_DIR}" >&2
    echo "Use código commitado ou ALLOW_DIRTY_CODE=true conscientemente." >&2
    exit 1
fi
manifest_sha256="$(sha256sum "${MANIFEST}" | cut -d' ' -f1)"
cat > "${PIPELINE_DIR}/batch.env" <<EOF
RUN_BATCH_ID=${RUN_BATCH_ID}
GIT_COMMIT=${git_commit}
MANIFEST_SHA256=${manifest_sha256}
WORLD_MODEL_SAVE_DIR=${WORLD_MODEL_SAVE_DIR}
AGENT_SAVE_ROOT=${AGENT_SAVE_ROOT}
SEEDS=${SEEDS}
PARTITION=${PARTITION}
EOF

printf 'family\tvariant\tdataset\tworld_job\tagent_job\n' > "${PLAN_FILE}"
printf 'timestamp\tkind\tdataset\tjob_id\tdependency\n' > "${SUBMISSION_LOG}"

echo "Submetendo lote ${RUN_BATCH_ID}..."
dataset_index=0
while IFS=$'\t' read -r family variant hf_path local_name expected_sha expected_size; do
    [ -n "${family}" ] || continue
    [[ "${family}" == \#* ]] && continue

    export_values="ALL,RUN_BATCH_ID=${RUN_BATCH_ID},DATASET_FAMILY=${family},EXPECTED_DATASET_SHA256=${expected_sha},WORLD_MODEL_SAVE_DIR=${WORLD_MODEL_SAVE_DIR},AGENT_SAVE_ROOT=${AGENT_SAVE_ROOT}"

    dataset_index=$((dataset_index + 1))
    if [ "${DRY_RUN}" = "true" ]; then
        world_job="DRY-WM-${dataset_index}"
        agent_job="DRY-AG-${dataset_index}"
    else
        world_job_raw="$(sbatch --parsable \
            --partition="${PARTITION}" \
            --array="${SEEDS}" \
            --job-name="wm_${family}_${variant}" \
            --export="${export_values}" \
            "${REPO_DIR}/scripts/ovx/run_world_model_go2.sh" \
            "${local_name}")"
        world_job="${world_job_raw%%;*}"
        printf '%s\tworld_model\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "${local_name}" "${world_job}" "NONE" \
            >> "${SUBMISSION_LOG}"
        echo "${local_name}: world=${world_job} submetido"

        agent_job_raw="$(sbatch --parsable \
            --partition="${PARTITION}" \
            --array="${SEEDS}" \
            --dependency="afterok:${world_job}" \
            --kill-on-invalid-dep=yes \
            --job-name="ag_${family}_${variant}" \
            --export="${export_values}" \
            "${REPO_DIR}/scripts/ovx/run_agent_go2.sh" \
            "${local_name}")"
        agent_job="${agent_job_raw%%;*}"
        printf '%s\tagent\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "${local_name}" "${agent_job}" "${world_job}" \
            >> "${SUBMISSION_LOG}"
    fi

    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${family}" "${variant}" "${local_name}" "${world_job}" "${agent_job}" \
        >> "${PLAN_FILE}"
    echo "${local_name}: world=${world_job}, agent=${agent_job}"
done < "${MANIFEST}"

echo "Pipeline submetido. Plano: ${PLAN_FILE}"
echo "World models: ${WORLD_MODEL_SAVE_DIR}"
echo "Agentes:      ${AGENT_SAVE_ROOT}"
