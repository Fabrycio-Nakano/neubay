#!/bin/bash
# Baixa e verifica os oito datasets Go2 declarados em go2_datasets.tsv.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/raid/${USER}/neubay}"
MANIFEST="${REPO_DIR}/scripts/ovx/go2_datasets.tsv"
HF_REVISION="${HF_REVISION:-1ecf8436e64d30dbc2293b3b89211619acf8b39f}"
HF_BASE="https://huggingface.co/datasets/akcit-rl/playground/resolve/${HF_REVISION}"

command -v curl >/dev/null || { echo "[ERROR] curl não encontrado" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "[ERROR] sha256sum não encontrado" >&2; exit 1; }
test -f "${MANIFEST}" || { echo "[ERROR] Manifesto ausente: ${MANIFEST}" >&2; exit 1; }

matches_filter() {
    local family="$1"
    local local_name="$2"
    shift 2
    if [ "$#" -eq 0 ]; then
        return 0
    fi
    local filter
    for filter in "$@"; do
        if [ "${filter}" = "${family}" ] || [ "${filter}" = "${local_name}" ]; then
            return 0
        fi
    done
    return 1
}

download_small_file() {
    local url="$1"
    local destination="$2"
    local temporary="${destination}.part"
    curl -fL --retry 5 --retry-delay 5 -o "${temporary}" "${url}"
    mv "${temporary}" "${destination}"
}

while IFS=$'\t' read -r family variant hf_path local_name expected_sha expected_size; do
    [ -n "${family}" ] || continue
    [[ "${family}" == \#* ]] && continue
    matches_filter "${family}" "${local_name}" "$@" || continue

    dataset_dir="${REPO_DIR}/datasets/${local_name}"
    data_dir="${dataset_dir}/data"
    dataset_file="${data_dir}/main_data.hdf5"
    partial_file="${dataset_file}.part"
    mkdir -p "${data_dir}"

    echo "============================================================"
    echo "Dataset: ${local_name}"
    echo "Origem:  ${hf_path}"
    echo "SHA256:  ${expected_sha}"

    valid_existing=false
    if [ -f "${dataset_file}" ]; then
        actual_size="$(stat -c '%s' "${dataset_file}")"
        actual_sha="$(sha256sum "${dataset_file}" | cut -d' ' -f1)"
        if [ "${actual_size}" = "${expected_size}" ] && [ "${actual_sha}" = "${expected_sha}" ]; then
            valid_existing=true
            echo "Arquivo existente verificado: OK"
        else
            echo "[ERROR] Arquivo existente não corresponde ao manifesto: ${dataset_file}" >&2
            echo "Obtido: size=${actual_size} sha256=${actual_sha}" >&2
            exit 1
        fi
    fi

    if [ "${valid_existing}" != "true" ]; then
        curl -fL --retry 8 --retry-delay 10 -C - \
            -o "${partial_file}" \
            "${HF_BASE}/${hf_path}/data/main_data.hdf5?download=true"
        actual_size="$(stat -c '%s' "${partial_file}")"
        actual_sha="$(sha256sum "${partial_file}" | cut -d' ' -f1)"
        test "${actual_size}" = "${expected_size}" || { echo "[ERROR] Tamanho inválido" >&2; exit 1; }
        test "${actual_sha}" = "${expected_sha}" || { echo "[ERROR] SHA-256 inválido" >&2; exit 1; }
        mv "${partial_file}" "${dataset_file}"
        echo "Download verificado: OK"
    fi

    download_small_file "${HF_BASE}/${hf_path}/data/metadata.json" "${data_dir}/metadata.json"
    download_small_file "${HF_BASE}/${hf_path}/metadata.json" "${dataset_dir}/metadata.json"
done < "${MANIFEST}"

echo "Todos os datasets solicitados foram verificados."
