#!/bin/bash
set -euo pipefail

# Submit one probe job per HDF5 dataset that has a matching checkpoint directory.
#
# Usage on OVX:
#   cd /raid/user_fabrycioalmada/neubay
#   DRY_RUN=true bash papers/run_all_hdf5_probes_ovx.sh
#   SAMPLE_SIZE=1000 SKIP_MLP=true bash papers/run_all_hdf5_probes_ovx.sh
#   SAMPLE_SIZE=10000 SKIP_MLP=false bash papers/run_all_hdf5_probes_ovx.sh

PROJECT_DIR="${PROJECT_DIR:-/raid/user_fabrycioalmada/neubay}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-papers/submit_linear_probes_ovx.slurm}"

SAMPLE_SIZE="${SAMPLE_SIZE:-10000}"
SKIP_MLP="${SKIP_MLP:-false}"
AGGREGATE="${AGGREGATE:-member}"
MEMBER_INDEX="${MEMBER_INDEX:-0}"
MODEL_SEED="${MODEL_SEED:-0}"
# Use ":" here because commas delimit variables in sbatch --export.
PROBE_SEEDS="${PROBE_SEEDS:-0:1:2}"
MLP_MAX_EPOCHS="${MLP_MAX_EPOCHS:-500}"
MLP_PATIENCE="${MLP_PATIENCE:-30}"
RUN_TAG="${RUN_TAG:-scaled_mlp_v2}"
DRY_RUN="${DRY_RUN:-true}"
ALLOW_RANDOM_MODEL_IF_MISSING="${ALLOW_RANDOM_MODEL_IF_MISSING:-false}"

cd "${PROJECT_DIR}"

if [ ! -f "${SUBMIT_SCRIPT}" ]; then
  echo "[ERROR] Submit script not found: ${SUBMIT_SCRIPT}" >&2
  exit 1
fi

mapfile -t HDF5_FILES < <(find datasets -mindepth 3 -maxdepth 3 -type f -path "*/data/main_data.hdf5" | sort)

if [ "${#HDF5_FILES[@]}" -eq 0 ]; then
  echo "[ERROR] No HDF5 datasets found under datasets/*/data/main_data.hdf5" >&2
  exit 1
fi

echo "[INFO] Project: ${PROJECT_DIR}"
echo "[INFO] Found HDF5 datasets: ${#HDF5_FILES[@]}"
echo "[INFO] SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "[INFO] SKIP_MLP=${SKIP_MLP}"
echo "[INFO] AGGREGATE=${AGGREGATE}"
echo "[INFO] MEMBER_INDEX=${MEMBER_INDEX}"
echo "[INFO] MODEL_SEED=${MODEL_SEED}"
echo "[INFO] PROBE_SEEDS=${PROBE_SEEDS}"
echo "[INFO] MLP_MAX_EPOCHS=${MLP_MAX_EPOCHS}"
echo "[INFO] MLP_PATIENCE=${MLP_PATIENCE}"
echo "[INFO] RUN_TAG=${RUN_TAG}"
echo "[INFO] DRY_RUN=${DRY_RUN}"
echo "[INFO] ALLOW_RANDOM_MODEL_IF_MISSING=${ALLOW_RANDOM_MODEL_IF_MISSING}"
echo

submitted=0
skipped=0

for hdf5_path in "${HDF5_FILES[@]}"; do
  dataset_dir="$(dirname "$(dirname "${hdf5_path}")")"
  dataset_name="$(basename "${dataset_dir}")"

  checkpoint_dir=""
  while IFS= read -r candidate; do
    if compgen -G "${candidate}/ensemble_seed*.eqx" >/dev/null || compgen -G "${candidate}/latest_seed*.eqx" >/dev/null; then
      checkpoint_dir="${candidate}"
      break
    fi
  done < <(find offline_world/ckpt -type d -name "${dataset_name}" | sort)

  domain="hdf5"
  if [ -n "${checkpoint_dir}" ]; then
    # Prefer a readable domain name from wm_trained/<domain>/<dataset>, when present.
    parent="$(basename "$(dirname "${checkpoint_dir}")")"
    if [ "${parent}" != "ckpt" ] && [ "${parent}" != "wm_trained" ]; then
      domain="${parent}"
    fi
  fi

  if [ -z "${checkpoint_dir}" ] && [ "${ALLOW_RANDOM_MODEL_IF_MISSING}" != "true" ]; then
    echo "[SKIP] ${dataset_name}"
    echo "       dataset:    ${hdf5_path}"
    echo "       checkpoint: not found"
    skipped=$((skipped + 1))
    continue
  fi

  random_flag="false"
  if [ -z "${checkpoint_dir}" ]; then
    checkpoint_dir="none"
    random_flag="true"
  fi

  echo "[JOB] ${dataset_name}"
  echo "      dataset:    ${hdf5_path}"
  echo "      checkpoint: ${checkpoint_dir}"
  echo "      domain:     ${domain}"

  export_args=(
    "ALL"
    "SAMPLE_SIZE=${SAMPLE_SIZE}"
    "SKIP_MLP=${SKIP_MLP}"
    "RANDOM_MODEL_IF_MISSING=${random_flag}"
    "DOMAIN=${domain}"
    "DATASET_NAME=${dataset_name}"
    "HDF5_DATASET=${hdf5_path}"
    "AGGREGATE=${AGGREGATE}"
    "MEMBER_INDEX=${MEMBER_INDEX}"
    "MODEL_SEED=${MODEL_SEED}"
    "PROBE_SEEDS=${PROBE_SEEDS}"
    "MLP_MAX_EPOCHS=${MLP_MAX_EPOCHS}"
    "MLP_PATIENCE=${MLP_PATIENCE}"
    "RUN_TAG=${RUN_TAG}"
  )

  if [ "${checkpoint_dir}" != "none" ]; then
    export_args+=("CHECKPOINT_DIR=${checkpoint_dir}")
  fi

  export_arg="$(IFS=,; echo "${export_args[*]}")"
  cmd=(sbatch "--export=${export_arg}" "${SUBMIT_SCRIPT}")

  if [ "${DRY_RUN}" = "true" ]; then
    printf '      command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
    submitted=$((submitted + 1))
  fi
done

echo
echo "[DONE] submitted=${submitted} skipped=${skipped} dry_run=${DRY_RUN}"
