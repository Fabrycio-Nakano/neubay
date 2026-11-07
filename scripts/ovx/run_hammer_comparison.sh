#!/bin/bash
# Uma GPU por seed: WM novo -> agente com WM novo -> agente com WM pré-treinado.
# Submeta por submit_hammer_comparison.sh para validar e registrar o lote.
#SBATCH --job-name=hammer_compare
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/raid/%u/neubay/logs/hammer_compare_%A_%a.out
#SBATCH --error=/raid/%u/neubay/logs/hammer_compare_%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/raid/${USER}/neubay}"
RUN_BATCH_ID="${RUN_BATCH_ID:?Use submit_hammer_comparison.sh}"
SEED="${SLURM_ARRAY_TASK_ID:?Execute como array SLURM}"
SMOKE_TEST="${SMOKE_TEST:-false}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1200}"
PRETRAINED_WM_DIR="${PRETRAINED_WM_DIR:-offline_world/ckpt/pretrained/adroit}"
DATASET=hammer-cloned-v1
if [[ ! "$RUN_BATCH_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo '[ERROR] Lote ou seed inválidos.' >&2
    exit 1
fi
if [[ "$SMOKE_TEST" != true && "$SMOKE_TEST" != false ]]; then
    echo '[ERROR] SMOKE_TEST deve ser true ou false.' >&2
    exit 1
fi
case "$PRETRAINED_WM_DIR" in
    /*|*..*|*','*|*' '* ) echo '[ERROR] PRETRAINED_WM_DIR deve ser relativo ao repositório, sem espaços ou vírgulas.' >&2; exit 1 ;;
esac

RUN_KIND=full
CHECKPOINT_BASE=experiments
if [[ "$SMOKE_TEST" == true ]]; then
    RUN_KIND=smoke
    CHECKPOINT_BASE=smoke
fi
WORLD_ROOT="offline_world/ckpt/${CHECKPOINT_BASE}/${RUN_BATCH_ID}/adroit"
AGENT_ROOT="offline_agent/ckpt/${CHECKPOINT_BASE}/${RUN_BATCH_ID}"
RECORD_DIR="logs/pipelines/${RUN_BATCH_ID}/seed${SEED}"
CACHE_DIR="${REPO_DIR}/.cache"
CONTAINER_HOME="/home/${USER}"
cd "$REPO_DIR"
mkdir -p logs wandb "${CACHE_DIR}/home" "$RECORD_DIR"
test -f neubay.sif
test -d "${CACHE_DIR}/mujoco_py_pkg"
test -s "${PRETRAINED_WM_DIR}/${DATASET}/ensemble_seed${SEED}.eqx"
# Nunca substituir um resultado anterior ao reenviar o mesmo lote.
for output in \
    "${WORLD_ROOT}/${DATASET}/ensemble_seed${SEED}.eqx" \
    "${AGENT_ROOT}/scratch/adroit/${DATASET}/agent_seed${SEED}.eqx" \
    "${AGENT_ROOT}/pretrained/adroit/${DATASET}/agent_seed${SEED}.eqx"; do
    if [[ -e "$output" ]]; then
        echo "[ERROR] Resultado já existe: $output. Use outro RUN_BATCH_ID." >&2
        exit 1
    fi
done
if [[ -n "${EXPECTED_GIT_COMMIT:-}" ]]; then
    test "$(git rev-parse HEAD)" = "$EXPECTED_GIT_COMMIT"
    test -z "$(git status --porcelain --untracked-files=no)"
fi
source scripts/ovx/load_wandb_env.sh

export APPTAINERENV_LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export APPTAINERENV_LD_LIBRARY_PATH="/.singularity.d/libs:/usr/lib/x86_64-linux-gnu:${CONTAINER_HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export APPTAINERENV_JAX_CUDA_P2P_DISABLE=1
export APPTAINERENV_NCCL_P2P_DISABLE=1
export APPTAINERENV_MUJOCO_GL=osmesa
export APPTAINERENV_PYOPENGL_PLATFORM=osmesa
export APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
export APPTAINERENV_WANDB_DIR=/workspace/wandb
run_python() {
    apptainer exec --nv --no-home \
        --bind "${REPO_DIR}:/workspace" \
        --bind "${CACHE_DIR}/home:${CONTAINER_HOME}" \
        --bind "${CACHE_DIR}/mujoco_py_pkg:/opt/neubay/lib/python3.10/site-packages/mujoco_py" \
        --env HOME="$CONTAINER_HOME" \
        "${REPO_DIR}/neubay.sif" \
        bash -ec 'export PYTHONPATH="/workspace:${PYTHONPATH:-}"; cd /workspace; exec python -u "$@"' _ "$@"
}

git rev-parse HEAD > "${RECORD_DIR}/git_commit.txt"
printf 'Lote: %s | seed: %s | modo: %s\n' "$RUN_BATCH_ID" "$SEED" "$RUN_KIND"
# O lock evita downloads simultâneos do mesmo HDF5 pelas seeds.
# Usa o cache D4RL já existente na home persistente do contêiner.
run_python - "$RECORD_DIR" "${PRETRAINED_WM_DIR}/${DATASET}/ensemble_seed${SEED}.eqx" "$SMOKE_TEST" <<'PY'
import fcntl
import hashlib
import json
from pathlib import Path
import sys

record_dir, checkpoint, smoke = sys.argv[1:]
def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

with open(checkpoint, "rb") as stream:
    header = json.loads(stream.readline())
if smoke != "true":
    for key, expected in {"ensemble_size": 128, "hidden_size": 400, "has_ln": True}.items():
        if header.get(key) != expected:
            raise ValueError(f"WM pré-treinado: {key}={header.get(key)!r}; esperado {expected!r}")
if header.get("dataset_name", "hammer-cloned-v1") != "hammer-cloned-v1":
    raise ValueError("Checkpoint de outro dataset")

with open(".cache/hammer_dataset.lock", "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    import gym
    import d4rl
    env = gym.make("hammer-cloned-v1")
    data = env.get_dataset()
    dataset_path = env.dataset_filepath
    digest = sha256(dataset_path)
    recorded_digest = header.get("dataset_sha256")
    if recorded_digest and recorded_digest != digest:
        raise ValueError("Dataset diverge do SHA-256 registrado no WM pré-treinado")
    metadata = {
        "dataset": "hammer-cloned-v1", "path": dataset_path,
        "sha256": digest, "transitions": len(data["rewards"]),
        "observation_shape": list(env.observation_space.shape),
        "action_shape": list(env.action_space.shape),
        "pretrained_checkpoint": checkpoint,
        "pretrained_sha256": sha256(checkpoint), "pretrained_header": header,
    }
    env.close()
Path(record_dir, "inputs.json").write_text(json.dumps(metadata, indent=2) + "\n")
Path(record_dir, "dataset.sha256").write_text(digest + "\n")
print(json.dumps(metadata, indent=2))
PY
DATASET_SHA256="$(cat "${RECORD_DIR}/dataset.sha256")"
WM_ARGS=("ensemble.total_epochs=${TOTAL_EPOCHS}")
AGENT_ARGS=()
if [[ "$SMOKE_TEST" == true ]]; then
    WM_ARGS=(ensemble.total_epochs=1 ensemble.total_size=2 ensemble.hidden_size=32
        ensemble.batch_size=1024 +ensemble.max_samples=10000)
    AGENT_ARGS=(train.grad_steps=2000 train.buffer_size=200000
        collect.ensemble_size=2 collect.max_rollout_len=10 eval.times=2 eval.real_episodes=2)
fi

echo '[STAGE] Treinar WM do zero'
run_python offline_world/cont_ensemble.py --config-path=../configs/adroit --config-name=base \
    task=hammer_cloned "seed=${SEED}" "ensemble.save_dir=${WORLD_ROOT}" \
    "ensemble.plot_dir=plt/${RUN_BATCH_ID}/scratch/adroit" \
    "+dataset_sha256=${DATASET_SHA256}" +wandb_project=world_models_adroit \
    "+wandb_group=${RUN_BATCH_ID}" +wandb_job_type=world-model-training \
    "+wandb_tags=[adroit,hammer,scratch,${RUN_KIND},${RUN_BATCH_ID}]" "${WM_ARGS[@]}"
test -s "${WORLD_ROOT}/${DATASET}/ensemble_seed${SEED}.eqx"

for condition in scratch pretrained; do
    model_root="$WORLD_ROOT"
    [[ "$condition" != pretrained ]] || model_root="$PRETRAINED_WM_DIR"
    echo "[STAGE] Treinar agente: ${condition}"
    run_python offline_cont.py --config-path=configs/adroit --config-name=base \
        task=hammer_cloned "seed=${SEED}" "ensemble.save_dir=${model_root}" \
        "ensemble.plot_dir=plt/${RUN_BATCH_ID}/${condition}/adroit" \
        save_agent=true "save_agent_root=${AGENT_ROOT}/${condition}" \
        "+dataset_sha256=${DATASET_SHA256}" +wandb_project=agents_adroit \
        "+wandb_group=${RUN_BATCH_ID}" +wandb_job_type=world-model-agent-training \
        "+exp_name=${RUN_KIND}-Hammer-${condition}-S${SEED}-${RUN_BATCH_ID}" \
        "+wandb_tags=[adroit,hammer,${condition},${RUN_KIND},${RUN_BATCH_ID}]" "${AGENT_ARGS[@]}"
    agent_path="${AGENT_ROOT}/${condition}/adroit/${DATASET}/agent_seed${SEED}.eqx"
    test -s "$agent_path"
    sha256sum "${model_root}/${DATASET}/ensemble_seed${SEED}.eqx" "$agent_path" \
        > "${RECORD_DIR}/${condition}.sha256"
done
echo '[DONE] WM novo e os dois agentes concluídos.'
