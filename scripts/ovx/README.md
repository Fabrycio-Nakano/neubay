# NEUBAY na OVX — Reprodução de Resultados

Guia completo para reproduzir os experimentos do NEUBAY na OVX usando **SLURM + Apptainer**, seguindo as boas práticas do cluster.

---

## Estrutura dos scripts

```
scripts/ovx/
├── build_container.sh   # (1x) Constrói a imagem Apptainer .sif
├── run_world_model.sh   # Treina o world model ensemble
├── run_world_model_go2.sh # Treina world model Go2 com dataset explícito
├── run_agent.sh         # Treina agentes dos benchmarks originais
└── run_agent_go2.sh     # Treina o agente Go2 a partir de HDF5
```

---

## Pré-requisitos (manual, fora do SLURM)

Execute **uma vez** no login node para preparar o ambiente no raid:

```bash
RAID_BASE="/raid/${USER}/neubay"
mkdir -p "${RAID_BASE}/logs"

# 1. Clone o repositório no raid (se ainda não estiver lá)
git clone <URL_DO_REPOSITORIO> "${RAID_BASE}"

# 2. Copie os checkpoints pré-treinados para o lugar certo
# Estrutura esperada: offline_world/ckpt/<domínio>/<dataset>/<seed>/
cp -r /caminho/para/ckpts_pretreinados "${RAID_BASE}/offline_world/ckpt"

# 3. Baixe os datasets D4RL (necessário mesmo com ckpts pré-treinados)
#    Use um job de CPU curto para isso:
sbatch --job-name=get_data --nodes=1 --ntasks=1 --cpus-per-task=4 \
       --mem=16G --time=01:00:00 \
       --output="${RAID_BASE}/logs/get_data_%j.out" \
       --wrap="apptainer exec --nv --bind ${RAID_BASE}:${RAID_BASE} \
               ${RAID_BASE}/neubay.sif bash -c \
               'export PYTHONPATH=${RAID_BASE}:\$PYTHONPATH && \
                cd ${RAID_BASE} && python get_all_datasets.py'"
```

Crie também a credencial local do W&B, sem adicioná-la ao Git:

```bash
cd "${RAID_BASE}"
cp wandb.env.example wandb.env
chmod 600 wandb.env
# Edite wandb.env e informe o token novo.
```

---

## Passo 0 — Build do container (apenas uma vez)

```bash
sbatch scripts/ovx/build_container.sh
# Aguarde ~1h. O .sif será salvo em /raid/<user>/neubay/neubay.sif
squeue -u $USER  # acompanhe o progresso
```

> **Reaproveite o .sif** para todos os experimentos futuros — não é necessário rebuildar.

---

## Passo 1 — Treinar o agente (usando ckpts pré-treinados)

Como os checkpoints de world model já estão disponíveis, **pule o Passo 1b** e vá direto para o treinamento do agente.

### D4RL Locomotion

```bash
sbatch scripts/ovx/run_agent.sh d4rl_loco halfcheetah_medium_expert
sbatch scripts/ovx/run_agent.sh d4rl_loco halfcheetah_medium
sbatch scripts/ovx/run_agent.sh d4rl_loco halfcheetah_medium_replay
sbatch scripts/ovx/run_agent.sh d4rl_loco halfcheetah_random
sbatch scripts/ovx/run_agent.sh d4rl_loco hopper_medium
sbatch scripts/ovx/run_agent.sh d4rl_loco hopper_medium_expert
sbatch scripts/ovx/run_agent.sh d4rl_loco hopper_medium_replay
sbatch scripts/ovx/run_agent.sh d4rl_loco hopper_random
sbatch scripts/ovx/run_agent.sh d4rl_loco walker2d_medium
sbatch scripts/ovx/run_agent.sh d4rl_loco walker2d_medium_expert
sbatch scripts/ovx/run_agent.sh d4rl_loco walker2d_medium_replay
sbatch scripts/ovx/run_agent.sh d4rl_loco walker2d_random
```

### NeoRL Locomotion

```bash
sbatch scripts/ovx/run_agent.sh neorl HalfCheetah_v3_low
sbatch scripts/ovx/run_agent.sh neorl HalfCheetah_v3_medium
sbatch scripts/ovx/run_agent.sh neorl HalfCheetah_v3_high
sbatch scripts/ovx/run_agent.sh neorl Hopper_v3_low
sbatch scripts/ovx/run_agent.sh neorl Hopper_v3_medium
sbatch scripts/ovx/run_agent.sh neorl Hopper_v3_high
sbatch scripts/ovx/run_agent.sh neorl Walker2d_v3_low
sbatch scripts/ovx/run_agent.sh neorl Walker2d_v3_medium
sbatch scripts/ovx/run_agent.sh neorl Walker2d_v3_high
```

### Adroit

```bash
sbatch scripts/ovx/run_agent.sh adroit pen_human
sbatch scripts/ovx/run_agent.sh adroit pen_cloned
sbatch scripts/ovx/run_agent.sh adroit hammer_cloned
```

### AntMaze

```bash
sbatch scripts/ovx/run_agent.sh antmaze umaze
sbatch scripts/ovx/run_agent.sh antmaze umaze_diverse
sbatch scripts/ovx/run_agent.sh antmaze medium_diverse
sbatch scripts/ovx/run_agent.sh antmaze medium_play
```

Cada `sbatch` dispara **3 seeds simultaneamente** como array job. Os logs ficam em `/raid/<user>/neubay/logs/`.

### Go2 Joystick (world-model only)

O Go2 usa o dataset Minari/HDF5 e checkpoints em
`offline_world/ckpt/wm_trained/go2/Go2JoystickFlatTerrain-direction-expert-v1/`.
A configuração atual treina e salva o agente no world model, mas não executa
avaliação no robô/simulador real (`eval.enabled=false`).

Antes do treino completo, rode uma seed curta:

```bash
sbatch --array=0 --time=01:00:00 --export=ALL,SMOKE_TEST=true \
  scripts/ovx/run_agent_go2.sh
```

Depois que o smoke test salvar um agente sem erros, rode as seeds validadas pelos probes:

```bash
sbatch scripts/ovx/run_agent_go2.sh
```

Para o dataset medium, treine primeiro os três world models e, somente depois
que todos terminarem com sucesso, treine os agentes:

```bash
DATASET_NAME=Go2JoystickFlatTerrain-direction-medium-replay-v0
sbatch --array=0-2 scripts/ovx/run_world_model_go2.sh "${DATASET_NAME}"

# Execute após confirmar os três checkpoints novos:
sbatch --array=0-2 scripts/ovx/run_agent_go2.sh "${DATASET_NAME}"
```

O log do world model registra o caminho absoluto e o SHA-256 do HDF5 usado.
Não reutilize os checkpoints medium anteriores à correção de `dataset_path`.

Para validar a infraestrutura sem tocar nos checkpoints completos:

```bash
sbatch --array=0 --time=01:00:00 --export=ALL,SMOKE_TEST=true \
  scripts/ovx/run_world_model_go2.sh \
  Go2JoystickFlatTerrain-direction-medium-replay-v0
```

World models de smoke são salvos em `offline_world/ckpt/smoke/`; agentes de
smoke são salvos em `offline_agent/ckpt_smoke/`. Eles nunca sobrescrevem os
checkpoints completos.

Os agentes são salvos em
`offline_agent/ckpt/go2/Go2JoystickFlatTerrain-direction-expert-v1/`. As runs
são registradas no projeto W&B `neubay-official-results`, agrupadas por dataset,
com nomes `NEUBAY-Go2JoystickFlatTerrain-direction-expert-v1-S<seed>`. Smoke
tests recebem o prefixo `SMOKE-` e o ID do job SLURM.

---

## Passo 1b — Treinar world model do zero (opcional)

Só necessário se **não** estiver usando os checkpoints pré-treinados.

```bash
# D4RL Locomotion
sbatch scripts/ovx/run_world_model.sh d4rl_loco hopper-random-v2 1200
sbatch scripts/ovx/run_world_model.sh d4rl_loco halfcheetah-medium-replay-v2
sbatch scripts/ovx/run_world_model.sh d4rl_loco walker2d-medium-v2 1200
sbatch scripts/ovx/run_world_model.sh d4rl_loco halfcheetah-medium-expert-v2 600

# NeoRL
sbatch scripts/ovx/run_world_model.sh neorl Hopper-v3-low 1200

# Adroit
sbatch scripts/ovx/run_world_model.sh adroit pen-human-v1
sbatch scripts/ovx/run_world_model.sh adroit pen-cloned-v1 2400
sbatch scripts/ovx/run_world_model.sh adroit hammer-cloned-v1 1200

# AntMaze
sbatch scripts/ovx/run_world_model.sh antmaze antmaze-umaze-v2 1200
```

---

## Alocação de recursos justificada

| Script | CPUs | GPU | RAM | Justificativa |
|---|---|---|---|---|
| `build_container.sh` | 4 | — | 16G | Build conda/pip; sem GPU |
| `run_world_model.sh` | 6 | 1x | 24G | Dataset D4RL + ensemble 128 modelos (JAX) |
| `run_agent.sh` | 8 | 1x | 32G | 100 envs paralelos + buffer 40M steps + modelo |

> **1 GPU por seed** é suficiente. O README recomenda `XLA_PYTHON_CLIENT_PREALLOCATE=false` justamente para permitir múltiplos processos na mesma GPU se necessário, mas com array jobs cada processo tem sua própria GPU — sem conflito.

---

## Monitoramento

```bash
# Ver jobs em execução
squeue -u $USER

# Acompanhar um job específico
tail -f /raid/${USER}/neubay/logs/agent_neubay_agent_<JOBID>_<SEED>.out

# Cancelar todos os jobs de um array
scancel <ARRAY_JOB_ID>

# Resultados no wandb — configurar antes de rodar:
wandb login  # rode no login node uma vez
```

---

## Boas práticas aplicadas

- ✅ **Apenas SLURM** — nenhum processo em background fora do scheduler
- ✅ **Apptainer** — sem Docker, sem root real, compatível com HPC
- ✅ **Tudo no `/raid`** — repositório, ckpts, datasets, logs, wandb cache, container `.sif`
- ✅ **Alocação enxuta** — 1 GPU por seed, CPUs e RAM estimados com base nos configs do projeto
- ✅ **`XLA_PYTHON_CLIENT_PREALLOCATE=false`** — evita monopolizar VRAM nos array jobs
- ✅ **`--time` razoável** — agent training ~6h, world model ~12h; ajuste conforme observar os primeiros runs
