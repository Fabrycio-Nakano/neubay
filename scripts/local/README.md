# Avaliação local do Go2

## Gerar um vídeo

```bash
JAX_PLATFORMS=cpu \
MUJOCO_GL=egl \
MESA_SHADER_CACHE_DIR=/tmp/neubay_mesa_cache \
.venv-go2-video/bin/python scripts/local/generate_go2_video.py \
  offline_agent/ckpt/go2/Go2JoystickFlatTerrain-direction-expert-v1/agent_seed0.eqx \
  --output videos/go2_expert_seed0.mp4 \
  --steps 500 \
  --vx 0.5
```

## Gerar a grade de variações

```bash
.venv-go2-video/bin/python scripts/local/generate_go2_variants.py
```

O gerador de variações é retomável: ele lê `videos/variants/manifest.csv`,
pula resultados válidos existentes e repete somente condições ausentes ou que
falharam.

## Gerar variações dos lotes importados

Para gerar a mesma grade para todos os datasets e seeds importados, com uma
pasta por dataset e manifesto único do lote:

```bash
.venv-go2-video/bin/python scripts/local/generate_go2_experiment_variants.py
```

Teste curto de um dataset e três condições:

```bash
.venv-go2-video/bin/python scripts/local/generate_go2_experiment_variants.py \
  --dataset Go2JoystickFlatTerrain-direction-expert-v0 \
  --policy-seeds 0 \
  --conditions stand,fwd025,yawleft025
```

## Reavaliar os 154 sobreviventes com horizonte de 1000 steps

`generate_go2_survivors_1000.py` seleciona as 149 execuções do manifesto
importado e as 5 do manifesto legado que alcançaram pelo menos 100 steps. Na
primeira execução, ele congela os 154 jobs em um novo manifesto; nas seguintes,
retoma esse snapshot sem depender de alterações posteriores nos manifests de
origem.

Confira a seleção sem criar arquivos:

```bash
.venv-go2-video/bin/python scripts/local/generate_go2_survivors_1000.py --dry-run
```

Inicie o lote desacoplado do terminal:

```bash
.venv-go2-video/bin/python scripts/local/generate_go2_survivors_1000.py --detach
```

O runner é sequencial para limitar o uso de RAM. Cada vídeo é gerado em um
arquivo temporário, validado com `ffprobe` e movido atomicamente ao destino. O
manifesto também é atualizado atomicamente antes e depois de cada tentativa.
Há lock contra runners duplicados, timeout de 30 minutos por tentativa e uma
nova tentativa automática em caso de erro.

Consulte o progresso ou acompanhe o log:

```bash
.venv-go2-video/bin/python scripts/local/generate_go2_survivors_1000.py --status
tail -f videos/go2_survivors_1000/runner.log
```

Para interromper de forma controlada:

```bash
.venv-go2-video/bin/python scripts/local/generate_go2_survivors_1000.py --stop
```

O mesmo comando com `--detach` retoma jobs pendentes, interrompidos ou que
falharam, sem refazer vídeos válidos. Os artefatos ficam em:

- `videos/go2_survivors_1000/manifest.csv`: resultados incrementais;
- `videos/go2_survivors_1000/status.json`: resumo atual;
- `videos/go2_survivors_1000/runner.log`: log geral;
- `videos/go2_survivors_1000/job_logs/`: saída completa de cada job.

`status=ok` significa que a execução e o MP4 foram validados. O campo
`reached_target` informa separadamente se o episódio chegou aos 1000 steps;
episódios que terminam antes são resultados válidos e não entram em loop de
retry.

Este ambiente roda em WSL2. O modo `--detach` sobrevive ao fechamento do
terminal, mas não executa durante suspensão/hibernação e não sobrevive a reboot
ou `wsl --shutdown`. Para deixar o lote sozinho, mantenha o notebook na tomada
e desative a suspensão automática e ao fechar a tampa. Se houver interrupção,
basta executar novamente com `--detach` para retomar.
