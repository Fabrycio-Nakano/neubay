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
