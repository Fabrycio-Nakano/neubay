# Scripts adicionais do projeto

Esta pasta separa as extensões operacionais e experimentais adicionadas ao
NEUBAY dos lançadores legados mantidos na raiz por compatibilidade.

## Estrutura

- `ovx/`: submissão e execução de jobs SLURM/Apptainer na OVX.
- `local/`: avaliação e geração de vídeos no notebook.

Os scripts da raiz (`submit_neubay.sh`, `submit_world_model.sh` e arquivos
`.slurm`) são preservados como interfaces legadas. Para novos experimentos Go2,
prefira `ovx/run_world_model_go2.sh` e `ovx/run_agent_go2.sh`. Eles recebem o
nome do dataset explicitamente e mantêm datasets e checkpoints separados.

## Credenciais W&B

Copie `wandb.env.example` para `wandb.env`, preencha com um token válido e use
permissão `600`. `wandb.env` nunca deve ser versionado. Os lançadores da OVX
carregam esse arquivo por meio de `ovx/load_wandb_env.sh`.

## Artefatos não versionados

Datasets, checkpoints, contêineres, ambientes virtuais, logs W&B/SLURM e vídeos
em massa permanecem fora do Git. Resultados publicáveis devem ser resumidos em
manifestos, tabelas e uma pequena seleção curada de figuras ou vídeos.
