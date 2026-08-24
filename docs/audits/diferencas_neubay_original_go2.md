# Diferenças entre o NEUBAY original e a adaptação Go2

**Base original comparada:** `5604bad474ed226dfdf8fc4d9d7e959ea6a2afe5`  
**Estado adaptado analisado:** `45fb140ab2a5d190864eece6cea33cbe3b5ff7f9`  
**Branch:** `robowm2026`  
**Data da análise:** 24 de agosto de 2026

## 1. Resumo executivo

O commit `5604bad474ed226dfdf8fc4d9d7e959ea6a2afe5` é a referência correta do NEUBAY original neste repositório:

- ele é o commit apontado por `main`, `origin/main` e `origin/HEAD` no momento da análise;
- é o ancestral comum exato da branch `robowm2026`;
- a branch adaptada está 15 commits à frente dele;
- seu pai é `7aa8c6b` (`release`);
- entre `7aa8c6b` e `5604bad` foram alterados somente o `README.md` e a imagem do algoritmo, não o código de treinamento.

Assim, `5604bad` é a melhor base documental do repositório original, enquanto `7aa8c6b` representa praticamente o mesmo código algorítmico de release.

A adaptação atual adiciona suporte a datasets HDF5 do Go2, configuração Hydra específica, execução reproduzível na OVX, rastreabilidade dos datasets/checkpoints, organização no W&B, geração local de vídeos e análise por probes. Ela **não reimplementa o algoritmo NEUBAY**.

Os componentes algorítmicos fundamentais que permaneceram inalterados incluem:

- modelo probabilístico do ensemble;
- funções de loss do world model;
- coletor e planner de rollouts imaginados;
- replay buffers real e imaginado;
- memória LRU;
- módulos e losses SAC/REDQ do ator-crítico recorrente.

As mudanças que afetam o comportamento do treinamento concentram-se em cinco pontos:

1. leitura e conversão dos HDF5 Go2;
2. configuração das dimensões e hiperparâmetros Go2;
3. carregamento seguro do world model correspondente ao dataset e à seed;
4. possibilidade de treinar sem avaliação online no simulador;
5. orquestração dos oito datasets e três seeds na OVX.

## 2. Método da comparação

A comparação foi feita com:

```bash
git diff 5604bad474ed226dfdf8fc4d9d7e959ea6a2afe5..45fb140
```

O diff cumulativo contém:

- 57 arquivos alterados ou adicionados;
- aproximadamente 6.143 linhas adicionadas e 69 removidas;
- 15 commits após a base original.

Esse número não deve ser interpretado como seis mil linhas de alteração no NEUBAY. Grande parte corresponde a notebooks, scripts de probes, documentação, resultados CSV/JSON e infraestrutura SLURM.

### Commits da adaptação

| Commit | Papel principal |
|---|---|
| `f045fb7` | Primeiros scripts e configuração de execução na OVX |
| `9c900dd` | Runtime SLURM, W&B e proteção de credenciais |
| `eec4798` | Consolidação inicial de submissão SLURM |
| `42be48b` | Padronização Apptainer e W&B |
| `78d867e` | Probes reproduzíveis e resultados |
| `7469d95` | Notebook de probes compatível com Colab |
| `ef35bfe` | Ajuste documental dos probes |
| `0a61f2f` | Suporte do agente Go2 aos HDF5 e world models |
| `a110667` | Padronização dos metadados das runs Go2 |
| `3753d25` | Organização do workflow e proveniência do dataset |
| `854944a` | Credenciais W&B por ambiente local |
| `c91e543` | Isolamento dos checkpoints de smoke test |
| `b540487` | Correção do projeto W&B do world model |
| `54022b0` | Pipeline dos oito datasets Go2 |
| `45fb140` | Proveniência, segurança de checkpoint e projetos W&B separados |

## 3. O que não mudou no método NEUBAY

Esta é a evidência mais importante para avaliar fidelidade ao projeto original.

| Componente | Arquivo | Estado em relação a `5604bad` |
|---|---|---|
| Arquitetura do ensemble probabilístico | `offline_world/modules.py` | Inalterado |
| Losses e incerteza do world model | `offline_world/losses.py` | Inalterado |
| Planner/coletor de rollouts | `experience/collector.py` | Inalterado |
| Buffer offline do world model | `experience/world_buffer.py` | Inalterado |
| Buffer do agente | `experience/agent_buffer.py` | Inalterado |
| Memória LRU | `memory/lru.py`, `memory/module.py` | Inalterado |
| Avaliador original | `experience/evaluator.py` | Inalterado |
| SAC/REDQ e losses Q | `online_rl/*.py` | Inalterado |

Portanto, não foram modificados:

- a forma da distribuição probabilística aprendida por cada membro;
- o bootstrap do ensemble;
- a seleção dos melhores membros por validação;
- o cálculo de incerteza usado no planejamento;
- a construção dos rollouts imaginados;
- o treinamento recorrente LRU;
- a atualização do ator, críticos, alvos e temperatura de entropia.

O método continua sendo o NEUBAY. O código novo fornece uma ponte de dados e uma camada operacional para aplicá-lo ao Go2.

## 4. Alterações funcionais no código principal

### 4.1 `experience/wrapper.py`: adaptação HDF5 Go2

Este é o arquivo novo/modificado mais importante para a semântica dos dados.

O NEUBAY original obtinha datasets por ambientes Gym/D4RL/NeoRL. A adaptação adicionou `HDF5OfflineEnv`, capaz de ler os arquivos Minari-style publicados em `akcit-rl/playground`.

O adaptador:

- localiza explicitamente `main_data.hdf5`;
- lê dimensões dos metadados ou do primeiro episódio;
- expõe `observation_space` e `action_space` compatíveis com Gym;
- filtra somente grupos `episode_*`;
- converte observações T+1 em pares `observations`/`next_observations` de tamanho T;
- converte `terminations` e `truncations` para os campos esperados pelo NEUBAY;
- verifica comprimentos de observações, ações, recompensas e flags;
- marca como timeout o fim de um grupo que não possua fronteira final explícita;
- concatena os episódios no formato consumido por `world_buffer.py`.

O roteamento de `make_env` passou a reconhecer `domain=go2` e usar esse adaptador. Os domínios originais continuam seguindo os caminhos anteriores.

#### Impacto

Sem essa mudança, o NEUBAY não conseguiria abrir os datasets Go2. Uma conversão incorreta aqui contaminaria tanto o world model quanto o agente. Por isso, este é o ponto de maior sensibilidade científica da adaptação.

#### Decisão relevante

O espaço de ações é declarado como `[-1, 1]`, que é a suposição feita pelo treinamento NEUBAY. Os arquivos foram verificados separadamente para confirmar que as ações reais respeitam esse intervalo.

### 4.2 `offline_world/cont_ensemble.py`: entrada Go2 e proveniência

A lógica de aprendizado do ensemble foi preservada. As mudanças estão ao redor dela:

- aceita `dataset_path` explícito;
- aceita SHA-256 do dataset;
- recebe entidade, projeto, grupo, job type e tags do W&B;
- usa nomes determinísticos `WorldModel-<dataset>-S<seed>`;
- permite limitar amostras em smoke tests;
- salva metadados de domínio, dataset, caminho, hash e seed no cabeçalho do checkpoint;
- grava o checkpoint atomicamente por arquivo temporário + `os.replace`;
- procura primeiro o arquivo exato `ensemble_seed<seed>.eqx`;
- rejeita incompatibilidade de nome ou SHA-256 entre checkpoint e dataset;
- mantém fallback com aviso para checkpoints legados.

#### Correção de associação por seed

O original selecionava o checkpoint usando posição na lista e módulo do número de arquivos. Isso era suficiente para coleções pré-organizadas, mas podia associar uma seed a outro arquivo quando checkpoints estavam ausentes. A adaptação prefere o nome exato da seed.

#### O que não mudou

Não mudaram a construção do ensemble, a loss, o otimizador, o bootstrap, a validação, o early stopping ou a seleção dos melhores membros.

### 4.3 `offline_cont.py`: agente sem simulador online obrigatório

O original sempre criava ambientes reais para avaliação. Para o Go2, o container de treino possui o dataset e o world model, mas não necessariamente o simulador/runtime completo de avaliação.

As mudanças:

- tornam a avaliação real condicional a `eval.enabled`;
- quando desativada, obtêm dimensões e horizonte a partir do world model;
- continuam executando os ciclos internos de logging sem chamar `ContEvaluator`;
- aceitam configuração completa do W&B;
- passam SHA-256 ao carregador do world model;
- salvam a política atomicamente;
- permitem configurar o diretório raiz do agente.

#### Consequência

`eval.enabled=false` não impede a geração da política. Ele apenas significa que retorno real e episódios no simulador não são calculados durante o treinamento. Avaliação e vídeo permanecem fases posteriores.

#### O que não mudou

As redes recorrentes, amostragem real/imaginada, gradientes, SAC/REDQ e atualizações de alvo continuam sendo o código original.

### 4.4 `offline_world/static_fns.py`: terminação Go2

Foi adicionada `termination_fn_go2`.

Ela retorna `False` para todas as transições porque a adaptação assumiu que falhas físicas não são inferíveis com segurança apenas pela observação armazenada de 48 dimensões. Dessa forma, rollouts imaginados terminam por:

- horizonte máximo;
- limite de incerteza.

Essa é uma extensão necessária para impedir que `get_termination_fn` rejeite o novo domínio, mas é também uma limitação: quedas não são detectadas por uma regra física específica.

## 5. Configuração Go2 adicionada

### `configs/go2/base.yaml`

Esse arquivo conecta o algoritmo original ao novo domínio. Os valores centrais são:

| Grupo | Configuração |
|---|---|
| World model | ensemble total 128, hidden 200, LayerNorm, batch 256 |
| Planejamento | 100 membros, incerteza `epi_mean`, quantil 1.0 |
| Conservadorismo | `penalty_coef=0.0` |
| Rollout | horizonte longo, `max_rollout_len=-1` |
| Agente | 2.000.000 passos de gradiente, batch 2.048 |
| Mistura | `real_weight=0.05` |
| Recorrência | LRU com duas camadas, modelo 256, hidden 128 |
| Críticos | 10 críticos, 2 amostrados |
| Avaliação | desativada durante o treino |

Esses valores preservam o regime NEUBAY não conservador. `penalty_coef=0.0`, ensemble de planejamento com 100 membros e quantil 1.0 são opções documentadas no repositório original.

Os arquivos de task `go2_joystick.yaml` e `placeholder.yaml` completam a composição Hydra, mas concentram pouca lógica.

## 6. Pipeline atual dos oito datasets

### 6.1 Manifesto: `scripts/ovx/go2_datasets.tsv`

Define a fonte única de verdade para:

- família (`direction` ou `forward`);
- variante;
- caminho no Hugging Face;
- nome local;
- SHA-256;
- tamanho em bytes.

Ele contém quatro variantes de cada família: `expert-v0`, `medium-expert-v0`, `medium-replay-v0` e `medium-v0`.

### 6.2 Download: `scripts/ovx/download_go2_datasets.sh`

O downloader:

- usa uma revisão fixa do Hugging Face, em vez de `main` mutável;
- pula HDF5 já válidos;
- retoma arquivos `.part`;
- verifica tamanho e SHA-256 antes de promover o parcial;
- baixa metadados de cada dataset;
- permite filtrar família ou nome.

### 6.3 World model: `scripts/ovx/run_world_model_go2.sh`

Responsabilidades:

- selecionar dataset e seed do array SLURM;
- confirmar container, dataset e SHA-256;
- carregar credenciais W&B locais;
- executar o container Apptainer com GPU;
- passar caminho, hash, seed e diretório versionado ao Hydra;
- separar smoke test de treinamento completo;
- registrar runs no projeto W&B `world_models_go2`.

### 6.4 Agente: `scripts/ovx/run_agent_go2.sh`

Responsabilidades:

- selecionar a mesma seed do world model;
- verificar o checkpoint esperado antes de treinar;
- passar explicitamente `ensemble.save_dir` ao Hydra;
- verificar o hash do dataset;
- impedir sobrescrita acidental;
- isolar artefatos smoke/full;
- registrar runs no projeto W&B `agents_go2`.

A passagem explícita de `ensemble.save_dir=${WORLD_MODEL_SAVE_DIR}` é crítica. Sem ela, o shell poderia verificar um checkpoint novo enquanto o Python carregava um diretório antigo.

### 6.5 Orquestração: `scripts/ovx/submit_go2_full_pipeline.sh`

O script atual:

- verifica os oito datasets antes da submissão;
- usa seeds `0-2`;
- cria oito arrays de world model;
- cria oito arrays de agente com `afterok` do world model correspondente;
- cria diretórios exclusivos por `RUN_BATCH_ID`;
- registra commit, hash do manifesto e diretórios do lote;
- recusa código rastreado não commitado;
- impede reutilização acidental do mesmo ID;
- produz `jobs.tsv` e `submission.log`.

O Full-8 representa 24 treinos de world model e 24 treinos de agente.

## 7. Infraestrutura e segurança adicionadas

### Apptainer e SLURM

Foram adicionados:

- definição do container (`neubay.def`);
- scripts de build e submissão;
- bindings para repositório, cache e dependências do MuJoCo;
- variáveis CUDA/JAX adequadas à OVX;
- tempos, memória, CPUs e GPUs por estágio.

Essas mudanças não alteram o método, mas determinam se ele executa de maneira reproduzível no cluster.

### Credenciais

`scripts/ovx/load_wandb_env.sh`, `wandb.env.example` e `.gitignore` separam credenciais do código. O arquivo real `wandb.env` não é versionado.

### Proveniência

A combinação de commit Git, hash do manifesto, SHA-256 do HDF5, seed, nome do dataset e `RUN_BATCH_ID` reduz o risco de repetir o erro anterior de rotular um checkpoint com o dataset errado.

## 8. Extensões de análise e visualização

Essas extensões fazem parte do estudo, mas não do pipeline de treino Full-8.

### Probes

Os arquivos em `papers/` adicionam:

- extração de representações internas do world model;
- probes lineares e MLPs escalados;
- controles diagnósticos;
- submissão na OVX;
- upload dos resultados ao W&B;
- notebooks e resultados tabulares.

O arquivo `papers/run_linear_probes_neubay.py`, com aproximadamente 947 linhas adicionadas, explica grande parte do volume do diff.

### Vídeos

`scripts/local/generate_go2_video.py` e `generate_go2_variants.py` executam políticas no notebook e geram variações de vídeo. Eles não participam do treinamento nem mudam checkpoints.

### Documentação

Foram adicionados protocolos, auditorias e instruções de sincronização em `docs/` e nos READMEs de scripts.

## 9. Arquivos mais relevantes para a task atual

### Prioridade 1 — semântica científica

1. **`experience/wrapper.py`**  
   Define como o HDF5 vira transições NEUBAY. É o arquivo mais importante para garantir que o método recebe os dados corretos.

2. **`configs/go2/base.yaml`**  
   Define arquitetura, treinamento, rollouts, mistura real/imaginada e ausência de avaliação real.

3. **`offline_world/cont_ensemble.py`**  
   Liga o dataset Go2 ao world model original, seleciona o checkpoint e valida proveniência.

4. **`offline_cont.py`**  
   Carrega o world model e treina/salva o agente recorrente.

5. **`offline_world/static_fns.py`**  
   Define como os rollouts Go2 tratam terminação.

### Prioridade 2 — correção operacional do Full-8

6. **`scripts/ovx/go2_datasets.tsv`** — identidade dos oito datasets.  
7. **`scripts/ovx/download_go2_datasets.sh`** — aquisição e integridade.  
8. **`scripts/ovx/run_world_model_go2.sh`** — execução do world model.  
9. **`scripts/ovx/run_agent_go2.sh`** — correspondência world model/agente.  
10. **`scripts/ovx/submit_go2_full_pipeline.sh`** — dependências e 48 treinos.

### Prioridade 3 — interpretação posterior

11. **`papers/run_linear_probes_neubay.py`** — análise das representações.  
12. **`scripts/local/generate_go2_video.py`** — avaliação visual local.  
13. **`docs/study/experiment_protocol.md`** — protocolo experimental.

## 10. Alterações periféricas que merecem revisão separada

Algumas mudanças cumulativas não são necessárias para o Full-8:

- `get_all_datasets.py` teve diversos datasets D4RL/NeoRL descomentados;
- `requirements.yml` removeu as entradas pip explícitas de `d4rl` e `mjrl`;
- existem scripts antigos na raiz e um arquivo `submit_neubay.sh.save`;
- resultados e notebooks de probes aumentam bastante o repositório;
- scripts genéricos e scripts Go2 coexistem em `scripts/ovx`.

Esses itens não afetam o pipeline atual porque ele usa o HDF5 local, a imagem `neubay.sif` já construída e os scripts `*_go2.sh`. Mesmo assim, são candidatos a uma limpeza futura para tornar a apresentação do repositório mais clara.

## 11. Riscos e limitações remanescentes

| Risco/limitação | Efeito | Situação atual |
|---|---|---|
| Sem avaliação real durante treino | W&B não contém retorno real do simulador | Avaliar depois com protocolo comum |
| Terminação Go2 sempre falsa | Quedas não encerram rollouts por regra física | Incerteza/horizonte ainda truncam |
| Hiperparâmetros iguais nos oito datasets | Comparação controlada, não ótimo individual | Adequado para benchmark inicial |
| Carregamento integral do HDF5 em RAM | Pressão de memória em datasets maiores | Recursos SLURM dimensionados e smoke aprovado |
| Dependência no array inteiro | Uma seed de WM falha e bloqueia três agentes | Conservador e rastreável |
| Checkpoint legado sem SHA | Proveniência incompleta em resultados antigos | Novos checkpoints incluem SHA |

## 12. Veredito

O trabalho atual deve ser descrito como **uma adaptação e instrumentação do NEUBAY para datasets offline Go2**, não como uma nova implementação do algoritmo.

A essência algorítmica foi preservada. As mudanças centrais estão nas interfaces com dados, na possibilidade de treinar sem simulador online, na terminação específica do novo domínio, na proveniência e na execução distribuída.

Para analisar ou revisar a task Full-8, os dez arquivos das prioridades 1 e 2 são suficientes. Os demais arquivos explicam infraestrutura auxiliar, visualização, estudos de representação e histórico operacional.

### Referências Git reproduzíveis

```text
NEUBAY original: 5604bad474ed226dfdf8fc4d9d7e959ea6a2afe5
Adaptação Go2:   45fb140ab2a5d190864eece6cea33cbe3b5ff7f9
Merge-base:      5604bad474ed226dfdf8fc4d9d7e959ea6a2afe5
Commits à frente: 15
```

O grafo opcional do `understand-anything` não estava presente neste checkout. A análise foi realizada diretamente sobre o histórico Git, os diffs cumulativos, os imports e o fluxo efetivo dos scripts.
