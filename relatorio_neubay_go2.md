
# Código, adaptações e resultados
Comparação com a versão original • Estado atual, incluindo alterações locais

## Conclusão principal

O repositório mantém o núcleo algorítmico do NEUBAY e acrescenta a infraestrutura necessária para treinar políticas offline para o Unitree Go2. As mudanças centrais estão no adaptador HDF5, na configuração do novo domínio, no carregamento rastreável de checkpoints e na avaliação posterior em MuJoCo.

Os resultados disponíveis indicam forte dependência do dataset e da seed. Na grade direction comparável, medium apresenta maior duração dos episódios. Em forward-medium, duas seeds atingem 5.000 passos nas três velocidades testadas; a terceira termina antes. Isso demonstra execução prolongada em algumas condições, mas ainda não estabelece controle robusto ou superioridade sobre o NEUBAY original.

| Achado | Evidência atual |
|---|---|
| Núcleo do método preservado | Ensemble, losses, planner, buffers, LRU e SAC/REDQ
sem diferenças em relação à base. |
| Direction-medium se destaca | 231,06 passos médios; 77/207 episódios
atingem 500 passos na amostra pareada. |
| Expert termina cedo | 41,32 passos médios; 0/207 atingem 500
na mesma amostra de condições. |
| Forward-medium é instável entre seeds | Seeds 0 e 2: 5.000 passos; seed 1:
393, 33 e 43, conforme a velocidade. |
| Comparação com o artigo é contextual | O artigo mede score normalizado em outros
ambientes; aqui há duração e retorno bruto. |

## Escopo e método

Base original: 5604bad474ed226dfdf8fc4d9d7e959ea6a2afe5, ancestral comum de main e HEAD. Estado atual: HEAD 618ff86, mais alterações não commitadas. São 18 commits após a base; o diff dos arquivos rastreados contém 60 arquivos, 6.714 inserções e 73 remoções. Scripts locais não rastreados foram inspecionados separadamente.

Foram lidos código, configurações, cabeçalhos de checkpoints, manifestos CSV, logs de simulação e resultados de probes. As estatísticas foram recalculadas dos arquivos locais. Não houve novo treinamento, nova simulação, consulta ao W&B remoto ou alteração do código de treinamento.


# O que foi alterado para o Go2


| Componente | Original → adaptação e efeito |
|---|---|
| experience/wrapper.py | Gym/D4RL/NeoRL → HDF5OfflineEnv. Lê grupos episode_*,
converte T+1 observações em T pares de transição e valida
comprimentos. Fim sem flag recebe timeout. |
| offline_world/cont_ensemble.py | Recebe dataset_path e SHA-256; seleciona checkpoint exato
da seed, verifica metadados e grava atomicamente. Adiciona
metadados W&B e limite de amostras para smoke tests. |
| offline_cont.py | Avaliação real passa a ser opcional. Sem ela, deriva as
dimensões do world model e mantém treino/logging. Salva
a política atomicamente e permite configurar seu destino. |
| offline_world/static_fns.py | Adiciona terminação Go2 sempre falsa. Rollouts imaginados
são limitados pelo horizonte e pela incerteza; a regra não
representa a queda usada no simulador. |
| configs/go2/*.yaml | Novo domínio e caminhos. Mantém hiperparâmetros da
base D4RL locomotion; salva agente e desativa avaliação
real durante treino. |
| scripts/ovx/ | Manifesto de oito datasets, download com hash e revisão
fixa, arrays SLURM, Apptainer, dependência WM → agente,
lotes separados e retomada/pulo de agentes concluídos. |
| scripts/local/ | Reconstituição do ator, execução MuJoCo e vídeos.
Grade de comandos, manifestos, execução local RTX 4090
e novo runner para casos que duraram ao menos 100 passos. |
| papers/ | Probes Ridge/MLP, controles, CSV/JSON e notebooks
para estudar representações do world model. Não muda
o aprendizado da política. |

## Mudanças locais ainda não commitadas

Imports de d4rl/neorl foram removidos do caminho Go2 e passaram a ocorrer sob demanda; o gerador de vídeo adiciona a raiz do projeto ao sys.path. Há novos scripts Docker/RTX 4090, dependências e runners de variantes. Essas alterações facilitam execução e avaliação, sem evidência de mudança das losses ou arquiteturas.


# Fidelidade ao método e configuração


A comparação direta com 5604bad não encontrou diferenças em offline_world/modules.py, offline_world/losses.py, experience/collector.py, experience/world_buffer.py, experience/agent_buffer.py, memory/ e online_rl/. Preservar esses módulos não torna os domínios equivalentes: os dados, a terminação e a avaliação mudaram.

| Item | Configuração Go2 atual |
|---|---|
| Observação / ação | 48 / 12 dimensões |
| Entrada / alvo do world model | 60 = estado + ação / 49 = delta de estado + recompensa |
| Entrada recorrente da política | 61 = estado + ação anterior + recompensa anterior |
| Ensemble | 128 membros; largura 200; LayerNorm; 100 no planejamento |
| Treino do world model | Batch 256; lr 0,001; validação e early stopping preservados |
| Planejamento | epi_mean; quantil 1,0; penalidade 0; sem teto curto explícito |
| Horizonte do adaptador | 1.000 passos, fixado em HDF5OfflineEnv |
| Treino do agente | 2 milhões de updates; batch 2.048; dados reais 5% |
| Memória / críticos | LRU: 2 camadas, d_model 256, d_hidden 128; 10 críticos |
| Avaliação no treino | eval.enabled=false; real_episodes=0 |

## Uma diferença experimental relevante

Os blocos numéricos de treino Go2 seguem a configuração-base D4RL locomotion. O artigo ajusta hiperparâmetros por dataset; aplicar um único real_weight=0,05 e learning rate do encoder de 1e-5 aos oito datasets não equivale ao protocolo de seleção dos melhores hiperparâmetros do trabalho original. [R1, R3]

## Fluxo de dados

HDF5 → adaptador de transições → world model probabilístico → rollouts imaginados + dados reais → ator/críticos recorrentes → checkpoint → avaliação MuJoCo. O adaptador fornece dados e metadados; ele não implementa um simulador Go2 para avaliação online.


# Resultados originais: referência externa
Valores do artigo; não são medições Go2 nem novas reproduções locais

| Benchmark | NEUBAY | Referência citada |
|---|---|---|
| D4RL locomotion — média | 80,1 | VIPO: 83,6 |
| NeoRL locomotion — média | 64,7 | VIPO: 73,3 |
| D4RL Adroit — média | 21,1 | MoMo: 28,1 |
| D4RL AntMaze — média | 28,8 | LEQ: 64,9 |

Fonte: Ni et al., arXiv:2512.04341v1, seção 5.1. São scores normalizados. No benchmark D4RL locomotion, os autores usam 6 seeds e 20 episódios na avaliação final. As referências Adroit/AntMaze acima são baselines model-based citadas no texto. [R1]

## Exemplos do benchmark D4RL locomotion

| Dataset | NEUBAY: média ± desvio-padrão |
|---|---|
| halfcheetah-medium-expert-v2 | 109,5 ± 8,7 |
| hopper-medium-replay-v2 | 110,6 ± 0,7 |
| walker2d-medium-expert-v2 | 120,7 ± 1,3 |

## Como comparar com o Go2

Os manifestos Go2 medem principalmente quantos passos o episódio executou; os logs forward-medium também registram retorno bruto. Não existe uma escala compartilhada com os scores D4RL/NeoRL nem um resultado original para Go2 no artigo. Portanto, não se pode calcular ganho percentual da adaptação em relação ao artigo.

A comparação defensável é dupla: fidelidade da implementação ao método original e comparação entre variantes Go2 sob condições pareadas. A presença de logs locais D4RL de world models não resolve essa lacuna: MSE de validação não é retorno da política, e erros de modelos em domínios diferentes não compartilham necessariamente a mesma escala.

Não foi identificada uma avaliação A/B no mesmo Go2, com o mesmo dataset e protocolo, que isole causalmente o efeito de cada alteração. O código original, sem a interface acrescentada, não oferece suporte direto aos HDF5 Go2.


# Go2 direction: comparação pareada
Lote go2-full8-20260824-r1 • Limite observado de 500 passos

O manifesto contém 846 registros: 845 execuções com status ok e 1 failed, incluindo 6 registros forward-expert. Para direction, a interseção exata de comando, policy_seed e simulation_seed resulta em 207 condições por dataset. Os parâmetros vx, vy e yaw_rate também foram exigidos iguais. Não há chaves dataset/condição/seed duplicadas.

| Dataset | N | Média | Mediana | Atingiu 500 |
|---|---|---|---|---|
| expert | 207 | 41.32 | 40 | 0/207 (0.0%) |
| medium-expert | 207 | 39.69 | 38 | 0/207 (0.0%) |
| medium-replay | 207 | 89.38 | 57 | 8/207 (3.9%) |
| medium | 207 | 231.06 | 93 | 77/207 (37.2%) |

Medium dura 5,59 vezes mais que expert em média nesta grade, mas isso é uma razão de duração observada, não de recompensa ou qualidade de locomoção. Sua mediana de 93 passos, muito abaixo da média, revela uma distribuição heterogênea. Medium-replay melhora sobre expert, porém só 3,9% atingem o limite.

Atingir 500 significa chegar ao teto da execução; não prova completar uma tarefa ou seguir a velocidade solicitada. Status ok significa que o processo de geração terminou sem erro, mesmo quando o episódio terminou cedo. As condições compartilham políticas e não são 207 seeds de treinamento independentes. [R4]


# Sensibilidade às seeds e cobertura


| Direction | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| expert | 36.7 / 0 | 41.7 / 0 | 45.3 / 0 |
| medium-expert | 37.4 / 0 | 39.5 / 0 | 42.1 / 0 |
| medium-replay | 104.1 / 3 | 106.8 / 4 | 56.9 / 1 |
| medium | 255.0 / 29 | 281.4 / 33 | 159.0 / 16 |

Cada célula: média de passos / quantidade que atingiu 500, usando todos os registros ok de cada seed. São 70 condições por seed, exceto medium seed 1 (69). Esta tabela é descritiva, não a amostra pareada de 207 da página anterior.

## Experimentos disponíveis e o que eles permitem afirmar

| Conjunto | Registros | Interpretação |
|---|---|---|
| Legacy expert-v1 | 120 ok | Seeds 0 e 2: médias 54,07 e 60,00;
nenhuma execução chega a 500. |
| Full8 / grade principal | 845 ok; 1 failed | Direction tem cobertura ampla; forward
ainda incompleto nessa grade. |
| Refeito / 06-10 | 108 ok; 42 failed | Condições e checkpoints se sobrepõem
a outros conjuntos; não somar como réplicas. |
| vx=1 / 06-10 | 13 ok; 2 failed | Todas as execuções ok têm 27–116 passos;
cobertura desigual entre datasets. |

Nos cabeçalhos locais há 15 agentes do lote Full8 distribuídos por 6 datasets: 12 direction, 2 forward-expert e 1 forward-medium-expert. Outro lote possui 3 agentes forward-medium. A existência do pipeline Full8 não comprova que os 24 agentes estejam concluídos e disponíveis neste checkout.

Não foram combinados registros de lotes ou reexecuções diferentes para inflar a amostra. As seeds do simulador dependem do índice da condição no runner; ao filtrar a lista, o mesmo nome de comando pode receber outra seed. Por isso a comparação principal usa todos os campos do protocolo, e não somente o nome da condição. [R4, R5]


# Go2 forward-medium: testes longos
Lote go2-ordenado-20260930T145042 • Seed do simulador 0

| vx solicitado | Seed política | Passos | Retorno bruto |
|---|---|---|---|
| 1.0 | 0 | 5000 | 98.014 |
| 1.0 | 1 | 393 | 6.106 |
| 1.0 | 2 | 5000 | 99.305 |
| 1.5 | 0 | 5000 | 29.576 |
| 1.5 | 1 | 33 | 0.034 |
| 1.5 | 2 | 5000 | 31.837 |
| 2.0 | 0 | 5000 | 10.187 |
| 2.0 | 1 | 43 | 0.024 |
| 2.0 | 2 | 5000 | 6.892 |

Seis das nove execuções chegam a 5.000 passos: todas das seeds 0 e 2. A seed 1 termina aos 393 passos em vx=1,0; aos 33 em vx=1,5; e aos 43 em vx=2,0. O controle local usa dt=0,02 s; 5.000 passos correspondem a 100 s de simulação. [R6]

## Duração alta não garante boa resposta ao comando

Entre vx=1,0 e vx=2,0, o retorno cai de 98,014 para 10,187 na seed 0 (−89,6%) e de 99,305 para 6,892 na seed 2 (−93,1%), apesar da mesma duração. A recompensa inclui vários termos; esses valores, isoladamente, não medem o erro de tracking. É necessário registrar a velocidade efetiva para concluir se o robô obedece ao comando.

Esses nove testes usam uma única seed do simulador, ruído observacional desligado e política determinística. Não são diretamente comparáveis à grade direction de 500 passos, que usa outros comandos, seeds e checkpoints. Não foram contados vídeos duplicados ou variantes RTX 4090 como replicações independentes.

O limite de 5.000 foi observado nos logs, enquanto o adaptador de treino fixa horizonte 1.000. O runner de vídeo chama o ambiente diretamente e interrompe por done ou pelo limite solicitado; não há garantia de um TimeLimit idêntico ao do treino.


# World model: probes e interpretação
Resultados scaled_mlp_v2 • Dataset legacy direction-expert-v1

Os probes usam 10.000 transições, world-model seeds 0 e 2, membro 0 do ensemble e três seeds para o MLP. Os CSVs registram 6.429 exemplos de treino, 1.525 de validação e 2.046 de teste. O código utiliza divisão por grupos quando disponíveis. São medidas de informação decodificável, não retorno da política. [R7]

| Seed WM | Alvo na camada h4 | R² Ridge | R² MLP |
|---|---|---|---|
| 0 | delta_obs | 0.7137 | 0.6613 |
| 0 | next_obs | 0.7469 | 0.7707 |
| 0 | reward | 0.9245 | 0.8963 |
| 2 | delta_obs | 0.7146 | 0.6591 |
| 2 | next_obs | 0.7533 | 0.7712 |
| 2 | reward | 0.9275 | 0.8939 |

Na camada h4, Ridge decodifica a recompensa com R² próximo de 0,925–0,928 e supera o MLP também em delta_obs. O MLP tem vantagem para next_obs nessa camada. Isso indica representação preditiva útil, mas não assegura precisão em rollouts longos ou em estados escolhidos pela política fora do dataset.

## Cuidado com o significado físico dos probes

build_targets usa obs[:, :2] para alvos chamados xy. No simulador Go2 local, as três primeiras componentes são velocidades lineares locais, não posição global. Portanto, esses rótulos não autorizam afirmar aprendizado de localização ou navegação. A semântica do dataset histórico deve ser conferida com a versão do coletor; nesta análise não foram usados esses alvos para concluir desempenho espacial.

## Histórico de treinamento incorretamente rotulado

A auditoria de 23/08/2026 relata world models medium-replay idênticos aos expert-v1, com MSE 0,1424 / 0,1421 / 0,1421. Esses números não são apresentados como desempenho medium válido. Os world models históricos correspondentes não estão disponíveis no caminho auditado deste checkout; a igualdade de hashes é evidência documental anterior, não uma nova verificação executada aqui. [R8]


# Limitações concretas e próximos passos


## 1. Terminação imaginada diverge da simulação

termination_fn_go2 retorna sempre False. O simulador local termina quando get_upvector(data)[-1] < 0. A observação contém gravidade projetada, então a alegação no comentário de que falhas seriam impossíveis de inferir das 48 dimensões precisa ser reavaliada: há informação de orientação potencialmente útil. Uma regra equivalente exige validar convenções e ruído; ela não foi implementada nesta análise.

## 2. Proveniência é melhor, mas não é absoluta

O SHA-256 é passado pelos scripts e salvo nos checkpoints. O Python compara metadados, mas não recalcula o hash do HDF5 nesse caminho. Checkpoints legados sem hash ainda são aceitos com aviso. O carregador também permite fallback de seed e de caminho de dataset; executar diretamente pode contornar as validações mais estritas dos scripts SLURM.

## 3. Cobertura experimental ainda é desigual

Os cabeçalhos dos 18 agentes importados dos lotes Full8/ordenado concordam com dataset, seed e SHA-256 do manifesto, mas isso não prova integralmente o histórico de execução. Faltam resultados homogêneos para todos os oito datasets e repetição das condições em múltiplas seeds de simulação. O runner survivors_1000 seleciona casos que já duraram pelo menos 100 passos; seus resultados, se produzidos, devem ser identificados como amostra selecionada.

## 4. Medir o objetivo de controle

O próximo benchmark deve salvar retorno por episódio, passos, motivo de término e RMSE das velocidades linear/lateral/angular em relação ao comando, além de usar o mesmo horizonte e grade em todas as variantes. Reportar média e dispersão entre seeds de treinamento, sem tratar comandos de uma mesma política como treinos independentes.

## 5. Isolar causas antes de atribuir ganhos

Comparar BC ou a política coletora no mesmo simulador, validar observações/ações/recompensa e versão do ambiente, e fazer ablações de terminação, mistura real/imaginada e limiar de incerteza. Só então atribuir uma mudança de desempenho a uma modificação específica do método.

Veredito: a adaptação funcional está estabelecida e há políticas capazes de execuções longas em condições específicas. A evidência atual ainda não demonstra robustez geral do Go2 nem reproduz, em escala comparável, o benchmark do artigo original.


# Fontes e rastreabilidade


[R1] Ni et al. Long-Horizon Model-Based Offline Reinforcement Learning Without Conservatism. arXiv:2512.04341v1 (04/12/2025), seção 5.1 e apêndice G. https://arxiv.org/html/2512.04341v1 — consultado em 14/10/2026.

[R2] Git local: base 5604bad474ed226dfdf8fc4d9d7e959ea6a2afe5; HEAD 618ff86. Comparação: git diff 5604bad; git diff --; git merge-base main HEAD. A análise inclui alterações locais e scripts não rastreados.

[R3] experience/wrapper.py; offline_world/cont_ensemble.py; offline_world/static_fns.py; offline_cont.py; configs/go2/base.yaml; configs/d4rl_loco/base.yaml; scripts/ovx/go2_datasets.tsv.

[R4] videos/experiment_variants/go2-full8-20260824-r1/manifest.csv. Comparação principal: status=ok, quatro datasets direction, interseção de condition/policy_seed/simulation_seed/vx/vy/yaw_rate.

[R5] videos/variants/manifest.csv; videos/experiment_variants/go2-refeito-20261006-143849/manifest.csv; videos/experiment_variants/go2-vel1-20261006-160240/manifest.csv; cabeçalhos em offline_agent/ckpt/imported/.

[R6] videos/forward_medium/ovx_go2-ordenado-20260930T145042_seed*_vx*_sim0.log; scripts/local/generate_go2_video.py; external/mujoco_playground/mujoco_playground/_src/locomotion/go2/joystick.py.

[R7] papers/probe_outputs/*seed{0,2}__member__scaled_mlp_v2__probe_results.csv e respectivos summary.json; papers/run_linear_probes_neubay.py.

[R8] docs/audits/auditoria_go2_expert_medium.md; docs/audits/diferencas_neubay_original_go2.md; local_validation/go2/auditoria_pipeline_go2_full8.md. Documentos anteriores utilizados como contexto, com ressalvas quando não reverificáveis.

## Impressão digital da principal fonte de resultados

SHA-256 do manifesto Full8: a4425dea0906843e2bbaee4104ab60c55c4a75aba2da8e8cef964771667b80c5

Os resultados foram obtidos de arquivos presentes no checkout no momento da análise. O PDF não implica sincronização com o cluster ou com projetos remotos, nem validação física no robô real. As porcentagens de teto e médias são descritivas; episódios encerrados no limite são censurados à direita.
