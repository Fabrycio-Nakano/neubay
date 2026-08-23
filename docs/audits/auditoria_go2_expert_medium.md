# Auditoria dos treinamentos Go2: expert e medium-replay

**Projeto:** NEUBAY / Go2 Joystick Flat Terrain

**Data do relatório:** 23 de agosto de 2026
**Escopo:** datasets, world models e agentes das variantes `direction-expert-v1` e `direction-medium-replay-v0`

## 1. Resumo executivo

Foram encontrados três checkpoints de world model rotulados como `medium-replay-v0` que são byte a byte idênticos aos checkpoints `expert-v1` correspondentes às seeds 0, 1 e 2. Os arquivos são independentes no sistema de arquivos e foram criados em datas diferentes, portanto não são links físicos ou simbólicos.

Os datasets atuais expert e medium são diferentes tanto no arquivo HDF5 quanto em seu conteúdo lógico. Apesar disso, os logs dos jobs rotulados como medium informam exatamente **1.000.007 transições usadas**, quantidade que coincide com o dataset expert. O dataset medium atual produziria **1.031.203 transições usadas**.

A conclusão sustentada pelas evidências é que os jobs de world model rotulados como medium consumiram os dados expert, ou uma representação com o mesmo conteúdo lógico do expert. Os checkpoints medium não constituem treinamentos independentes no dataset medium. Como consequência, os agentes medium foram treinados usando world models incorretamente rotulados e seus resultados não devem ser publicados como resultados medium definitivos.

Não foi encontrada evidência de que o treinamento expert tenha sido causado ou corrompido por esse incidente. Entretanto, os artefatos expert devem ser mantidos sob auditoria até que o caminho efetivo do dataset utilizado nos jobs históricos também seja documentado.

## 2. Artefatos examinados

### 2.1 Agentes

- Expert: `agent_seed0.eqx` e `agent_seed2.eqx`.
- Medium: `agent_seed0.eqx`, `agent_seed1.eqx` e `agent_seed2.eqx`.

Os agentes medium possuem hashes diferentes e suas configurações internas registram corretamente:

- `dataset_name=Go2JoystickFlatTerrain-direction-medium-replay-v0`;
- caminho HDF5 medium;
- seeds 0, 1 e 2;
- ausência de carregamento de agente anterior (`load_agent_path=null`).

### 2.2 World models

Foram examinados os checkpoints finais `ensemble_seed*.eqx` das variantes expert e medium. Também foram observados checkpoints `latest_seed*.eqx` apenas no conjunto expert.

### 2.3 Datasets

- Expert: `Go2JoystickFlatTerrain-direction-expert-v1/data/main_data.hdf5`.
- Medium: `Go2JoystickFlatTerrain-direction-medium-replay-v0/data/main_data.hdf5`.

## 3. Evidências coletadas

### 3.1 Checkpoints expert e medium possuem hashes idênticos

- Seed 0:

  `aa40b8e0e05cda0ebd3dc86f901947c74e88d8aab0a676822c7c80498ee8d31e`
- Seed 1:

  `a4a047f327b069409b2c2c1264dc36707e6ff18dd70d3e8ef524208b63157e6a`
- Seed 2:

  `178646db09c17ca12b490caf4e49f371a11ea52f8ce33e8eb3e8a6ca90bb3624`

Igualdade de SHA-256 significa igualdade byte a byte, incluindo parâmetros serializados e cabeçalho do checkpoint.

### 3.2 Os arquivos não são hard links

O comando `stat` mostrou inodes diferentes e contagem de links igual a 1. Os checkpoints expert são datados de 7 de julho de 2026; os medium, de 19 de agosto de 2026. Portanto, existem como arquivos físicos separados.

### 3.3 Os HDF5 são diferentes

- Expert, 477 MB:

  `c9337a3c6cd0a8ccb908bc0f7768533c673cb598c3b7daf186ddf45ac828dc66`
- Medium, 673 MB:

  `12a15665780eeece5827f01f3b42733b201601d0dec605bbb2e073d0122801eb`

A inspeção estrutural também encontrou diferenças claras. O primeiro episódio expert possui 1.000 ações e 1.001 observações, enquanto o primeiro episódio medium possui 34 ações e 35 observações.

### 3.4 O conteúdo lógico utilizado pelo world model é diferente

| Medida | Expert | Medium |
|---|---:|---:|
| Episódios | 1.003 | 4.322 |
| Transições brutas | 1.001.007 | 1.032.026 |
| Transições utilizáveis | 1.000.007 | 1.031.203 |
| Hash lógico completo | `03c27a04...029e5` | `953521dc...a4a63` |
| Hash das transições de treino | `bedf2b5a...164fd` | `95d3aab4...5e9c` |

Os hashes lógicos completos e de treinamento são diferentes. Assim, a igualdade dos checkpoints não pode ser explicada apenas por compressão, metadados ou organização interna do HDF5.

### 3.5 Os logs medium registram a contagem do expert

Todos os jobs completos rotulados como medium imprimiram:

```text
inputs.shape=(1000007, 60)
targets.shape=(1000007, 49)
```

O número `1.000.007` coincide exatamente com as transições utilizáveis do expert. Se o HDF5 medium atual tivesse sido consumido, o esperado seria `1.031.203` transições antes da separação treino/validação.

Essa é a evidência mais direta de que o conteúdo expert foi lido durante os jobs rotulados como medium.

### 3.6 O treinamento realmente executou

Os jobs medium não apenas copiaram imediatamente uma inicialização:

| Seed | Época de encerramento | MSE de validação final |
|---:|---:|---:|
| 0 | 61 | 0,1424 |
| 1 | 91 | 0,1421 |
| 2 | 85 | 0,1421 |

Cada época processou 3.903 batches. Não foram encontrados `NaN`, traceback, término por falta de memória ou interrupção do treinamento.

Uma reconstrução da inicialização aleatória para cada seed também produziu hashes diferentes dos checkpoints finais. Portanto, os `ensemble_seed*.eqx` contêm parâmetros treinados, não apenas pesos iniciais.

### 3.7 Agentes medium carregaram os checkpoints medium

Os logs dos agentes seeds 0, 1 e 2 registram o caminho medium e mostram que os agentes foram salvos normalmente. Porém, como os arquivos de world model medium contêm os mesmos pesos expert, a origem dinâmica desses rollouts é incorreta para um experimento medium independente.

## 4. O que ocorreu

A reconstrução mais consistente é:

1. O job foi submetido e rotulado como `direction-medium-replay-v0`.
2. O diretório de saída e os nomes no W&B foram configurados como medium.
3. Durante a leitura dos dados, o processo consumiu 1.000.007 transições, correspondentes ao expert.
4. O world model foi efetivamente otimizado por dezenas de épocas sobre esses dados.
5. Os pesos resultantes foram salvos no diretório medium.
6. Os agentes medium carregaram esses checkpoints e foram treinados sob a identificação medium.

Isso explica simultaneamente a contagem nos logs, a igualdade exata dos checkpoints e a existência de agentes medium distintos.

## 5. Causas possíveis

O mecanismo exato que resolveu o caminho errado ainda não foi demonstrado historicamente. As causas mais plausíveis são:

1. **Divergência entre `dataset_name` e `dataset_path`.** O nome foi sobrescrito para medium, mas algum caminho efetivo permaneceu com o valor expert definido em `configs/go2/base.yaml`.
2. **Versão histórica diferente do carregador.** O código executado em 19 de agosto pode não ser idêntico ao código atualmente inspecionado, especialmente na resolução do HDF5.
3. **Caminho implícito ou fallback.** A resolução automática pode ter localizado o expert quando o medium não estava visível no mesmo namespace do contêiner ou quando o caminho explícito não foi repassado ao método de treinamento.
4. **Estado divergente entre host, bind e contêiner.** O script exibia o argumento medium, mas o processo Python pode ter resolvido outro arquivo dentro do ambiente montado.
5. **Mudança posterior de arquivos ou código.** O dataset expert atual tem data posterior ao world model expert, indicando que parte do estado histórico já foi substituída ou baixada novamente.

Não foi encontrada uma operação explícita de `cp`, `rsync` ou link de checkpoints no script Slurm ou em `cont_ensemble.py`. Por isso, a hipótese de leitura do dataset errado é mais consistente que uma cópia posterior manual, embora a cópia não possa ser excluída apenas com os registros disponíveis.

## 6. Impacto científico

### Expert

- Os world models expert não são invalidados diretamente por este incidente.
- Os agentes expert foram treinados apenas no world model, com avaliação real desativada (`eval.enabled=false`, `real_episodes=0`).
- A queda rápida observada no MuJoCo indica baixa transferência para o simulador, mas não prova falha no processo de otimização offline.

### Medium

- Os world models medium não podem ser tratados como modelos treinados independentemente no dataset medium.
- Os agentes medium e suas métricas não devem ser usados como resultados medium definitivos.
- Os runs W&B devem ser marcados como inválidos ou arquivados com uma nota de auditoria, sem apagá-los.
- World models e agentes medium precisam ser retreinados após a correção e validação do pipeline.

## 7. Correções recomendadas

1. Passar `dataset_path` explicitamente ao treinamento do world model.
2. Derivar esse caminho do mesmo `DATASET` usado no nome do job.
3. Registrar no início de cada run:
   - caminho absoluto resolvido;
   - SHA-256 do HDF5;
   - número de episódios;
   - transições brutas e utilizadas.
4. Adicionar uma asserção que verifique a correspondência entre `dataset_name` e caminho.
5. Salvar essas informações no cabeçalho do checkpoint e na configuração W&B.
6. Comparar o hash do novo checkpoint com checkpoints de outros datasets e interromper o pipeline se houver igualdade inesperada.
7. Executar primeiro um smoke test e confirmar que o log medium contém 1.031.203 transições utilizáveis.
8. Retreinar as seeds 0, 1 e 2 do world model medium.
9. Somente depois retreinar os agentes medium nas mesmas seeds.
10. Preservar os artefatos atuais em uma área marcada como `invalid_audit`, sem publicá-los como resultados válidos.

## 8. Segurança e publicação

Durante a auditoria foi encontrada uma credencial W&B escrita diretamente no script Slurm. O valor não é reproduzido neste relatório.

Antes de qualquer envio ao GitHub:

1. revogar e rotacionar a credencial;
2. removê-la do script;
3. verificar se ela aparece no histórico Git;
4. manter segredos apenas em `.netrc`, arquivo ignorado ou variável protegida;
5. executar uma varredura de segredos antes do push.

Checkpoints `.eqx`, datasets, contêineres e ambientes virtuais não devem ser enviados diretamente ao repositório Git. Recomenda-se publicar apenas código, configurações, manifestos, hashes, documentação e uma seleção pequena de vídeos; artefatos grandes devem ser armazenados externamente.

## 9. Conclusão

O erro não pode ser reduzido a uma ação individual do usuário. O comando e os logs aparentavam corretamente um treinamento medium, mas o pipeline não registrou nem validou o arquivo efetivamente lido. Essa ausência de validação permitiu que um run rotulado como medium consumisse dados compatíveis com o expert e produzisse artefatos incorretamente identificados.

O experimento expert permanece separadamente auditável. Já o experimento medium deve ser corrigido e repetido antes da publicação. A prioridade imediata é proteger a credencial exposta, instrumentar a resolução do dataset e executar um smoke test medium com contagens e hash verificados.
