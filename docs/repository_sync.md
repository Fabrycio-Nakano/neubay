# Sincronização do notebook, GitHub e OVX

O código deve ser sincronizado por commit Git. Datasets, checkpoints, logs,
contêineres, caches, credenciais e vídeos em massa permanecem locais e não fazem
parte da paridade entre ambientes.

## Fonte canônica

Durante o desenvolvimento, o notebook prepara e valida as alterações. Após a
revisão, o repositório pessoal no GitHub torna-se a fonte canônica. A OVX deve
executar exatamente o commit publicado, sem receber cópias avulsas de scripts.

## 1. Notebook

```bash
cd /home/fabrycio/neubay
git status --short --branch
git diff --check
bash -n scripts/ovx/*.sh
```

Revise o conjunto que será versionado antes de criar o commit:

```bash
git diff -- . ':!docs/audits/*.pdf'
git status --short
```

## 2. GitHub pessoal

Somente após revisão e aprovação:

```bash
git push -u origin robowm2026
```

O push é uma etapa explícita; os procedimentos de organização e teste não o
executam automaticamente.

## 3. OVX

Antes de sincronizar, preserve o estado de trabalho existente:

```bash
cd /raid/user_fabrycioalmada/neubay
git status --short --branch
git rev-parse HEAD
git diff > /raid/user_fabrycioalmada/neubay_before_sync.patch
```

Se não houver alterações de código locais que precisem ser incorporadas:

```bash
git fetch origin
git switch robowm2026
git pull --ff-only origin robowm2026
```

Confirme a paridade pelo commit:

```bash
git rev-parse HEAD
git status --short --branch
```

O hash impresso na OVX deve ser igual ao hash do notebook e do GitHub.

## 4. Artefatos locais da OVX

Os seguintes caminhos não devem ser removidos durante uma atualização de
código:

```text
datasets/
offline_world/ckpt/
offline_agent/ckpt/
logs/
wandb/
.cache/
neubay.sif
wandb.env
```

Eles são ignorados pelo Git e continuam disponíveis depois de `fetch`,
`switch` e `pull --ff-only`.

## Regra contra divergência

Não edite uma cópia isolada de um script diretamente na OVX. Faça a alteração
no notebook, valide, publique no GitHub e atualize a OVX pelo commit. Quando uma
correção emergencial for inevitável na OVX, registre-a em uma branch e traga o
commit de volta ao notebook antes de continuar.
