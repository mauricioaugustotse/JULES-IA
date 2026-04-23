# Pipeline Backfill TSE

Documento operacional curto do fluxo canônico do backfill de sessões do TSE para o Notion.

## Visão Geral

O pipeline hoje tem 5 camadas:

1. primeira passada Gemini
2. fechamento operacional do lote
3. blindagem de schema + saneamento estrutural pós-lote
4. super auditor OpenAI
5. enriquecimento editorial de notícias

A regra prática é: primeiro acertar a identidade e a estrutura do julgamento; notícia fica por último.

## Fluxo Automático Até Pré-Notícias

Script:

- `pipeline_pre_news.py`

Objetivo:

- encadear automaticamente o pipeline do Gemini até o `super_auditor`
- parar antes da etapa de notícias
- evitar rodadas manuais separadas de `rerun`, `audit`, `repair` e `quality-core`

Ordem executada:

1. primeira passada Gemini
2. rerun dos vídeos em `error`
3. auditoria do ano
4. reparo retroativo determinístico
5. `schema-core` para regravar campos perdidos por schema do Notion, com canonização de `relator` e `pedido_vista`
6. `identity-core` para sanear `youtube_link`, `data_sessao`, `numero_processo` e `tipo_registro`
7. preenchimento determinístico de campos estruturados em branco a partir de `02_judgment.items`
8. residual automático de `composicao` para corrigir listas fora do padrão `6` ou `7`
9. `super_auditor` em `quality-core`

Exemplo do zero:

```bash
python3 pipeline_pre_news.py \
  --playlist-url "URL_DA_PLAYLIST" \
  --year 2022 \
  --auto-scale \
  --initial-workers 3 \
  --max-workers 3
```

Exemplo partindo do lote Gemini já concluído:

```bash
python3 pipeline_pre_news.py \
  --playlist-url "URL_DA_PLAYLIST" \
  --year 2022 \
  --skip-initial-backfill
```

Defaults importantes:

- `repair-focus=all`
- `repair-focus auxiliar=schema-core`
- `repair-focus auxiliar=identity-core`
- `repair-focus auxiliar=deterministic-core`
- `repair-focus auxiliar=composition`
- `super-focus=quality-core`
- `super-model=gpt-5.4-mini`
- `super-min-confidence=medium`

Launcher gráfico para VSCode Play:

- `pipeline_pre_news_launcher.py`

Esse launcher abre uma janela com dois modos:

1. rodar do zero
2. continuar após a primeira passada Gemini

## 1. Primeira Passada Gemini

Script principal:

- `tse_backfill_2025_notion.py`

Apoio:

- `tse_youtube_notion_core.py`
- `tse_normalization.py`

Objetivo:

- ler a playlist do ano
- filtrar vídeos relevantes
- extrair julgamentos com Gemini
- normalizar os campos
- publicar no Notion

Campos tratados já na primeira passada:

- `data_sessao`
- `tipo_registro`
- `numero_processo`
- `origem`
- `tribunal`
- `classe_processo`
- `tema`
- `punchline`
- `resultado`
- `votacao`
- `relator`
- `pedido_vista`
- `partes`
- `advogados`
- `composicao`

Perfil operacional recomendado:

- sem notícias
- sem grounding isolado de `origem`
- `3` workers no arranque

Exemplo:

```bash
python3 tse_backfill_2025_notion.py \
  --playlist-url "URL_DA_PLAYLIST" \
  --year 2022 \
  --auto-scale \
  --initial-workers 3 \
  --max-workers 3
```

No Windows:

```powershell
.venv\Scripts\python.exe tse_backfill_2025_notion.py `
  --playlist-url "URL_DA_PLAYLIST" `
  --year 2022 `
  --auto-scale `
  --initial-workers 3 `
  --max-workers 3
```

## 2. Monitoramento do Lote

Script:

- `tse_backfill_monitor.py`

Comando canônico:

```powershell
.venv\Scripts\python.exe tse_backfill_monitor.py --watch --interval 5 --manifest artifacts\tse_youtube_notion\backfill_2025\ANO_PLAYLIST_ID\manifest.json
```

Sinais saudáveis:

- `manifest.json` atualizando
- workers com heartbeat recente
- artefatos novos sendo gravados
- `done` subindo

Sinais de atenção:

- timeouts de `1200s`
- `error` crescente
- `manifest` sem avanço por vários minutos
- erro de publish no Notion

## 3. Fechamento Operacional do Lote

Quando a primeira passada termina, o fluxo canônico é:

1. rerodar vídeos em `error`
2. auditar o ano
3. reparar deterministicamente os resíduos

### 3.1. Rerun dos vídeos com erro

```bash
python3 tse_backfill_2025_notion.py \
  --playlist-url "URL_DA_PLAYLIST" \
  --year 2022 \
  --rerun-error-videos
```

Uso:

- timeouts
- erro de publish
- falhas pontuais de worker

### 3.2. Auditoria do ano

```bash
python3 tse_backfill_2025_notion.py \
  --playlist-url "URL_DA_PLAYLIST" \
  --year 2022 \
  --audit-existing-year
```

O audit mede principalmente:

- `tipo_registro` em branco, duplicado ou fora de ordem
- `origem` vazia, degradada ou inválida
- `classe_processo` vazia ou inconsistente
- `votacao` e `resultado`
- `youtube_link`
- `numero_processo`
- `partes`
- `advogados`
- `composicao`
- `tema`
- `punchline`

### 3.3. Reparo retroativo determinístico

```bash
python3 tse_backfill_2025_notion.py \
  --playlist-url "URL_DA_PLAYLIST" \
  --year 2022 \
  --repair-existing-year \
  --repair-focus all
```

Focos disponíveis:

- `all`
- `association`
- `origem`
- `classe`
- `votacao`
- `links`
- `tipo`
- `punchline`
- `numero`
- `core-fields`
- `composition`

Pode ser combinado com:

- `--review-only`
- `--video-id`
- `--only-composicao-incompleta`
- `--no-theme-api`

## 4. Limpeza de Falso-Positivo e Duplicata

Essa etapa continua sendo pós-lote.

Objetivo:

- remover precedente citado tratado como julgamento
- remover duplicata segura
- remover associação sem prova local

Regra de segurança:

- só excluir automaticamente quando a evidência local for clara
- não reassociar vídeo/data/timestamp sem prova afirmativa

## 5. Super Auditor OpenAI

Script:

- `super_auditor.py`

Objetivo:

- revisar páginas já publicadas sem reabrir vídeo
- usar só artefatos locais + texto já salvo no Notion
- aplicar apenas correções compatíveis com o formato canônico

Modelo padrão:

- `gpt-5.4`

Também usamos:

- `gpt-5.1` para passadas focais

Campos que o super auditor consegue melhorar bem:

- `tema`
- `punchline`
- `origem`
- `classe_processo`
- `votacao`
- `resultado`
- `pedido_vista`
- `tribunal`
- `partes`
- `advogados`
- `precedentes_citados`
- `resolucoes_citadas`

Campo tratado deterministicamente:

- `materia_semelhante`

Exemplo de rodada ampla:

```bash
python3 super_auditor.py --years 2025 2024 2023 --apply --min-confidence high
```

Exemplos focais:

```bash
python3 super_auditor.py --years 2025 2024 2023 --apply --focus quality-core --model gpt-5.1 --min-confidence medium
python3 super_auditor.py --years 2025 2024 2023 --apply --focus origem --model gpt-5.1 --min-confidence high
```

Política:

- sem fonte externa
- sem reabrir vídeo
- só artefatos locais e página já publicada
- aplicar automaticamente apenas quando a evidência for suficiente

## 6. Notícias

Notícia é a última camada.

Só ligar depois de:

1. primeira passada concluída
2. reruns concluídos
3. auditoria e reparos estruturais feitos
4. super auditor finalizado

Comando no backfill principal:

```bash
python3 tse_backfill_2025_notion.py \
  --playlist-url "URL_DA_PLAYLIST" \
  --year 2022 \
  --with-news
```

Regra prática:

- não gastar notícia em registro estruturalmente errado

## Ordem Canônica de Execução

1. primeira passada Gemini sem notícias
2. monitoramento do lote
3. rerun de `error`
4. auditoria do ano
5. reparo retroativo determinístico
6. `schema-core`
7. `identity-core`
8. `deterministic-core`
9. `composition`
10. limpeza de falso-positivo e duplicata
11. super auditor OpenAI
12. notícias

## Arquivos Mais Importantes

- `tse_backfill_2025_notion.py`: orquestração do backfill e reparos
- `tse_backfill_monitor.py`: monitor de terminal
- `tse_youtube_notion_core.py`: extração, preview e publicação
- `tse_normalization.py`: normalização dos campos
- `super_auditor.py`: auditoria semântica com OpenAI

## Defaults Atuais

- `origem` não tem mais grounding isolado no pipeline principal
- `quality-core` é o foco padrão do `super_auditor` para pós-lote; focos antigos de `theme-punchline` e `punchline` foram aposentados
- `TRE/UF` é subsidiário; `Cidade/UF` prevalece quando o texto local sustenta
- `numero_processo` curto é preferível a CNJ inventado
- `tipo_registro` deve sair sequencial por vídeo
- `partes` e `advogados` devem nascer no schema com cor `default`
