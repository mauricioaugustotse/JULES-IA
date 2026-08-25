# Kit de manutenção da base "sessões" (campanha 21-24/08/2026)

Scripts promovidos do scratchpad da campanha de auditoria. Todos usam o venv
`..\.venv-win` e a API do Notion via `tse_youtube_notion_core`. Convenção:
rodar SEM flag = dry-run (relatório); `--apply` = grava. Logs restauráveis em
`..\artifacts\notion_sessoes_auditoria\`.

## O que roda SOZINHO (não precisa destes scripts)
- `..\watch_jurisprudencia_csv.py` (watcher do DJe) chama `..\fill_inteiro_teor.py`
  a cada CSV novo — teores novos já saem com o cortador v3 (ementas estruturadas,
  itens numerados, sem órfãos) e fonte por (CNJ, data) com janela [-5,+60]d.
- `..\tse_normalization.py` — advogados/partes sem travessões (en-dash incluído)
  para todo registro novo do fluxo de vídeo.
- **Vocabulário de ministros**: o watcher confere ao fim de cada passada se `dje` e
  `sessões` ainda gravam o mesmo nome para o mesmo ministro, e AVISA no log quando não
  (advisory: nunca reprova a rodada). Ver **Vocabulário único de ministros** abaixo.

## Vocabulário único de ministros (24/08/2026)
Fonte da verdade: `C:\Users\mauri\ProjetoConversor\_ministros_canonico.py` — forma
**completa e acentuada** ("Min. André Ramos Tavares", não "Min. Ramos Tavares").
`tse_normalization.normalize_ministro_name` termina nela, então todo o lado sessões já
sai canônico; `tests\test_tse_normalization.py` trava a regressão no alias map e no roster.

O que quebrou antes, para não se repetir: o alias map devolvia a forma curta e
**rebaixava** o nome que chegava canônico do DJe, então cada rodada do watcher recriava a
opção divergente no select. Os pipelines que montam o vocabulário a partir do **schema**
ou das **páginas** (`..\fill_composicao_from_jurisprudencia.py`, `composicao_por_acordao.py`)
canonizam o que leem — sem isso, uma grafia sobrevivente vira "a mais frequente" e se
auto-perpetua.

- Conferir: `py -3 ..\..\ProjetoConversor\_ministros_conferir.py` (rc 2 = divergiu).
- Consertar as páginas: `_ministros_migrar.py --base sessoes --apply` (dry-run sem `--apply`).
- Tirar a opção morta do schema: `_schema_limpar_orfas.py --apply` — **depois** das páginas,
  nunca antes. Enquanto a opção órfã existir, os pipelines schema-driven podem reusá-la.
- Grafia nova e legítima: cadastre em `_ministros_canonico.MIGRACAO` apontando para o canônico.

## Rotinas periódicas recomendadas (mensal ou pós-lote grande)
- `varrer_props.py` — advogados (pontuação/travessão), partes repetidas,
  composição duplicada/>7 (consenso-do-dia).
- `composicao_por_acordao.py` — composição pela lista oficial do acórdão
  ("Composição do julgamento: ..."); `composicao_consenso_epoca.py` — fallback
  p/ >7 sem lista (top-7 da base em ±45 dias).
- `cadeia_julgado.py` + `aplicar_cadeia.py` — cadeia do julgado (linha de vista
  = "Suspenso mas julgado depois"/"Suspenso*"; ED/reconsideração transparentes).
  Gera candidatos; o julgamento é por painel (ver memória da campanha).
- `punchline_vs_teor.py` + `aplicar_punchlines.py` — punchlines que narram
  suspensão em linha conclusiva.
- `extrair_citacoes.py` — fundamentacao_normativa/precedentes/resoluções vazios
  extraídos do teor por regex.
- `fill_campos_vazios.py` — partes/origem/relator vazios ← docs SJUR salvos.
- `preencher_t.py --full` + `renumerar_por_t.py` — t= dos links pela janela dos
  artifacts e renumeração "Julgamento N" pela ordem real do vídeo.
- `censo_merito.py` — placar de vazios/curtos dos campos meritórios.

## Regra do teor × cadeia do julgado (24/08)
**Linha de VISTA não carrega acórdão.** O acórdão pertence à página que registra a
proclamação; a linha interrompida fica sem teor, por definição. A janela `[-5,+60]d`
do motor casa o acórdão da conclusão com a linha anterior (vista de 26/09 recebendo
acórdão de 21/11, +56d) — foi assim que 29 páginas ganharam documento de outra sessão.
- Guarda no `aplicar_teor.py`: pula página suspensa cujo doc é posterior à sessão em >5d.
- Corretivo: `limpar_teor_linha_vista.py [--apply]` — remove só quando existe irmã
  conclusiva posterior (o acórdão não se perde da base); backup do texto no log.

## Armadilhas descobertas em 24/08 (conferir SEMPRE ao usar o motor SJUR)
- **Timeout silencioso**: a rodada de 23/08 perdeu **199 páginas** por timeout de 25 s —
  que no jsonl aparecem como `doc: null`, indistinguíveis de "acórdão não existe". SEMPRE
  auditar `consultas[].erro` antes de concluir que o teor não existe; re-rodar com
  `--timeout 60000 --fila <fila> --out <novo jsonl>`.
- **Truncamento em 2500 chars**: `sjur_enriquecer.py` (campanha do núcleo duro) usava
  `limpa(s, n=2500)` e gravou teores cortados no meio da palavra (ex. "...os pedidos em açã"
  em 0600941-38). O `sjur_teor_motor.py` NÃO trunca. Detecção: `len(textoDecisao) >= 2490`.
- Diagnóstico de página: fim do teor que não termina em pontuação = truncado.

## Motor SJUR (busca de jurisprudência do TSE por robô)
- `sjur_ui_motor.py` (modos partes/tema), `sjur_teor_motor.py` (teor por CNJ),
  `sjur_scorer.py`. Detalhes e pegadinhas: memória `sjur-busca-jurisprudencia-tse`
  (hCaptcha invisível via Playwright headless msedge; venv `..\.venv-windows-notion-ui`).

## Regravação de teores
- **`sanear_formato_teor.py [--apply]` — ROTINA PADRÃO quando o cortador evoluir.**
  Detecta E regrava numa única passada quem está fora do formato atual; retomável
  por assinatura do cortador (muda sozinha quando `segmenta_semantico`/`to_paragraphs`
  mudam, invalidando o "já feito" das versões anteriores). Preserva o marker.
- `regravar_uma.py "NUMERO"` — regrava o teor de uma página específica (conserto pontual).
- `regravar_defasados.py` — versão antiga (lista pré-gerada); prefira o sanear.
- `aplicar_teor.py [--refazer]` — grava teores dos docs SJUR (jsonl).

### Padrões que o cortador já resolve (`fill_inteiro_teor.segmenta_semantico`)
- itens numerados: `APROVAÇÃO.1. Trata-se...` → um parágrafo por item;
- ementa estruturada: `DIPLOMAÇÃO.I. QUESTÃO DE ORDEM1. Questão...` (seções romanas);
- **subitens: `R$ 124.913,03.6.1. O fato...` → `6.1`, `6.2`, `6.3` em parágrafos próprios** (v4);
- nunca corta após abreviação (`art.`, `n.`, `Min.`) nem deixa marcador órfão (`... 4.`);
- NÃO quebra em valores/datas/dispositivos (`R$ 124.913,03`, `10.9.2021`, `Lei nº 6.830/1980`).
Ao achar um padrão novo: ajustar `segmenta_semantico`, rodar `pytest tests -q`,
`regravar_uma.py` no caso testemunha e depois `sanear_formato_teor.py --apply`.

## Classe do processo (24-25/08/2026)
Fonte da verdade: o **cabeçalho FORMAL** do acórdão gravado no corpo
("INSTRUÇÃO (11544) N. 0600749-95.2019...", "AÇÃO CAUTELAR (12061) Nº ...").
- `classe_pelo_cabecalho.py` — confere etiqueta × cabeçalho. **PEGADINHA QUE CUSTOU CARO:**
  a primeira versão lia a EMENTA e propôs 806 trocas erradas ("ELEIÇÕES 2016. REGISTRO DE
  CANDIDATURA. RECURSO ESPECIAL..." → RCand, quando a classe é o recurso e RCand é o
  ASSUNTO). A versão atual exige que a classe ABRA o texto e venha seguida do número do
  processo; revalidando as 806 sobraram **9 reais** (`revalidar_classe.py`).
- `classe_por_consenso_cnj.py` — propaga a classe oficial para as páginas do MESMO CNJ que
  não têm teor (um processo tem uma classe só). Foi como se resolveram as instruções
  normativas rotuladas "PC"/"PA" (0600749-95, 0600748-13, 0600747-28): 14 aplicadas.
  Guarda: não propaga quando o `resultado` é incompatível com a classe (ex.: "Provido" numa
  Instrução) — isso indica número errado NAQUELA linha, e vai para revisão.
- Regra geral: divergência DENTRO da família (REspe × AgRg-REspe × ED-REspe) é fase
  processual, não erro. Nunca criar option nova de classe — fragmenta o vocabulário.

## Voo profundo meritório (funil barato Haiku→Sonnet)
- `gerar_lotes_triagem.py` → workflow `triagem-meritoria-full.js` (Haiku, 5
  dimensões vs teor+cadeia) → revisão Sonnet → `aplicar_revisao.py`.
  Scripts de workflow em `~\.claude\projects\...\workflows\scripts\`.
