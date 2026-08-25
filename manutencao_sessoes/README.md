# Kit de manutenção da base "sessões" (campanha 21-24/08/2026)

Scripts promovidos do scratchpad da campanha de auditoria. Todos usam o venv
`..\.venv-win` e a API do Notion via `tse_youtube_notion_core`. Convenção:
rodar SEM flag = dry-run (relatório); `--apply` = grava. Logs restauráveis em
`..\artifacts\notion_sessoes_auditoria\`.

## O que roda SOZINHO (não precisa destes scripts)
- **Relation DJe + teor pelo acordao pareado** (etapa 9 do watcher, 25/08/2026):
  `auditar_relation_dje.py --apply` liga as sessoes as decisoes novas do DJe e deixa em
  `artifacts
otion_sessoes_auditoria
elation_fila_novas.json` SO as paginas ligadas
  naquela rodada; em seguida `preencher_teor_do_dje.py --apply --fila <ela>` grava o teor
  dessas. Os dois sao advisory (falha vira aviso, nao reprova o CSV). A fila e o que torna
  o segundo passo viavel: sem `--fila` ele releria os blocos das ~3,2 mil paginas com
  relation (~40 min).
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

## Relation `sessões` <-> `DJe` (25/08/2026)
`auditar_relation_dje.py [--apply]` — parte das SESSÕES sem relation e faz uma consulta
por processo (minutos), em vez de reler as 188 mil páginas do DJe como o
`..\..\ProjetoConversor\DJE_relations.py --modo cross` (~30 min só de leitura).
**A ausência da relation é um DETECTOR, não só uma falha de ligação** — foi assim que a
rodada de 25/08 separou 615 páginas sem relation em:
- **391 com par no DJe** (369 com acórdão) — a ligação é que nunca fora gravada. O `cross`
  grava do lado DJe **sem comparar com o estado atual**, então uma rodada que não conclui
  não deixa rastro de que faltou: as páginas do DJe já existiam (58/60 da amostra criadas
  antes de agosto) e mesmo assim ficaram soltas. Depois do reparo: 3.164/3.388 (93,4%).
- **130 atos administrativos** (Instrução/PA/"Aprovada") — não geram acórdão no repositório
  de jurisprudência; a ausência é CORRETA e não deve virar alerta.
- **26 sessões recentes** (acórdão ainda não publicado) e **11 linhas suspensas por vista**
  (o acórdão pertence à linha conclusiva — mesma regra do teor).
- **57 suspeitas**, das quais 9 com ano do CNJ POSTERIOR à sessão (número errado, mesmo
  padrão do `detectar_cnj_impossivel.py`). **Todas as 57 estão sem inteiro teor**: falta de
  acórdão e falta de teor têm a mesma causa, então uma varredura confirma a outra.

Pegadinha do acervo: `dje` NÃO é o DJe — é o repositório de jurisprudência do TSE. Um
julgamento por maioria tem acórdão por definição, mas ele pode não estar no acervo (o
0600421-63.2020.6.05.0107, julgado em 05/03/2026, só tem lá a monocrática de 2023; no
consolidado de 1,2 GB há uma única ocorrência do CNJ-20, a mesma monocrática). Antes de
concluir "número errado", conferir se a monocrática existente bate em relator/classe/município.

`preencher_teor_do_dje.py [--apply] [--janela]` — depois de reparada a relation, o
ACORDAO pareado vira fonte local de teor: `textoEmenta` + `textoDecisao` da pagina do DJe
sao o mesmo material que o motor extrai do CSV, sem SJUR nem CSV de 1,2 GB. Rodada de
25/08: **132 paginas ganharam teor** (mediana 3.323 chars).

**So a DATA EXATA autoriza gravar** (`dataDecisao` == data da sessao). **PEGADINHA QUE A
PRIMEIRA VERSAO PAGOU:** o gate original aceitava ate 180d quando o dispositivo concordava
com a etiqueta `votacao` (unanimidade x maioria) — isso NAO separa nada, porque a maioria
dos julgamentos e unanime e a concordancia sai por acaso. Dos 30 casos aprovados assim,
praticamente todos eram acordaos de EMBARGOS DE DECLARACAO julgados meses depois
("Desprovido" na pagina x "rejeitou os embargos" no acordao). **Só o RESULTADO separa fases
do processo; a votacao, nunca** — com esse teste a janela [-5,+60]d reprovou 25 de 33, e por
isso ela so entra sob `--janela`, ainda exigindo resultado igual e nao-embargos.
Divergencia de resultado no balde de data exata (6 de 25 na auditoria) e limitacao do
parser ("deu parcial provimento" -> "Provido em parte"), nao acordao errado.

Linhas "Suspenso"/"Suspenso*" sao puladas: o acordao pertence a irma conclusiva.

## Coluna `tema` (25/08/2026)
`auditar_tema.py [--csv]` — audita a coluna com **a mesma funcao que o fluxo usa ao
publicar** (`core.tema_looks_generic`). Regra de ouro: **regra nova entra no core** (onde ha
teste que a trava) **e o auditor a herda**; se o auditor tivesse padroes proprios, a base
ficaria limpa por um criterio e suja pelo outro.

Dois defeitos que o gate nao pegava e agora pega:
- **nome de autoridade no lugar da materia** ("Ministro Sergio Banhos"). Nasce quando o
  video traz so a proclamacao — "pedido de vista do Ministro X" — e o modelo toma quem
  falou pelo assunto julgado; a punchline denunciava ("Julgamento sobre ministro Sergio
  Banhos..."). O veto exige CARGO + NOME PROPRIO, entao "nulidade por impedimento de
  Ministro" e "competencia monocratica do relator" continuam validos: ali a autoridade
  integra a TESE. Sem essa distincao, 6 temas legitimos teriam sido destruidos.
- **rotulo processual puro** ("Embargos de declaracao", "Agravo regimental"): dizem o RITO,
  nao a materia. O prompt ja os proibia em palavras — e mesmo assim eram o tema de 6
  paginas. **Instrucao de prompt nao e garantia; o veto tem de ser deterministico.**

Rodada de 25/08: 12 temas reescritos a partir do teor (fonte oficial) + prompt reforcado
(nome de ministro e rito proibidos explicitamente) + 3 testes. Auditoria seguinte: 0 recusados.

## Voo profundo meritório (funil barato Haiku→Sonnet)
- `gerar_lotes_triagem.py` → workflow `triagem-meritoria-full.js` (Haiku, 5
  dimensões vs teor+cadeia) → revisão Sonnet → `aplicar_revisao.py`.
  Scripts de workflow em `~\.claude\projects\...\workflows\scripts\`.
