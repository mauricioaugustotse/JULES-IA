# Monitor do lote TSE YouTube → Notion

Abra o atalho **TSE YouTube Notion** normalmente e inicie o lote. O monitor é
ativado automaticamente; o botão **Monitor** abre o painel no navegador. Uma
janela que já estava aberta antes da atualização precisa ser reaberta.

## Correções automáticas e vistoria

O fluxo reconcilia as linhas com o inventário oficial **antes do enriquecimento
e novamente antes da publicação**. CNJ curto ou com um zero excedente é corrigido
quando existe uma única correspondência sustentada pelo número original do vídeo
e pelos demais dados da sessão. A relatoria e a composição efetiva, incluindo
substitutos, prevalecem sobre o gabinete atual do DataJud e o cadastro de titulares.
Linhas já confirmadas dispensam a consulta ao DataJud.

Pedido de vista gera `Suspenso por vista`, votação `Suspenso`, ministro solicitante
e texto que distingue voto do relator de decisão final. Uma proclamação com
resultados diferentes para recursos distintos preserva a informação específica
extraída do julgamento. Adiamentos expressos ficam contabilizados como exclusões.
Números citados em outro julgamento são excluídos quando scan, detalhe, partes e
texto comprovam essa relação; a ausência no inventário, isoladamente, não basta.

Após os tratamentos finais, campos confirmados pela reconciliação são restaurados
se algum script os alterar. Cada reparo registra o valor anterior antes de escrever
e passa por nova leitura no Notion. A busca de páginas é repetida após completar o
CNJ para evitar duplicação durante a recuperação.

Na **Fila de vistoria**, cada linha representa um caso. O painel de evidências
mostra o problema, os campos extraídos, os dados oficiais, a proclamação e o próximo
passo. **Abrir trecho** leva ao ponto do vídeo; **Fonte oficial** abre o inventário.
**Corrigir dados** permite editar uma proposta local e revalidá-la antes de
**Publicar julgamento**. Salvar a correção não publica. Alertas superados são
encerrados automaticamente; publicações só saem da fila após releitura confirmada.
O filtro de histórico conserva os casos resolvidos, publicados e descartados.

O painel mostra a última etapa registrada e as pendências de cada vídeo. Ele se
atualiza a cada dez segundos. Se o processo for encerrado, o horário para de
avançar; o arquivo não é um serviço independente que reinicia o workflow.

## Conferências e recuperação

- Cada trecho previsto na varredura precisa ter uma resposta válida. Trechos
  falhos são retomados pelo plano alternativo. Persistindo lacunas, o vídeo falha
  antes da publicação. Caches antigos incompletos também não são aceitos.
- Os capítulos da descrição do YouTube formam um inventário independente.
  Processos ausentes da varredura ganham janelas para análise do próprio vídeo.
- A pauta oficial do TSE é consultada para a data da sessão. O monitor compara
  processos individuais julgados, CNJ completo, classe, resultado, votação,
  relator, origem e composição, quando esses dados são publicados. Retiradas e
  processos julgados em lista ficam documentados como exclusões. Divergências
  de uma linha impedem sua publicação; ausência ou indisponibilidade de uma
  fonte permanece como pendência de cobertura.
- Detalhamento vazio ou com identidade divergente recebe uma releitura
  contextual, limitada a uma tentativa adicional por bloco. Persistindo a
  divergência, ela aparece como pendência, com o trecho e os processos envolvidos.
- Antes e depois de publicar, o monitor compara varredura, detalhamento, linhas
  finais, capítulos e contagem do rito. A quantidade total de linhas não basta:
  as identidades dos processos também são comparadas.
- Cada resultado da publicação é gravado imediatamente em um diário local.
  Depois, o fluxo lê as páginas do Notion e confere os campos gravados. Há nova
  leitura após os tratamentos posteriores à publicação. Bloqueios, descartes,
  falhas de tratamentos e leituras não confirmadas impedem um resultado de
  conclusão sem pendências.

As regras anteriores para julgamento coletivo em lista continuam aplicadas;
lista tríplice continua sendo caso individual. Os sinais de cobertura não são
certeza de que todo número mencionado era um julgamento: precedentes citados
podem gerar alertas que precisam de conferência. Ausência de capítulos também é
registrada. O monitor não transforma um candidato em decisão nem aprova registros
bloqueados automaticamente.

## Arquivos para acompanhar

Dentro de `artifacts/tse_youtube_notion/batch_gui/<lote>/`:

- `monitor.html`: painel de acompanhamento.
- `monitor_status.json`: estado do lote e dos vídeos, inclusive erros de início.
- `<video>/00_chapter_inventory.json`: capítulos e descrição consultados.
- `<video>/00_official_session_inventory.json`: inventário oficial da sessão e
  estado da consulta, inclusive quando a fonte está indisponível.
- `<video>/00_scan_coverage.json`: trechos cobertos e lacunas da varredura.
- `<video>/02_detail_coverage.json`: conferência e releituras de cada bloco.
- `<video>/04e_coverage_report.json`: inconsistências de cobertura.
- `<video>/04f_official_comparison.json`: confronto com os processos e campos
  divulgados pelo TSE.
- `<video>/04g_official_excluded_rows.json`: retiradas e listas excluídas com
  identificação e justificativa.
- `<video>/04h_publish_preview_rows.json`: linhas exatas enviadas ao publicador.
- `<video>/04i_automatic_reconciliation.json`: identidades, correções e citações
  excluídas, com evidências e valores anteriores, nas duas etapas do fluxo.
- `<video>/05_publish_journal.json`: resultados por linha, inclusive publicação parcial.
- `<video>/05b_notion_verification.json`: confirmação das páginas por leitura.
- `<video>/05c_final_notion_verification.json`: nova leitura após os tratamentos.
- `<video>/05d_final_official_comparison.json`: confronto oficial no fechamento.
- `<video>/05e_automatic_post_repair.json`: reparos de campos oficiais alterados
  pelos tratamentos posteriores.
- `batch_summary.json`: contagem separada de concluídos, pendentes, erros,
  interrompidos e vídeos não processados.

A fila de vistoria é recarregada ao finalizar o lote e acompanha alterações do
arquivo a cada dois segundos. O filtro e a seleção são preservados. O contador
da aba mostra os casos pendentes de decisão, mesmo durante o processamento.

A etiqueta `tipo_registro` enumera os registros publicados: itens bloqueados ou
descartados não reservam números. Ao publicar pela vistoria, uma proposta nova
recebe o próximo número disponível na data; a revisão de uma página existente
preserva sua etiqueta. A prévia salva acompanha a numeração enviada ao Notion,
que também é conferida na releitura final.

Pendências também entram na fila de vistoria. Reprocessar em um **novo lote**
refaz a varredura; retomar um cache com cobertura comprovadamente incompleta
 exige primeiro reprocessar a sessão. A auditoria preparatória da execução de
17/09 está em `artifacts/monitor_preflight_20260919/`.

## Recuperação das sessões de setembro

A recuperação de 10, 15 e 17/09/2026 está documentada em
`artifacts/monitor_repair_20260922/RELATORIO_FINAL.md`, com links das páginas e
evidências de leitura. O lote original mantém seu histórico de falha;
`monitor_resolution.json` e `resolution_monitor.html` registram a conferência
posterior. Ao reabrir a GUI, o botão Monitor exibe essa resolução até o próximo
lote criar seu próprio painel.

A consulta oficial não valida, sozinha, todo o raciocínio jurídico. Resultados
compostos, fundamentos e dados omitidos pela fonte exigem conferência da sessão.
Quando um agravo interno ou regimental é provido para julgar o recurso
subjacente, a etiqueta `resultado` segue o desfecho desse recurso; o texto
registra separadamente o provimento do agravo.
Contagens do rito também incluem chamadas sem julgamento e não substituem a
conferência das identidades. Uma nova sessão ainda precisa passar pelo fluxo
atualizado para validar a execução completa em produção.

## Sessão de 22/09/2026

O lote do vídeo `WYjLx6WzMns` falhou na leitura dos horários da varredura.
Depois da correção do leitor, a recuperação dirigida publicou e releu quatro
páginas no Notion: dois julgamentos concluídos e dois recursos ordinários cujo
exame prosseguiu no vídeo, mas foi adiado. Estes últimos permanecem marcados
como `Suspenso`, sem resultado de mérito. Seis julgamentos em lista ficaram fora
das páginas individuais. O erro da execução original permanece no histórico;
`monitor_resolution.json` e `resolution_monitor.html` registram a conferência.
O relatório e as leituras estão em
`artifacts/monitor_repair_WYjLx6WzMns/` na máquina onde a recuperação ocorreu.
Uma nova execução completa do atalho ainda precisa ser observada.

## Sessão de 24/09/2026

O replay dos dados originais confirmou cinco registros e excluiu três referências
a condenações citadas dentro do recurso de Arruda. O PA de Doutor Severiano/RN,
`0601130-04.2026.6.20.0000`, foi publicado e relido. As cinco páginas foram
conferidas contra o inventário; os 15 itens pendentes foram encerrados com trilha
de auditoria. Evidências em
`artifacts/tse_youtube_notion/batch_gui/20260929_153718_906300/automatic_repair_20260929/`.
O painel `resolution_monitor.html` registra a situação atual; o lote original
permanece preservado.
