# Monitor do lote TSE YouTube → Notion

Abra o atalho **TSE YouTube Notion** normalmente e inicie o lote. O monitor é
ativado automaticamente; o botão **Monitor** abre o painel no navegador. Uma
janela que já estava aberta antes da atualização precisa ser reaberta.

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
- `<video>/05_publish_journal.json`: resultados por linha, inclusive publicação parcial.
- `<video>/05b_notion_verification.json`: confirmação das páginas por leitura.
- `<video>/05c_final_notion_verification.json`: nova leitura após os tratamentos.
- `<video>/05d_final_official_comparison.json`: confronto oficial no fechamento.
- `batch_summary.json`: contagem separada de concluídos, pendentes, erros,
  interrompidos e vídeos não processados.

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
