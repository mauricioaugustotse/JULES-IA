# Para o monitor DJe AGORA: encerra a Tarefa Agendada (se rodando) e mata o processo
# pythonw/python do watcher. Nao remove a tarefa (ela volta a iniciar no proximo logon).
# Para desativar de vez, use REMOVER_TAREFA_WATCH_DJE.ps1.
try { Stop-ScheduledTask -TaskName "WatchDJe_Notion" -ErrorAction Stop } catch {}

$killed = 0
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -and $_.CommandLine.Contains('watch_jurisprudencia_csv.py') } |
  Sort-Object ProcessId -Descending |
  ForEach-Object {
      try { Stop-Process -Id $_.ProcessId -Force; $killed++ } catch {}
  }

Write-Output "watch DJe parado (processos encerrados: $killed)"
