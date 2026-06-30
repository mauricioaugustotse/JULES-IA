# Remove a Tarefa Agendada do monitor DJe (desfaz CRIAR_TAREFA_WATCH_DJE.ps1)
# e encerra qualquer processo do watcher que tenha ficado rodando.
$ErrorActionPreference = "Stop"
$taskName = "WatchDJe_Notion"

try { Stop-ScheduledTask -TaskName $taskName -ErrorAction Stop } catch {}

# O python.exe/pythonw.exe do venv e um stub que lanca o interpretador real como filho;
# parar a tarefa pode deixar esse filho orfao. Mata por linha de comando para garantir.
$killed = 0
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -like 'python*' -and $_.CommandLine -and $_.CommandLine.Contains('watch_jurisprudencia_csv.py') } |
  ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force; $killed++ } catch {} }
if ($killed -gt 0) { Write-Host "Processos do watcher encerrados: $killed" }

$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -eq $existing) {
    Write-Host "Tarefa '$taskName' nao existe (nada a remover)."
} else {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Tarefa '$taskName' removida." -ForegroundColor Green
}
