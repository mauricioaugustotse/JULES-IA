# Registra a Tarefa Agendada "WatchDJe_Notion": monitora a pasta DJe e aplica os CSVs
# de jurisprudencia nos registros do Notion, de forma INDEPENDENTE do fluxo do lote 10.
#
#  - Inicia ao fazer LOGON (na sua sessao, para o OneDrive estar montado).
#  - Roda OCULTA via pythonw.exe (sem janela de console).
#  - REINICIA sozinha (ate 3x, intervalo de 1 min) se o processo cair.
#  - Instancia unica garantida tambem pelo lockfile do proprio watcher.
#
# Rode este script uma vez (clique direito > "Executar com PowerShell", ou no terminal).
# Para iniciar imediatamente sem deslogar:  Start-ScheduledTask -TaskName WatchDJe_Notion
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$py = Join-Path $scriptDir ".venv-win\Scripts\pythonw.exe"
if (-not (Test-Path $py)) { $py = Join-Path $scriptDir ".venv\Scripts\pythonw.exe" }
if (-not (Test-Path $py)) {
    throw "pythonw.exe nao encontrado em .venv-win nem .venv. Crie/ative o venv do projeto primeiro."
}

$watcher = Join-Path $scriptDir "watch_jurisprudencia_csv.py"
if (-not (Test-Path $watcher)) { throw "Watcher nao encontrado: $watcher" }

$djeDir  = "C:\Users\mauri\OneDrive\Documentos\12 - Consultoria Legislativa\DJe"
$logFile = Join-Path $scriptDir "artifacts\jurisprudencia_csv\watch_dje.log"
$taskName = "WatchDJe_Notion"

# Argumentos para o pythonw (cada caminho entre aspas por causa dos espacos).
$argList = @(
    "`"$watcher`"",
    "--watch-dir", "`"$djeDir`"",
    "--apply",
    "--poll-secs", "5",
    "--log-file", "`"$logFile`""
) -join " "

$action  = New-ScheduledTaskAction -Execute $py -Argument $argList -WorkingDirectory $scriptDir
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERNAME"
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal `
    -Description "Monitora a pasta DJe e aplica os CSVs de jurisprudencia nos registros do Notion (independente do lote 10)." `
    -Force | Out-Null

Write-Host "Tarefa '$taskName' criada/atualizada." -ForegroundColor Green
Write-Host "  Executa : $py"
Write-Host "  Args    : $argList"
Write-Host "  Pasta   : $djeDir"
Write-Host "  Log     : $logFile"
Write-Host ""
Write-Host "Para iniciar agora (sem deslogar):  Start-ScheduledTask -TaskName '$taskName'"
Write-Host "Para conferir o estado          :  Get-ScheduledTask -TaskName '$taskName' | Get-ScheduledTaskInfo"
