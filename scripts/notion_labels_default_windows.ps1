param(
    [ValidateSet("dump", "apply")]
    [string]$Mode = "apply",
    [ValidateSet("partes", "advogados", "origem")]
    [string]$Property = "partes",
    [int]$Limit = 0,
    [int]$PendingLimit = 0,
    [int]$DelayMs = 450,
    [string]$StartLabel = "",
    [switch]$ApplyDeletes,
    [switch]$DryRun,
    [switch]$LaunchBrowser,
    [switch]$DebugMode,
    [switch]$StopOnError
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $RepoRoot ".venv-windows-notion-ui"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$ScriptPath = Join-Path $RepoRoot "notion_labels_default_playwright.py"
$NotionUrl = "https://app.notion.com/p/2eb721955c64809796bec75a81f9555f?v=ffe93c7f3ae4415699545f93f566d152"

if (-not (Test-Path $ScriptPath)) {
    throw "Script nao encontrado: $ScriptPath"
}

if (-not (Test-Path $PythonExe)) {
    Write-Host "Criando ambiente virtual Windows em $VenvDir ..."
    py -3 -m venv $VenvDir
}

& $PythonExe -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('playwright') else 1)" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Instalando dependencias (playwright) ..."
    & $PythonExe -m pip install --upgrade pip playwright
}

$argsList = @($ScriptPath, "--mode", $Mode, "--property-name", $Property, "--database-url", $NotionUrl, "--delay-ms", "$DelayMs")
if ($LaunchBrowser) { $argsList += "--launch-browser" }
if ($DebugMode) { $argsList += "--debug" }
if ($StopOnError) { $argsList += "--stop-on-error" }
if ($ApplyDeletes) { $argsList += "--apply-deletes" }
if ($DryRun) { $argsList += "--dry-run" }
if ($StartLabel -ne "") { $argsList += @("--start-label", $StartLabel) }
if ($Limit -gt 0) { $argsList += @("--limit", "$Limit") }
if ($PendingLimit -gt 0) { $argsList += @("--pending-limit", "$PendingLimit") }

Set-Location $RepoRoot
& $PythonExe @argsList
$exitCode = 0
if ($LASTEXITCODE -is [int]) { $exitCode = $LASTEXITCODE }
elseif (-not $?) { $exitCode = 1 }
if ($exitCode -ne 0) { throw "Execucao falhou com codigo $exitCode" }
