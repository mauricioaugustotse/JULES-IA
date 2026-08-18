$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# pythonw da venv direto (sem console/tela preta), como nos demais atalhos
$pythonw = Join-Path $scriptDir ".venv-win\Scripts\pythonw.exe"
$gui = Join-Path $scriptDir "tse_youtube_notion_batch_gui.py"

if (-not (Test-Path $pythonw)) {
    throw "pythonw da venv nao encontrado: $pythonw"
}
if (-not (Test-Path $gui)) {
    throw "GUI nao encontrada: $gui"
}

$shell = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "TSE YouTube Notion.lnk"
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = "`"$gui`""
$shortcut.WorkingDirectory = $scriptDir
$shortcut.Description = "TSE YouTube -> Notion: processa videos de sessoes (ate 10 por lote) e publica no Notion"
$icon = Join-Path $scriptDir "assets\icone_tse_youtube_notion.ico"
if (Test-Path $icon) {
    $shortcut.IconLocation = "$icon,0"
} else {
    $shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,167"
}
$shortcut.Save()

Write-Host "Atalho criado em: $shortcutPath"
