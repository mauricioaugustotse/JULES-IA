$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$launcher = Join-Path $scriptDir "TSE_YOUTUBE_NOTION_BATCH.cmd"

if (-not (Test-Path $launcher)) {
    throw "Launcher nao encontrado: $launcher"
}

$shell = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "TSE YouTube Notion - lote 10.lnk"
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $launcher
$shortcut.WorkingDirectory = $scriptDir
$shortcut.Description = "Processa ate 10 links do YouTube do TSE, busca noticias e publica no Notion"
$shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,167"
$shortcut.Save()

Write-Host "Atalho criado em: $shortcutPath"
