@echo off
setlocal
rem Abre o Edge com remote debugging (porta 9222) na base de sessoes.
rem Depois: logue no Notion, abra a base e deixe o painel de Opcoes da
rem propriedade-alvo (partes/advogados/origem) visivel antes de rodar o apply.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\notion_labels_default_windows.ps1" -Mode dump -LaunchBrowser -DebugMode
pause
