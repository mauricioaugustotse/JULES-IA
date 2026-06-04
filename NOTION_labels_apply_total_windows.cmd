@echo off
setlocal
rem Recolore para default TODAS as etiquetas nao-default pendentes da coluna (sem excluir).
rem Uso: NOTION_labels_apply_total_windows.cmd <partes^|advogados^|origem>
set "PROP=%~1"
if "%PROP%"=="" set "PROP=partes"
echo Coluna: %PROP%  (recolor total, sem exclusao)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\notion_labels_default_windows.ps1" -Mode apply -Property %PROP% -DebugMode
pause
