@echo off
setlocal
rem Recolore E EXCLUI as etiquetas orfas marcadas no CSV (remover_se_apply=1).
rem A exclusao tem guarda: aborta a opcao se a UI indicar uso por >0 paginas.
rem Uso: NOTION_labels_delete_orphans_windows.cmd <partes^|advogados^|origem>
set "PROP=%~1"
if "%PROP%"=="" set "PROP=partes"
echo Coluna: %PROP%  (recolor + EXCLUSAO de orfas)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\notion_labels_default_windows.ps1" -Mode apply -Property %PROP% -ApplyDeletes -DebugMode -StopOnError
pause
