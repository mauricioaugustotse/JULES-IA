@echo off
setlocal
rem Teste seguro: recolore as proximas 5 etiquetas pendentes (so cor, sem excluir).
rem Uso: NOTION_labels_apply_test_windows.cmd <partes^|advogados^|origem>
set "PROP=%~1"
if "%PROP%"=="" set "PROP=partes"
echo Coluna: %PROP%  (lote de teste = 5, sem exclusao)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\notion_labels_default_windows.ps1" -Mode apply -Property %PROP% -PendingLimit 5 -DebugMode -StopOnError
pause
