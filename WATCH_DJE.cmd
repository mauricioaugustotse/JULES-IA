@echo off
setlocal

rem Monitor MANUAL da pasta DJe (janela de console visivel). Para o servico oculto
rem que inicia sozinho no logon, use CRIAR_TAREFA_WATCH_DJE.ps1 (Tarefa Agendada).
rem Ctrl+C encerra.

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_PATH=%SCRIPT_DIR%watch_jurisprudencia_csv.py"
set "DJE_DIR=C:\Users\mauri\OneDrive\Documentos\12 - Consultoria Legislativa\DJe"
set "PYTHON_EXE="

if exist "%SCRIPT_DIR%.venv-win\Scripts\python.exe" set "PYTHON_EXE=%SCRIPT_DIR%.venv-win\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"

if not exist "%SCRIPT_PATH%" (
    echo Script nao encontrado em "%SCRIPT_PATH%".
    pause
    exit /b 1
)

pushd "%SCRIPT_DIR%"
if defined PYTHON_EXE (
    "%PYTHON_EXE%" "%SCRIPT_PATH%" --watch-dir "%DJE_DIR%" --apply --poll-secs 5 %*
    set "EXIT_CODE=%ERRORLEVEL%"
) else (
    where py >nul 2>nul
    if errorlevel 1 (
        python "%SCRIPT_PATH%" --watch-dir "%DJE_DIR%" --apply --poll-secs 5 %*
        set "EXIT_CODE=%ERRORLEVEL%"
    ) else (
        py -3 "%SCRIPT_PATH%" --watch-dir "%DJE_DIR%" --apply --poll-secs 5 %*
        set "EXIT_CODE=%ERRORLEVEL%"
    )
)
popd

if not "%EXIT_CODE%"=="0" (
    echo.
    echo O monitor terminou com erro. Verifique as mensagens acima.
    pause
)

exit /b %EXIT_CODE%
