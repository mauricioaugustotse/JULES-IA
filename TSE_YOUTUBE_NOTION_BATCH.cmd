@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_PATH=%SCRIPT_DIR%tse_youtube_notion_batch_gui.py"
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
    "%PYTHON_EXE%" "%SCRIPT_PATH%" %*
    set "EXIT_CODE=%ERRORLEVEL%"
) else (
    where py >nul 2>nul
    if errorlevel 1 (
        python "%SCRIPT_PATH%" %*
        set "EXIT_CODE=%ERRORLEVEL%"
    ) else (
        py -3 "%SCRIPT_PATH%" %*
        set "EXIT_CODE=%ERRORLEVEL%"
    )
)
popd

if not "%EXIT_CODE%"=="0" (
    echo.
    echo O processamento terminou com erro. Verifique as mensagens acima.
    pause
)

exit /b %EXIT_CODE%
