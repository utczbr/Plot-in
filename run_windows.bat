@echo off
:: ---------------------------------------------------------------------------
:: run_app.bat — Windows launcher for the Chart Analysis GUI
::
:: Double-click this file (or run it from a Command Prompt) to start the app.
::
:: The script:
::   1. Changes to the project root (wherever this .bat file lives).
::   2. Activates the local virtual environment if it is not already active.
::   3. Launches  python src\main_modern.py
:: ---------------------------------------------------------------------------

setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "VENV_DIR=%~dp0venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_ACTIVATE=%VENV_DIR%\Scripts\activate.bat"

:: ── Activate venv only if not already inside one ─────────────────────────
:: VIRTUAL_ENV is exported by activate.bat; absent in a bare shell.
if defined VIRTUAL_ENV (
    echo ^» Virtual environment already active: %VIRTUAL_ENV%
) else (
    if not exist "%VENV_ACTIVATE%" (
        echo ERROR: Virtual environment not found at "%VENV_DIR%".
        echo Run the installer first:  install_windows.bat
        echo.
        pause
        exit /b 1
    )
    call "%VENV_ACTIVATE%"
    echo ^» Activated virtual environment: %VENV_DIR%
)

:: ── Prefer the venv python explicitly for robustness ─────────────────────
if exist "%VENV_PYTHON%" (
    set "PYTHON=%VENV_PYTHON%"
) else (
    set "PYTHON=python"
)

:: ── Launch the application ────────────────────────────────────────────────
echo ^» Starting Chart Analysis GUI...
"%PYTHON%" src\main_modern.py %*
exit /b %ERRORLEVEL%
