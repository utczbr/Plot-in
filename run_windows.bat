@echo off
:: ---------------------------------------------------------------------------
:: run_app.bat — Windows launcher for the Chart Analysis GUI
::
:: Double-click this file (or run it from a Command Prompt) to start the app.
::
:: The script:
::   1. Changes to the project root (wherever this .bat file lives).
::   2. Locates the virtual environment created by install_windows.bat.
::   3. Launches src\main_modern.py using the venv python.exe directly
::      (avoids the Microsoft Store "Python was not found" stub).
:: ---------------------------------------------------------------------------

setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "VENV_DIR=%~dp0.venv"
set "VENV_DIR_ALT=%~dp0venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_ACTIVATE=%VENV_DIR%\Scripts\activate.bat"

:: Fall back to plain 'venv' folder if '.venv' is absent (legacy / manual installs)
if not exist "%VENV_ACTIVATE%" (
    if exist "%VENV_DIR_ALT%\Scripts\activate.bat" (
        set "VENV_DIR=%VENV_DIR_ALT%"
        set "VENV_PYTHON=%VENV_DIR_ALT%\Scripts\python.exe"
        set "VENV_ACTIVATE=%VENV_DIR_ALT%\Scripts\activate.bat"
    )
)

:: ── Check the venv exists ─────────────────────────────────────────────────
if not exist "%VENV_ACTIVATE%" (
    echo.
    echo  ERROR: Virtual environment not found.
    echo.
    echo  Expected location: %VENV_DIR%
    echo.
    echo  Please run the installer first:
    echo    install_windows.bat
    echo.
    echo  If you already ran the installer and still see this message,
    echo  make sure you are running this file from the Plot-in folder.
    echo.
    pause
    exit /b 1
)

:: ── Always use the venv python.exe by absolute path ───────────────────────
:: Do NOT fall back to bare 'python' — on many Windows systems that resolves
:: to the Microsoft Store stub which prints "Python was not found" and exits.
if not exist "%VENV_PYTHON%" (
    echo.
    echo  ERROR: python.exe not found inside the virtual environment.
    echo  Expected: %VENV_PYTHON%
    echo.
    echo  The virtual environment may be corrupted.
    echo  Delete the .venv folder and run install_windows.bat again.
    echo.
    pause
    exit /b 1
)

:: ── Activate venv (sets PATH, VIRTUAL_ENV, etc. for child processes) ──────
if not defined VIRTUAL_ENV (
    call "%VENV_ACTIVATE%"
    echo ^» Activated: %VENV_DIR%
) else (
    echo ^» Virtual environment already active: %VIRTUAL_ENV%
)

:: ── Launch the application ────────────────────────────────────────────────
echo ^» Starting Chart Analysis GUI...
"%VENV_PYTHON%" src\main_modern.py %*
set "APP_EXIT=%ERRORLEVEL%"

if %APP_EXIT% NEQ 0 (
    echo.
    echo  The application exited with error code %APP_EXIT%.
    echo  Check the messages above for details.
    echo.
    pause
)
exit /b %APP_EXIT%
