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

:: ── Check for Visual C++ 2022 Redistributable ───────────────────────────
:: onnxruntime 1.21+ requires it; without it Python raises:
::   ImportError: DLL load failed while importing onnxruntime_pybind11_state
set "VCRT_OK=0"
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=3" %%V in ('reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Version 2^>nul ^| findstr /i "Version"') do set "VCRT_VER=%%V"
    set "VCRT_VER=!VCRT_VER:v=!"
    for /f "tokens=1,2 delims=." %%A in ("!VCRT_VER!") do (
        if %%A GEQ 14 if %%B GEQ 30 set "VCRT_OK=1"
    )
)
if "!VCRT_OK!" == "0" (
    echo.
    echo  [ERROR] Microsoft Visual C++ 2022 Redistributable is not installed.
    echo  onnxruntime 1.21+ requires it. The app will crash without it.
    echo.
    echo  Download and install it from:
    echo    https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
    echo  After installing, run this file again.
    echo.
    pause
    exit /b 1
)

:: ── Check that PyQt6 is available in the venv ─────────────────────────────
:: If the installer's pip batch was interrupted (e.g. torch failure), PyQt6
:: may be missing even though the venv exists.  Attempt a quick fix.
"%VENV_PYTHON%" -c "import PyQt6" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  [NOTICE] PyQt6 is not installed in the virtual environment.
    echo  Attempting to install it now...
    echo.
    "%VENV_PYTHON%" -m pip install PyQt6==6.6.1 PyQt6-Qt6==6.6.1 PyQt6-sip==13.6.0
    if !ERRORLEVEL! NEQ 0 (
        echo.
        echo  [ERROR] Failed to install PyQt6.
        echo  Try running install_windows.bat again, or install manually:
        echo    "%VENV_PYTHON%" -m pip install PyQt6==6.6.1
        echo.
        pause
        exit /b 1
    )
    echo  PyQt6 installed successfully.
    echo.
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
