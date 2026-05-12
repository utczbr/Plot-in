@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0

:: ── Step 1: Use a pre-built installer executable if one is present ────────
for %%F in (
    chart-analysis-installer-windows.exe
    chart-analysis-installer-windows-amd64.exe
    chart-analysis-installer-windows-x86_64.exe
    chart-analysis-installer-windows-arm64.exe
    chart-analysis-installer.exe
) do (
    if exist "%%~fF" (
        "%%~fF" %*
        exit /b %ERRORLEVEL%
    )
)

:: ── Step 2: Locate a Python 3 interpreter ────────────────────────────────
set "PYTHON_EXE="

:: 2a. Try the Windows py launcher (most reliable when Python was installed via python.org)
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py -3 --version >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set "PYTHON_EXE=py -3"
        goto :run_installer
    )
)

:: 2b. Try plain 'python' (Microsoft Store installs, conda, etc.)
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python --version >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set "PYTHON_EXE=python"
        goto :run_installer
    )
)

:: 2c. Try 'python3'
where python3 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_EXE=python3"
    goto :run_installer
)

:: 2d. Search common python.org install locations (3.11 and 3.12)
for %%V in (313 312 311 310 39) do (
    for %%D in (
        "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
        "%APPDATA%\Python\Python%%V\python.exe"
        "C:\Python%%V\python.exe"
        "%ProgramFiles%\Python%%V\python.exe"
        "%ProgramFiles(x86)%\Python%%V\python.exe"
    ) do (
        if not defined PYTHON_EXE (
            if exist %%D (
                set "PYTHON_EXE=%%~D"
            )
        )
    )
)

if defined PYTHON_EXE goto :run_installer

:: ── No Python found ───────────────────────────────────────────────────────
echo.
echo [ERROR] Python 3 was not found on this computer.
echo.
echo  Please install Python 3.11 or 3.12 from:
echo    https://www.python.org/downloads/windows/
echo.
echo  IMPORTANT: On the installer screen, tick the box:
echo    "Add Python to PATH" / "Add python.exe to PATH"
echo  Then close this window and run install_windows.bat again.
echo.
pause
exit /b 1

:run_installer
:: ── Step 3: Run the installer ─────────────────────────────────────────────
echo Using Python: %PYTHON_EXE%
%PYTHON_EXE% install.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Installation failed (exit code %ERRORLEVEL%).
    echo         See the messages above for details.
    echo.
    pause
)
exit /b %ERRORLEVEL%
