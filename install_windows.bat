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
:: ── Step 3: Check for Visual C++ 2022 Redistributable ──────────────────
:: onnxruntime 1.21+ requires it. Without it the app crashes with
:: "ImportError: DLL load failed" on startup.
set "VCRT_OK=0"
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    :: Registry key exists — check that the version is >= 14.30 (VS 2022)
    for /f "tokens=3" %%V in ('reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Version 2^>nul ^| findstr /i "Version"') do (
        set "VCRT_VER=%%V"
    )
    :: Version string looks like "v14.38.33130.0" — strip the leading 'v' and grab major.minor
    set "VCRT_VER=!VCRT_VER:v=!"
    for /f "tokens=1,2 delims=." %%A in ("!VCRT_VER!") do (
        if %%A GEQ 14 if %%B GEQ 30 set "VCRT_OK=1"
    )
)

if "!VCRT_OK!" == "0" (
    echo.
    echo  [NOTICE] Microsoft Visual C++ 2022 Redistributable not detected.
    echo  onnxruntime 1.21+ requires it to run on Windows.
    echo.
    echo  Would you like to open the download page now?
    echo    Y = open https://aka.ms/vs/17/release/vc_redist.x64.exe in browser
    echo    N = continue without it ^(app may crash on first run^)
    echo.
    set /p VCRT_CHOICE="Your choice [Y/N]: "
    if /i "!VCRT_CHOICE!" == "Y" (
        start "" "https://aka.ms/vs/17/release/vc_redist.x64.exe"
        echo.
        echo  Download started. Install the Redistributable, then run install_windows.bat again.
        echo.
        pause
        exit /b 0
    )
    echo  Continuing without VC++ 2022 Redistributable...
    echo.
)

:: ── Step 4: Run the installer ─────────────────────────────────────────────
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
