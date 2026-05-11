@echo off
setlocal
cd /d %~dp0

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

py -3 install.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] 'py' launcher failed or returned an error. Trying 'python' directly...
    python install.py %*
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo [ERROR] Installation failed. Please ensure Python 3 is installed and in your PATH.
        pause
    )
)
exit /b %ERRORLEVEL%
