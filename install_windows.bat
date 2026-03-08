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
exit /b %ERRORLEVEL%
