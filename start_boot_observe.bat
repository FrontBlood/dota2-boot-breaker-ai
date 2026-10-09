@echo off
setlocal
cd /d "%~dp0"
title Dota 2 Boot Breaker AI
where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Please install Python 3.11 or newer.
    pause
    exit /b 1
)
echo Starting Boot Breaker AI in observation mode...
echo This mode will not send keyboard input.
echo.
python -m boot_breaker %*
set "BOOT_EXIT=%errorlevel%"
if not "%BOOT_EXIT%"=="0" (
    echo.
    echo Boot Breaker AI exited with error code %BOOT_EXIT%.
    pause
)
endlocal & exit /b %BOOT_EXIT%
