@echo off
setlocal
cd /d "%~dp0"
title Dota 2 Boot Breaker AI
if "%BOOT_BREAKER_LAUNCHER_TEST%"=="1" (
    "Dota2-Boot-Breaker-AI.exe" --help >nul
    exit /b %errorlevel%
)
"Dota2-Boot-Breaker-AI.exe" --control --no-window
set "APP_EXIT=%errorlevel%"
if not "%APP_EXIT%"=="0" (
    echo.
    echo Application exited with error code %APP_EXIT%.
    pause
)
endlocal & exit /b %APP_EXIT%
