@echo off
setlocal
cd /d "%~dp0"
title Dota 2 Lockpick AI

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Please install Python 3.11 or newer.
    echo.
    pause
    exit /b 1
)

echo Starting Lockpick AI in observation mode...
echo This mode will not send mouse input.
echo.
python -m lockpick_ai %*
set "LOCKPICK_EXIT=%errorlevel%"

if not "%LOCKPICK_EXIT%"=="0" (
    echo.
    echo Lockpick AI exited with error code %LOCKPICK_EXIT%.
    pause
)

endlocal & exit /b %LOCKPICK_EXIT%
