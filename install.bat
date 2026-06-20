@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem find a Python launcher
where py >nul 2>nul && (set "PY=py") || (set "PY=python")

rem if args were passed, forward them straight to install.py
if not "%~1"=="" (
    %PY% install.py %*
    goto :end
)

echo ============================================
echo   beatgen installer
echo ============================================
echo Select runtime:
echo   [1] CPU
echo   [2] CUDA ^(NVIDIA GPU^)
set "choice="
set /p choice="Enter 1 or 2 [1]: "
if "!choice!"=="2" (set "RUNTIME=cuda") else (set "RUNTIME=cpu")

%PY% install.py --runtime !RUNTIME!

:end
echo.
pause
