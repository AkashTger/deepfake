@echo off
title DeepGuard AI - Deepfake Detection
echo.
echo ========================================================
echo   DeepGuard AI - SOTA Deepfake Detection
echo ========================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

:: Check model file
if not exist "%~dp0model_files\ensemble_best.safetensors" (
    echo [WARNING] Model weights not found!
    echo.
    echo   Download ensemble_best.safetensors from your Kaggle notebook output
    echo   and place it in: %~dp0model_files\
    echo.
    echo   The website will still load, but inference will be disabled.
    echo.
)

:: Install dependencies
echo Installing dependencies...
pip install -r "%~dp0backend\requirements.txt" -q
echo.

:: Create assets dir
if not exist "%~dp0frontend\assets" mkdir "%~dp0frontend\assets"

:: Start server
echo Starting server at http://localhost:8000 ...
echo Press Ctrl+C to stop.
echo.
python "%~dp0backend\server.py"

pause
