@echo off
REM Bicleaner Service Launcher for Windows
REM Auto-detects GPU and starts appropriate mode

cd /d "%~dp0\.."

echo Bicleaner Service Launcher
echo ==========================

REM Check if GPU is available
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] GPU detected, starting with GPU support...
    docker compose up -d
) else (
    echo [!] GPU not available, starting in CPU mode...
    docker compose --profile cpu up -d bicleaner-service-cpu
)

echo.
echo Waiting for service to start...
timeout /t 10 /nobreak >nul

REM Check health
curl -sf http://localhost:8057/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Service is healthy!
    curl -s http://localhost:8057/health
) else (
    echo [!] Service not ready yet. Check logs with: docker compose logs -f
)
