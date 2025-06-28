@echo off
echo === GPU Docker Setup for Dell G7 ===
echo.

REM Check if running as Administrator
net session >nul 2>&1
if %errorLevel__ neq 0 (
    echo ERROR: This script must be run as Administrator
    pause
    exit /b 1
)

echo 🎮 Checking GPU...
nvidia-smi
if %ERRORLEVEL__ NEQ 0 (
    echo ❌ NVIDIA GPU not detected or drivers missing
    echo Please install/update NVIDIA drivers first
    pause
    exit /b 1
)

echo ✅ NVIDIA GPU detected!
echo.

echo 📦 Installing NVIDIA Container Toolkit...
REM Download and install NVIDIA Container Toolkit
curl -L https://github.com/NVIDIA/nvidia-docker/releases/download/v2.13.0/nvidia-docker2_2.13.0-1_amd64.deb -o nvidia-docker2.deb

echo.
echo 🔄 Restarting Docker with GPU support...
net stop docker
net start docker

echo.
echo ✅ GPU Docker setup complete!
echo Now update your docker-compose.yml to use GPU
pause