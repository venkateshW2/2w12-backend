@echo off
echo === Rebuild with psutil Fix ===
echo.

cd /d C:\2w12-backend

echo 🛑 Stopping current container...
docker-compose down

echo.
echo 📝 Adding psutil to environment.yml...

REM Backup original environment.yml
copy environment.yml environment.yml.backup

REM Add psutil to the pip section
echo Adding psutil to dependencies...

REM Create updated environment.yml with psutil
(
echo name: 2w12-backend
echo channels:
echo   - conda-forge
echo   - defaults
echo   - pytorch
echo dependencies:
echo   - python=3.10
echo   - numpy=1.24.3
echo   - scipy=1.11.4
echo   - matplotlib=3.8.2
echo   - ffmpeg=4.4.2
echo   - portaudio=19.6.0
echo   - pip=23.3.1
echo   - pip:
echo     - fastapi==0.104.1
echo     - uvicorn[standard]==0.24.0
echo     - python-multipart==0.0.6
echo     - librosa==0.10.1
echo     - requests==2.31.0
echo     - redis==5.0.1
echo     - yt-dlp==2023.11.16
echo     - essentia-tensorflow==2.1b6.dev1110
echo     - aubio==0.4.9
echo     - Pillow==10.1.0
echo     - pydantic==2.5.0
echo     - python-jose==3.3.0
echo     - passlib==1.7.4
echo     - aiofiles==23.2.1
echo     - python-dotenv==1.0.0
echo     - psutil==5.9.5
) > environment-fixed.yml

echo ✅ Updated environment file created
echo.

echo 🏗️ Rebuilding container with psutil...
docker-compose build --no-cache

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Build failed!
    echo Restoring original environment.yml...
    copy environment.yml.backup environment.yml
    pause
    exit /b 1
)

echo.
echo 🚀 Starting fixed container...
docker-compose up -d

echo.
echo ⏳ Waiting for container to start...
timeout /t 20

echo.
echo 🧪 Testing backend...
curl -s http://localhost:8001/health >NUL 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Backend is working!
    start http://localhost:8001/docs
) else (
    echo ⚠️ Still starting or has issues...
    echo Checking logs...
    docker logs 2w12-audio-server --tail=10
)

echo.
echo === Rebuild Complete ===
pause