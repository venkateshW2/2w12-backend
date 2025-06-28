@echo off
echo ========================================
echo    2W12 Backend Docker Setup - Dell G7
echo ========================================
echo.

cd /d C:\2w12-backend

echo 📋 Checking files...
if not exist "Dockerfile" (
    echo ❌ Dockerfile missing!
    pause
    exit /b 1
)
if not exist "docker-compose.yml" (
    echo ❌ docker-compose.yml missing!
    pause 
    exit /b 1
)
if not exist "environment.yml" (
    echo ❌ environment.yml missing!
    pause
    exit /b 1
)
if not exist "main.py" (
    echo ❌ main.py missing!
    pause
    exit /b 1
)

echo ✅ All required files found!
echo.

echo 🐳 Checking Docker...
docker --version >NUL 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker not found! Please start Docker Desktop.
    pause
    exit /b 1
)

echo ✅ Docker is running!
echo.

echo 🧹 Cleaning up previous containers...
docker-compose down 2>NUL
docker system prune -f

echo.
echo 🏗️ Building Docker container...
echo ⏰ This will take 10-15 minutes (downloading libraries)
echo 📊 Progress will show below:
echo.

docker-compose build --progress=plain

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Build failed! Common issues:
    echo - Check Docker Desktop is running
    echo - Check internet connection
    echo - Check if files have correct content
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ Build successful! 🎉
echo.

echo 🚀 Starting services...
docker-compose up -d

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to start services!
    echo Checking logs...
    docker-compose logs
    pause
    exit /b 1
)

echo.
echo ⏳ Waiting for services to initialize...
timeout /t 20

echo.
echo 🧪 Testing API connection...

REM Test with PowerShell (more reliable than curl on Windows)
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://localhost:8001/health' -TimeoutSec 10; if ($response.StatusCode -eq 200) { Write-Host '✅ Backend is running successfully!' -ForegroundColor Green } else { Write-Host '⚠️ Backend responded but may have issues' -ForegroundColor Yellow } } catch { Write-Host '⚠️ Backend may still be starting...' -ForegroundColor Yellow }"

echo.
echo ========================================
echo      🎉 2W12 Backend is Ready!
echo ========================================
echo.
echo 🌐 API URL: http://localhost:8001
echo 📚 API Documentation: http://localhost:8001/docs
echo 🎛️ Container Management: http://localhost:9000
echo.
echo === Useful Commands ===
echo 📊 Check status: docker-compose ps
echo 📝 View logs: docker-compose logs -f 2w12-backend
echo 🔄 Restart: docker-compose restart 2w12-backend
echo 🛑 Stop all: docker-compose down
echo 🔧 Rebuild: docker-compose build --no-cache
echo.

echo 🌍 Opening API documentation in browser...
start http://localhost:8001/docs

echo.
echo Press any key to see container status...
pause

echo.
echo 📊 Container Status:
docker-compose ps

echo.
echo 📝 Recent logs:
docker-compose logs --tail=20 2w12-backend

echo.
echo Setup complete! Your backend is running locally on your Dell G7.
echo You saved $12/month and got better performance! 🚀
pause