@echo off
echo === 2W12 Maximum Performance Setup ===
echo.
echo Dell G7 2019 Specs:
echo - 16GB RAM
echo - 8 CPU Cores  
echo - Dedicated to Docker only
echo.

cd /d C:\2w12-backend

echo 🛑 Stopping current containers...
docker-compose down

echo.
echo 🔧 Applying unlimited performance configuration...
echo - Removed all CPU limits
echo - Removed all memory limits  
echo - Increased Redis to 2GB
echo - Full system resource access
echo.

echo 🚀 Starting with maximum performance...
docker-compose up -d

echo.
echo ⏳ Waiting for full startup...
timeout /t 15

echo.
echo 📊 Checking container resource usage...
docker stats --no-stream

echo.
echo === Performance Setup Complete! ===
echo.
echo 🎯 Your backend now has access to:
echo - All 8 CPU cores
echo - Up to 14GB RAM  
echo - Full disk I/O
echo - No artificial limits
echo.
echo 🌐 Access from Mac: http://172.20.176.1:8001
echo 📚 API Documentation: http://172.20.176.1:8001/docs
echo.
echo 🚀 Audio processing will be lightning fast!
echo.
pause