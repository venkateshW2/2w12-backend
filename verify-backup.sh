@echo off
echo =====================================
echo   2W12 Backup Verification Script
echo =====================================
echo.

REM Find the most recent backup directory
for /f "tokens=*" %%a in ('dir C:\2w12-backup-* /b /ad /o-d 2^>nul') do (
    set "latest_backup=C:\%%a"
    goto :found
)

echo ❌ No backup directories found!
echo Make sure you ran the backup script first.
pause
exit /b 1

:found
echo 📁 Latest backup found: %latest_backup%
echo.

echo 🔍 VERIFICATION CHECKLIST:
echo ========================

REM Check Docker images
echo 📦 Docker Images:
if exist "%latest_backup%\2w12-audio-server.tar" (
    for %%A in ("%latest_backup%\2w12-audio-server.tar") do echo    ✅ Main image: %%~zA bytes
) else (
    echo    ❌ Main image: MISSING
)

if exist "%latest_backup%\redis.tar" (
    for %%A in ("%latest_backup%\redis.tar") do echo    ✅ Redis image: %%~zA bytes
) else (
    echo    ❌ Redis image: MISSING
)

if exist "%latest_backup%\portainer.tar" (
    for %%A in ("%latest_backup%\portainer.tar") do echo    ✅ Portainer image: %%~zA bytes
) else (
    echo    ❌ Portainer image: MISSING
)

echo.

REM Check project files
echo 📂 Project Files:
if exist "%latest_backup%\2w12-backend\main.py" (
    echo    ✅ main.py found
) else (
    echo    ❌ main.py MISSING
)

if exist "%latest_backup%\2w12-backend\docker-compose.yml" (
    echo    ✅ docker-compose.yml found
) else (
    echo    ❌ docker-compose.yml MISSING  
)

if exist "%latest_backup%\2w12-backend\Dockerfile" (
    echo    ✅ Dockerfile found
) else (
    echo    ❌ Dockerfile MISSING
)

if exist "%latest_backup%\2w12-backend\api" (
    echo    ✅ API directory found
) else (
    echo    ❌ API directory MISSING
)

if exist "%latest_backup%\2w12-backend\core" (
    echo    ✅ Core directory found  
) else (
    echo    ❌ Core directory MISSING
)

echo.

REM Check volumes
echo 💾 Data Volumes:
if exist "%latest_backup%\redis_data.tar.gz" (
    for %%A in ("%latest_backup%\redis_data.tar.gz") do echo    ✅ Redis data: %%~zA bytes
) else (
    echo    ❌ Redis data: MISSING
)

if exist "%latest_backup%\portainer_data.tar.gz" (
    for %%A in ("%latest_backup%\portainer_data.tar.gz") do echo    ✅ Portainer data: %%~zA bytes
) else (
    echo    ❌ Portainer data: MISSING
)

echo.

REM Check configuration
echo ⚙️ Configuration:
if exist "%latest_backup%\docker-compose-resolved.yml" (
    echo    ✅ Docker Compose config found
) else (
    echo    ❌ Docker Compose config MISSING
)

if exist "%latest_backup%\RESTORATION_INSTRUCTIONS.md" (
    echo    ✅ Restoration instructions found
) else (
    echo    ❌ Restoration instructions MISSING
)

echo.

REM Calculate total backup size
echo 📊 Backup Statistics:
for /f "tokens=3" %%a in ('dir "%latest_backup%" /-c ^| find "File(s)"') do set total_size=%%a
echo    Total backup size: %total_size% bytes

REM Convert to MB
set /a size_mb=%total_size% / 1048576
echo    Total backup size: %size_mb% MB

echo.

REM Check if backup is complete
set issues=0

if not exist "%latest_backup%\2w12-audio-server.tar" set /a issues+=1
if not exist "%latest_backup%\redis.tar" set /a issues+=1  
if not exist "%latest_backup%\2w12-backend\main.py" set /a issues+=1
if not exist "%latest_backup%\redis_data.tar.gz" set /a issues+=1

if %issues% EQU 0 (
    echo ✅ BACKUP VERIFICATION PASSED!
    echo    All critical components found
    echo    Backup appears complete and ready for Linux migration
) else (
    echo ❌ BACKUP VERIFICATION FAILED!
    echo    %issues% critical components missing
    echo    Please re-run the backup script
)

echo.
echo 📍 Backup location: %latest_backup%
echo 📖 View instructions: notepad "%latest_backup%\RESTORATION_INSTRUCTIONS.md"
echo.
pause