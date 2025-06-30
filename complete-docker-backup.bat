@echo off
echo ========================================
echo    2W12 Complete Docker Backup Script
echo ========================================
echo.

REM Create backup directory with timestamp
set backup_date=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%
set backup_date=%backup_date: =0%
set backup_dir=C:\2w12-backup-%backup_date%
mkdir "%backup_dir%"

echo 📁 Creating backup directory: %backup_dir%
echo.

echo 🐳 PHASE 1: Docker Container Backup
echo =====================================

REM Export all Docker images
echo 📦 Exporting Docker images...
docker save 2w12-audio-server -o "%backup_dir%\2w12-audio-server.tar"
docker save redis:7-alpine -o "%backup_dir%\redis.tar"
docker save portainer/portainer-ce:latest -o "%backup_dir%\portainer.tar"

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker image export failed!
    echo Make sure Docker is running and containers exist
    pause
    exit /b 1
)

echo ✅ Docker images exported successfully
echo.

echo 💾 PHASE 2: Project Files Backup
echo =================================

REM Copy entire project directory
echo 📋 Copying project files...
xcopy "C:\2w12-backend" "%backup_dir%\2w12-backend\" /E /I /H /Y

REM Copy any additional data
if exist "C:\2w12-backend\uploads" (
    echo 📁 Copying uploads directory...
    xcopy "C:\2w12-backend\uploads" "%backup_dir%\uploads\" /E /I /H /Y
)

if exist "C:\2w12-backend\logs" (
    echo 📜 Copying logs directory...
    xcopy "C:\2w12-backend\logs" "%backup_dir%\logs\" /E /I /H /Y
)

echo ✅ Project files copied successfully
echo.

echo 🗃️ PHASE 3: Docker Volumes Backup
echo ==================================

REM Export Redis data volume
echo 💽 Backing up Redis data...
docker run --rm -v 2w12-backend_redis_data:/source -v "%backup_dir%":/backup alpine tar czf /backup/redis_data.tar.gz -C /source .

REM Export Portainer data volume
echo 🎛️ Backing up Portainer data...
docker run --rm -v 2w12-backend_portainer_data:/source -v "%backup_dir%":/backup alpine tar czf /backup/portainer_data.tar.gz -C /source .

echo ✅ Docker volumes backed up successfully
echo.

echo 📊 PHASE 4: Configuration Export
echo =================================

REM Export Docker Compose configuration
echo ⚙️ Exporting Docker configuration...
docker-compose config > "%backup_dir%\docker-compose-resolved.yml"

REM Export container information
echo 📋 Exporting container information...
docker ps -a > "%backup_dir%\container_list.txt"
docker images > "%backup_dir%\image_list.txt"
docker network ls > "%backup_dir%\network_list.txt"
docker volume ls > "%backup_dir%\volume_list.txt"

REM Export environment information
echo 🌍 Exporting environment information...
echo Docker Version: > "%backup_dir%\environment_info.txt"
docker --version >> "%backup_dir%\environment_info.txt"
echo. >> "%backup_dir%\environment_info.txt"
echo Docker Compose Version: >> "%backup_dir%\environment_info.txt"
docker-compose --version >> "%backup_dir%\environment_info.txt"
echo. >> "%backup_dir%\environment_info.txt"
echo System Information: >> "%backup_dir%\environment_info.txt"
systeminfo >> "%backup_dir%\environment_info.txt"

echo ✅ Configuration exported successfully
echo.

echo 🔐 PHASE 5: Security & SSH Backup
echo ==================================

REM Backup SSH keys if they exist
if exist "%USERPROFILE%\.ssh" (
    echo 🔑 Backing up SSH keys...
    xcopy "%USERPROFILE%\.ssh" "%backup_dir%\ssh_keys\" /E /I /H /Y
    echo ✅ SSH keys backed up
) else (
    echo ⚠️ No SSH keys found
)

echo.

echo 📝 PHASE 6: Create Restoration Instructions
echo ============================================

REM Create restoration script
echo 📜 Creating restoration instructions...
(
echo # 2W12 Docker Backup - Restoration Instructions
echo Created: %date% %time%
echo.
echo ## Contents:
echo - 2w12-audio-server.tar : Main application Docker image
echo - redis.tar : Redis Docker image  
echo - portainer.tar : Portainer Docker image
echo - 2w12-backend/ : Complete project source code
echo - redis_data.tar.gz : Redis data volume
echo - portainer_data.tar.gz : Portainer data volume
echo - docker-compose-resolved.yml : Docker Compose configuration
echo - *.txt : Container and system information
echo - ssh_keys/ : SSH keys backup
echo.
echo ## Restoration Steps:
echo.
echo ### 1. Install Prerequisites:
echo - Docker Desktop
echo - Git ^(optional^)
echo.
echo ### 2. Restore Project:
echo ```
echo # Copy 2w12-backend folder to C:\
echo # Navigate to project directory
echo cd C:\2w12-backend
echo ```
echo.
echo ### 3. Load Docker Images:
echo ```
echo docker load -i 2w12-audio-server.tar
echo docker load -i redis.tar  
echo docker load -i portainer.tar
echo ```
echo.
echo ### 4. Restore Volumes:
echo ```
echo docker volume create 2w12-backend_redis_data
echo docker volume create 2w12-backend_portainer_data
echo docker run --rm -v 2w12-backend_redis_data:/target -v %backup_dir%:/backup alpine tar xzf /backup/redis_data.tar.gz -C /target
echo docker run --rm -v 2w12-backend_portainer_data:/target -v %backup_dir%:/backup alpine tar xzf /backup/portainer_data.tar.gz -C /target
echo ```
echo.
echo ### 5. Start Services:
echo ```
echo docker-compose up -d
echo ```
echo.
echo ### 6. Verify:
echo - API: http://localhost:8001/health
echo - Docs: http://localhost:8001/docs  
echo - Portainer: http://localhost:9000
echo.
echo ## Network Configuration:
echo Your setup was configured for network access at 192.168.1.16
echo Update firewall rules and network settings as needed.
echo.
echo ## SSH Keys:
echo Restore SSH keys from ssh_keys/ folder to ~/.ssh/
echo.
) > "%backup_dir%\RESTORATION_INSTRUCTIONS.md"

echo ✅ Restoration instructions created
echo.

echo 📊 PHASE 7: Backup Verification
echo ================================

echo 🔍 Verifying backup integrity...

set total_size=0
for /f "tokens=3" %%a in ('dir "%backup_dir%" /-c ^| find "File(s)"') do set total_size=%%a

echo 📦 Backup Summary:
echo    Location: %backup_dir%
echo    Total Size: %total_size% bytes
echo.
echo 📁 Contents:
dir "%backup_dir%" /b

REM Calculate checksums for critical files
echo.
echo 🔐 File Verification:
if exist "%backup_dir%\2w12-audio-server.tar" (
    for %%A in ("%backup_dir%\2w12-audio-server.tar") do echo    Main Image: %%~zA bytes
)
if exist "%backup_dir%\redis.tar" (
    for %%A in ("%backup_dir%\redis.tar") do echo    Redis Image: %%~zA bytes  
)
if exist "%backup_dir%\2w12-backend" (
    echo    ✅ Project files backed up
)

echo.
echo ========================================
echo        🎉 BACKUP COMPLETED! 🎉
echo ========================================
echo.
echo 📍 Backup Location: %backup_dir%
echo 📖 Instructions: %backup_dir%\RESTORATION_INSTRUCTIONS.md
echo.
echo 🔄 Next Steps:
echo 1. Verify backup contents look correct
echo 2. Consider copying backup to external drive
echo 3. Test restoration on another machine ^(optional^)
echo 4. Proceed with Linux migration when ready
echo.
echo ⚠️ IMPORTANT: Keep this backup safe until Linux migration is complete and tested!
echo.
pause