#!/bin/bash

echo "========================================"
echo "   2W12 Smart Docker Backup Script"
echo "========================================"
echo

# Create backup directory with timestamp
backup_date=$(date +"%Y%m%d_%H%M%S")
backup_dir="/mnt/c/2w12-backup-${backup_date}"
mkdir -p "$backup_dir"

echo "📁 Creating backup directory: $backup_dir"
echo

# Change to project directory
cd /mnt/c/2w12-backend || {
    echo "❌ Cannot find project directory /mnt/c/2w12-backend"
    echo "Please update the path in this script"
    exit 1
}

echo "🔍 DISCOVERY PHASE: Finding your Docker setup"
echo "=============================================="

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running or not accessible!"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

# Discover running containers
echo "📦 Discovering containers..."
docker ps -a --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" > "$backup_dir/containers_discovered.txt"
echo "Current containers:"
cat "$backup_dir/containers_discovered.txt"
echo

# Get docker-compose project name and services
if [ -f "docker-compose.yml" ]; then
    echo "🔍 Analyzing docker-compose.yml..."
    
    # Get project name (directory name by default)
    project_name=$(basename "$(pwd)")
    echo "Project name: $project_name"
    
    # Find service names
    services=$(docker-compose config --services 2>/dev/null)
    echo "Services found: $services"
    echo
    
    # Try to find images using docker-compose
    echo "📋 Getting Docker Compose image information..."
    docker-compose config > "$backup_dir/docker-compose-config.yml" 2>/dev/null
    
else
    echo "⚠️ No docker-compose.yml found in current directory"
fi

echo "🐳 PHASE 1: Smart Docker Image Backup"
echo "====================================="

# Method 1: Export images by container names
echo "📦 Attempting to export images by container names..."

containers=$(docker ps -a --format "{{.Names}}" | grep -E "(2w12|backend|audio|server)")
for container in $containers; do
    if [ ! -z "$container" ]; then
        echo "Found container: $container"
        image=$(docker inspect --format='{{.Config.Image}}' "$container" 2>/dev/null)
        if [ ! -z "$image" ]; then
            echo "  Image: $image"
            safe_name=$(echo "$container" | tr '/' '_')
            docker save "$image" -o "$backup_dir/${safe_name}.tar" && {
                echo "  ✅ Exported: ${safe_name}.tar"
            } || {
                echo "  ❌ Failed to export: $container"
            }
        fi
    fi
done

# Method 2: Export common images
echo
echo "📦 Exporting common images..."

common_images=("redis:7-alpine" "redis" "portainer/portainer-ce" "portainer/portainer-ce:latest")
for image in "${common_images[@]}"; do
    if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${image}$"; then
        safe_name=$(echo "$image" | tr '/:' '_')
        echo "Exporting: $image -> ${safe_name}.tar"
        docker save "$image" -o "$backup_dir/${safe_name}.tar" && {
            echo "  ✅ Success"
        } || {
            echo "  ⚠️ Failed: $image"
        }
    else
        echo "⏭️ Not found: $image"
    fi
done

# Method 3: Export all images with '2w12' in the name
echo
echo "📦 Searching for 2w12-related images..."
docker images --format "{{.Repository}}:{{.Tag}}" | grep -i "2w12" | while read image; do
    if [ ! -z "$image" ]; then
        safe_name=$(echo "$image" | tr '/:' '_')
        echo "Found 2w12 image: $image"
        docker save "$image" -o "$backup_dir/${safe_name}.tar" && {
            echo "  ✅ Exported: ${safe_name}.tar"
        } || {
            echo "  ❌ Failed: $image"
        }
    fi
done

# Save list of all images for reference
docker images > "$backup_dir/all_images.txt"

echo "✅ Docker image export phase completed"
echo

echo "💾 PHASE 2: Project Files Backup"
echo "================================="

# Copy entire project directory
echo "📋 Copying project files..."
cp -r "/mnt/c/2w12-backend" "$backup_dir/" || {
    echo "❌ Failed to copy project files!"
    exit 1
}

echo "✅ Project files copied successfully"
echo

echo "🗃️ PHASE 3: Docker Volumes Backup"
echo "=================================="

# Get all volumes
echo "💽 Discovering volumes..."
docker volume ls > "$backup_dir/volumes_list.txt"
echo "Volumes found:"
cat "$backup_dir/volumes_list.txt"
echo

# Try to backup volumes that contain '2w12', 'redis', or 'portainer'
docker volume ls --format "{{.Name}}" | while read volume; do
    if echo "$volume" | grep -qE "(2w12|redis|portainer)"; then
        echo "Backing up volume: $volume"
        safe_name=$(echo "$volume" | tr '/' '_')
        docker run --rm -v "$volume":/source -v "$backup_dir":/backup alpine tar czf "/backup/volume_${safe_name}.tar.gz" -C /source . 2>/dev/null && {
            echo "  ✅ Volume backed up: volume_${safe_name}.tar.gz"
        } || {
            echo "  ⚠️ Failed to backup volume: $volume"
        }
    fi
done

echo "✅ Volume backup phase completed"
echo

echo "📊 PHASE 4: Configuration Export"
echo "================================="

# Export Docker Compose configuration
echo "⚙️ Exporting configurations..."
if [ -f "docker-compose.yml" ]; then
    docker-compose config > "$backup_dir/docker-compose-resolved.yml" 2>/dev/null || {
        echo "⚠️ Docker Compose config export failed"
    }
fi

# Export container information
docker ps -a > "$backup_dir/container_list.txt"
docker images > "$backup_dir/image_list.txt"
docker network ls > "$backup_dir/network_list.txt"
docker volume ls > "$backup_dir/volume_list.txt"

# Export environment information
{
    echo "=== DOCKER ENVIRONMENT ==="
    echo "Docker Version:"
    docker --version
    echo
    echo "Docker Compose Version:"
    docker-compose --version
    echo
    echo "=== SYSTEM INFORMATION ==="
    uname -a
    echo
    if [ -f /etc/os-release ]; then
        echo "Linux Distribution:"
        cat /etc/os-release
    fi
    echo
    echo "=== DISK SPACE ==="
    df -h
    echo
    echo "=== DOCKER SYSTEM INFO ==="
    docker system df
} > "$backup_dir/environment_info.txt"

echo "✅ Configuration exported successfully"
echo

echo "📝 PHASE 5: Create Restoration Instructions"
echo "============================================"

# Create comprehensive restoration instructions
cat > "$backup_dir/RESTORATION_INSTRUCTIONS.md" << 'EOF'
# 2W12 Docker Backup - Smart Restoration Guide

This backup was created automatically by detecting your actual Docker setup.

## Backup Contents Analysis

Check these files to understand what was backed up:
- `containers_discovered.txt` - All containers found
- `all_images.txt` - All Docker images
- `volumes_list.txt` - All Docker volumes
- `*.tar` files - Exported Docker images
- `volume_*.tar.gz` - Exported volume data
- `2w12-backend/` - Complete project source

## Quick Restoration (Recommended)

### 1. Restore Project Files
```bash
# Copy project to desired location
sudo cp -r 2w12-backend /opt/
cd /opt/2w12-backend
```

### 2. Load All Docker Images
```bash
# Load all .tar files in backup directory
for tarfile in *.tar; do
    echo "Loading $tarfile..."
    docker load -i "$tarfile"
done
```

### 3. Restore Volumes
```bash
# Create and restore volumes
for volume_file in volume_*.tar.gz; do
    if [ -f "$volume_file" ]; then
        volume_name=$(echo "$volume_file" | sed 's/volume_//' | sed 's/.tar.gz//')
        echo "Restoring volume: $volume_name"
        docker volume create "$volume_name"
        docker run --rm -v "$volume_name":/target -v $(pwd):/backup alpine tar xzf "/backup/$volume_file" -C /target
    fi
done
```

### 4. Start Services
```bash
# Try to start with docker-compose
docker-compose up -d

# If that fails, check the resolved config:
cat docker-compose-resolved.yml
```

## Manual Restoration

If automatic restoration doesn't work:

1. **Check what images were backed up:**
   ```bash
   ls -la *.tar
   ```

2. **Load specific images:**
   ```bash
   docker load -i <specific-image>.tar
   docker images  # Verify
   ```

3. **Check original container configuration:**
   ```bash
   cat containers_discovered.txt
   cat container_list.txt
   ```

4. **Manually recreate containers if needed:**
   Use the information in the .txt files to recreate containers with correct names and settings.

## Verification

After restoration:
- Check: `docker ps`
- Test API: `curl http://localhost:8001/health`
- Test web interface: `http://localhost:8001/docs`

## Troubleshooting

- If containers don't start: Check `docker logs <container-name>`
- If ports conflict: Update docker-compose.yml port mappings
- If volumes missing: Check `docker volume ls` and restore manually

EOF

echo "✅ Restoration instructions created"
echo

echo "📊 PHASE 6: Backup Summary"
echo "=========================="

# Calculate total backup size
total_size=$(du -sh "$backup_dir" | cut -f1)

echo "📦 Backup Summary:"
echo "    Location: $backup_dir"
echo "    Total Size: $total_size"
echo

echo "📁 Backup Contents:"
ls -la "$backup_dir"

echo
echo "🔍 Docker Images Backed Up:"
find "$backup_dir" -name "*.tar" -exec ls -lh {} \; | while read line; do
    echo "    $line"
done

echo
echo "💾 Volumes Backed Up:"
find "$backup_dir" -name "volume_*.tar.gz" -exec ls -lh {} \; | while read line; do
    echo "    $line"
done

echo
echo "========================================"
echo "        🎉 SMART BACKUP COMPLETED! 🎉"
echo "========================================"
echo
echo "📍 Backup Location: $backup_dir"
echo "📍 Windows Path: $(echo "$backup_dir" | sed 's|/mnt/c|C:|')"
echo "📖 Instructions: $backup_dir/RESTORATION_INSTRUCTIONS.md"
echo
echo "🔍 Review these files to understand what was backed up:"
echo "  - containers_discovered.txt"
echo "  - all_images.txt"
echo "  - volumes_list.txt"
echo
echo "⚠️ If some exports failed, that's normal - the script backs up everything it can find."
echo

# Make the backup directory accessible from Windows
chmod -R 755 "$backup_dir"