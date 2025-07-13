# 2W12-Backend Setup Session - CORRECTED Storage Strategy

## Storage Analysis
- **Root Drive** (`/dev/sdb2`): 116GB, 79GB available (29% used) - **AVOID FOR LARGE INSTALLS**
- **Large Drive** (`/mnt/2w12-data`): 916GB, 854GB available (2% used) - **USE THIS**

## Current Conda Status
- **Small conda**: `/home/w2/miniconda3` (on root drive) - Delete this
- **Large conda**: `/mnt/2w12-data/miniconda3` (on large drive) - Use this
- **Problem**: PATH prioritizes small conda

## SOLUTION: Use Large Drive Conda

### Step 1: Remove Small Conda Installation
```bash
rm -rf ~/miniconda3
rm ~/miniconda_installer.sh
```

### Step 2: Update PATH to Use Large Drive Conda
```bash
# Remove old conda paths and add large drive conda
export PATH="/mnt/2w12-data/miniconda3/bin:$PATH"
echo 'export PATH="/mnt/2w12-data/miniconda3/bin:$PATH"' >> ~/.bashrc
```

### Step 3: Create Environment on Large Drive
```bash
source /mnt/2w12-data/miniconda3/bin/activate
conda create -n 2w12-backend python=3.10 -y
conda activate 2w12-backend
```

### Step 4: Install All Packages (Environment on Large Drive)
```bash
# PyTorch with CUDA 12.1 (compatible with CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torchaudio

# TensorFlow with GPU support
pip install tensorflow[and-cuda]

# Audio processing
pip install librosa soundfile ffmpeg-python pyloudnorm noisereduce soxr

# FastAPI framework
pip install fastapi uvicorn python-multipart aiofiles pydantic python-dotenv

# Data science
pip install numpy==1.24.3 scipy pandas scikit-learn matplotlib seaborn

# ML libraries
pip install essentia-tensorflow==2.1b6.dev1110
pip install pyacoustid musicbrainzngs

# Utilities
pip install requests psutil tqdm redis PyYAML anyio sniffio

# Madmom (order matters)
pip install cython mido
pip install git+https://github.com/CPJKU/madmom.git
```

### Step 5: Verify Installation Location
```bash
conda info --envs
# Should show: 2w12-backend    /mnt/2w12-data/miniconda3/envs/2w12-backend
```

## Why This Solution is Best
1. **Space**: 854GB available vs 79GB on root
2. **Performance**: Data drive has more space for caching
3. **Safety**: Won't fill up root drive
4. **Existing**: You already have conda properly installed there

## Environment Size Estimate
- PyTorch: ~2.5GB
- TensorFlow: ~1.5GB  
- Audio libraries: ~1GB
- Other packages: ~1GB
- **Total**: ~6GB (easily fits on large drive)

## Quick Recovery Commands
```bash
# If session crashes, use these to resume:
export PATH="/mnt/2w12-data/miniconda3/bin:$PATH"
source /mnt/2w12-data/miniconda3/bin/activate
conda activate 2w12-backend
# Continue installations where you left off
```