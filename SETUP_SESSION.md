# 2W12-Backend Setup Session - July 12, 2025

## System Status
- **CUDA Version**: 12.8 (verified with nvidia-smi)
- **Available Space**: 854GB on /mnt/2w12-data, 90GB on main drive
- **Conda**: Fresh installation completed at ~/miniconda3

## Environment Setup Commands

### 1. Activate Conda and Environment
```bash
source ~/miniconda3/bin/activate
conda activate 2w12-backend
```

### 2. Install PyTorch with CUDA 12.1 Support (Compatible with CUDA 12.8)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torchaudio
```

### 3. Install TensorFlow with GPU Support
```bash
pip install tensorflow[and-cuda]
```

### 4. Install Core Audio Processing Libraries
```bash
pip install librosa soundfile ffmpeg-python pyloudnorm noisereduce soxr
```

### 5. Install FastAPI and Web Framework
```bash
pip install fastapi uvicorn python-multipart aiofiles pydantic python-dotenv
```

### 6. Install Data Science Libraries
```bash
pip install numpy==1.24.3 scipy pandas scikit-learn matplotlib seaborn
```

### 7. Install Essential Audio ML Libraries
```bash
pip install essentia-tensorflow==2.1b6.dev1110
pip install pyacoustid musicbrainzngs
```

### 8. Install Utilities
```bash
pip install requests psutil tqdm redis PyYAML anyio sniffio
```

### 9. Install Madmom (Order Matters - Install Dependencies First)
```bash
pip install cython mido
pip install git+https://github.com/CPJKU/madmom.git
```

### 10. Test GPU Installation
```bash
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
python -c "import tensorflow as tf; print(f'TensorFlow GPUs: {tf.config.list_physical_devices(\"GPU\")}')"
```

### 11. Test Audio Analysis Pipeline
```bash
cd /home/w2/2w12-backend
python main.py
```

## Project Context (from SESSION_SUMMARY.md)
- **Performance Focus**: Audio analysis with near-realtime performance (1.15x slower than realtime achieved)
- **Hybrid Approach**: Fast librosa + Madmom downbeats (27x faster than full Madmom)
- **GPU Acceleration**: Essentia ML models with TensorFlow GPU support
- **Timeline Data**: Beat times, downbeat times, transient detection
- **Streaming**: Real-time analysis with Server-Sent Events

## Key Performance Optimizations Applied
1. **Sample Rate Reduction**: 22050Hz → 11025Hz (2x speed)
2. **Hybrid Processing**: Eliminated slow librosa key detection
3. **GPU Batch Processing**: 8 chunks processed simultaneously
4. **Parallel Execution**: 3 components running concurrently
5. **Essentia Ultra-Fast**: 4000x speedup in spectral analysis

## Current Status
- **Environment**: 2w12-backend conda environment created
- **PyTorch**: Installing with CUDA 12.1 support
- **Next**: Install remaining dependencies and test pipeline

## Installation Progress
- [x] Conda environment created
- [ ] PyTorch with CUDA support
- [ ] TensorFlow with GPU support  
- [ ] Audio processing libraries
- [ ] ML dependencies (essentia, madmom)
- [ ] Test GPU functionality
- [ ] Test audio analysis pipeline

## Notes
- Using CUDA 12.1 wheels (compatible with CUDA 12.8 driver)
- Essentia-tensorflow version pinned to 2.1b6.dev1110 (working version)
- Madmom installed from GitHub (latest fixes)
- Large downloads expected (~2GB+ for PyTorch + TensorFlow)

## Recovery Commands (if session crashes)
```bash
# Reactivate environment
source ~/miniconda3/bin/activate
conda activate 2w12-backend

# Check what's already installed
pip list | grep -E "(torch|tensorflow|librosa|essentia)"

# Continue from where left off using commands above
```