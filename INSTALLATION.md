# INSTALLATION.md - 2W12 Audio Analysis Platform

**Date**: July 12, 2025  
**Environment**: /mnt/2w12-data/2w12-backend  
**Status**: GPU Acceleration Working with Known Issues

---

## 🚀 **Current Installation Status**

### **Environment Setup - ✅ COMPLETE**
- **Project Location**: `/mnt/2w12-data/2w12-backend/` (moved from `/home/w2/`)
- **Conda Environment**: `/mnt/2w12-data/miniconda3/envs/2w12-backend` (Python 3.10.18)
- **Storage**: 854GB available on large drive
- **Root Drive**: Cleaned - 92GB free space

### **GPU Acceleration - ✅ MOSTLY WORKING**
- **NVIDIA Driver**: 570.133.07 with CUDA 12.8
- **GPU**: NVIDIA GeForce GTX 1060 Max-Q (6GB VRAM)
- **TensorFlow GPU**: 2.19.0 - **✅ WORKING** (5.5GB allocated)
- **CREPE Model**: **✅ WORKING** (0.081s load, 0.862s inference)
- **GPU Temperature**: 63°C (normal operating range)

---

## 📦 **Package Installation Status**

### **✅ Core Libraries Installed (2025 Latest Versions)**
```
fastapi==0.116.1          ✅ Latest
uvicorn==0.35.0           ✅ Latest
librosa==0.11.0           ✅ Latest (FFT backend optimized)
pandas==2.3.1            ✅ Latest
scikit-learn==1.7.0       ✅ Latest
tensorflow==2.19.0       ✅ Latest (Hermetic CUDA support)
essentia-tensorflow==2.1b6.dev1110  ✅ Installed
numpy==1.26.4             ⬆️ Can upgrade to 2.3.1
scipy==1.15.3             ⬆️ Can upgrade to 1.16.0
```

### **✅ GPU Libraries Installed**
```
nvidia-cuda-cupti-cu12==12.6.80      ✅ Installed
nvidia-cuda-nvrtc-cu12==12.6.77      ✅ Installed  
nvidia-cuda-runtime-cu12==12.6.77    ✅ Installed
nvidia-cublas-cu12==12.6.4.1         ✅ Installed
nvidia-cufft-cu12==11.3.0.4          ✅ Installed
nvidia-curand-cu12==10.3.7.77        ✅ Installed
nvidia-cusolver-cu12==11.7.1.2       ✅ Installed
nvidia-cusparse-cu12==12.5.4.2       ✅ Installed
nvidia-cudnn-cu12==9.5.1.17          ✅ Installed
```

### **✅ ML Models Status**
```
models/Crepe Large Model.pb          ✅ 51MB (Working - GPU accelerated)
models/audioset-vggish-3.pb          ✅ 1.8MB (Copied from workspace)
models/Danceability Discogs Effnet.pb ❌ cppPool error (see known issues)
models/Genre Discogs 400 Model.pb    ✅ 1.2MB (Copied from workspace)
```

---

## ⚠️ **Known Issues**

### **Critical Issue: cppPool Error**
```
Error: 'numpy.ndarray' object has no attribute 'cppPool'
Location: Danceability model inference
Impact: Danceability ML feature not working
Status: Needs investigation - likely Essentia/NumPy version compatibility
```

### **GPU Library Warnings (Non-Critical)**
```
Warning: Unable to register cuFFT/cuDNN/cuBLAS factory (duplicate registration)
Impact: Cosmetic warnings only - GPU acceleration still working
Status: Normal for TensorFlow 2.19.0 with multiple CUDA library sources
```

---

## 🧪 **GPU Performance Test Results**

### **CREPE Pitch Detection Model**
- **Load Time**: 0.081s (optimized)
- **Inference Time**: 0.862s for 16K sample (GPU accelerated)
- **GPU Memory**: 5.5GB allocated during processing
- **Status**: **✅ WORKING PERFECTLY**

### **TensorFlow GPU Detection**
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# Output: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## 🚀 **Performance Achievements**

### **From SESSION_SUMMARY.md Context:**
- **Current Performance**: 14.97s for 13s file (1.15x slower than realtime)
- **Major Breakthrough**: Nearly realtime processing achieved!
- **Hybrid Approach**: Fast librosa + Madmom downbeats (27x speedup)
- **Essentia Optimizations**: Spectral analysis 4000x faster
- **Timeline Data**: Beat times, downbeat times, transient detection working

### **GPU Acceleration Impact:**
- **CREPE Model**: Ultra-fast key detection with GPU
- **Expected Speedup**: 2-3x once cppPool issue resolved
- **Target Performance**: Sub-realtime processing (faster than audio duration)

---

## 🔧 **Installation Commands**

### **Environment Activation**
```bash
# Set PATH to use large drive conda
export PATH="/mnt/2w12-data/miniconda3/bin:$PATH"

# Source conda initialization  
source /mnt/2w12-data/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate 2w12-backend

# Navigate to project
cd /mnt/2w12-data/2w12-backend
```

### **Verify Installation**
```bash
# Check Python
which python  # Should show: /mnt/2w12-data/miniconda3/envs/2w12-backend/bin/python
python --version  # Should show: Python 3.10.18

# Check GPU
nvidia-smi  # Should show GTX 1060 with CUDA 12.8

# Test TensorFlow GPU
python -c "import tensorflow as tf; print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

---

## 🎯 **Next Steps**

### **Priority 1: Fix cppPool Error**
- Investigate Essentia/NumPy compatibility issue
- Test different Essentia versions or NumPy downgrade
- Enable danceability ML feature

### **Priority 2: Complete GPU Optimization**  
- Resolve any remaining GPU library warnings
- Optimize batch processing for maximum GPU utilization
- Achieve 2-3x speedup potential

### **Priority 3: Verify Code Recovery**
- Check if enhanced_audio_loader.py matches SESSION_SUMMARY.md
- Verify all optimizations from 10 phases are implemented
- Test end-to-end pipeline performance

---

## 📋 **Session Context**

### **Migration Success:**
✅ **Environment moved** from root drive to large drive  
✅ **GPU acceleration** working for CREPE models  
✅ **Model files** restored from workspace backup  
✅ **2025 library versions** installed with latest optimizations  

### **From CLAUDE.md:**
- **Project Goal**: Near real-time audio analysis with TuneBat-level ML capabilities
- **Current Achievement**: 1.15x slower than realtime (nearly realtime!)
- **Tech Stack**: FastAPI + GPU-accelerated ML models + hybrid processing

### **Architecture Status:**
- **Core Files**: /mnt/2w12-data/2w12-backend/core/*.py
- **API Endpoints**: /api/audio/analyze-enhanced, /api/audio/analyze-streaming
- **Models**: GPU-accelerated Essentia ML models
- **Performance**: Ready for final optimization push

---

**Installation Status**: GPU acceleration working, ready for cppPool fix  
**Next Session**: Resolve danceability model issue, verify code recovery