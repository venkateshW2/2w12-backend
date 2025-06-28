# api/health.py
from fastapi import APIRouter
from models.responses import HealthResponse

# Add torch import for GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def check_gpu_availability():
    """Check if GPU is available for processing"""
    gpu_info = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_name": None,
        "gpu_memory": None
    }
    
    if not TORCH_AVAILABLE:
        gpu_info["error"] = "PyTorch not available"
        return gpu_info
    
    try:
        if torch.cuda.is_available():
            gpu_info["cuda_available"] = True
            gpu_info["gpu_count"] = torch.cuda.device_count()
            gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
            gpu_info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
            print(f"🚀 GPU Acceleration ENABLED: {gpu_info['gpu_name']} ({gpu_info['gpu_memory']})")
        else:
            print("⚠️ GPU not available, using CPU processing")
    except Exception as e:
        print(f"⚠️ GPU check failed: {e}")
        gpu_info["error"] = str(e)
    
    return gpu_info

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    # Check GPU status each time (in case it changes)
    gpu_status = check_gpu_availability()
    
    return HealthResponse(
        status="healthy",
        version="3.0.0",
        endpoints=[
            "/health",
            "/api/audio/analyze",
            "/api/audio/analyze-advanced",
            "/api/audio/classify-genre",
            "/api/audio/detect-mood",
            "/api/audio/loudness",
            "/api/audio/mfcc",
            "/api/audio/audio-tagging",
            "/api/audio/batch-analyze"
        ],
        features=[
            "Multi-algorithm key detection",
            "Genre classification", 
            "Mood detection",
            "Loudness analysis",
            "MFCC analysis",
            "Audio tagging",
            "Batch processing"
        ],
        gpu_info=gpu_status
    )