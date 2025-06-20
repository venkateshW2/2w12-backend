# api/health.py
from fastapi import APIRouter
from models.responses import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
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
        ]
    )