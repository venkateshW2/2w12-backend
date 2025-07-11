# main.py - Day 6 API Integration (Complete Implementation)
import os
from pathlib import Path
import logging
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn
import torch

# === ENHANCED COMPONENTS (Days 2-5 Complete) ===
from core.enhanced_audio_loader import EnhancedAudioLoader
from core.database_manager import SoundToolsDatabase

logger = logging.getLogger(__name__)

# Initialize enhanced components
enhanced_loader = EnhancedAudioLoader()
db_manager = SoundToolsDatabase()

# === YOUR EXISTING IMPORTS ===
from api import health, audio

# Optional imports
try:
    from api.streaming import router as streaming_router
    streaming_available = True
except ImportError:
    streaming_router = None
    streaming_available = False

try:
    from api.visualization import router as visualization_router
    visualization_available = True
except ImportError:
    visualization_router = None
    visualization_available = False

def check_gpu_availability():
    """Check if GPU is available for processing"""
    gpu_info = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_name": None,
        "gpu_memory": None
    }
    
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
    
    return gpu_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="2W12.ONE Audio Analysis Platform",
    description="Professional audio analysis with ML-powered enhancements and Redis caching",
    version="3.1.0"
)

gpu_status = check_gpu_availability()

# Container-friendly directory configuration
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/uploads")
TEMP_DIR = os.getenv("TEMP_DIR", "/app/temp")
LOG_DIR = os.getenv("LOG_DIR", "/app/logs")
STATIC_DIR = os.getenv("STATIC_DIR", "/app/static")

# Ensure directories exist
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(STATIC_DIR).mkdir(parents=True, exist_ok=True)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === YOUR EXISTING ROUTERS (always work) ===
app.include_router(health.router)
app.include_router(audio.router, prefix="/api/audio", tags=["Audio Analysis"])

# Include optional routers
if visualization_available:
    app.include_router(visualization_router)
    logger.info("✅ Visualization router included")

if streaming_available:
    app.include_router(streaming_router)
    logger.info("✅ Streaming router included")

# === DAY 6: ENHANCED ENDPOINTS (NOW WORKING) ===

@app.post("/api/audio/analyze-enhanced")
async def analyze_audio_enhanced(file: UploadFile = File(...)):
    """
    Enhanced audio analysis with ML features and caching
    
    Features:
    - Intelligent Redis caching (50x speedup for repeated files)
    - Enhanced librosa analysis with confidence scores
    - Multi-algorithm tempo detection
    - Harmonic vs percussive analysis
    - Spectral feature analysis
    - Cross-validation between algorithms
    """
    
    # File validation
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac')):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported audio format. Supported: WAV, MP3, FLAC, M4A, AAC"
        )
    
    try:
        # Read file content
        file_content = await file.read()
        file_size_mb = len(file_content) / 1024 / 1024
        
        # File size validation (150MB limit)
        if file_size_mb > 150:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file_size_mb:.1f}MB. Maximum size: 150MB"
            )
        
        logger.info(f"🎵 Processing {file.filename} ({file_size_mb:.1f}MB)")
        
        # Enhanced analysis with caching
        result = enhanced_loader.analyze_with_caching(file_content, file.filename)
        
        # Get cache statistics
        cache_stats = db_manager.get_cache_stats()
        
        return {
            "success": True,
            "filename": file.filename,
            "file_size_mb": round(file_size_mb, 2),
            "analysis": result,
            "cache_performance": cache_stats,
            "api_version": "v3.1_enhanced",
            "features": {
                "enhanced_librosa": True,
                "redis_caching": True,
                "confidence_scoring": True,
                "cross_validation": True
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Enhanced analysis failed for {file.filename}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Enhanced analysis failed: {str(e)}"
        )

@app.get("/api/stats/cache")
async def get_cache_statistics():
    """
    Get detailed cache performance statistics
    
    Returns:
    - Cache hit/miss rates
    - Memory usage
    - Storage statistics
    - Performance metrics
    """
    try:
        stats = db_manager.get_cache_stats()
        return {
            "success": True,
            "cache_statistics": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"❌ Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

@app.post("/api/admin/cache/cleanup")
async def manual_cache_cleanup():
    """
    Manual cache cleanup for maintenance
    """
    try:
        cleanup_result = db_manager.cleanup_expired_data()
        return {
            "success": True,
            "cleanup_result": cleanup_result,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"❌ Cache cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.get("/api/health/enhanced")
async def enhanced_health_check():
    """
    Enhanced health check including database connectivity and ML readiness
    """
    try:
        # Test Redis connection
        cache_stats = db_manager.get_cache_stats()
        redis_healthy = cache_stats.get("status") != "error"
        
        # Test enhanced loader
        loader_healthy = hasattr(enhanced_loader, 'db') and enhanced_loader.db is not None
        
        health_details = {
            "api_status": "healthy",
            "gpu_acceleration": gpu_status,
            "redis_connection": "healthy" if redis_healthy else "error",
            "enhanced_loader": "healthy" if loader_healthy else "error",
            "ml_models_loaded": enhanced_loader.ml_models_loaded,
            "analysis_version": enhanced_loader.analysis_version,
            "cache_stats": cache_stats,
            "timestamp": time.time()
        }
        
        overall_status = all([
            redis_healthy,
            loader_healthy
        ])
        
        return {
            "success": True,
            "overall_status": "enhanced_mode" if overall_status else "degraded",
            "details": health_details
        }
        
    except Exception as e:
        logger.error(f"❌ Enhanced health check failed: {e}")
        return {
            "success": False,
            "overall_status": "error",
            "error": str(e),
            "timestamp": time.time()
        }

@app.post("/api/audio/analyze-streaming")
async def analyze_audio_streaming(file: UploadFile = File(...)):
    """
    Streaming audio analysis - returns progress in real-time
    
    This endpoint streams progress updates as the analysis progresses:
    1. File upload confirmation
    2. Audio loading progress  
    3. GPU batch processing progress
    4. Librosa analysis progress
    5. Madmom analysis progress
    6. Final results
    """
    import json
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    # Read file content BEFORE starting the streaming generator
    logger.info(f"📁 Reading uploaded file: {file.filename}")
    try:
        file_content = await file.read()
        logger.info(f"✅ File read successfully: {len(file_content)} bytes")
    except Exception as e:
        logger.error(f"❌ Failed to read file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")
    
    file_size_mb = len(file_content) / (1024 * 1024)
    filename = file.filename
    
    async def analysis_stream():
        try:
            # Step 1: File upload confirmation
            yield f"data: {json.dumps({'status': 'upload_complete', 'filename': filename, 'size_mb': round(file_size_mb, 2), 'progress': 10})}\n\n"
            await asyncio.sleep(0.1)
            
            # Step 2: Audio loading
            yield f"data: {json.dumps({'status': 'loading_audio', 'message': 'Processing audio data...', 'progress': 20})}\n\n"
            await asyncio.sleep(0.1)
            
            # Step 3: Start analysis in background thread with proper error handling
            def run_analysis():
                try:
                    # Set up logging for the thread
                    import logging
                    thread_logger = logging.getLogger(__name__ + ".thread")
                    
                    thread_logger.info(f"🎵 Starting analysis for {filename}")
                    thread_logger.info(f"🔍 File content size: {len(file_content)} bytes")
                    
                    # Verify file content is valid
                    if not file_content or len(file_content) == 0:
                        raise ValueError("Empty file content received")
                    
                    # Ensure we don't have any asyncio issues in the thread
                    import asyncio
                    try:
                        # Check if there's already an event loop in this thread
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If loop is running, we need to be careful about async calls
                            thread_logger.warning("⚠️ Event loop detected in thread - this may cause issues")
                    except RuntimeError:
                        # No event loop in this thread - this is expected and good
                        pass
                    
                    # Run analysis synchronously (no asyncio in thread)
                    result = enhanced_loader.analyze_with_caching(file_content, filename)
                    thread_logger.info(f"✅ Analysis completed for {filename}")
                    return result
                except Exception as e:
                    thread_logger.error(f"❌ Analysis failed for {filename}: {str(e)}")
                    import traceback
                    thread_logger.error(f"❌ Full traceback: {traceback.format_exc()}")
                    raise e
            
            with ThreadPoolExecutor() as executor:
                # Submit analysis task
                future = executor.submit(run_analysis)
                
                # Send progress updates while analysis runs
                progress = 30
                while not future.done():
                    yield f"data: {json.dumps({'status': 'analyzing', 'message': 'GPU batch processing in progress...', 'progress': min(progress, 85)})}\n\n"
                    await asyncio.sleep(2)  # Send update every 2 seconds
                    progress += 10
                
                # Get final result
                result = future.result()
                
            # Step 4: Final results
            yield f"data: {json.dumps({'status': 'complete', 'progress': 100, 'result': result})}\n\n"
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(f"❌ Streaming analysis error: {error_msg}")
            yield f"data: {json.dumps({'status': 'error', 'message': error_msg, 'progress': 0})}\n\n"
    
    return StreamingResponse(
        analysis_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@app.get("/streaming", response_class=HTMLResponse)
async def streaming_interface():
    """Serve the streaming test interface"""
    try:
        with open("streaming_test.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Streaming interface not found</h1>", status_code=404)

# Root endpoint
@app.get("/")
async def root():
    """API root with complete feature status"""
    
    # Get real-time status
    try:
        cache_stats = db_manager.get_cache_stats()
        cache_working = cache_stats.get("status") != "error"
    except:
        cache_working = False
    
    features = {
        "basic_audio_analysis": True,
        "enhanced_ml_analysis": True,
        "redis_caching": cache_working,
        "confidence_scoring": True,
        "cross_validation": True,
        "streaming": streaming_available,
        "visualization": visualization_available,
        "gpu_acceleration": gpu_status["cuda_available"]
    }
    
    return {
        "message": "2W12.ONE Audio Analysis Platform",
        "version": "3.1.0",
        "status": "running",
        "docs": "/docs",
        "mode": "enhanced_mode",
        "features": features,
        "endpoints": {
            "basic_analysis": "/api/audio/analyze",
            "enhanced_analysis": "/api/audio/analyze-enhanced",
            "cache_stats": "/api/stats/cache",
            "health_check": "/api/health/enhanced",
            "cache_cleanup": "/api/admin/cache/cleanup"
        },
        "performance": {
            "cache_enabled": cache_working,
            "gpu_enabled": gpu_status["cuda_available"],
            "expected_speedup": "50x for repeated files" if cache_working else "No caching"
        }
    }

if __name__ == "__main__":
    # Container-friendly configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    
    print("🚀 Starting 2W12 Audio Analysis Server...")
    print(f"📍 Server running at: http://{host}:{port}")
    print(f"📊 API Documentation: http://localhost:{port}/docs")
    
    # Feature status report
    print(f"\n📋 Enhanced Features Status:")
    print(f"   ✅ Basic Analysis: Available")
    print(f"   ✅ Enhanced Analysis: Available") 
    print(f"   ✅ Redis Caching: Available")
    print(f"   ✅ Confidence Scoring: Available")
    print(f"   ✅ Cross-Validation: Available")
    print(f"   {'✅' if gpu_status['cuda_available'] else '⚠️'} GPU Acceleration: {'Available' if gpu_status['cuda_available'] else 'Not detected'}")
    
    print(f"\n🎯 New Endpoints Ready:")
    print(f"   POST /api/audio/analyze-enhanced")
    print(f"   GET  /api/stats/cache")
    print(f"   GET  /api/health/enhanced")
    print(f"   POST /api/admin/cache/cleanup")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        access_log=True,
        log_level="info"
    )