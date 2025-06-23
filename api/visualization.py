"""
Visualization API endpoints
Handles audio analysis with visualization data generation
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from core.file_manager import file_manager
from core.audio_compressor import audio_compressor
from core.visualization_generator import visualization_generator
from core.audio_analyzer import AudioAnalyzer  # Your existing analyzer
from models.visualization_models import (
    CompleteAnalysisResponse, 
    FileStatusResponse
)
from models.streaming_models import ExtendSessionRequest, ExtendSessionResponse

router = APIRouter(prefix="/api/visualization", tags=["visualization"])
logger = logging.getLogger(__name__)

# Initialize your existing audio analyzer
audio_analyzer = AudioAnalyzer()

@router.post("/analyze-complete", response_model=CompleteAnalysisResponse)
async def analyze_with_visualization(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Complete audio analysis with visualization data generation
    Combines your existing analysis pipeline with new visualization features
    """
    
    start_time = time.time()
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file size (150MB limit)
        file_content = await file.read()
        if len(file_content) > 150 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 150MB)")
        
        # Store original file
        file_id = await file_manager.store_file(
            file_id=f"upload_{int(time.time())}_{file.filename}",
            file_content=file_content,
            original_filename=file.filename
        )
        
        logger.info(f"Processing file {file_id}: {len(file_content)/1024/1024:.1f}MB")
        
        # Get file path
        file_info = await file_manager.get_file_info(file_id)
        if not file_info:
            raise HTTPException(status_code=500, detail="Failed to store file")
        
        original_path = file_info["original_path"]
        
        # Load audio for analysis
        import librosa
        y, sr = librosa.load(original_path, sr=None)
        
        # Run your existing analysis pipeline
        logger.info("Running audio analysis...")
        analysis_results = {}
        
        # Basic analysis
        basic_analysis = audio_analyzer.analyze_basic(y, sr)
        analysis_results.update(basic_analysis)
        
        # Advanced analysis
        try:
            advanced_analysis = audio_analyzer.analyze_advanced(y, sr)
            analysis_results.update(advanced_analysis)
        except Exception as e:
            logger.warning(f"Advanced analysis failed: {e}")
        
        # Genre classification
        try:
            genre_result = audio_analyzer.classify_genre(y, sr)
            analysis_results["genre"] = genre_result
        except Exception as e:
            logger.warning(f"Genre classification failed: {e}")
        
        # Mood detection
        try:
            mood_result = audio_analyzer.detect_mood(y, sr)
            analysis_results["mood"] = mood_result
        except Exception as e:
            logger.warning(f"Mood detection failed: {e}")
        
        # Generate visualization data
        logger.info("Generating visualization data...")
        visualization_data = await visualization_generator.generate_visualization_data(
            original_path, y, sr
        )
        
        # Compress audio for playback
        logger.info("Compressing audio for playback...")
        compressed_path = original_path.replace("_original", "_compressed").replace(
            original_path.split('.')[-1], "mp3"
        )
        
        compression_success = await audio_compressor.compress_audio(
            original_path, compressed_path
        )
        
        if compression_success:
            await file_manager.set_compressed_path(file_id, compressed_path)
        else:
            logger.warning("Audio compression failed, will serve original file")
            compressed_path = original_path
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        response = CompleteAnalysisResponse(
            analysis=analysis_results,
            visualization=visualization_data,
            playback={
                "file_id": file_id,
                "stream_url": f"/api/streaming/audio/{file_id}",
                "compressed_size": os.path.getsize(compressed_path) if os.path.exists(compressed_path) else 0,
                "format": "mp3" if compression_success else "original",
                "expires_at": datetime.now() + timedelta(minutes=30)
            },
            processing_time=round(processing_time, 2)
        )
        
        logger.info(f"Analysis complete for {file_id} in {processing_time:.1f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_with_visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/status/{file_id}", response_model=FileStatusResponse)
async def get_file_status(file_id: str):
    """Get status of uploaded file"""
    
    file_info = await file_manager.get_file_info(file_id)
    
    if not file_info:
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileStatusResponse(
        file_id=file_id,
        status=file_info["status"],
        created=file_info["created"],
        last_accessed=file_info["last_accessed"],
        expires_at=file_info["created"] + timedelta(minutes=30),
        original_size=file_info["original_size"],
        compressed_size=file_info.get("compressed_size"),
        error_message=file_info.get("error_message")
    )

@router.post("/extend-session/{file_id}", response_model=ExtendSessionResponse)
async def extend_file_session(file_id: str, request: ExtendSessionRequest):
    """Extend file retention period"""
    
    file_info = await file_manager.get_file_info(file_id)
    
    if not file_info:
        raise HTTPException(status_code=404, detail="File not found")
    
    if file_info["status"] == "deleted":
        raise HTTPException(status_code=410, detail="File has been deleted")
    
    # Update last accessed time (extends retention)
    await file_manager.touch_file(file_id)
    
    new_expires_at = datetime.now() + timedelta(minutes=request.extend_minutes)
    
    return ExtendSessionResponse(
        file_id=file_id,
        new_expires_at=new_expires_at.isoformat(),
        extended_by_minutes=request.extend_minutes,
        success=True,
        message=f"Session extended by {request.extend_minutes} minutes"
    )

@router.delete("/cleanup/{file_id}")
async def cleanup_file(file_id: str):
    """Manually cleanup file"""
    
    await file_manager.remove_file(file_id)
    return {"message": f"File {file_id} cleaned up successfully"}
