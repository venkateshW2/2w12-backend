"""
Visualization API endpoints - FIXED VERSION
"""

import asyncio
import logging
import time
import os
import librosa
import uuid  # ADD THIS
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from core.file_manager import file_manager
from core.audio_compressor import audio_compressor
from core.visualization_generator import visualization_generator
from core.audio_analyzer import AudioAnalyzer  
from models.visualization_models import (
    CompleteAnalysisResponse, 
    FileStatusResponse
)
from models.streaming_models import ExtendSessionRequest, ExtendSessionResponse

router = APIRouter(prefix="/api/visualization", tags=["visualization"])
logger = logging.getLogger(__name__)

# Initialize your existing audio analyzer
audio_analyzer = AudioAnalyzer()

@router.post("/analyze-complete")  # REMOVE response_model temporarily
async def analyze_with_visualization(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Complete audio analysis with visualization data generation"""
    
    start_time = time.time()
    file_id = str(uuid.uuid4())
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Read file content
        file_content = await file.read()
        
        # Check file size (50MB limit for now, increase later)
        if len(file_content) > 150 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Max 150MB for now.")
        
        logger.info(f"Processing file: {file.filename} ({len(file_content)} bytes)")
        
        # Store file
        file_path = await file_manager.store_file(file_id, file_content, file.filename)
        # Comment out this line for now - method might not exist
        # await file_manager.update_file_status(file_id, "processing")
        
        try:
            # Get analysis from AudioAnalyzer (keeping the efficient approach)
            logger.info("Running comprehensive analysis with pre-computed features...") 
            result = audio_analyzer.comprehensive_analysis_with_features(file_path)
            analysis_result = result["analysis"]
            features = result["audio_features"]
            
            # Extract audio data for modules
            y = np.array(features["y"])
            sr = features["sr"]
            
            logger.info(f"Got analysis + features: {features['samples']} samples at {sr}Hz")
            
            # USE YOUR PROFESSIONAL VISUALIZATION GENERATOR
            logger.info("Generating visualization using VisualizationGenerator...")
            visualization_data = await visualization_generator.generate_visualization_data(
                audio_path=file_path,
                y=y,
                sr=sr
            )
            logger.info("Visualization generated using professional module")
            
            # USE YOUR AUDIO COMPRESSOR
            logger.info("Compressing audio using AudioCompressor...")
            
            # Generate compressed file path
            file_ext = os.path.splitext(file_path)[1].lower()
            compressed_path = file_path.replace(file_ext, '.mp3')
            
            # Get optimal compression settings based on file size
            file_size = os.path.getsize(file_path)
            duration = features["duration"]
            compression_settings = audio_compressor.get_compression_settings(file_size, duration)
            
            # Compress using your professional compressor
            compression_success = await audio_compressor.compress_audio(
                input_path=file_path,
                output_path=compressed_path,
                target_bitrate=compression_settings["bitrate"],
                target_sample_rate=int(compression_settings["sample_rate"])
            )
            
            if compression_success:
                await file_manager.set_compressed_path(file_id, compressed_path)
                compressed_size = os.path.getsize(compressed_path)
                format_type = "mp3"
                logger.info(f"Audio compressed successfully to {compressed_size/1024/1024:.1f}MB")
            else:
                logger.warning("Audio compression failed, will serve original file")
                compressed_size = file_size
                format_type = "original"
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return comprehensive response
            return {
                "analysis": analysis_result,
                "visualization": visualization_data,  # From professional VisualizationGenerator
                "playback": {
                    "file_id": file_id,
                    "stream_url": f"/api/streaming/audio/{file_id}",
                    "compressed_size": compressed_size,
                    "original_size": file_size,
                    "compression_ratio": round((1 - compressed_size / file_size) * 100, 1) if compression_success else 0,
                    "format": format_type
                },
                "processing_time": round(processing_time, 2)
            }
            
        except Exception as e:
            await file_manager.update_file_status(file_id, "error", error_message=str(e))
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Keep the other endpoints as they are - they look good

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
