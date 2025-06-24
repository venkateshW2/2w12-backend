"""
Audio streaming API endpoints
Handles compressed audio file streaming with range request support
"""

import logging
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from datetime import timedelta

from core.file_manager import file_manager
from core.streaming_handler import streaming_handler

router = APIRouter(prefix="/api/streaming", tags=["streaming"])
logger = logging.getLogger(__name__)

@router.get("/audio/{file_id}")
async def stream_audio(file_id: str, request: Request):
    """
    Stream audio file with range request support for seeking
    """
    
    # Get file info
    file_info = await file_manager.get_file_info(file_id)
    
    if not file_info:
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    if file_info["status"] == "deleted":
        raise HTTPException(status_code=410, detail="Audio file has expired")
    
    if file_info["status"] != "ready":
        raise HTTPException(status_code=202, detail="Audio file still processing")
    
    # Update last accessed time
    await file_manager.touch_file(file_id)
    
    # Get compressed file path (fallback to original if compression failed)
    audio_path = file_info.get("compressed_path") or file_info.get("original_path")
    
    if not audio_path:
        raise HTTPException(status_code=500, detail="Audio file path not found")
    
    # Determine content type
    content_type = streaming_handler.get_content_type(audio_path)
    
    # Stream the file
    return await streaming_handler.stream_audio_file(audio_path, request, content_type)

@router.get("/info/{file_id}")
async def get_stream_info(file_id: str):
    """Get streaming information for audio file"""
    
    file_info = await file_manager.get_file_info(file_id)
    
    if not file_info:
        raise HTTPException(status_code=404, detail="File not found")
    
    return {
        "file_id": file_id,
        "status": file_info["status"],
        "stream_url": f"/api/streaming/audio/{file_id}",
        "file_size": file_info.get("compressed_size", file_info["original_size"]),
        "format": "mp3" if file_info.get("compressed_path") else "original",
        "created": file_info["created"].isoformat(),
        "expires_at": (file_info["created"] + timedelta(minutes=30)).isoformat()
    }
