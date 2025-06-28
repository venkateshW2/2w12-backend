"""
HTTP streaming handler for audio files
Supports range requests for efficient seeking and progressive loading
"""

import os
import re
import logging
from typing import Optional, Tuple
from fastapi import Request, HTTPException
from fastapi.responses import StreamingResponse
import aiofiles

logger = logging.getLogger(__name__)

class StreamingHandler:
    def __init__(self):
        self.chunk_size = 8192  # 8KB chunks
        self.max_chunk_size = 1024 * 1024  # 1MB max chunk
    
    async def stream_audio_file(self, 
                               file_path: str, 
                               request: Request,
                               content_type: str = "audio/mpeg") -> StreamingResponse:
        """
        Stream audio file with support for range requests
        
        Args:
            file_path: Path to audio file
            request: FastAPI request object
            content_type: MIME type for response
            
        Returns:
            StreamingResponse: Streaming audio response
        """
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        file_size = os.path.getsize(file_path)
        range_header = request.headers.get("range")
        
        if range_header:
            # Handle range request
            return await self._handle_range_request(
                file_path, range_header, file_size, content_type
            )
        else:
            # Stream entire file
            return await self._stream_full_file(file_path, file_size, content_type)
    
    async def _handle_range_request(self, 
                                   file_path: str,
                                   range_header: str, 
                                   file_size: int,
                                   content_type: str) -> StreamingResponse:
        """Handle HTTP range requests for seeking"""
        
        # Parse range header: "bytes=start-end"
        range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
        
        if not range_match:
            raise HTTPException(status_code=400, detail="Invalid range header")
        
        start = int(range_match.group(1))
        end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
        
        # Validate range
        if start >= file_size or end >= file_size or start > end:
            raise HTTPException(
                status_code=416, 
                detail="Range not satisfiable",
                headers={"Content-Range": f"bytes */{file_size}"}
            )
        
        content_length = end - start + 1
        
        async def stream_range():
            try:
                async with aiofiles.open(file_path, "rb") as f:
                    await f.seek(start)
                    remaining = content_length
                    
                    while remaining > 0:
                        chunk_size = min(self.chunk_size, remaining)
                        chunk = await f.read(chunk_size)
                        
                        if not chunk:
                            break
                        
                        remaining -= len(chunk)
                        yield chunk
                        
            except Exception as e:
                logger.error(f"Error streaming range {start}-{end}: {e}")
                raise
        
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Type": content_type
        }
        
        return StreamingResponse(
            stream_range(),
            status_code=206,  # Partial Content
            headers=headers
        )
    
    async def _stream_full_file(self, 
                               file_path: str,
                               file_size: int, 
                               content_type: str) -> StreamingResponse:
        """Stream complete file"""
        
        async def stream_file():
            try:
                async with aiofiles.open(file_path, "rb") as f:
                    while True:
                        chunk = await f.read(self.chunk_size)
                        if not chunk:
                            break
                        yield chunk
                        
            except Exception as e:
                logger.error(f"Error streaming file {file_path}: {e}")
                raise
        
        headers = {
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Content-Type": content_type
        }
        
        return StreamingResponse(stream_file(), headers=headers)
    
    def get_content_type(self, file_path: str) -> str:
        """Determine content type from file extension"""
        
        ext = os.path.splitext(file_path)[1].lower()
        
        content_types = {
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.flac': 'audio/flac',
            '.m4a': 'audio/mp4',
            '.aac': 'audio/aac',
            '.ogg': 'audio/ogg'
        }
        
        return content_types.get(ext, 'audio/mpeg')

# Global streaming handler
streaming_handler = StreamingHandler()
