"""
Temporary file management system for audio processing
Handles file lifecycle, cleanup, and resource monitoring
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from pathlib import Path
import shutil
import uuid

logger = logging.getLogger(__name__)

class AudioFileManager:
    def __init__(self, 
             temp_dir: str = "/tmp/audio",
             max_storage: int = 2_000_000_000,  # 2GB
             cleanup_threshold: int = 1_800_000_000):  # 1.8GB
            self.temp_dir = Path(temp_dir)
            self.max_storage = max_storage
            self.cleanup_threshold = cleanup_threshold
            self.files: Dict[str, dict] = {}
            self._cleanup_task = None
            
            # Create temp directory if it doesn't exist
            self.temp_dir.mkdir(exist_ok=True, parents=True)
            
            logger.info(f"AudioFileManager initialized: {self.temp_dir}")
            # REMOVED: Background task creation
    
    async def start_background_cleanup(self):
        """Start background cleanup task (call this after event loop is running)"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self.background_cleanup())
            logger.info("Background cleanup task started")
    
    async def store_file(self, file_id: str, file_content: bytes, 
                        original_filename: str) -> str:
        """Store uploaded file and return path"""
        
        # Start cleanup task if not already running
        if self._cleanup_task is None:
            await self.start_background_cleanup()
        
        # Generate unique file path
        file_path = self.temp_dir / f"{file_id}_{original_filename}"
        
        # Write file to disk
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Store file metadata
        self.files[file_id] = {
            "original_path": str(file_path),
            "original_filename": original_filename,
            "original_size": len(file_content),
            "created": datetime.now(),
            "last_accessed": datetime.now(),
            "status": "uploaded",
            "compressed_path": None,
            "compressed_size": None
        }
        
        logger.info(f"Stored file {file_id}: {len(file_content)} bytes")
        return str(file_path)
    
    async def get_file_info(self, file_id: str) -> Optional[dict]:
        """Get file information"""
        return self.files.get(file_id)
    
    async def update_file_status(self, file_id: str, status: str, **kwargs):
        """Update file status and metadata"""
        if file_id in self.files:
            self.files[file_id]["status"] = status
            self.files[file_id]["last_accessed"] = datetime.now()
            
            # Update additional metadata
            for key, value in kwargs.items():
                self.files[file_id][key] = value
            
            logger.debug(f"Updated file {file_id} status: {status}")
    
    async def touch_file(self, file_id: str):
        """Update last accessed time"""
        if file_id in self.files:
            self.files[file_id]["last_accessed"] = datetime.now()
    
    async def remove_file(self, file_id: str):
        """Remove file and cleanup"""
        if file_id not in self.files:
            return
        
        file_info = self.files[file_id]
        
        # Remove original file
        if file_info.get("original_path") and os.path.exists(file_info["original_path"]):
            os.remove(file_info["original_path"])
        
        # Remove compressed file
        if file_info.get("compressed_path") and os.path.exists(file_info["compressed_path"]):
            os.remove(file_info["compressed_path"])
        
        # Remove from tracking
        del self.files[file_id]
        
        logger.info(f"Removed file {file_id}")

    # Add this method to your AudioFileManager class
    async def set_compressed_path(self, file_id: str, compressed_path: str):
        """Update file info with compressed version"""
        if file_id in self.files:
            self.files[file_id]["compressed_path"] = compressed_path
            if os.path.exists(compressed_path):
                self.files[file_id]["compressed_size"] = os.path.getsize(compressed_path)
            self.files[file_id]["status"] = "ready"
            logger.info(f"Set compressed path for {file_id}: {compressed_path}")
        
    async def get_storage_usage(self) -> int:

        total_size = 0
        for file_info in self.files.values():
            total_size += file_info.get("original_size", 0)
            # FIX: Handle None values
            compressed_size = file_info.get("compressed_size")
            if compressed_size is not None:
                total_size += compressed_size
        return total_size
    
    async def cleanup_expired_files(self):
        """Remove expired files"""
        now = datetime.now()
        expired_files = []
        
        for file_id, file_info in self.files.items():
            # Check age-based expiration
            age = now - file_info["last_accessed"]
            max_age = timedelta(hours=2)  # 2 hour maximum
            
            if age > max_age:
                expired_files.append(file_id)
            elif age > timedelta(minutes=30) and file_info["status"] == "ready":
                # Remove files older than 30 minutes if ready
                expired_files.append(file_id)
        
        # Remove expired files
        for file_id in expired_files:
            await self.remove_file(file_id)
        
        if expired_files:
            logger.info(f"Cleaned up {len(expired_files)} expired files")
    
    async def cleanup_by_storage_limit(self):
        """Remove oldest files if storage limit exceeded"""
        current_usage = await self.get_storage_usage()
        
        if current_usage > self.cleanup_threshold:
            # Sort files by last accessed time (oldest first)
            sorted_files = sorted(
                self.files.items(),
                key=lambda x: x[1]["last_accessed"]
            )
            
            files_removed = 0
            for file_id, file_info in sorted_files:
                await self.remove_file(file_id)
                files_removed += 1
                
                # Check if we're under threshold
                current_usage = await self.get_storage_usage()
                if current_usage < self.cleanup_threshold:
                    break
            
            if files_removed > 0:
                logger.info(f"Cleaned up {files_removed} files due to storage limit")
    
    async def background_cleanup(self):
        """Background task for periodic cleanup"""
        while True:
            try:
                await self.cleanup_expired_files()
                await self.cleanup_by_storage_limit()
                
                # Log current status
                usage = await self.get_storage_usage()
                file_count = len(self.files)
                logger.debug(f"Storage: {usage / 1_000_000:.1f}MB, Files: {file_count}")
                
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
            
            # Run cleanup every 5 minutes
            await asyncio.sleep(300)

# Global file manager instance
file_manager = AudioFileManager()