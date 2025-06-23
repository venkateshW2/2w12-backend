"""
Audio compression module using FFmpeg
Handles conversion from various formats to compressed MP3 for streaming
"""

import subprocess
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

class AudioCompressor:
    def __init__(self):
        self.ffmpeg_path = "ffmpeg"  # Assume ffmpeg is in PATH
        
    async def compress_audio(self, 
                           input_path: str, 
                           output_path: str,
                           target_bitrate: str = "192k",
                           target_sample_rate: int = 44100) -> bool:
        """
        Compress audio file for efficient streaming
        
        Args:
            input_path: Path to original audio file
            output_path: Path for compressed output
            target_bitrate: Target bitrate (e.g., "192k", "128k")
            target_sample_rate: Target sample rate
            
        Returns:
            bool: Success status
        """
        
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Build FFmpeg command
            cmd = [
                self.ffmpeg_path,
                "-i", input_path,           # Input file
                "-codec:a", "libmp3lame",   # MP3 encoder
                "-b:a", target_bitrate,     # Audio bitrate
                "-ar", str(target_sample_rate),  # Sample rate
                "-ac", "2",                 # Stereo output
                "-q:a", "2",               # High quality
                "-y",                      # Overwrite output
                output_path
            ]
            
            logger.info(f"Compressing audio: {input_path} -> {output_path}")
            logger.debug(f"FFmpeg command: {' '.join(cmd)}")
            
            # Run FFmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                # Verify output file was created
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    input_size = os.path.getsize(input_path)
                    output_size = os.path.getsize(output_path)
                    compression_ratio = (1 - output_size / input_size) * 100
                    
                    logger.info(f"Compression successful: {input_size/1024/1024:.1f}MB -> "
                               f"{output_size/1024/1024:.1f}MB ({compression_ratio:.1f}% reduction)")
                    return True
                else:
                    logger.error(f"FFmpeg completed but output file is invalid")
                    return False
            else:
                logger.error(f"FFmpeg failed with return code {result.returncode}")
                logger.error(f"FFmpeg stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg compression timed out for {input_path}")
            return False
        except Exception as e:
            logger.error(f"Error compressing audio: {e}")
            return False
    
    def get_compression_settings(self, file_size: int, duration: float) -> Dict[str, str]:
        """
        Determine optimal compression settings based on file characteristics
        
        Args:
            file_size: Original file size in bytes
            duration: Audio duration in seconds
            
        Returns:
            dict: Compression settings
        """
        
        # Calculate original bitrate estimate
        estimated_bitrate = (file_size * 8) / duration / 1000  # kbps
        
        if file_size > 100_000_000:  # >100MB
            return {
                "bitrate": "128k",
                "sample_rate": "44100",
                "quality": "standard"
            }
        elif file_size > 50_000_000:  # >50MB
            return {
                "bitrate": "160k", 
                "sample_rate": "44100",
                "quality": "good"
            }
        else:  # <50MB
            return {
                "bitrate": "192k",
                "sample_rate": "44100", 
                "quality": "high"
            }
    
    async def get_audio_info(self, file_path: str) -> Optional[Dict]:
        """
        Get audio file information using FFprobe
        
        Returns:
            dict: Audio information (duration, bitrate, format, etc.)
        """
        
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                # Extract audio stream info
                audio_stream = None
                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "audio":
                        audio_stream = stream
                        break
                
                if audio_stream:
                    return {
                        "duration": float(data["format"].get("duration", 0)),
                        "bitrate": int(data["format"].get("bit_rate", 0)),
                        "sample_rate": int(audio_stream.get("sample_rate", 0)),
                        "channels": int(audio_stream.get("channels", 0)),
                        "codec": audio_stream.get("codec_name", "unknown"),
                        "format": data["format"].get("format_name", "unknown")
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            return None

# Global compressor instance
audio_compressor = AudioCompressor()
