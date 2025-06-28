"""
Audio visualization data generator
Creates waveform and spectrogram data for frontend display
"""

import numpy as np
import librosa
import logging
from typing import Dict, List, Tuple, Optional
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)

class VisualizationGenerator:
    def __init__(self):
        self.default_overview_points = 2000
        self.default_detailed_points = 8000
        self.spectrogram_height = 256
        self.spectrogram_width = 1000
    
    async def generate_visualization_data(self, 
                                        audio_path: str,
                                        y: np.ndarray = None,
                                        sr: int = None) -> Dict:
        """
        Generate complete visualization data for audio file
        
        Args:
            audio_path: Path to audio file
            y: Pre-loaded audio data (optional)
            sr: Sample rate (optional)
            
        Returns:
            dict: Complete visualization data
        """
        
        try:
            # Load audio if not provided
            if y is None or sr is None:
                y, sr = librosa.load(audio_path, sr=None)
            
            logger.info(f"Generating visualization for {len(y)/sr:.1f}s audio")
            
            # Generate waveform data
            waveform_overview = self._generate_waveform(y, self.default_overview_points)
            waveform_detailed = self._generate_waveform(y, self.default_detailed_points)
            
            # Generate spectrogram data
            spectrogram_data = self._generate_spectrogram(y, sr)
            
            # Calculate duration and timing
            duration = len(y) / sr
            time_overview = np.linspace(0, duration, len(waveform_overview))
            time_detailed = np.linspace(0, duration, len(waveform_detailed))
            
            return {
                "waveform": {
                    "overview": {
                        "data": waveform_overview.tolist(),
                        "time": time_overview.tolist(),
                        "points": len(waveform_overview)
                    },
                    "detailed": {
                        "data": waveform_detailed.tolist(), 
                        "time": time_detailed.tolist(),
                        "points": len(waveform_detailed)
                    }
                },
                "spectrogram": spectrogram_data,
                "metadata": {
                    "duration": float(duration),
                    "sample_rate": int(sr),
                    "samples": int(len(y)),
                    "channels": 1 if y.ndim == 1 else y.shape[0]
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating visualization data: {e}")
            raise
    
    def _generate_waveform(self, y: np.ndarray, target_points: int) -> np.ndarray:
        """
        Generate downsampled waveform data for display
        
        Args:
            y: Audio signal
            target_points: Number of points for output
            
        Returns:
            np.ndarray: Downsampled amplitude data
        """
        
        if len(y) <= target_points:
            return y
        
        # Calculate chunk size for downsampling
        chunk_size = len(y) // target_points
        
        # Reshape and take RMS of chunks for better representation
        num_complete_chunks = (len(y) // chunk_size) * chunk_size
        y_chunked = y[:num_complete_chunks].reshape(-1, chunk_size)
        
        # Calculate RMS for each chunk
        waveform_data = np.sqrt(np.mean(y_chunked ** 2, axis=1))
        
        # Handle any remaining samples
        if len(y) > num_complete_chunks:
            remaining = y[num_complete_chunks:]
            remaining_rms = np.sqrt(np.mean(remaining ** 2))
            waveform_data = np.append(waveform_data, remaining_rms)
        
        # Ensure we have exactly target_points
        if len(waveform_data) > target_points:
            waveform_data = waveform_data[:target_points]
        elif len(waveform_data) < target_points:
            # Interpolate to exact target
            x_old = np.linspace(0, 1, len(waveform_data))
            x_new = np.linspace(0, 1, target_points)
            f = np.interp(x_new, x_old, waveform_data)
            waveform_data = f
        
        return waveform_data
    
    def _generate_spectrogram(self, y: np.ndarray, sr: int) -> Dict:
        """
        Generate compressed spectrogram data for visualization
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            dict: Spectrogram data with frequencies and time bins
        """
        
        try:
            # Compute STFT
            stft = librosa.stft(y, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            
            # Convert to dB
            magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
            
            # Compress frequency dimension (take only first N bins)
            max_freq_bins = min(self.spectrogram_height, magnitude_db.shape[0])
            magnitude_compressed = magnitude_db[:max_freq_bins, :]
            
            # Compress time dimension if needed
            if magnitude_compressed.shape[1] > self.spectrogram_width:
                # Downsample time axis
                time_step = magnitude_compressed.shape[1] // self.spectrogram_width
                magnitude_compressed = magnitude_compressed[:, ::time_step]
                magnitude_compressed = magnitude_compressed[:, :self.spectrogram_width]
            
            # Normalize to 0-255 range for efficient transfer
            normalized = ((magnitude_compressed - magnitude_compressed.min()) / 
                         (magnitude_compressed.max() - magnitude_compressed.min()) * 255)
            normalized = normalized.astype(np.uint8)
            
            # Create frequency and time axes
            frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)[:max_freq_bins]
            time_frames = librosa.frames_to_time(
                np.arange(normalized.shape[1]), 
                sr=sr, 
                hop_length=512
            )
            
            return {
                "data": normalized.tolist(),
                "frequencies": frequencies.tolist(),
                "time_bins": time_frames.tolist(),
                "shape": list(normalized.shape),
                "db_range": [float(magnitude_db.min()), float(magnitude_db.max())]
            }
            
        except Exception as e:
            logger.error(f"Error generating spectrogram: {e}")
            # Return empty spectrogram on error
            return {
                "data": [],
                "frequencies": [],
                "time_bins": [],
                "shape": [0, 0],
                "db_range": [0, 0]
            }
    
    def create_waveform_image(self, waveform_data: np.ndarray, 
                             width: int = 800, height: int = 200) -> str:
        """
        Create a PNG image of waveform for quick preview
        
        Returns:
            str: Base64 encoded PNG image
        """
        
        try:
            # Create image
            img = Image.new('RGB', (width, height), 'white')
            pixels = img.load()
            
            # Normalize waveform data
            normalized = (waveform_data - waveform_data.min()) / (waveform_data.max() - waveform_data.min())
            normalized = normalized * height
            
            # Draw waveform
            for x in range(min(width, len(normalized))):
                y_val = int(height - normalized[x])
                y_val = max(0, min(height-1, y_val))
                
                # Draw vertical line from center to amplitude
                center_y = height // 2
                start_y = min(center_y, y_val)
                end_y = max(center_y, y_val)
                
                for y in range(start_y, end_y + 1):
                    pixels[x, y] = (0, 100, 200)  # Blue waveform
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_data = buffer.getvalue()
            
            return base64.b64encode(img_data).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error creating waveform image: {e}")
            return ""

# Global visualization generator
visualization_generator = VisualizationGenerator()
