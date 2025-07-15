# core/content_detector.py - Content-Aware Analysis Foundation
import numpy as np
import librosa
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ContentRegion:
    """Represents a detected content region in audio"""
    start: float           # Start time in seconds
    end: float            # End time in seconds
    duration: float       # Duration in seconds
    content_type: str     # 'music', 'silence', 'speech', 'ambient', 'noise'
    energy_level: float   # RMS energy level
    spectral_complexity: float  # Measure of harmonic complexity
    confidence: float     # Detection confidence (0.0 - 1.0)
    should_analyze: bool  # Whether this region should be analyzed

class ContentDetector:
    """
    Content-aware audio analysis foundation
    
    Detects musical regions vs silence/noise/speech for targeted analysis.
    ALL subsequent analysis (chords, key, tempo, downbeats) operates only on musical regions.
    """
    
    def __init__(self, min_duration: float = 1.0):
        self.min_duration = min_duration  # Minimum region duration (1 second)
        self.silence_threshold = -40      # dB threshold for silence detection
        self.energy_threshold = 0.005     # Lower RMS energy threshold for music (more inclusive)
        self.spectral_threshold = 0.05    # Lower spectral complexity threshold (more inclusive)
        
        logger.info(f"🔊 ContentDetector initialized with {min_duration}s minimum duration (simplified mode)")
    
    def detect_content_regions(self, audio_data: np.ndarray, sr: int) -> List[ContentRegion]:
        """
        Main content detection pipeline
        
        Returns list of content regions with only musical regions marked for analysis
        """
        try:
            logger.info(f"🔍 Starting content detection for {len(audio_data)/sr:.1f}s audio")
            
            # Step 1: Silence detection
            silence_regions = self._detect_silence_regions(audio_data, sr)
            logger.info(f"🔇 Detected {len(silence_regions)} silence regions")
            
            # Step 2: Convert silence to audio regions
            audio_regions = self._invert_silence_to_audio_regions(silence_regions, len(audio_data)/sr)
            logger.info(f"🎵 Found {len(audio_regions)} potential audio regions")
            
            # Step 3: Classify each audio region
            classified_regions = []
            for region in audio_regions:
                # Always classify regions, regardless of duration
                content_region = self._classify_audio_region(audio_data, sr, region)
                classified_regions.append(content_region)
            
            # Step 4: Add silence regions for completeness
            for silence in silence_regions:
                classified_regions.append(ContentRegion(
                    start=silence['start'],
                    end=silence['end'],
                    duration=silence['duration'], 
                    content_type='silence',
                    energy_level=0.0,
                    spectral_complexity=0.0,
                    confidence=1.0,
                    should_analyze=False
                ))
            
            # Sort by start time
            classified_regions.sort(key=lambda x: x.start)
            
            # Log summary
            sound_regions = [r for r in classified_regions if r.should_analyze]
            total_sound_duration = sum(r.duration for r in sound_regions)
            total_duration = len(audio_data) / sr
            
            logger.info(f"✅ Content detection complete:")
            logger.info(f"   📊 Total regions: {len(classified_regions)}")
            logger.info(f"   🔊 Sound regions: {len(sound_regions)} ({total_sound_duration:.1f}s)")
            logger.info(f"   ⚡ Analysis coverage: {total_sound_duration/total_duration*100:.1f}% of file will be analyzed")
            
            return classified_regions
            
        except Exception as e:
            logger.error(f"❌ Content detection failed: {e}")
            # Fallback: treat entire file as one sound region
            return [ContentRegion(
                start=0.0,
                end=len(audio_data)/sr,
                duration=len(audio_data)/sr,
                content_type='sound',
                energy_level=0.5,
                spectral_complexity=0.5, 
                confidence=0.3,
                should_analyze=True
            )]
    
    def _detect_silence_regions(self, audio_data: np.ndarray, sr: int) -> List[Dict[str, float]]:
        """Detect silence regions using energy-based analysis"""
        
        # Frame-based analysis
        frame_length = int(0.1 * sr)  # 100ms frames
        hop_length = frame_length // 2
        
        # Calculate RMS energy for each frame
        rms_energy = []
        times = []
        
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i:i + frame_length]
            rms = np.sqrt(np.mean(frame**2))
            
            # Convert to dB
            rms_db = 20 * np.log10(rms + 1e-10)  # Add small value to avoid log(0)
            
            rms_energy.append(rms_db)
            times.append(i / sr)
        
        # Find silence frames
        silence_frames = np.array(rms_energy) < self.silence_threshold
        
        # Convert to silence regions
        silence_regions = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silence_frames):
            if is_silent and not in_silence:
                # Start of silence
                silence_start = times[i]
                in_silence = True
            elif not is_silent and in_silence:
                # End of silence
                silence_end = times[i]
                duration = silence_end - silence_start
                
                if duration >= 25.0:  # Only count silences >= 25 seconds
                    silence_regions.append({
                        'start': silence_start,
                        'end': silence_end,
                        'duration': duration
                    })
                
                in_silence = False
        
        # Handle silence at end of file
        if in_silence:
            silence_end = len(audio_data) / sr
            duration = silence_end - silence_start
            if duration >= 25.0:
                silence_regions.append({
                    'start': silence_start,
                    'end': silence_end,
                    'duration': duration
                })
        
        return silence_regions
    
    def _invert_silence_to_audio_regions(self, silence_regions: List[Dict[str, float]], 
                                       total_duration: float) -> List[Dict[str, float]]:
        """Convert silence regions to audio regions"""
        
        if not silence_regions:
            # No silence detected - entire file is audio
            return [{'start': 0.0, 'end': total_duration, 'duration': total_duration}]
        
        audio_regions = []
        current_time = 0.0
        
        for silence in silence_regions:
            # Audio region before this silence
            if current_time < silence['start']:
                duration = silence['start'] - current_time
                audio_regions.append({
                    'start': current_time,
                    'end': silence['start'],
                    'duration': duration
                })
            
            current_time = silence['end']
        
        # Final audio region after last silence
        if current_time < total_duration:
            duration = total_duration - current_time
            audio_regions.append({
                'start': current_time,
                'end': total_duration,
                'duration': duration
            })
        
        return audio_regions
    
    def _classify_audio_region(self, audio_data: np.ndarray, sr: int, 
                             region: Dict[str, float]) -> ContentRegion:
        """Classify an audio region as music, speech, noise, etc."""
        
        # Extract region audio
        start_sample = int(region['start'] * sr)
        end_sample = int(region['end'] * sr)
        region_audio = audio_data[start_sample:end_sample]
        
        # Calculate features
        energy_level = self._calculate_energy_level(region_audio)
        spectral_complexity = self._calculate_spectral_complexity(region_audio, sr)
        
        # Debug logging
        logger.info(f"🔍 Region {region['start']:.1f}s-{region['end']:.1f}s: energy={energy_level:.6f}, complexity={spectral_complexity:.6f}")
        
        # Classification logic
        content_type, confidence, should_analyze = self._classify_content_type(
            energy_level, spectral_complexity, region['duration']
        )
        
        return ContentRegion(
            start=region['start'],
            end=region['end'],
            duration=region['duration'],
            content_type=content_type,
            energy_level=energy_level,
            spectral_complexity=spectral_complexity,
            confidence=confidence,
            should_analyze=should_analyze
        )
    
    def _calculate_energy_level(self, audio: np.ndarray) -> float:
        """Calculate normalized RMS energy level"""
        if len(audio) == 0:
            return 0.0
        
        rms = np.sqrt(np.mean(audio**2))
        return float(rms)
    
    def _calculate_spectral_complexity(self, audio: np.ndarray, sr: int) -> float:
        """Calculate spectral complexity as measure of harmonic content"""
        if len(audio) < 1024:
            return 0.0
        
        # Calculate spectral centroid and spread
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        
        # Normalize complexity metric
        complexity = np.mean(spectral_bandwidth) / (np.mean(spectral_centroid) + 1e-10)
        return float(np.clip(complexity / 1000, 0, 1))  # Normalize to 0-1
    
    def _classify_content_type(self, energy: float, complexity: float, 
                             duration: float) -> Tuple[str, float, bool]:
        """Simple classification - just silence vs sound regions"""
        
        # SIMPLIFIED: Only distinguish silence vs sound
        # All non-silence regions are analyzed regardless of content type
        if energy < self.energy_threshold:
            return 'silence', 0.9, False
        else:
            # Any sound with energy above threshold = analyze it
            return 'sound', 0.9, True
    
    def get_sound_regions_only(self, content_regions: List[ContentRegion]) -> List[ContentRegion]:
        """Filter to only sound regions that should be analyzed"""
        return [region for region in content_regions if region.should_analyze]
    
    def calculate_analysis_efficiency(self, content_regions: List[ContentRegion], 
                                    total_duration: float) -> Dict[str, float]:
        """Calculate how much processing time we save with content awareness"""
        
        sound_regions = self.get_sound_regions_only(content_regions)
        sound_duration = sum(region.duration for region in sound_regions)
        
        return {
            'total_duration': total_duration,
            'sound_duration': sound_duration,
            'coverage_percentage': (sound_duration / total_duration) * 100,
            'silence_percentage': ((total_duration - sound_duration) / total_duration) * 100,
            'regions_analyzed': len(sound_regions),
            'regions_skipped': len(content_regions) - len(sound_regions)
        }