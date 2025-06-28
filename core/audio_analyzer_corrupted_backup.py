# core/audio_analyzer.py - Enhanced version preserving all existing features
import librosa
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import tempfile
import os
import gc
import psutil
import soundfile as sf
from collections import Counter

class AudioAnalyzer:
    """Core audio analysis functionality with enhanced memory management"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        # NEW: Memory management settings
        self.max_file_size = 150 * 1024 * 1024  # 150MB limit
        self.max_duration = 600  # 10 minutes
        self.chunk_duration = 30  # Analyze first 30 seconds for long files
    
    # NEW: Memory management methods
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validate audio file before processing"""
        try:
            info = sf.info(file_path)
            file_size = os.path.getsize(file_path)
            
            if file_size > self.max_file_size:
                raise ValueError(f"File too large: {file_size/1024/1024:.1f}MB > {self.max_file_size/1024/1024}MB")
            
            if info.duration > self.max_duration:
                raise ValueError(f"File too long: {info.duration:.1f}s > {self.max_duration}s")
            
            return {
                "valid": True,
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "file_size_mb": file_size / 1024 / 1024
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def smart_load_audio(self, file_path: str) -> Tuple[np.ndarray, int, Dict]:
        """Load audio with smart memory management"""
        validation = self.validate_file(file_path)
        if not validation["valid"]:
            raise ValueError(validation["error"])
        
        duration = validation["duration"]
        original_sr = validation["sample_rate"]
        
        # Smart sample rate handling
        if original_sr > 48000:
            processing_sr = 44100
        elif original_sr >= 22050:
            processing_sr = original_sr
        else:
            processing_sr = 22050
        
        # Duration-based loading
        if duration <= 120:  # <= 2 minutes
            y, sr = librosa.load(file_path, sr=processing_sr, mono=True)
            analysis_duration = duration
        else:
            # Load first chunk only for long files
            y, sr = librosa.load(file_path, sr=processing_sr, mono=True, duration=self.chunk_duration)
            analysis_duration = self.chunk_duration
        
        file_info = {
            "original_duration": duration,
            "analyzed_duration": analysis_duration,
            "original_sample_rate": original_sr,
            "processing_sample_rate": sr,
            "file_size_mb": validation["file_size_mb"],
            "chunked": duration > 120
        }
        
        return y, sr, file_info
    
    def memory_cleanup(self, *arrays):
        """Clean up memory after processing"""
        for array in arrays:
            if array is not None:
                del array
        gc.collect()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
    
    # NEW: Enhanced tempo detection with multiple algorithms
    def multi_algorithm_tempo_detection(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """5-method tempo detection with consensus"""
        try:
            # Method 1: Librosa beat tracker
            tempo1, beats1 = librosa.beat.beat_track(y=y, sr=sr)
            
            # Method 2: Onset-based tempo
            onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
            if len(onsets) > 1:
                tempo2 = 60.0 / np.median(np.diff(onsets))
            else:
                tempo2 = 0
            
            # Method 3: Autocorrelation-based
            autocorr = librosa.autocorrelate(y)
            if len(autocorr) > sr//2:
                tempo3 = sr / (np.argmax(autocorr[sr//4:sr//2]) + sr//4) * 60
            else:
                tempo3 = 0
            
            # Method 4: Harmonic/percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            tempo4, beats4 = librosa.beat.beat_track(y=y_percussive, sr=sr)
            
            # Method 5: Spectral flux-based
            stft = librosa.stft(y)
            spectral_flux = np.sum(np.diff(np.abs(stft), axis=1)**2, axis=0)
            flux_onsets = librosa.onset.onset_detect(onset_envelope=spectral_flux, sr=sr, units='time')
            if len(flux_onsets) > 1:
                tempo5 = 60.0 / np.median(np.diff(flux_onsets))
            else:
                tempo5 = 0
            
            # Collect valid tempos (60-200 BPM range)
            tempos = [tempo1, tempo2, tempo3, tempo4, tempo5]
            method_names = ["beat_track", "onset", "autocorr", "hpss", "spectral_flux"]
            valid_tempos = [(t, name) for t, name in zip(tempos, method_names) if 60 <= t <= 200]
            
            if len(valid_tempos) >= 2:
                tempo_values = [t[0] for t in valid_tempos]
                consensus_tempo = np.median(tempo_values)
                confidence = 1.0 - (np.std(tempo_values) / np.mean(tempo_values))
            else:
                consensus_tempo = float(tempo1)
                confidence = 0.5
            
            return {
                "tempo": round(consensus_tempo, 1),
                "confidence": round(confidence, 3),
                "methods": {
                    "beat_track": round(float(tempo1), 1),
                    "onset_based": round(tempo2, 1) if tempo2 > 0 else None,
                    "autocorrelation": round(tempo3, 1) if tempo3 > 0 else None,
                    "hpss": round(float(tempo4), 1),
                    "spectral_flux": round(tempo5, 1) if tempo5 > 0 else None
                },
                "valid_methods": len(valid_tempos),
                "consensus_votes": len([t for t in tempo_values if abs(t - consensus_tempo) < 5]) if len(valid_tempos) >= 2 else 1
            }
        except Exception as e:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            return {
                "tempo": round(float(tempo), 1),
                "confidence": 0.3,
                "error": str(e),
                "fallback": True
            }
    
   
    # Fixed transient detection method for core/audio_analyzer.py
    # Replace the existing extract_transient_timeline method with this:

    def extract_transient_timeline(self, y: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Extract transient markers as a list for UI - FIXED VERSION"""
        try:
            # Onset detection
            onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
            
            # Beat tracking
            _, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr)
            
            # Spectral peaks
            stft = librosa.stft(y)
            spectral_peaks = librosa.onset.onset_detect(
                onset_envelope=np.mean(np.abs(stft), axis=0),
                sr=sr, 
                units='time'
            )
            
            # Convert all to numpy arrays and ensure they're 1D
            onsets = np.atleast_1d(onsets)
            beat_times = np.atleast_1d(beat_times)
            spectral_peaks = np.atleast_1d(spectral_peaks)
            
            # Combine all transients
            all_transients = np.concatenate([onsets, beat_times, spectral_peaks])
            all_transients = np.unique(all_transients)
            all_transients = np.sort(all_transients)
            
            # Create transient markers list
            transient_markers = []
            for timestamp in all_transients:
                sample_index = int(float(timestamp) * sr)
                if sample_index < len(y):
                    amplitude = float(abs(y[sample_index]))
                    
                    # Fixed: Use np.any for array comparisons
                    is_onset = np.any(np.abs(onsets - float(timestamp)) < 0.01)
                    is_beat = np.any(np.abs(beat_times - float(timestamp)) < 0.05)
                    is_spectral = np.any(np.abs(spectral_peaks - float(timestamp)) < 0.01)
                    
                    # Determine transient type
                    if is_onset:
                        transient_type = "onset"
                    elif is_beat:
                        transient_type = "beat"
                    else:
                        transient_type = "spectral"
                    
                    transient_markers.append({
                        "time": round(float(timestamp), 3),
                        "amplitude": round(amplitude, 4),
                        "type": transient_type,
                        "is_onset": bool(is_onset),
                        "is_beat": bool(is_beat),
                        "is_spectral": bool(is_spectral)
                    })
            
            return transient_markers
            
        except Exception as e:
            print(f"Transient detection error: {e}")
            return []
    
    # NEW: MFCC analysis method
    def extract_mfcc_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract MFCC features for timbre analysis"""
        try:
            # Standard 13 MFCC coefficients
            mfcc_13 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Extended 20 MFCC coefficients
            mfcc_20 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            # Delta features (first derivative)
            mfcc_delta = librosa.feature.delta(mfcc_13)
            
            # Delta-delta features (second derivative)
            mfcc_delta2 = librosa.feature.delta(mfcc_13, order=2)
            
            return {
                "mfcc_13": {
                    "mean": [round(float(np.mean(mfcc_13[i])), 3) for i in range(13)],
                    "std": [round(float(np.std(mfcc_13[i])), 3) for i in range(13)]
                },
                "mfcc_20": {
                    "mean": [round(float(np.mean(mfcc_20[i])), 3) for i in range(20)],
                    "std": [round(float(np.std(mfcc_20[i])), 3) for i in range(20)]
                },
                "delta_features": {
                    "mean": [round(float(np.mean(mfcc_delta[i])), 3) for i in range(13)],
                    "std": [round(float(np.std(mfcc_delta[i])), 3) for i in range(13)]
                },
                "delta_delta_features": {
                    "mean": [round(float(np.mean(mfcc_delta2[i])), 3) for i in range(13)],
                    "std": [round(float(np.std(mfcc_delta2[i])), 3) for i in range(13)]
                },
                "timbre_complexity": round(float(np.mean(np.std(mfcc_13, axis=1))), 3),
                "temporal_dynamics": round(float(np.mean(np.std(mfcc_delta, axis=1))), 3)
            }
        except Exception as e:
            return {"error": str(e)}
    
    # NEW: Scale detection method
    def detect_musical_scale(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Detect musical scale (major/minor/modes)"""
        try:
            # Enhanced chroma analysis
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Major and minor scale templates
            major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
            minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
            
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            best_key = None
            best_mode = None
            best_correlation = 0
            
            for i in range(12):
                # Rotate templates for each key
                major_rotated = np.roll(major_template, i)
                minor_rotated = np.roll(minor_template, i)
                
                # Calculate correlations
                major_corr = np.corrcoef(chroma_mean, major_rotated)[0, 1]
                minor_corr = np.corrcoef(chroma_mean, minor_rotated)[0, 1]
                
                if major_corr > best_correlation:
                    best_correlation = major_corr
                    best_key = keys[i]
                    best_mode = "major"
                
                if minor_corr > best_correlation:
                    best_correlation = minor_corr
                    best_key = keys[i]
                    best_mode = "minor"
            
            # Camelot notation
            camelot_map = {
                'C major': '8B', 'G major': '9B', 'D major': '10B', 'A major': '11B',
                'E major': '12B', 'B major': '1B', 'F# major': '2B', 'C# major': '3B',
                'G# major': '4B', 'D# major': '5B', 'A# major': '6B', 'F major': '7B',
                'A minor': '8A', 'E minor': '9A', 'B minor': '10A', 'F# minor': '11A',
                'C# minor': '12A', 'G# minor': '1A', 'D# minor': '2A', 'A# minor': '3A',
                'F minor': '4A', 'C minor': '5A', 'G minor': '6A', 'D minor': '7A'
            }
            
            camelot = camelot_map.get(f"{best_key} {best_mode}", "Unknown")
            
            return {
                "key": best_key,
                "mode": best_mode,
                "camelot": camelot,
                "confidence": round(float(best_correlation), 3) if not np.isnan(best_correlation) else 0.5,
                "scale_full": f"{best_key} {best_mode}"
            }
        except Exception as e:
            # Fallback to basic key detection
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            key_profile = np.mean(chroma, axis=1)
            key_index = np.argmax(key_profile)
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            return {
                "key": keys[key_index],
                "mode": "unknown",
                "camelot": "Unknown",
                "confidence": 0.5,
                "error": str(e)
            }
    
    # EXISTING: Keep original load_audio method for compatibility
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file with librosa"""
        return librosa.load(file_path, sr=self.sample_rate)
    
    # ENHANCED: Updated basic_analysis with enhanced tempo
    def basic_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Basic audio analysis - key, tempo, features with enhancements"""
        # Enhanced tempo detection
        tempo_analysis = self.multi_algorithm_tempo_detection(y, sr)
        
        # Enhanced scale detection
        scale_analysis = self.detect_musical_scale(y, sr)
        
        # Basic features (keeping original structure)
        rms_energy = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        return {
            "duration": len(y) / sr,
            "tempo": tempo_analysis["tempo"],  # For backward compatibility
            "tempo_analysis": tempo_analysis,  # Enhanced tempo info
            "key": scale_analysis["key"],  # For backward compatibility
            "scale_analysis": scale_analysis,  # Enhanced scale info
            "rms_energy": float(rms_energy),
            "spectral_centroid": float(spectral_centroid),
            "zero_crossing_rate": float(zero_crossing_rate),
            "sample_rate": sr
        }
    
   # Enhanced advanced_analysis method with better error handling
# Update your existing advanced_analysis method:

# Replace your advanced_analysis method in core/audio_analyzer.py with this safer version:

def advanced_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Advanced multi-algorithm analysis with comprehensive error handling"""
    
    # Start with basic result structure
    result = {
        "duration": len(y) / sr,
        "sample_rate": sr
    }
    
    try:
        # Key detection (existing code)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_profile_chroma = np.mean(chroma, axis=1)
        key_index_chroma = np.argmax(key_profile_chroma)
        
        y_harmonic, _ = librosa.effects.hpss(y)
        chroma_harmonic = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
        key_profile_harmonic = np.mean(chroma_harmonic, axis=1)
        key_index_harmonic = np.argmax(key_profile_harmonic)
        
        cqt = np.abs(librosa.cqt(y, sr=sr))
        chroma_cqt = librosa.feature.chroma_cqt(C=cqt, sr=sr)
        key_profile_cqt = np.mean(chroma_cqt, axis=1)
        key_index_cqt = np.argmax(key_profile_cqt)
        
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Consensus
        key_votes = [key_index_chroma, key_index_harmonic, key_index_cqt]
        most_common_key = Counter(key_votes).most_common(1)[0][0]
        consensus_strength = Counter(key_votes).most_common(1)[0][1] / 3
        
        result["key_detection"] = {
            "consensus_key": keys[most_common_key],
            "consensus_strength": round(float(consensus_strength), 3),
            "methods": {
                "chroma": {"key": keys[key_index_chroma], "confidence": round(float(np.max(key_profile_chroma) / np.sum(key_profile_chroma)), 3)},
                "harmonic": {"key": keys[key_index_harmonic], "confidence": round(float(np.max(key_profile_harmonic) / np.sum(key_profile_harmonic)), 3)},
                "cqt": {"key": keys[key_index_cqt], "confidence": round(float(np.max(key_profile_cqt) / np.sum(key_profile_cqt)), 3)}
            }
        }
        
    except Exception as e:
        print(f"Key detection failed: {e}")
        result["key_detection"] = {"error": "Key detection failed"}
    
    try:
        # Enhanced tempo
        if hasattr(self, 'multi_algorithm_tempo_detection'):
            tempo_analysis = self.multi_algorithm_tempo_detection(y, sr)
        else:
            # Fallback to basic tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo_analysis = {"tempo": round(float(tempo), 1), "confidence": 0.5}
        
        beats = librosa.beat.beat_track(y=y, sr=sr)[1]
        tempo_stability = np.std(np.diff(librosa.frames_to_time(beats, sr=sr))) if len(beats) > 1 else 0.0
        
        result["tempo_analysis"] = {
            **tempo_analysis,
            "stability": round(float(tempo_stability), 3),
            "beat_count": len(beats)
        }
        
    except Exception as e:
        print(f"Tempo analysis failed: {e}")
        result["tempo_analysis"] = {"error": "Tempo analysis failed"}
    
    try:
        # Spectral features
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        result["spectral_features"] = {
            "bandwidth": round(float(spectral_bandwidth), 2),
            "contrast": round(float(spectral_contrast), 2),
            "rolloff": round(float(spectral_rolloff), 2)
        }
        
    except Exception as e:
        print(f"Spectral features failed: {e}")
        result["spectral_features"] = {"error": "Spectral analysis failed"}
    
    try:
        # MFCC features (safe)
        if hasattr(self, 'extract_mfcc_features'):
            mfcc_features = self.extract_mfcc_features(y, sr)
            result["mfcc_features"] = mfcc_features
        else:
            result["mfcc_features"] = {"error": "MFCC method not available"}
            
    except Exception as e:
        print(f"MFCC extraction failed: {e}")
        result["mfcc_features"] = {"error": "MFCC extraction failed"}
    
    try:
        # Transient markers (safe)
        if hasattr(self, 'extract_transient_timeline'):
            transient_markers = self.extract_transient_timeline(y, sr)
            result["transient_markers"] = transient_markers
        else:
            result["transient_markers"] = []
            
    except Exception as e:
        print(f"Transient detection failed: {e}")
        result["transient_markers"] = []
    
    return result

    # EXISTING: Keep your classify_genre method
    def classify_genre(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Basic genre classification"""
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        rms_energy = np.mean(librosa.feature.rms(y=y))
        
        # Simple rule-based classification
        if tempo > 120 and spectral_centroid > 2000:
            genre, confidence = "Electronic/Dance", 0.75
        elif tempo < 80 and rms_energy < 0.05:
            genre, confidence = "Classical/Ambient", 0.70
        elif 90 <= tempo <= 120 and spectral_centroid < 1500:
            genre, confidence = "Rock/Pop", 0.65
        elif tempo > 100 and spectral_rolloff > 3000:
            genre, confidence = "Hip-Hop/Rap", 0.68
        elif spectral_centroid < 1000 and rms_energy < 0.08:
            genre, confidence = "Jazz/Blues", 0.65
        else:
            genre, confidence = "Other/Mixed", 0.50
        
        return {
            "genre": genre,
            "confidence": confidence,
            "features": {
                "tempo": round(float(tempo), 1),
                "spectral_centroid": round(float(spectral_centroid), 2),
                "spectral_rolloff": round(float(spectral_rolloff), 2),
                "rms_energy": round(float(rms_energy), 4)
            }
        }
    # Add this method to your AudioAnalyzer class in core/audio_analyzer.py

    def professional_loudness_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Professional loudness analysis with EBU R128 compliance"""
        
        try:
            import pyloudnorm as pyln
            
            # Ensure stereo for loudness measurement
            if y.ndim == 1:
                y_stereo = np.stack([y, y], axis=-1)
            else:
                y_stereo = y
            
            # Create loudness meter
            meter = pyln.Meter(sr)
            
            # Integrated loudness (LUFS)
            loudness_lufs = meter.integrated_loudness(y_stereo)
            
            # Peak measurement
            peak_db = 20 * np.log10(np.max(np.abs(y))) if np.max(np.abs(y)) > 0 else -np.inf
            
            # RMS analysis
            rms_db = 20 * np.log10(np.sqrt(np.mean(y**2))) if np.sqrt(np.mean(y**2)) > 0 else -np.inf
            
            # Dynamic range estimation
            dynamic_range = peak_db - rms_db
            
            # Recommendations
            recommendations = []
            if loudness_lufs < -23:
                recommendations.append("Audio is quieter than broadcast standard (-23 LUFS)")
            elif loudness_lufs > -14:
                recommendations.append("Audio is louder than streaming standard (-14 LUFS)")
            
            if peak_db > -1:
                recommendations.append("Peak levels too high - risk of clipping")
            
            if dynamic_range < 6:
                recommendations.append("Low dynamic range - heavily compressed")
            elif dynamic_range > 20:
                recommendations.append("High dynamic range - good dynamics")
            
            # Add mastering recommendations
            if -16 <= loudness_lufs <= -12:
                recommendations.append("Perfect for streaming platforms (Spotify, Apple Music)")
            if -23.5 <= loudness_lufs <= -22.5:
                recommendations.append("Compliant with broadcast standards (EBU R128)")
            
            return {
                "lufs_integrated": round(float(loudness_lufs), 2) if not np.isnan(loudness_lufs) else None,
                "peak_dbfs": round(float(peak_db), 2),
                "rms_db": round(float(rms_db), 2),
                "dynamic_range_db": round(float(dynamic_range), 2),
                "recommendations": recommendations,
                "ebu_r128_compliant": -23.5 <= loudness_lufs <= -22.5 if not np.isnan(loudness_lufs) else False,
                "streaming_optimized": -16 <= loudness_lufs <= -12 if not np.isnan(loudness_lufs) else False,
                "broadcast_ready": -23.5 <= loudness_lufs <= -22.5 if not np.isnan(loudness_lufs) else False,
                "mastering_quality": "professional" if dynamic_range > 12 else ("good" if dynamic_range > 8 else "heavily_compressed")
            }
            
        except Exception as e:
            return {
                "lufs_integrated": None,
                "peak_dbfs": None,
                "rms_db": None,
                "dynamic_range_db": None,
                "error": str(e),
                "recommendations": ["Loudness analysis failed - ensure pyloudnorm is installed"]
            }    
    # Add this simpler loudness method to your AudioAnalyzer class:

def simple_loudness_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Simple loudness analysis without pyloudnorm dependency"""
    
    try:
        # Peak measurement
        peak_db = 20 * np.log10(np.max(np.abs(y))) if np.max(np.abs(y)) > 0 else -np.inf
        
        # RMS analysis
        rms_value = np.sqrt(np.mean(y**2))
        rms_db = 20 * np.log10(rms_value) if rms_value > 0 else -np.inf
        
        # Dynamic range estimation
        dynamic_range = peak_db - rms_db
        
        # Simple LUFS estimation (approximate)
        # This is a rough approximation, not true LUFS
        lufs_estimate = rms_db - 23  # Rough conversion
        
        # Recommendations based on simple analysis
        recommendations = []
        if peak_db > -1:
            recommendations.append("Peak levels too high - risk of clipping")
        if dynamic_range < 6:
            recommendations.append("Low dynamic range - heavily compressed")
        elif dynamic_range > 20:
            recommendations.append("High dynamic range - good dynamics")
        
        if rms_db > -12:
            recommendations.append("Very loud - may cause listener fatigue")
        elif rms_db < -30:
            recommendations.append("Very quiet - may need normalization")
        
        return {
            "peak_dbfs": round(float(peak_db), 2),
            "rms_db": round(float(rms_db), 2),
            "dynamic_range_db": round(float(dynamic_range), 2),
            "lufs_estimate": round(float(lufs_estimate), 2),
            "recommendations": recommendations,
            "analysis_type": "simple",
            "note": "Simplified analysis - install pyloudnorm for professional LUFS measurement"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "analysis_type": "failed"
        }
    def professional_loudness_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Simple loudness analysis"""
        try:
            # Peak measurement
            peak_db = 20 * np.log10(np.max(np.abs(y))) if np.max(np.abs(y)) > 0 else -np.inf
            
            # RMS analysis
            rms_value = np.sqrt(np.mean(y**2))
            rms_db = 20 * np.log10(rms_value) if rms_value > 0 else -np.inf
            
            # Dynamic range
            dynamic_range = peak_db - rms_db
            
            # Simple recommendations
            recommendations = []
            if peak_db > -1:
                recommendations.append("Peak levels too high - risk of clipping")
            if dynamic_range < 6:
                recommendations.append("Low dynamic range - heavily compressed")
            elif dynamic_range > 20:
                recommendations.append("High dynamic range - good dynamics")
            
            return {
                "peak_dbfs": round(float(peak_db), 2),
                "rms_db": round(float(rms_db), 2),
                "dynamic_range_db": round(float(dynamic_range), 2),
                "recommendations": recommendations,
                "analysis_type": "simple_working"
            }
        except Exception as e:
            return {"error": str(e)}

    def advanced_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Advanced analysis using existing working methods"""
        try:
            # Use your existing working methods
            basic_result = self.basic_analysis(y, sr)
            
            # Add enhanced tempo if available
            if hasattr(self, 'multi_algorithm_tempo_detection'):
                tempo_analysis = self.multi_algorithm_tempo_detection(y, sr)
            else:
                tempo_analysis = {"tempo": basic_result.get("tempo", 0)}
            
            # Add MFCC if available  
            mfcc_features = {}
            if hasattr(self, 'extract_mfcc_features'):
                try:
                    mfcc_features = self.extract_mfcc_features(y, sr)
                except:
                    mfcc_features = {"error": "MFCC extraction failed"}
            
            # Add transients if available
            transient_markers = []
            if hasattr(self, 'extract_transient_timeline'):
                try:
                    transient_markers = self.extract_transient_timeline(y, sr)
                except:
                    pass
            
            # Basic spectral features
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            
            return {
                "duration": len(y) / sr,
                "sample_rate": sr,
                "tempo_analysis": tempo_analysis,
                "key_analysis": basic_result.get("scale_analysis", {"key": basic_result.get("key")}),
                "spectral_features": {
                    "bandwidth": round(float(spectral_bandwidth), 2),
                    "contrast": round(float(spectral_contrast), 2),
                    "rolloff": round(float(spectral_rolloff), 2)
                },
                "mfcc_features": mfcc_features,
                "transient_markers": transient_markers,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def professional_loudness_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Simple loudness analysis"""
        try:
            # Peak measurement
            peak_db = 20 * np.log10(np.max(np.abs(y))) if np.max(np.abs(y)) > 0 else -np.inf
            
            # RMS analysis
            rms_value = np.sqrt(np.mean(y**2))
            rms_db = 20 * np.log10(rms_value) if rms_value > 0 else -np.inf
            
            # Dynamic range
            dynamic_range = peak_db - rms_db
            
            # Simple recommendations
            recommendations = []
            if peak_db > -1:
                recommendations.append("Peak levels too high - risk of clipping")
            if dynamic_range < 6:
                recommendations.append("Low dynamic range - heavily compressed")
            elif dynamic_range > 20:
                recommendations.append("High dynamic range - good dynamics")
            
            return {
                "peak_dbfs": round(float(peak_db), 2),
                "rms_db": round(float(rms_db), 2),
                "dynamic_range_db": round(float(dynamic_range), 2),
                "recommendations": recommendations,
                "analysis_type": "simple_working"
            }
        except Exception as e:
            return {"error": str(e)}

    def advanced_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Advanced analysis using existing working methods"""
        try:
            # Use your existing working methods
            basic_result = self.basic_analysis(y, sr)
            
            # Add enhanced tempo if available
            if hasattr(self, 'multi_algorithm_tempo_detection'):
                tempo_analysis = self.multi_algorithm_tempo_detection(y, sr)
            else:
                tempo_analysis = {"tempo": basic_result.get("tempo", 0)}
            
            # Add MFCC if available  
            mfcc_features = {}
            if hasattr(self, 'extract_mfcc_features'):
                try:
                    mfcc_features = self.extract_mfcc_features(y, sr)
                except:
                    mfcc_features = {"error": "MFCC extraction failed"}
            
            # Add transients if available
            transient_markers = []
            if hasattr(self, 'extract_transient_timeline'):
                try:
                    transient_markers = self.extract_transient_timeline(y, sr)
                except:
                    pass
            
            # Basic spectral features
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            
            return {
                "duration": len(y) / sr,
                "sample_rate": sr,
                "tempo_analysis": tempo_analysis,
                "key_analysis": basic_result.get("scale_analysis", {"key": basic_result.get("key")}),
                "spectral_features": {
                    "bandwidth": round(float(spectral_bandwidth), 2),
                    "contrast": round(float(spectral_contrast), 2),
                    "rolloff": round(float(spectral_rolloff), 2)
                },
                "mfcc_features": mfcc_features,
                "transient_markers": transient_markers,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e), "status": "failed"}
