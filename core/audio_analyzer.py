# core/audio_analyzer.py - Clean, properly structured version
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
        # Memory management settings
        self.max_file_size = 150 * 1024 * 1024  # 150MB limit
        self.max_duration = 600  # 10 minutes
        self.chunk_duration = 30  # Analyze first 30 seconds for long files
    
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
    
    def extract_transient_timeline(self, y: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Extract transient markers as a list for UI"""
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
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file with librosa"""
        return librosa.load(file_path, sr=self.sample_rate)
    
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
    
    def advanced_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Advanced analysis using existing working methods"""
        try:
            # Use existing working methods
            basic_result = self.basic_analysis(y, sr)
            
            # Enhanced tempo if available
            tempo_analysis = self.multi_algorithm_tempo_detection(y, sr)
            
            # MFCC if available
            mfcc_features = self.extract_mfcc_features(y, sr)
            
            # Transients if available
            transient_markers = self.extract_transient_timeline(y, sr)
            
            # Basic spectral features
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            
            return {
                "duration": len(y) / sr,
                "sample_rate": sr,
                "tempo_analysis": tempo_analysis,
                "key_analysis": basic_result.get("scale_analysis", {"key": basic_result.get("key")}),
                "key_detection": basic_result.get("scale_analysis", {"key": basic_result.get("key")}),  # ADD THIS LINE
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
    # Add this NEW method to your AudioAnalyzer
    # In AudioAnalyzer
    def comprehensive_analysis_with_features(self, file_path):
        """
        Complete analysis + pre-computed audio features for visualization
        Does expensive computations ONCE
        """
        # Load audio ONCE
        y, sr = librosa.load(file_path, sr=None)
        
        # Get comprehensive analysis using existing method
        analysis = self.basic_analysis(y, sr)
        
        # Do expensive STFT computation ONCE
        stft = librosa.stft(y, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # Pre-compute frequency and time axes
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)
        time_frames = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=512)
        
        return {
                "analysis": analysis,
                "audio_features": {
                    # For waveform generation
                    "y": y.tolist(),
                    "sr": int(sr),
                    "duration": float(len(y) / sr),
                    "samples": int(len(y)),
                    
                    # Pre-computed spectrogram data (expensive computation done ONCE)
                    "stft_magnitude_db": magnitude_db.tolist(),
                    "frequencies": frequencies.tolist(),
                    "time_frames": time_frames.tolist(),
                    "stft_shape": list(magnitude_db.shape)
                }
            }
    def comprehensive_analysis_memory_safe(self, file_path: str):
        """
        Memory-safe version of comprehensive analysis for large files
        Keeps existing functionality but reduces memory usage
        """
        import gc
        import psutil
        import soundfile as sf
        
        # Step 1: Check file info without loading
        try:
            info = sf.info(file_path)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            print(f"File info: {info.duration:.1f}s, {info.samplerate}Hz, {file_size_mb:.1f}MB")
            
            # Step 2: Determine processing strategy based on file size
            if file_size_mb < 30:  # Small files - use existing method
                print("Small file - using existing method")
                return self.comprehensive_analysis_with_features(file_path)
            
            elif file_size_mb < 75:  # Medium files - reduced sample rate
                print("Medium file - using reduced sample rate")
                return self._analyze_medium_file(file_path, info)
            
            else:  # Large files - chunked processing
                print("Large file - using chunked processing")
                return self._analyze_large_file_chunked(file_path, info)
                
        except Exception as e:
            print(f"Memory-safe analysis failed: {e}")
            # Fallback to basic analysis
            try:
                y, sr = librosa.load(file_path, sr=22050, duration=30)
                basic_result = self.basic_analysis(y, sr)
                del y
                gc.collect()
                return {
                    "analysis": basic_result,
                    "audio_features": {
                        "duration": info.duration if 'info' in locals() else 30,
                        "sample_rate": 22050,
                        "processing_note": "Fallback analysis due to memory constraints"
                    }
                }
            except Exception as fallback_error:
                return {
                    "analysis": {"error": str(fallback_error)},
                    "audio_features": {"error": "Analysis failed"}
                }

    def _analyze_medium_file(self, file_path: str, info):
        """Analyze medium-sized files with reduced sample rate"""
        import gc
        
        # Load with reduced sample rate to save memory
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        
        print(f"Loaded medium file: {len(y)} samples at {sr}Hz")
        
        # Get analysis using existing method
        analysis = self.basic_analysis(y, sr)
        # Generate lightweight waveform for visualization
        waveform_data = self.get_lightweight_waveform(file_path)

        # Clean up large arrays before creating response
        del y
        gc.collect()

        return {
            "analysis": analysis,
            "audio_features": {
                "duration": float(info.duration),
                "sample_rate": int(sr),
                "samples": int(info.frames),
                "processing_mode": "medium_file_reduced_sr"
            },
            "visualization": waveform_data  # Simple waveform only
        }

    def _analyze_large_file_chunked(self, file_path: str, info):
        """Analyze large files using chunked processing"""
        import gc
        
        duration = info.duration
        chunk_size = 60  # 60 seconds per chunk
        target_sr = 22050
        
        print(f"Processing large file in chunks: {duration:.1f}s total")
        
        # Initialize results
        timeline_results = {
            "key_changes": [],
            "tempo_changes": [],
            "energy_timeline": []
        }
        
        # Global accumulators
        all_tempos = []
        all_keys = []
        all_energies = []
        
        # Process first few chunks to get representative analysis
        max_chunks = min(3, int(duration // chunk_size))
        
        for i in range(max_chunks):
            chunk_start = i * chunk_size
            print(f"Processing chunk {i+1}/{max_chunks}: {chunk_start}s-{chunk_start+chunk_size}s")
            
            # Load chunk
            y_chunk, sr = librosa.load(file_path, sr=target_sr, mono=True, 
                                       offset=chunk_start, duration=chunk_size)
            
            # Analyze chunk
            chunk_analysis = self.basic_analysis(y_chunk, sr)
            
            # Accumulate results
            all_tempos.append(chunk_analysis["tempo"])
            all_keys.append(chunk_analysis["key"])
            
            # Energy analysis
            rms = np.mean(librosa.feature.rms(y=y_chunk))
            all_energies.append(float(rms))
            
            # Timeline data
            timeline_results["tempo_changes"].append({
                "time": chunk_start,
                "tempo": chunk_analysis["tempo"]
            })
            
            timeline_results["key_changes"].append({
                "time": chunk_start,
                "key": chunk_analysis["key"]
            })
            
            # Clean up chunk immediately
            del y_chunk
            gc.collect()
        
        # Create aggregated analysis
        from collections import Counter
        
        # Most common key
        key_counter = Counter(all_keys)
        most_common_key = key_counter.most_common(1)[0][0]
        
        # Average tempo
        avg_tempo = sum(all_tempos) / len(all_tempos)
        
        # Average energy
        avg_energy = sum(all_energies) / len(all_energies)
        
        # NEW visualization section (ADD THIS):
        # Generate lightweight waveform for entire file
        waveform_data = self.get_lightweight_waveform(file_path)

        # Final results
        aggregated_analysis = {
            "duration": duration,
            "tempo": avg_tempo,
            "key": most_common_key,
            "rms_energy": avg_energy,
            "sample_rate": target_sr,
            "processing_mode": "chunked_analysis",
            "chunks_processed": max_chunks,
            "key_stability": key_counter[most_common_key] / len(all_keys),
            "tempo_variance": max(all_tempos) - min(all_tempos) if all_tempos else 0
        }

        return {
            "analysis": aggregated_analysis,
            "audio_features": {
                "duration": float(duration),
                "sample_rate": int(target_sr),
                "processing_mode": "large_file_chunked",
                "chunks_analyzed": max_chunks
            },
            "timeline_analysis": timeline_results,
            "visualization": waveform_data  # Simple waveform only
        }

    def get_memory_usage(self):
        """Helper method to check memory usage"""
        import psutil
        process = psutil.Process()
        return {
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "available_system_mb": psutil.virtual_memory().available / 1024 / 1024
        }
        
    def get_lightweight_waveform(self, file_path: str, target_points=2000):
        """Get minimal waveform data for interactive visualization"""
        import soundfile as sf
        
        # Get file info without loading
        info = sf.info(file_path)
        
        # Load at very low sample rate - just for waveform shape
        y, sr = librosa.load(file_path, sr=2000, mono=True)  
        
        # Downsample to exact target points
        if len(y) > target_points:
            step = len(y) // target_points
            waveform = y[::step][:target_points]
        else:
            waveform = y
        
        return {
            "waveform_data": waveform.tolist(),  # ~8KB
            "duration": info.duration,           # Real duration
            "points": len(waveform),
            "sample_rate": 2000,                 # Low res for visualization only
            "note": "Optimized for waveform display and timeline interaction"
        }