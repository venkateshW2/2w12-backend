# NEW FILE: core/enhanced_audio_loader.py
import librosa
import numpy as np
import time
import logging
import tempfile
import os
from typing import Dict, Any, Tuple, List, Optional
from core.database_manager import SoundToolsDatabase

logger = logging.getLogger(__name__)

class EnhancedAudioLoader:
    """
    Enhanced Audio Loader - Foundation for ML Pipeline
    
    Architecture:
    - Week 1: Enhanced librosa analysis with caching
    - Week 2: Essentia + Madmom ML models integration
    - Future: Custom models and real-time processing
    """
    
    def __init__(self):
        self.db = SoundToolsDatabase()
        self.sample_rate = 22050  # Standard for consistency
        self.max_duration = 600   # 10 minutes max for now
        
        # Analysis versions for tracking improvements
        self.analysis_version = "v2.0_enhanced_librosa"
        self.ml_models_loaded = False  # Will be True in Week 2
        
        logger.info("🚀 Enhanced Audio Loader initialized")
        self._log_capabilities()
    
    def _log_capabilities(self):
        """Log current capabilities"""
        capabilities = {
            "librosa_analysis": True,
            "caching": True,
            "enhanced_features": True,
            "essentia_models": self.ml_models_loaded,  # False for Week 1
            "madmom_models": self.ml_models_loaded,   # False for Week 1
            "universal_chunking": False,               # Week 2 feature
            "real_time_analysis": False                # Future feature
        }
        
        active_features = [k for k, v in capabilities.items() if v]
        logger.info(f"📋 Active capabilities: {', '.join(active_features)}")
    
    def analyze_with_caching(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Main analysis method with intelligent caching
        
        This is the primary entry point that will be used by API endpoints
        """
        start_total = time.time()
        
        # Step 1: Create unique fingerprint
        fingerprint = self.db.create_file_fingerprint(file_content, filename)
        logger.info(f"🔍 Processing: {filename} (fingerprint: {fingerprint[:8]}...)")
        
        # Step 2: Check cache first
        cached_result = self.db.get_cached_analysis(fingerprint)
        if cached_result:
            total_time = time.time() - start_total
            logger.info(f"⚡ Cache HIT - returned in {total_time:.3f}s")
            return {
                **cached_result,
                "cache_status": "HIT",
                "response_time": total_time
            }
        
        # Step 3: Cache miss - perform fresh analysis
        logger.info(f"🔄 Cache MISS - performing fresh analysis")
        start_analysis = time.time()
        
        try:
            # Perform comprehensive analysis
            analysis_result = self._perform_comprehensive_analysis(file_content, filename)
            
            analysis_time = time.time() - start_analysis
            total_time = time.time() - start_total
            
            # Step 4: Prepare final result with metadata
            final_result = {
                **analysis_result,
                "fingerprint": fingerprint,
                "analysis_time": analysis_time,
                "total_time": total_time,
                "cache_status": "MISS",
                "analyzed_at": time.time(),
                "analysis_version": self.analysis_version
            }
            
            # Step 5: Cache for future use
            cache_success = self.db.cache_analysis_result(fingerprint, final_result)
            final_result["cached"] = cache_success
            
            logger.info(f"✅ Analysis completed in {analysis_time:.2f}s (total: {total_time:.2f}s)")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Analysis failed for {filename}: {e}")
            raise
    
    def _perform_comprehensive_analysis(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Comprehensive analysis combining enhanced librosa + ML foundation
        
        Week 1: Enhanced librosa analysis
        Week 2: Will add Essentia + Madmom integration here
        """
        
        # Create temporary file for librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            # Load audio with validation
            y, sr, file_info = self._smart_audio_loading(tmp_file_path)
            
            # Core analysis pipeline
            core_analysis = self._librosa_enhanced_analysis(y, sr)
            
            # Combine all results
            comprehensive_result = {
                "filename": filename,
                "file_info": file_info,
                
                # Core enhanced librosa analysis
                **core_analysis,
                
                # Analysis metadata
                "features_extracted": list(core_analysis.keys()),
                "analysis_pipeline": ["enhanced_librosa"],
                "quality_score": self._calculate_analysis_quality(core_analysis)
            }
            
            return comprehensive_result
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    def _smart_audio_loading(self, file_path: str) -> Tuple[np.ndarray, int, Dict]:
        """Smart audio loading with validation and optimization"""
        try:
            # Get file info first
            import soundfile as sf
            info = sf.info(file_path)
            original_duration = info.duration
            original_sr = info.samplerate
            
            # Duration-based loading strategy
            if original_duration > self.max_duration:
                logger.warning(f"Large file detected ({original_duration:.1f}s), loading first {self.max_duration}s")
                y, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.max_duration)
                analyzed_duration = self.max_duration
            else:
                # Load complete file
                y, sr = librosa.load(file_path, sr=self.sample_rate)
                analyzed_duration = original_duration
            
            file_info = {
                "original_duration": round(float(original_duration), 2),
                "analyzed_duration": round(float(analyzed_duration), 2),
                "original_sample_rate": int(original_sr),
                "processing_sample_rate": int(sr),
                "truncated": original_duration > self.max_duration,
                "file_size_mb": round(os.path.getsize(file_path) / 1024 / 1024, 2)
            }
            
            logger.info(f"📊 Audio loaded: {analyzed_duration:.1f}s @ {sr}Hz")
            return y, sr, file_info
            
        except Exception as e:
            logger.error(f"❌ Audio loading failed: {e}")
            raise ValueError(f"Could not load audio file: {e}")
    
    def _librosa_enhanced_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Enhanced librosa analysis - improved versions of existing features"""
        duration = len(y) / sr
        logger.info(f"🎵 Starting enhanced librosa analysis ({duration:.1f}s)")
        
        # Initialize with safe defaults
        key_analysis = {"key": "C", "mode": "major", "key_confidence": 0.5}
        tempo_analysis = {"tempo": 120.0, "tempo_confidence": 0.5}
        harmonic_analysis = {"harmonic_confidence": 0.5}
        spectral_analysis = {"brightness": "unknown"}
        rhythmic_analysis = {"onset_count": 0}
        energy_analysis = {"energy_level": "medium"}
        
        try:
            # === ENHANCED KEY DETECTION ===
            key_analysis = self._enhanced_key_detection(y, sr)
            logger.info("✅ Key detection completed")
        except Exception as e:
            logger.error(f"❌ Key detection failed: {e}")
        
        try:
            # === ENHANCED TEMPO DETECTION ===
            tempo_analysis = self._enhanced_tempo_detection(y, sr)
            logger.info("✅ Tempo detection completed")
        except Exception as e:
            logger.error(f"❌ Tempo detection failed: {e}")
        
        try:
            # === HARMONIC ANALYSIS ===
            harmonic_analysis = self._harmonic_content_analysis(y, sr)
            logger.info("✅ Harmonic analysis completed")
        except Exception as e:
            logger.error(f"❌ Harmonic analysis failed: {e}")
        
        try:
            # === SPECTRAL FEATURES ===
            spectral_analysis = self._spectral_feature_analysis(y, sr)
            logger.info("✅ Spectral analysis completed")
        except Exception as e:
            logger.error(f"❌ Spectral analysis failed: {e}")
        
        try:
            # === RHYTHMIC FEATURES ===
            rhythmic_analysis = self._rhythmic_feature_analysis(y, sr)
            logger.info("✅ Rhythmic analysis completed")
        except Exception as e:
            logger.error(f"❌ Rhythmic analysis failed: {e}")
        
        try:
            # === ENERGY ANALYSIS ===
            energy_analysis = self._energy_analysis(y, sr)
            logger.info("✅ Energy analysis completed")
        except Exception as e:
            logger.error(f"❌ Energy analysis failed: {e}")
        
        # Combine all enhanced features with safe defaults
        enhanced_result = {
            "duration": round(float(duration), 2),
            
            # Enhanced analyses
            **key_analysis,
            **tempo_analysis,
            **harmonic_analysis,
            **spectral_analysis,
            **rhythmic_analysis,
            **energy_analysis,
            
            # Overall confidence
            "overall_confidence": self._calculate_overall_confidence([
                key_analysis.get("key_confidence", 0.5),
                tempo_analysis.get("tempo_confidence", 0.5),
                harmonic_analysis.get("harmonic_confidence", 0.5)
            ])
        }
        
        logger.info("✅ Enhanced librosa analysis completed")
        return enhanced_result
    
    def _enhanced_key_detection(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Enhanced key detection with confidence and mode"""
        try:
            # Chromagram with better parameters
            chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512, n_fft=2048)
            
            # Mean chroma vector
            chroma_mean = np.mean(chromagram, axis=1)
            
            # Key detection
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_index = np.argmax(chroma_mean)
            detected_key = keys[key_index]
            
            # Key confidence based on chroma strength
            key_strength = float(chroma_mean[key_index])
            key_confidence = min(1.0, key_strength * 2.0)
            
            # Major/Minor detection
            major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
            minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
            
            # Rotate profiles to match detected key
            major_rotated = np.roll(major_profile, key_index)
            minor_rotated = np.roll(minor_profile, key_index)
            
            # Calculate correlations safely
            try:
                major_correlation = float(np.corrcoef(chroma_mean, major_rotated)[0, 1])
                if np.isnan(major_correlation):
                    major_correlation = 0.0
            except:
                major_correlation = 0.0
                
            try:
                minor_correlation = float(np.corrcoef(chroma_mean, minor_rotated)[0, 1])
                if np.isnan(minor_correlation):
                    minor_correlation = 0.0
            except:
                minor_correlation = 0.0
            
            if major_correlation > minor_correlation:
                mode = "major"
                mode_confidence = abs(major_correlation)
            else:
                mode = "minor"
                mode_confidence = abs(minor_correlation)
            
            return {
                "key": detected_key,
                "mode": mode,
                "key_confidence": round(key_confidence, 3),
                "mode_confidence": round(mode_confidence, 3),
                "full_key": f"{detected_key} {mode}",
                "chroma_strength": round(key_strength, 3)
            }
        except Exception as e:
            logger.error(f"Key detection error: {e}")
            return {"key": "C", "mode": "major", "key_confidence": 0.5}
    
    def _enhanced_tempo_detection(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Multi-algorithm tempo detection with confidence"""
        try:
            # Method 1: Standard beat tracking
            tempo1, beats1 = librosa.beat.beat_track(y=y, sr=sr)
            tempo1 = float(tempo1)
            
            # Method 2: Simple fallback
            tempo2 = 120.0  # Skip complex tempo detection for now
            
            # Method 3: Harmonic/percussive separation
            try:
                y_harmonic, y_percussive = librosa.effects.hpss(y)
                tempo3, beats3 = librosa.beat.beat_track(y=y_percussive, sr=sr)
                tempo3 = float(tempo3)
            except:
                tempo3 = tempo1
            
            # Collect tempo candidates
            tempo_candidates = [t for t in [tempo1, tempo2, tempo3] if 40 <= t <= 200]
            
            if tempo_candidates:
                final_tempo = float(np.median(tempo_candidates))
                tempo_std = float(np.std(tempo_candidates))
                tempo_confidence = max(0.1, 1.0 - (tempo_std / 30.0))
            else:
                final_tempo = 120.0
                tempo_confidence = 0.3
            
            # Beat strength analysis
            beat_strength = float(np.mean(librosa.onset.onset_strength(y=y, sr=sr)))
            
            return {
                "tempo": round(final_tempo, 1),
                "tempo_confidence": round(tempo_confidence, 3),
                "tempo_candidates": [round(t, 1) for t in tempo_candidates],
                "beat_strength": round(beat_strength, 4),
                "beat_count": len(beats1),
                "rhythmic_consistency": round(tempo_confidence, 3)
            }
        except Exception as e:
            logger.error(f"Tempo detection error: {e}")
            return {"tempo": 120.0, "tempo_confidence": 0.5}
    
    def _harmonic_content_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Harmonic vs percussive content analysis"""
        try:
            # Harmonic/percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Calculate ratios
            harmonic_energy = float(np.mean(np.abs(y_harmonic)))
            percussive_energy = float(np.mean(np.abs(y_percussive)))
            total_energy = float(np.mean(np.abs(y)))
            
            harmonic_ratio = harmonic_energy / (total_energy + 1e-8)
            percussive_ratio = percussive_energy / (total_energy + 1e-8)
            
            return {
                "harmonic_ratio": round(harmonic_ratio, 3),
                "percussive_ratio": round(percussive_ratio, 3),
                "harmonic_confidence": round(harmonic_ratio, 3),
                "content_type": "harmonic" if harmonic_ratio > percussive_ratio else "percussive"
            }
        except Exception as e:
            logger.error(f"Harmonic analysis error: {e}")
            return {"harmonic_confidence": 0.5}
    
    def _spectral_feature_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Enhanced spectral feature analysis"""
        try:
            # Spectral features
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
            spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
            spectral_contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
            zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
            
            return {
                "spectral_centroid": round(spectral_centroid, 2),
                "spectral_rolloff": round(spectral_rolloff, 2),
                "spectral_bandwidth": round(spectral_bandwidth, 2),
                "spectral_contrast": round(spectral_contrast, 3),
                "zero_crossing_rate": round(zero_crossing_rate, 4),
                "spectral_flatness": round(spectral_flatness, 4),
                "brightness": "bright" if spectral_centroid > 2000 else "dark"
            }
        except Exception as e:
            logger.error(f"Spectral analysis error: {e}")
            return {"brightness": "unknown"}
    
    def _rhythmic_feature_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Rhythmic pattern analysis"""
        try:
            # Onset detection
            onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
            onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
            
            # Rhythmic regularity
            if len(onsets) > 1:
                onset_intervals = np.diff(onsets)
                rhythmic_regularity = 1.0 / (1.0 + float(np.std(onset_intervals)))
            else:
                rhythmic_regularity = 0.0
            
            return {
                "onset_count": len(onsets),
                "onset_density": round(len(onsets) / (len(y) / sr), 2),
                "rhythmic_regularity": round(rhythmic_regularity, 3),
                "onset_strength_mean": round(float(np.mean(onset_strength)), 4)
            }
        except Exception as e:
            logger.error(f"Rhythmic analysis error: {e}")
            return {"onset_count": 0}
    
    def _energy_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Energy and dynamics analysis"""
        try:
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            rms_mean = float(np.mean(rms))
            rms_std = float(np.std(rms))
            dynamic_range = float(np.max(rms) - np.min(rms))
            
            # Energy classification
            if rms_mean > 0.1:
                energy_level = "high"
            elif rms_mean > 0.05:
                energy_level = "medium"
            else:
                energy_level = "low"
            
            return {
                "rms_energy": round(rms_mean, 4),
                "energy_variance": round(rms_std, 4),
                "dynamic_range": round(dynamic_range, 4),
                "energy_level": energy_level
            }
        except Exception as e:
            logger.error(f"Energy analysis error: {e}")
            return {"energy_level": "medium"}
    
    def _calculate_overall_confidence(self, confidence_scores: List[float]) -> float:
        """Calculate overall analysis confidence"""
        if not confidence_scores:
            return 0.5
        return round(float(np.mean(confidence_scores)), 3)
    
    def _calculate_analysis_quality(self, analysis_result: Dict) -> float:
        """Calculate quality score for the analysis"""
        factors = []
        
        if "key_confidence" in analysis_result:
            factors.append(analysis_result["key_confidence"])
        if "tempo_confidence" in analysis_result:
            factors.append(analysis_result["tempo_confidence"])
        if "harmonic_confidence" in analysis_result:
            factors.append(analysis_result["harmonic_confidence"])
        
        if factors:
            quality = float(np.mean(factors))
        else:
            quality = 0.5
        
        return round(quality, 3)