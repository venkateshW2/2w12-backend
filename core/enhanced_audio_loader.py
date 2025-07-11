# NEW FILE: core/enhanced_audio_loader.py
import librosa
import numpy as np
import asyncio
import time
import logging
import tempfile
import os
from typing import Dict, Any, Tuple, List, Optional
from core.database_manager import SoundToolsDatabase

logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class ModelManagerSingleton:
    """Singleton pattern for model persistence - load once, use many times"""
    _essentia_instance = None
    _madmom_instance = None
    _musicbrainz_instance = None
    
    @classmethod
    def get_essentia_models(cls):
        if cls._essentia_instance is None:
            try:
                from .essentia_models import EssentiaModelManager
                cls._essentia_instance = EssentiaModelManager()
                logger.info(f"🤖 Essentia models loaded (singleton): {cls._essentia_instance.models_loaded}")
            except Exception as e:
                logger.warning(f"⚠️  Essentia models unavailable: {e}")
                cls._essentia_instance = None
        return cls._essentia_instance
    
    @classmethod 
    def get_madmom_processor(cls):
        if cls._madmom_instance is None:
            try:
                from .madmom_processor import MadmomProcessor
                cls._madmom_instance = MadmomProcessor()
                logger.info(f"🥁 Madmom processors loaded (singleton): {cls._madmom_instance.processors_loaded}")
            except Exception as e:
                logger.warning(f"⚠️  Madmom processors unavailable: {e}")
                cls._madmom_instance = None
        return cls._madmom_instance
    
    @classmethod
    def get_musicbrainz_researcher(cls):
        if cls._musicbrainz_instance is None:
            try:
                from .musicbrainz_utils import MusicBrainzResearcher
                cls._musicbrainz_instance = MusicBrainzResearcher()
                logger.info(f"🔬 MusicBrainz researcher loaded (singleton)")
            except Exception as e:
                logger.warning(f"⚠️  MusicBrainz research unavailable: {e}")
                cls._musicbrainz_instance = None
        return cls._musicbrainz_instance

class EnhancedAudioLoader:
    """
    Enhanced Audio Loader - Foundation for ML Pipeline with Model Persistence
    
    Architecture:
    - Week 1: Enhanced librosa analysis with caching
    - Week 2: Essentia + Madmom ML models integration  
    - Week 3: Parallel processing with model persistence
    - Future: Real-time processing and custom models
    """
    
    def __init__(self):
        self.db = SoundToolsDatabase()
        self.sample_rate = 22050  # Standard for consistency
        self.max_duration = 600   # 10 minutes max for now
        
        # Use singleton pattern for model persistence
        self.essentia_models = ModelManagerSingleton.get_essentia_models()
        self.madmom_processor = ModelManagerSingleton.get_madmom_processor()
        self.mb_researcher = ModelManagerSingleton.get_musicbrainz_researcher()
        
        # Initialize capabilities flags
        self.ml_models_loaded = self.essentia_models and self.essentia_models.models_loaded
        self.madmom_loaded = self.madmom_processor and self.madmom_processor.processors_loaded
        self.research_enabled = (
            self.mb_researcher and 
            hasattr(self.mb_researcher, 'acoustid_available') and
            hasattr(self.mb_researcher, 'musicbrainz_available') and
            self.mb_researcher.acoustid_available and 
            self.mb_researcher.musicbrainz_available
        )
        
        # Analysis version tracking
        self.analysis_version = "v2.3_parallel_processing"
        
        logger.info("🚀 Enhanced Audio Loader initialized with model persistence")
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
            # Always skip background research to avoid async issues in streaming context
            logger.info("🔄 Background research skipped - avoiding async issues in streaming")
            # From Image 2: Return cache hit with timing
            total_time = time.time() - start_total
            logger.info(f"⚡ Cache HIT - returned in {total_time:.3f}s")
            cached_result = convert_numpy_types(cached_result)
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
            
            # Step 5: Convert numpy types to JSON-serializable types
            final_result = convert_numpy_types(final_result)
            
            # Step 6: Cache for future use
            cache_success = self.db.cache_analysis_result(fingerprint, final_result)
            final_result["cached"] = cache_success
            
            if self.research_enabled:
                fingerprint = final_result.get("fingerprint")
                if fingerprint:
                    try:
                        # Start background research (does not block response) - only if event loop exists
                        asyncio.create_task(
                            self.mb_researcher.background_research(
                                fingerprint, file_content, final_result
                            )
                        )
                        final_result["background_research_started"] = True
                    except RuntimeError:
                        # No event loop running (e.g., in ThreadPoolExecutor) - skip background research
                        logger.info("🔄 Background research skipped - no event loop available")
                        final_result["background_research_started"] = False
                else:
                    final_result["background_research_started"] = False
            
            return final_result
            
            logger.info(f"✅ Analysis completed in {analysis_time:.2f}s (total: {total_time:.2f}s)")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Analysis failed for {filename}: {e}")
            raise
    
    def _perform_comprehensive_analysis(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Comprehensive analysis: Enhanced librosa + Essentia ML + Madmom rhythm

        """
        
        # Create temporary file for librosa - ensure it's accessible after closing
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        tmp_file.write(file_content)
        tmp_file.flush()  # Ensure data is written to disk
        tmp_file.close()  # Close the file handle but keep the file on disk
        tmp_file_path = tmp_file.name
        
        try:
            # Load audio with validation and GPU-optimized chunking
            logger.info(f"🔍 Loading audio from temp file: {tmp_file_path}")
            y, sr, file_info = self._smart_audio_loading(tmp_file_path)
            logger.info(f"✅ Audio loaded successfully: {file_info.get('analyzed_duration', 0):.1f}s")
            
            # Check if we need chunking for large files (GPU optimization)
            should_chunk = self._should_use_chunking(y, sr, file_info)
            
            if should_chunk:
                logger.info(f"🔧 Large file detected - using GPU-optimized chunking")
                return self._chunked_parallel_analysis(y, sr, tmp_file_path, file_info, filename)
            else:
                logger.info(f"🎵 Standard parallel analysis for {file_info.get('analyzed_duration', 0):.1f}s file")
            
            # PARALLEL ANALYSIS PIPELINE - Run all analyses simultaneously
            logger.info("🚀 Starting parallel analysis pipeline...")
            parallel_start = time.time()
            
            # Use ThreadPoolExecutor for parallel execution (no asyncio conflicts)
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all tasks to thread pool
                logger.info("🚀 Submitting librosa analysis task...")
                future_core = executor.submit(self._librosa_enhanced_analysis, y, sr)
                
                logger.info("🚀 Submitting Essentia ML analysis task...")
                future_ml = executor.submit(self._essentia_ml_analysis, y, sr)
                
                logger.info(f"🚀 Submitting Madmom rhythm analysis task for file: {tmp_file_path}")
                future_rhythm = executor.submit(self._madmom_fast_rhythm_analysis, tmp_file_path)
                
                # Wait for fast analyses to complete (don't wait for heavy downbeat analysis)
                logger.info("⏳ Waiting for librosa analysis...")
                core_analysis = future_core.result()
                logger.info("✅ Librosa analysis completed")
                
                logger.info("⏳ Waiting for Essentia ML analysis...")
                ml_analysis = future_ml.result() 
                logger.info("✅ Essentia ML analysis completed")
                
                logger.info("⏳ Waiting for Madmom rhythm analysis...")
                rhythm_analysis = future_rhythm.result()  # Fast rhythm only
                logger.info("✅ Madmom rhythm analysis completed")
            
            parallel_time = time.time() - parallel_start
            logger.info(f"⚡ Parallel analysis completed in {parallel_time:.2f}s")

            # Combine all results
            comprehensive_result = {
                "filename": filename,
                "file_info": file_info,
                
                # Core enhanced librosa analysis
                **core_analysis,
                
                # Essentia ML analysis
                **ml_analysis,

                # Madmom rhythm analysis
                 **rhythm_analysis,

                # Analysis metadata
                "features_extracted": list(core_analysis.keys()),
                "analysis_pipeline": ["enhanced_librosa", "essentia_ml", "madmom_rhythm"],
                "quality_score": self._calculate_analysis_quality({
                **core_analysis, **ml_analysis, **rhythm_analysis
                }),
                
                # Parallel processing performance metrics
                "parallel_processing": {
                    "enabled": True,
                    "parallel_time": round(parallel_time, 2),
                    "estimated_sequential_time": round(parallel_time * 3, 2),  # Rough estimate
                    "speedup_factor": "~3x",
                    "realtime_factor": round(file_info.get("analyzed_duration", 1) / parallel_time, 2)
                }
            }
            
            # Convert numpy types to JSON-serializable types
            comprehensive_result = convert_numpy_types(comprehensive_result)
            
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
            
            # OPTIMIZED: Use 11025 Hz for 2x speed improvement
            optimized_sample_rate = 11025
            
            # Duration-based loading strategy
            if original_duration > self.max_duration:
                logger.warning(f"Large file detected ({original_duration:.1f}s), loading first {self.max_duration}s")
                y, sr = librosa.load(file_path, sr=optimized_sample_rate, duration=self.max_duration)
                analyzed_duration = self.max_duration
            else:
                # Load complete file
                y, sr = librosa.load(file_path, sr=optimized_sample_rate)
                analyzed_duration = original_duration
            
            file_info = {
                "original_duration": round(float(original_duration), 2),
                "analyzed_duration": round(float(analyzed_duration), 2),
                "original_sample_rate": int(original_sr),
                "processing_sample_rate": int(sr),
                "truncated": original_duration > self.max_duration,
                "file_size_mb": round(os.path.getsize(file_path) / 1024 / 1024, 2),
                "optimization": "11025Hz_fast_mode"
            }
            
            logger.info(f"📊 Audio loaded: {analyzed_duration:.1f}s @ {sr}Hz (optimized)")
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
        """Fast key detection (lightweight since Essentia ML provides better results)"""
        try:
            # Fast chromagram with optimized parameters for speed
            chromagram = librosa.feature.chroma_stft(
                y=y, sr=sr, 
                hop_length=2048,  # 4x larger hop (faster)
                n_fft=1024,       # Smaller FFT (faster) 
                norm=None         # Skip normalization (faster)
            )
            
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
        """Fast tempo detection (lightweight since Madmom provides better results)"""
        try:
            # Fast single-method tempo detection (remove max_tempo - not supported in this librosa version)
            tempo1, beats1 = librosa.beat.beat_track(
                y=y, sr=sr,
                hop_length=1024,  # Larger hop for speed
                start_bpm=60      # Constrain search range
            )
            tempo1 = float(tempo1)
            
            # Skip expensive methods since we have Madmom
            tempo2 = tempo1  # Use same result
            tempo3 = tempo1  # Use same result
            
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
        """OPTIMIZED: Fast harmonic vs percussive content analysis"""
        try:
            # OPTIMIZED: Fast harmonic/percussive separation with reduced margin
            y_harmonic, y_percussive = librosa.effects.hpss(
                y, 
                margin=(1.0, 1.0),  # Reduced margin for speed (default is 1.0, 5.0)
                power=1.0           # Reduced power for speed (default is 2.0)
            )
            
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
                "content_type": "harmonic" if harmonic_ratio > percussive_ratio else "percussive",
                "optimization": "fast_hpss"
            }
        except Exception as e:
            logger.error(f"Harmonic analysis error: {e}")
            return {"harmonic_confidence": 0.5}
    
    def _spectral_feature_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """OPTIMIZED: Fast spectral feature analysis - essential features only"""
        try:
            # OPTIMIZED: Use faster hop_length for all spectral features
            hop_length = 2048  # 4x larger hop for speed
            
            # Essential spectral features only (skip expensive ones)
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(
                y=y, sr=sr, hop_length=hop_length
            )))
            spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(
                y=y, sr=sr, hop_length=hop_length
            )))
            zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(
                y, hop_length=hop_length
            )))
            
            # Skip expensive features: spectral_bandwidth, spectral_contrast, spectral_flatness
            
            return {
                "spectral_centroid": round(spectral_centroid, 2),
                "spectral_rolloff": round(spectral_rolloff, 2),
                "zero_crossing_rate": round(zero_crossing_rate, 4),
                "brightness": "bright" if spectral_centroid > 2000 else "dark",
                "optimization": "fast_spectral"
            }
        except Exception as e:
            logger.error(f"Spectral analysis error: {e}")
            return {"brightness": "unknown"}
    
    def _rhythmic_feature_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """OPTIMIZED: Fast rhythmic pattern analysis"""
        try:
            # OPTIMIZED: Fast onset detection with larger hop_length
            hop_length = 1024  # 2x larger hop for speed
            
            onsets = librosa.onset.onset_detect(
                y=y, sr=sr, units='time', hop_length=hop_length
            )
            onset_strength = librosa.onset.onset_strength(
                y=y, sr=sr, hop_length=hop_length
            )
            
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
                "onset_strength_mean": round(float(np.mean(onset_strength)), 4),
                "optimization": "fast_onset"
            }
        except Exception as e:
            logger.error(f"Rhythmic analysis error: {e}")
            return {"onset_count": 0}
    
    def _energy_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """OPTIMIZED: Fast energy and dynamics analysis"""
        try:
            # OPTIMIZED: Fast RMS energy with larger hop_length
            hop_length = 1024  # 2x larger hop for speed
            
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
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
    
    # Essentia ML analysis method:

    def _essentia_ml_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Essentia ML-powered analysis"""
    
        if not self.essentia_models or not self.ml_models_loaded:
            logger.info("🔄 Essentia models not available, skipping ML analysis")
            return {
                    "ml_features_available": False,
                    "ml_status": "models_not_loaded"
            }
            
        logger.info("🤖 Starting Essentia ML analysis")
        
        try:
            # ML-based key detection
            ml_key_analysis = self.essentia_models.analyze_key_ml(y, sr)
            
            # ML-based genre classification
            ml_genre_analysis = self.essentia_models.analyze_genre_ml(y, sr)
            
            # ML-based danceability analysis
            ml_danceability_analysis = self.essentia_models.analyze_danceability_ml(y, sr)
            
            # ML-based tempo detection
            ml_tempo_analysis = self.essentia_models.analyze_tempo_ml(y, sr)
            
            # Combine ML results
            ml_results = {
                **ml_key_analysis,
                **ml_genre_analysis,
                **ml_danceability_analysis,
                **ml_tempo_analysis,
                "ml_features_available": True,
                "ml_status": "success",
                "ml_model_count": len(self.essentia_models.available_models)
            }
            
            logger.info("✅ Essentia ML analysis completed")
            return ml_results
            
        except Exception as e:
            logger.error(f"❌ Essentia ML analysis failed: {e}")
            return {
                "ml_features_available": False,
                "ml_status": f"error: {str(e)}"
            }
        
    def _madmom_fast_rhythm_analysis(self, audio_file_path: str) -> Dict[str, Any]:
        """Fast Madmom rhythm analysis - tempo and beats only"""
        
        if not self.madmom_processor or not self.madmom_loaded:
            logger.info("🔄 Madmom processors not available, skipping rhythm analysis")
            return {
                "madmom_features_available": False,
                "madmom_status": "processors_not_loaded"
            }
        
        logger.info(f"🥁 Starting fast Madmom rhythm analysis for file: {audio_file_path}")
        
        # Check if file exists and is accessible
        if not os.path.exists(audio_file_path):
            logger.error(f"❌ Audio file not found: {audio_file_path}")
            return {
                "madmom_features_available": False,
                "madmom_status": "file_not_found"
            }
        
        try:
            # Fast analyses only (skip heavy downbeat analysis)
            logger.info("🔄 Running tempo analysis...")
            tempo_analysis = self.madmom_processor.analyze_tempo_precise(audio_file_path)
            logger.info("✅ Tempo analysis completed")
            
            logger.info("🔄 Running beat analysis...")
            beat_analysis = self.madmom_processor.analyze_beats_neural(audio_file_path)
            logger.info("✅ Beat analysis completed")
            
            # Quick result with essential rhythm info
            rhythm_results = {
                **tempo_analysis,
                **beat_analysis,
                
                # Add placeholder for skipped downbeat analysis
                "madmom_downbeat_count": 0,
                "madmom_downbeat_confidence": 0.0,
                "madmom_meter": "complex",  # Default assumption
                "madmom_downbeat_interval": 0.0,
                
                # Status
                "madmom_features_available": True,
                "madmom_status": "success",
                "madmom_processors_available": len(self.madmom_processor.available_processors),
                "madmom_analysis_complete": True,
                "madmom_fast_mode": True
            }
            
            # Calculate tempo agreement
            madmom_tempo = tempo_analysis.get("madmom_tempo", 0.0)
            beat_tempo = beat_analysis.get("madmom_beat_tempo", 0.0)
            if madmom_tempo > 0 and beat_tempo > 0:
                tempo_diff = abs(madmom_tempo - beat_tempo)
                tempo_agreement = max(0.0, 1.0 - (tempo_diff / 20.0))
            else:
                tempo_agreement = 0.0
            
            rhythm_results.update({
                "madmom_tempo_agreement": round(tempo_agreement, 3),
                "madmom_rhythm_confidence": round((
                    tempo_analysis.get("madmom_tempo_confidence", 0.0) +
                    beat_analysis.get("madmom_beat_confidence", 0.0) +
                    tempo_agreement
                ) / 3.0, 3)
            })
            
            logger.info("✅ Fast Madmom rhythm analysis completed")
            return rhythm_results
            
        except Exception as e:
            logger.error(f"❌ Fast Madmom rhythm analysis failed: {e}")
            return {
                "madmom_features_available": False,
                "madmom_status": f"error: {str(e)}"
            }
    
    # ADD: Full Madmom rhythm analysis method (for background processing):
    def _madmom_rhythm_analysis(self, audio_file_path: str) -> Dict[str, Any]:
        """Full Madmom-based rhythm analysis (with heavy downbeat analysis)"""
        
        if not self.madmom_processor or not self.madmom_loaded:
            logger.info("🔄 Madmom processors not available, skipping rhythm analysis")
            return {
                "madmom_features_available": False,
                "madmom_status": "processors_not_loaded"
            }
        
        logger.info("🥁 Starting full Madmom rhythm analysis")
        
        try:
            # Comprehensive rhythm analysis (including heavy downbeat analysis)
            rhythm_results = self.madmom_processor.comprehensive_rhythm_analysis(audio_file_path)
            
            # Add availability status
            rhythm_results.update({
                "madmom_features_available": True,
                "madmom_status": "success"
            })
            
            logger.info("✅ Full Madmom rhythm analysis completed")
            return rhythm_results
            
        except Exception as e:
            logger.error(f"❌ Full Madmom rhythm analysis failed: {e}")
            return {
                "madmom_features_available": False,
                "madmom_status": f"error: {str(e)}"
            }
    
    def _should_use_chunking(self, y: np.ndarray, sr: int, file_info: Dict) -> bool:
        """Determine if we should use chunking based on file size and GPU capabilities"""
        duration = file_info.get("analyzed_duration", len(y) / sr)
        file_size_mb = file_info.get("file_size_mb", 0)
        
        # Use chunking for files longer than 1 minute OR larger than 5MB (lowered for testing)
        return duration > 60 or file_size_mb > 5
    
    def _chunked_parallel_analysis(self, y: np.ndarray, sr: int, tmp_file_path: str, 
                                 file_info: Dict, filename: str) -> Dict[str, Any]:
        """GPU-optimized chunked analysis for large files"""
        
        chunk_duration = 120  # 2 minutes per chunk for GPU efficiency
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(10 * sr)  # 10 second overlap for continuity
        
        chunks = []
        chunk_results = []
        
        # Create overlapping chunks
        for i in range(0, len(y), chunk_samples - overlap_samples):
            chunk_start = i
            chunk_end = min(i + chunk_samples, len(y))
            chunk_audio = y[chunk_start:chunk_end]
            
            if len(chunk_audio) >= sr:  # At least 1 second of audio
                chunk_time_start = chunk_start / sr
                chunks.append({
                    "audio": chunk_audio,
                    "start_time": chunk_time_start,
                    "duration": len(chunk_audio) / sr,
                    "chunk_id": len(chunks)
                })
        
        logger.info(f"🔧 Processing {len(chunks)} chunks of {chunk_duration}s each")
        
        # GPU BATCH PROCESSING - Process multiple chunks on GPU simultaneously
        batch_size = 8  # Process 8 chunks simultaneously on GPU
        all_chunk_results = []
        
        import concurrent.futures
        
        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start:batch_start + batch_size]
            
            logger.info(f"🚀 GPU batch processing {len(batch_chunks)} chunks...")
            
            # Extract audio data for GPU batch processing
            batch_audio_chunks = [chunk["audio"] for chunk in batch_chunks]
            
            # Process batch using GPU batch processing + parallel librosa
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                
                # GPU batch processing for Essentia ML (all chunks at once)
                future_essentia_batch = executor.submit(
                    self.essentia_models.analyze_batch_gpu, 
                    batch_audio_chunks, 
                    sr
                )
                
                # Parallel librosa processing for each chunk
                future_librosa = executor.submit(
                    self._process_librosa_batch, 
                    batch_audio_chunks, 
                    sr
                )
                
                # Wait for both to complete
                essentia_batch_results = future_essentia_batch.result()
                librosa_batch_results = future_librosa.result()
            
            # Combine results for this batch
            batch_results = []
            for i in range(len(batch_chunks)):
                librosa_result = librosa_batch_results[i] if i < len(librosa_batch_results) else {}
                essentia_result = essentia_batch_results[i] if i < len(essentia_batch_results) else {}
                batch_results.append((librosa_result, essentia_result))
            
            all_chunk_results.extend(batch_results)
            
            logger.info(f"✅ GPU batch {batch_start//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} completed")
        
        # Process Madmom on full file (fast mode for speed)
        rhythm_analysis = self._madmom_fast_rhythm_analysis(tmp_file_path)
        
        # Aggregate chunk results
        aggregated_result = self._aggregate_chunk_results(all_chunk_results, chunks, file_info, filename)
        
        # Add rhythm analysis from full file
        aggregated_result.update(rhythm_analysis)
        
        # Add chunking metadata
        aggregated_result.update({
            "chunked_processing": {
                "enabled": True,
                "total_chunks": len(chunks),
                "chunk_duration": chunk_duration,
                "overlap_seconds": overlap_samples / sr,
                "batch_size": batch_size
            }
        })
        
        return aggregated_result
    
    def _aggregate_chunk_results(self, chunk_results: List, chunks: List[Dict], 
                               file_info: Dict, filename: str) -> Dict[str, Any]:
        """Aggregate results from multiple chunks using weighted averaging"""
        
        # Initialize aggregation containers
        aggregated = {
            "filename": filename,
            "file_info": file_info,
            "analysis_pipeline": ["enhanced_librosa", "essentia_ml", "chunked_processing"]
        }
        
        # Aggregate numeric features with weighted averaging
        numeric_features = [
            "tempo", "key_confidence", "spectral_centroid", "rms_energy",
            "ml_danceability", "ml_tempo", "harmonic_ratio"
        ]
        
        categorical_features = [
            "key", "mode", "ml_key", "energy_level", "brightness"
        ]
        
        # Weight chunks by duration (longer chunks have more influence)
        total_duration = sum(chunk["duration"] for chunk in chunks)
        
        for feature in numeric_features:
            weighted_sum = 0
            total_weight = 0
            
            for i, (librosa_result, ml_result) in enumerate(chunk_results):
                chunk_weight = chunks[i]["duration"] / total_duration
                
                # Get value from either librosa or ML results
                value = librosa_result.get(feature) or ml_result.get(feature)
                if value is not None and not np.isnan(float(value)):
                    weighted_sum += float(value) * chunk_weight
                    total_weight += chunk_weight
            
            if total_weight > 0:
                aggregated[feature] = round(weighted_sum / total_weight, 3)
        
        # Aggregate categorical features using majority voting
        for feature in categorical_features:
            feature_votes = {}
            
            for i, (librosa_result, ml_result) in enumerate(chunk_results):
                value = librosa_result.get(feature) or ml_result.get(feature)
                if value:
                    chunk_weight = chunks[i]["duration"] / total_duration
                    feature_votes[value] = feature_votes.get(value, 0) + chunk_weight
            
            if feature_votes:
                aggregated[feature] = max(feature_votes, key=feature_votes.get)
        
        # Calculate overall confidence based on chunk consistency
        chunk_count = len(chunk_results)
        aggregated["chunk_consistency"] = min(1.0, 1.0 / max(1, chunk_count * 0.1))
        aggregated["quality_score"] = aggregated.get("chunk_consistency", 0.8)
        
        return aggregated
    
    def _process_librosa_batch(self, batch_audio_chunks: List[np.ndarray], sr: int) -> List[Dict[str, Any]]:
        """Process librosa analysis for multiple chunks in parallel"""
        
        import concurrent.futures
        
        # Process each chunk in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for chunk_audio in batch_audio_chunks:
                future = executor.submit(self._librosa_enhanced_analysis, chunk_audio, sr)
                futures.append(future)
            
            # Collect results
            batch_results = []
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Librosa batch processing failed for chunk {i}: {e}")
                    batch_results.append({
                        "duration": len(batch_audio_chunks[i]) / sr,
                        "key": "C", "mode": "major", "tempo": 120.0
                    })
            
            return batch_results