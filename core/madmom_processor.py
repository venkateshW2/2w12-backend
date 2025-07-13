# NEW FILE: core/madmom_processor.py
import madmom
import numpy as np
import logging
from typing import Dict, Any, List
from madmom.features.tempo import TempoEstimationProcessor
from madmom.features.beats import RNNBeatProcessor
from madmom.features.downbeats import RNNDownBeatProcessor

logger = logging.getLogger(__name__)

class MadmomProcessor:
    """
    Madmom-based downbeat and meter analysis for 2W12 Sound Tools
    
    FOCUSED ON:
    - Downbeat detection (ONLY)
    - Meter/time signature analysis (ONLY)
    - Timeline generation for downbeats
    
    NOTE: Tempo detection now handled by ML models
    """
    
    def __init__(self):
        self.processors_loaded = False
        self.available_processors = {}
        self.processors_ready = False
        
        self._load_processors()
    
    def _load_processors(self):
        """Load Madmom processors - DOWNBEAT FOCUSED ONLY"""
        try:
            logger.info("🔄 Loading Madmom downbeat processors...")
            
            # PRIORITY #2: Load only what we need for downbeat detection
            from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
            from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
            
            # RNN Downbeat processor (PRIMARY FOCUS)
            self.downbeat_processor = RNNDownBeatProcessor()
            self.downbeat_tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=50)
            logger.info("✅ RNN downbeat processor loaded (PRIORITY #2)")
            
            # Beat processor (minimal, only for downbeat support)
            self.beat_processor = RNNBeatProcessor()
            self.beat_tracker = BeatTrackingProcessor(fps=50)
            logger.info("✅ RNN beat processor loaded (supporting downbeats)")
            
            self.processors_loaded = True
            self.processors_ready = True
            logger.info("🚀 Madmom DOWNBEATS ONLY processors ready - optimized for speed")
            
        except Exception as e:
            logger.error(f"❌ Madmom processor loading failed: {e}")
            self.processors_loaded = False
            self.processors_ready = False
    
    def analyze_downbeats_only_numpy(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """PRIORITY #2: Pure downbeat detection from numpy array - nothing else
        Give audio array to Madmom → get ONLY downbeats, optimized for speed"""
        
        if not self.processors_ready:
            logger.warning("⚠️ Madmom processors not ready")
            return {"madmom_downbeat_count": 0, "madmom_status": "processors_not_ready"}
        
        try:
            import time
            logger.info("🥁 Madmom DOWNBEATS ONLY analysis (numpy input)")
            start_time = time.time()
            
            # Direct numpy array processing - no file I/O
            audio_data = y.astype(np.float32)
            if sr != 22050:
                # Resample if needed for consistent processing
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=22050)
                sr = 22050
            
            # APPROACH: Direct processing without SignalProcessor file I/O
            from madmom.audio.signal import Signal
            
            # Create Signal object directly from numpy array
            signal = Signal(audio_data, sample_rate=sr, num_channels=1)
            
            # Process beats (required for downbeats)
            beat_activations = self.beat_processor(signal)
            beats = self.beat_tracker(beat_activations)
            
            # Process downbeats (MAIN TARGET)
            downbeat_activations = self.downbeat_processor(signal)
            downbeats = self.downbeat_tracker(downbeat_activations)
            
            processing_time = time.time() - start_time
            duration = len(y) / sr
            realtime_factor = processing_time / duration
            
            if len(downbeats) > 0:
                # Extract downbeat times and positions
                if downbeats.ndim == 2:
                    downbeat_times = downbeats[:, 0]  # Time in seconds
                    downbeat_positions = downbeats[:, 1]  # Position in bar (1=downbeat)
                    
                    # Filter actual downbeats (position = 1)
                    actual_downbeats = downbeat_times[downbeat_positions == 1]
                else:
                    actual_downbeats = downbeats
                
                if len(actual_downbeats) > 0:
                    # Calculate intervals between downbeats
                    downbeat_intervals = np.diff(actual_downbeats) if len(actual_downbeats) > 1 else np.array([])
                    
                    # Estimate meter from intervals (simple approach)
                    if len(downbeat_intervals) > 0:
                        mean_interval = np.mean(downbeat_intervals)
                        # Simple meter estimation based on interval
                        if mean_interval > 3.0:
                            estimated_meter = 4.0
                            meter_detection = "4/4"
                        elif mean_interval > 2.0:
                            estimated_meter = 3.0
                            meter_detection = "3/4"
                        else:
                            estimated_meter = 4.0  # Default
                            meter_detection = "4/4"
                    else:
                        mean_interval = 0.0
                        estimated_meter = 4.0
                        meter_detection = "4/4"
                    
                    logger.info(f"✅ Madmom downbeats: {len(actual_downbeats)} detected in {processing_time:.2f}s ({realtime_factor:.2f}x realtime)")
                    
                    return {
                        "madmom_downbeat_count": len(actual_downbeats),
                        "madmom_downbeat_times": actual_downbeats.tolist(),
                        "madmom_downbeat_intervals": downbeat_intervals.tolist(),
                        "madmom_meter_estimated": estimated_meter,
                        "madmom_meter_detection": meter_detection,
                        "madmom_timeline_available": True,
                        "madmom_processing_time": round(processing_time, 2),
                        "madmom_realtime_factor": round(realtime_factor, 3),
                        "madmom_mode": "downbeats_only_optimized_numpy",
                        "madmom_status": "success",
                        "madmom_input_type": "numpy_array_direct"
                    }
                else:
                    logger.warning("⚠️ No actual downbeats found (position filtering)")
                    return {
                        "madmom_downbeat_count": 0,
                        "madmom_status": "no_actual_downbeats",
                        "madmom_processing_time": round(processing_time, 2)
                    }
            else:
                logger.warning("⚠️ No downbeats detected by Madmom")
                return {
                    "madmom_downbeat_count": 0,
                    "madmom_status": "no_downbeats_detected",
                    "madmom_processing_time": round(processing_time, 2)
                }
        
        except Exception as e:
            logger.error(f"❌ Madmom downbeat-only analysis failed: {e}")
            return {
                "madmom_downbeat_count": 0,
                "madmom_status": "error",
                "madmom_error": str(e),
                "madmom_error_type": type(e).__name__
            }

    def analyze_tempo_precise(self, audio_file_path: str) -> Dict[str, Any]:
        """
        High-precision tempo analysis using Madmom
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Dict with precise tempo analysis
        """
        
        if "tempo" not in self.available_processors:
            logger.warning("⚠️  Tempo processor not available")
            return {"madmom_tempo": 0.0, "madmom_tempo_confidence": 0.0}
        
        try:
            # Process tempo with Madmom - need to use the correct pipeline
            from madmom.features.tempo import TempoEstimationProcessor
            from madmom.features.beats import RNNBeatProcessor
            from madmom.audio import SignalProcessor
            
            # Create the proper pipeline (optimized for speed)
            sig = SignalProcessor(num_channels=1, sample_rate=22050, norm=True)  # Reduced sample rate
            frames = sig(audio_file_path)
            
            # Use RNN beat processor for tempo estimation
            beat_proc = RNNBeatProcessor()
            beat_activations = beat_proc(frames)
            
            # Use tempo estimation processor (optimized)
            tempo_proc = TempoEstimationProcessor(fps=50)  # Reduced from 100 fps
            tempi = tempo_proc(beat_activations)
            
            if len(tempi) > 0:
                # Primary tempo (strongest peak)
                primary_tempo = float(tempi[0][0])
                primary_strength = float(tempi[0][1])
                
                # Secondary tempo (if exists)
                secondary_tempo = float(tempi[1][0]) if len(tempi) > 1 else None
                secondary_strength = float(tempi[1][1]) if len(tempi) > 1 else 0.0
                
                # Tempo confidence based on strength
                tempo_confidence = min(1.0, primary_strength / 10.0)  # Normalize strength
                
                return {
                    "madmom_tempo": round(primary_tempo, 1),
                    "madmom_tempo_confidence": round(tempo_confidence, 3),
                    "madmom_tempo_strength": round(primary_strength, 3),
                    "madmom_secondary_tempo": round(secondary_tempo, 1) if secondary_tempo else None,
                    "madmom_tempo_candidates": len(tempi),
                    "madmom_model": "rnn_tempo_estimation"
                }
            else:
                return {
                    "madmom_tempo": 0.0,
                    "madmom_tempo_confidence": 0.0,
                    "madmom_model": "rnn_tempo_estimation",
                    "madmom_status": "no_tempo_detected"
                }
                
        except Exception as e:
            logger.error(f"❌ Madmom tempo analysis failed: {e}")
            return {
                "madmom_tempo": 0.0,
                "madmom_tempo_confidence": 0.0,
                "madmom_error": str(e)
            }
    
    def analyze_beats_neural(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Neural network-based beat tracking
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Dict with beat analysis
        """
        
        if "beats" not in self.available_processors:
            logger.warning("⚠️  Beat processor not available")
            return {"madmom_beat_count": 0, "madmom_beat_confidence": 0.0}
        
        try:
            # Process beats with RNN - use correct pipeline
            from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
            from madmom.audio import SignalProcessor
            
            # Create the proper pipeline (optimized)
            sig = SignalProcessor(num_channels=1, sample_rate=22050, norm=True)  # Reduced sample rate
            frames = sig(audio_file_path)
            
            # Process beats with RNN
            beat_proc = RNNBeatProcessor()
            beat_activations = beat_proc(frames)
            
            # Convert activations to beat times (optimized)
            beat_tracker = BeatTrackingProcessor(fps=50)  # Reduced from 100 fps
            beats = beat_tracker(beat_activations)
            
            if len(beats) > 0:
                # Rest of your code stays the same...
                beat_times = beats  # beats are already just times, not [time, confidence]
                
                # Beat statistics
                beat_intervals = np.diff(beat_times)
                mean_interval = np.mean(beat_intervals) if len(beat_intervals) > 0 else 0.0
                interval_consistency = 1.0 / (1.0 + np.std(beat_intervals)) if len(beat_intervals) > 1 else 0.0
                
                # Estimated tempo from beats
                estimated_tempo = 60.0 / mean_interval if mean_interval > 0 else 0.0
                
                return {
                    "madmom_beat_count": len(beats),
                    "madmom_beat_confidence": 0.8,  # Fixed confidence since we don't have individual confidences
                    "madmom_beat_consistency": round(interval_consistency, 3),
                    "madmom_beat_tempo": round(estimated_tempo, 1),
                    "madmom_beat_interval_mean": round(mean_interval, 3),
                    "madmom_model": "rnn_beat_tracking"
                }
            else:
                return {
                    "madmom_beat_count": 0,
                    "madmom_beat_confidence": 0.0,
                    "madmom_model": "rnn_beat_tracking",
                    "madmom_status": "no_beats_detected"
                }
                
        except Exception as e:
            logger.error(f"❌ Madmom beat analysis failed: {e}")
            return {
                "madmom_beat_count": 0,
                "madmom_beat_confidence": 0.0,
                "madmom_error": str(e)
            }
    
    def analyze_downbeats_timeline(self, audio_file_path: str) -> Dict[str, Any]:
        """
        FOCUSED: Downbeat detection and meter analysis with timeline generation
        Uses ffmpeg for robust file loading
        """
        
        if not self.processors_loaded:
            logger.warning("⚠️ Madmom processors not loaded, skipping downbeat analysis")
            return {
                "madmom_downbeats_available": False,
                "madmom_status": "processors_not_loaded"
            }
        
        logger.info(f"🥁 Starting Madmom downbeat analysis for: {audio_file_path}")
        
        try:
            # Import required processors
            from madmom.audio.signal import SignalProcessor
            from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
            from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
            
            # Load audio with ffmpeg support
            logger.info("🔄 Loading audio with ffmpeg support...")
            sig = SignalProcessor(num_channels=1, sample_rate=22050, norm=True)
            frames = sig(audio_file_path)
            logger.info(f"✅ Audio loaded: {len(frames)} frames")
            
            # OPTIMIZED: Downbeat-ONLY detection (no beat tracking)
            logger.info("🔄 Downbeat-ONLY detection (no beat tracking)...")
            downbeat_proc = RNNDownBeatProcessor()
            downbeat_activations = downbeat_proc(frames)
            
            downbeat_tracker = DBNDownBeatTrackingProcessor(
                beats_per_bar=[3, 4], fps=50  # Focus on 3/4 and 4/4 time
            )
            downbeats = downbeat_tracker(downbeat_activations)
            
            logger.info(f"✅ Downbeat-only detection completed: {len(downbeats)} downbeat positions")
            
            if len(downbeats) > 0:
                # Extract downbeat times and positions
                downbeat_times = downbeats[:, 0]  # Time in seconds
                downbeat_positions = downbeats[:, 1]  # Position in bar (1=downbeat)
                
                # Filter actual downbeats (position = 1)
                actual_downbeats = downbeat_times[downbeat_positions == 1]
                
                logger.info(f"✅ Detected {len(actual_downbeats)} downbeats")
                
                # Meter analysis (simplified without beat tracking)
                if len(actual_downbeats) > 1:
                    downbeat_intervals = np.diff(actual_downbeats)
                    mean_bar_length = np.mean(downbeat_intervals)
                    
                    # Estimate meter from downbeat intervals only
                    if mean_bar_length > 3.0:
                        estimated_meter = 4.0  # Likely 4/4
                    elif mean_bar_length > 1.5:
                        estimated_meter = 3.0  # Likely 3/4
                    else:
                        estimated_meter = 4.0  # Default to 4/4
                    
                    # Confidence based on consistency
                    interval_consistency = 1.0 / (1.0 + np.std(downbeat_intervals))
                    
                    return {
                        "madmom_downbeat_count": len(actual_downbeats),
                        "madmom_downbeat_times": actual_downbeats.tolist(),
                        "madmom_downbeat_intervals": downbeat_intervals.tolist(),
                        "madmom_meter_estimated": round(estimated_meter, 1),
                        "madmom_bar_length_mean": round(mean_bar_length, 3),
                        "madmom_downbeat_confidence": round(interval_consistency, 3),
                        "madmom_meter_detection": "4/4" if estimated_meter > 3.5 else "3/4",
                        "madmom_timeline_available": True,
                        "madmom_status": "success",
                        "madmom_model": "rnn_downbeat_dbn"
                    }
                else:
                    return {
                        "madmom_downbeat_count": len(actual_downbeats),
                        "madmom_downbeat_confidence": 0.3,
                        "madmom_meter_detection": "4/4",  # Default assumption
                        "madmom_timeline_available": False,
                        "madmom_status": "insufficient_downbeats"
                    }
            else:
                return {
                    "madmom_downbeat_count": 0,
                    "madmom_downbeat_confidence": 0.0,
                    "madmom_timeline_available": False,
                    "madmom_status": "no_downbeats_detected"
                }
                
        except Exception as e:
            logger.error(f"❌ Madmom downbeat analysis failed: {e}")
            return {
                "madmom_downbeats_available": False,
                "madmom_status": f"error: {str(e)}",
                "madmom_error": str(e)
            }
    
    def analyze_downbeats_meter(self, audio_file_path: str) -> Dict[str, Any]:
        """Downbeat detection and meter analysis (legacy method)"""
        
        if "downbeats" not in self.available_processors:
            logger.warning("⚠️  Downbeat processor not available")
            return {"madmom_meter": "unknown", "madmom_downbeat_count": 0}
        
        try:
            # Process downbeats with RNN - use correct pipeline
            from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
            from madmom.audio import SignalProcessor
            
            # Create the proper pipeline (optimized)
            sig = SignalProcessor(num_channels=1, sample_rate=22050, norm=True)  # Reduced sample rate
            frames = sig(audio_file_path)
            
            # Process downbeats with RNN
            downbeat_proc = RNNDownBeatProcessor()
            downbeat_activations = downbeat_proc(frames)
            
            # Convert activations to downbeat times (optimized)
            downbeat_tracker = DBNDownBeatTrackingProcessor(beats_per_bar=4, fps=50)  # Reduced from 100 fps
            downbeats = downbeat_tracker(downbeat_activations)
            
            if len(downbeats) > 0 and hasattr(downbeats, 'ndim') and downbeats.ndim > 0:
                # Downbeat analysis
                if downbeats.ndim == 2:
                    downbeat_times = downbeats[:, 0]  # Downbeat positions
                    downbeat_confidences = downbeats[:, 1] if downbeats.shape[1] > 1 else np.ones(len(downbeat_times)) * 0.5
                else:
                    downbeat_times = downbeats
                    downbeat_confidences = np.ones(len(downbeat_times)) * 0.5
                
                # Estimate meter from downbeat intervals
                if len(downbeat_times) > 1:
                    downbeat_intervals = np.diff(downbeat_times)
                    mean_downbeat_interval = np.mean(downbeat_intervals)
                    
                    # Simple meter estimation (this is simplified)
                    if mean_downbeat_interval > 4.0:
                        estimated_meter = "4/4"
                    elif mean_downbeat_interval > 2.5:
                        estimated_meter = "3/4"
                    elif mean_downbeat_interval > 1.5:
                        estimated_meter = "2/4"
                    else:
                        estimated_meter = "complex"
                else:
                    estimated_meter = "unknown"
                    mean_downbeat_interval = 0.0
                
                mean_confidence = np.mean(downbeat_confidences)
                
                return {
                    "madmom_downbeat_count": len(downbeat_times),
                    "madmom_downbeat_confidence": round(float(mean_confidence), 3),
                    "madmom_meter": estimated_meter,
                    "madmom_downbeat_interval": round(float(mean_downbeat_interval), 2),
                    "madmom_model": "rnn_downbeat_tracking"
                }
            else:
                return {
                    "madmom_downbeat_count": 0,
                    "madmom_downbeat_confidence": 0.0,
                    "madmom_meter": "unknown",
                    "madmom_model": "rnn_downbeat_tracking",
                    "madmom_status": "no_downbeats_detected"
                }
                
        except Exception as e:
            logger.error(f"❌ Madmom downbeat analysis failed: {e}")
            return {
                "madmom_downbeat_count": 0,
                "madmom_downbeat_confidence": 0.0,
                "madmom_meter": "unknown",
                "madmom_error": str(e)
            }
    
    def comprehensive_rhythm_analysis(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Complete rhythm analysis combining all Madmom features
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Dict with comprehensive rhythm analysis
        """
        
        logger.info("🥁 Starting comprehensive Madmom rhythm analysis")
        
        try:
            # Combine all analyses
            tempo_analysis = self.analyze_tempo_precise(audio_file_path)
            beat_analysis = self.analyze_beats_neural(audio_file_path)
            downbeat_analysis = self.analyze_downbeats_meter(audio_file_path)
            
            # Cross-validation between methods
            madmom_tempo = tempo_analysis.get("madmom_tempo", 0.0)
            beat_tempo = beat_analysis.get("madmom_beat_tempo", 0.0)
            
            # Tempo agreement confidence
            if madmom_tempo > 0 and beat_tempo > 0:
                tempo_diff = abs(madmom_tempo - beat_tempo)
                tempo_agreement = max(0.0, 1.0 - (tempo_diff / 20.0))  # Within 20 BPM = good agreement
            else:
                tempo_agreement = 0.0
            
            # Overall rhythm confidence
            rhythm_confidence = np.mean([
                tempo_analysis.get("madmom_tempo_confidence", 0.0),
                beat_analysis.get("madmom_beat_confidence", 0.0),
                downbeat_analysis.get("madmom_downbeat_confidence", 0.0),
                tempo_agreement
            ])
            
            comprehensive_result = {
                # Individual analyses
                **tempo_analysis,
                **beat_analysis,
                **downbeat_analysis,
                
                # Cross-validation
                "madmom_tempo_agreement": round(tempo_agreement, 3),
                "madmom_rhythm_confidence": round(rhythm_confidence, 3),
                
                # Analysis metadata
                "madmom_processors_available": len(self.available_processors),
                "madmom_analysis_complete": True
            }
            
            logger.info("✅ Madmom rhythm analysis completed")
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"❌ Madmom comprehensive analysis failed: {e}")
            return {
                "madmom_analysis_complete": False,
                "madmom_error": str(e)
            }
    
    def get_processor_status(self) -> Dict[str, Any]:
        """Get status of loaded processors"""
        return {
            "processors_loaded": self.processors_loaded,
            "available_processors": list(self.available_processors.keys()),
            "processor_count": len(self.available_processors)
        }