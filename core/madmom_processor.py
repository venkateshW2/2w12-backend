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
    Madmom-based rhythm and tempo analysis for 2W12 Sound Tools
    
    Specializes in:
    - High-precision tempo detection
    - Beat tracking with neural networks
    - Downbeat detection
    - Meter/time signature analysis
    - Rhythmic pattern analysis
    """
    
    def __init__(self):
        self.processors_loaded = False
        self.available_processors = {}
        
        self._load_processors()
    
    def _load_processors(self):
        """Load Madmom processors"""
        try:
            logger.info("🔄 Loading Madmom processors...")
            
            # Tempo estimation processor (ultra-fast mode)
            self.available_processors["tempo"] = TempoEstimationProcessor(fps=25)  # Further reduced to 25 fps for speed
            logger.info("✅ Tempo estimation processor loaded")
            
            # RNN Beat processor
            self.available_processors["beats"] = RNNBeatProcessor()
            logger.info("✅ RNN beat processor loaded")
            
            # RNN Downbeat processor (for meter detection)
            self.available_processors["downbeats"] = RNNDownBeatProcessor()
            logger.info("✅ RNN downbeat processor loaded")
            
            self.processors_loaded = True
            logger.info("🚀 Madmom processors ready")
            
        except Exception as e:
            logger.error(f"❌ Madmom processor loading failed: {e}")
            self.processors_loaded = False
    
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
    
    def analyze_downbeats_meter(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Downbeat detection and meter analysis
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Dict with downbeat and meter analysis
        """
        
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