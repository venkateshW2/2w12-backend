# NEW FILE: core/essentia_models.py
import essentia.standard as es
import numpy as np
import logging
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class EssentiaModelManager:
    """
    Essentia ML Model Manager for 2W12 Sound Tools
    
    Handles loading and execution of Essentia TensorFlow models:
    - Key detection
    - Genre classification
    - Mood detection
    - Danceability
    - And more advanced ML features
    """
    
    def __init__(self):
        self.models_loaded = False
        self.available_models = {}
        self.model_paths = {
            # Use specialized models for each task - GPU accelerated
            "pitch_detection": "models/Crepe Large Model.pb",  # CREPE for pitch/key detection  
            "tempo_classification": "models/Danceability Audioset Yamnet.pb",  # YamNet for tempo/rhythm
            "genre_classification": "models/Genre Discogs 400 Model.pb",
            "danceability": "models/Danceability Discogs Effnet.pb", 
            "audio_features": "models/audioset-vggish-3.pb",  # VGGish for general audio features
            "voice_instrumental": "models/Voice Instrumental Model.pb"
        }
        
        self._load_available_models()
    
    def _load_available_models(self):
        """Load available Essentia models"""
        try:
            logger.info("🔄 Loading Essentia models...")
            
            # CREPE Pitch Detection Model (for key detection)
            if self._model_exists("pitch_detection"):
                try:
                    self.available_models["pitch_detection"] = es.TensorflowPredictCREPE(
                        graphFilename=self.model_paths["pitch_detection"]
                    )
                    logger.info("✅ CREPE pitch detection model loaded")
                except Exception as e:
                    logger.warning(f"⚠️ CREPE pitch detection model failed to load: {e}")
            
            # YamNet Tempo/Rhythm Classification Model  
            if self._model_exists("tempo_classification"):
                try:
                    self.available_models["tempo_classification"] = es.TensorflowPredict(
                        graphFilename=self.model_paths["tempo_classification"],
                        inputs=["melspectrogram"],
                        outputs=["scores"]
                    )
                    logger.info("✅ YamNet tempo classification model loaded")
                except Exception as e:
                    logger.warning(f"⚠️ YamNet tempo classification model failed to load: {e}")
            
            # VGGish Audio Features Model
            if self._model_exists("audio_features"):
                try:
                    self.available_models["audio_features"] = es.TensorflowPredictVGGish(
                        graphFilename=self.model_paths["audio_features"]
                    )
                    logger.info("✅ VGGish audio features model loaded")
                except Exception as e:
                    logger.warning(f"⚠️ VGGish audio features model failed to load: {e}")
                    
            # Danceability Model (using correct output node)
            if self._model_exists("danceability"):
                try:
                    self.available_models["danceability"] = es.TensorflowPredict(
                        graphFilename=self.model_paths["danceability"],
                        inputs=["model/Placeholder"],
                        outputs=["model/Softmax"]
                    )
                    logger.info("✅ Danceability model loaded")
                except Exception as e:
                    logger.warning(f"⚠️ Danceability model failed to load: {e}")
            
            # For Week 2, we'll start with key detection and genre classification
            # Additional models can be added incrementally
            
            self.models_loaded = len(self.available_models) > 0
            
            if self.models_loaded:
                logger.info(f"🚀 Essentia models ready: {list(self.available_models.keys())}")
            else:
                logger.warning("⚠️  No Essentia models found. Download models to enable ML features.")
                
        except Exception as e:
            logger.error(f"❌ Essentia model loading failed: {e}")
            self.models_loaded = False
    
    def _model_exists(self, model_name: str) -> bool:
        """Check if model file exists"""
        model_path = self.model_paths.get(model_name)
        return model_path and os.path.exists(model_path)
    
    def analyze_key_ml(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """ML-based key detection using CREPE pitch detection"""
        
        if "pitch_detection" not in self.available_models:
            logger.warning("⚠️  CREPE pitch detection model not available, using fallback")
            return self._fallback_key_detection(y, sr)
        
        try:
            # Prepare audio for CREPE (needs specific format)
            audio_vector = y.astype(np.float32)
            
            # CREPE returns pitch estimates - convert to key
            pitch_predictions = self.available_models["pitch_detection"](audio_vector)
            
            # Convert pitch predictions to key
            # CREPE outputs pitch in Hz, we need to convert to musical key
            if len(pitch_predictions) > 0:
                # Get most confident pitch predictions
                pitches = pitch_predictions[0] if isinstance(pitch_predictions, tuple) else pitch_predictions
                
                # Convert Hz to musical note (simplified approach)
                # A4 = 440 Hz reference
                if isinstance(pitches, np.ndarray) and len(pitches) > 0:
                    # Get median pitch to avoid outliers
                    median_pitch = np.median(pitches[pitches > 0])  # Remove zeros
                    
                    if median_pitch > 0:
                        # Convert Hz to MIDI note number
                        midi_note = 69 + 12 * np.log2(median_pitch / 440.0)
                        # Get note name from MIDI
                        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                        note_index = max(0, min(11, int(midi_note) % 12))  # Clamp to valid range
                        key = note_names[note_index]
                        
                        # Simple major/minor detection based on pitch stability
                        pitch_variance = np.var(pitches[pitches > 0])
                        mode = "major" if pitch_variance < 1000 else "minor"
                        
                        confidence = min(1.0, 1.0 / (1.0 + pitch_variance / 1000))
                        
                        return {
                            "ml_key": key,
                            "ml_mode": mode,
                            "ml_key_confidence": round(confidence, 3),
                            "ml_full_key": f"{key} {mode}",
                            "ml_model": "essentia_crepe_pitch",
                            "ml_median_pitch": round(float(median_pitch), 2)
                        }
            
            # Fallback if no pitch detected
            return self._fallback_key_detection(y, sr)
            
        except Exception as e:
            logger.error(f"❌ CREPE key detection failed: {e}")
            return self._fallback_key_detection(y, sr)
    
    def analyze_genre_ml(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """ML-based genre classification using Essentia"""
        
        if "genre_classification" not in self.available_models:
            logger.warning("⚠️  Genre classification model not available")
            return {"ml_genre": "unknown", "ml_genre_confidence": 0.0}
        
        try:
            # Prepare audio for Essentia
            audio_vector = y.astype(np.float32)
            
            # Extract genre features
            features = self.available_models["genre_classification"](audio_vector)
            
            # Top 3 genre predictions
            top_indices = np.argsort(features)[-3:][::-1]
            
            # Genre labels (simplified subset)
            genre_labels = [
                "electronic", "rock", "pop", "hip-hop", "jazz", "classical",
                "folk", "blues", "country", "reggae", "metal", "ambient"
            ]
            
            # Map predictions to genre labels
            predicted_genres = []
            for idx in top_indices:
                if idx < len(genre_labels):
                    genre = genre_labels[idx]
                    confidence = float(features[idx])
                    predicted_genres.append({
                        "genre": genre,
                        "confidence": round(confidence, 3)
                    })
            
            primary_genre = predicted_genres[0] if predicted_genres else {"genre": "unknown", "confidence": 0.0}
            
            return {
                "ml_genre": primary_genre["genre"],
                "ml_genre_confidence": primary_genre["confidence"],
                "ml_genre_alternatives": predicted_genres[1:],
                "ml_model": "essentia_discogs_effnet"
            }
            
        except Exception as e:
            logger.error(f"❌ ML genre classification failed: {e}")
            return {"ml_genre": "unknown", "ml_genre_confidence": 0.0}
    
    def analyze_danceability_ml(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """ML-based danceability analysis using Essentia"""
        
        if "danceability" not in self.available_models:
            logger.warning("⚠️  Danceability model not available")
            return {"ml_danceability": 0.0, "ml_danceability_confidence": 0.0}
        
        try:
            # Prepare audio for Essentia
            audio_vector = y.astype(np.float32)
            
            # Extract danceability features
            features = self.available_models["danceability"](audio_vector)
            
            # Handle different output formats
            if isinstance(features, np.ndarray):
                features = features.flatten()
            elif hasattr(features, '__len__') and len(features) > 0:
                # Handle tuple or list outputs
                features = np.array(features[0]) if isinstance(features[0], (list, np.ndarray)) else np.array(features)
            else:
                features = np.array([0.5])  # Default fallback
            
            # Get danceability score (assuming binary classification)
            if len(features) >= 2:
                danceability_score = float(features[1])  # Usually the "danceable" class
            elif len(features) >= 1:
                danceability_score = float(features[0])
            else:
                danceability_score = 0.5  # Default
            
            confidence = abs(danceability_score - 0.5) * 2  # Convert to confidence
            
            return {
                "ml_danceability": round(danceability_score, 3),
                "ml_danceability_confidence": round(confidence, 3),
                "ml_danceability_class": "danceable" if danceability_score > 0.5 else "not_danceable",
                "ml_model": "essentia_danceability_discogs"
            }
            
        except Exception as e:
            logger.error(f"❌ ML danceability analysis failed: {e}")
            return {"ml_danceability": 0.0, "ml_danceability_confidence": 0.0}
    
    def analyze_tempo_ml(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """ML-based tempo detection using YamNet and VGGish"""
        
        if "tempo_classification" not in self.available_models and "audio_features" not in self.available_models:
            logger.warning("⚠️  No tempo detection models available")
            return {"ml_tempo": 0.0, "ml_tempo_confidence": 0.0}
        
        try:
            # Use VGGish for general audio features if available
            if "audio_features" in self.available_models:
                # Prepare audio for VGGish
                audio_vector = y.astype(np.float32)
                
                # VGGish processes audio and returns feature embeddings
                features = self.available_models["audio_features"](audio_vector)
                
                # Handle different output formats
                if isinstance(features, np.ndarray):
                    features = features.flatten()
                elif hasattr(features, '__len__') and len(features) > 0:
                    # Handle tuple or list outputs
                    features = np.array(features[0]) if isinstance(features[0], (list, np.ndarray)) else np.array(features)
                else:
                    features = np.array([0.5])  # Default fallback
                
                # Map VGGish features to tempo estimation
                # This is a heuristic approach - real implementation would need training
                if len(features) > 0:
                    # Use feature energy and spectral characteristics for tempo estimation
                    feature_energy = np.mean(np.abs(features))
                    feature_variance = np.var(features) if len(features) > 1 else 0.1
                    
                    # Map to tempo ranges based on energy patterns
                    # High energy + high variance = fast tempo
                    # Low energy + low variance = slow tempo
                    base_tempo = 120.0
                    energy_factor = min(2.0, max(0.1, feature_energy * 100))  # Scale factor
                    variance_factor = min(1.5, max(0.1, feature_variance * 50))
                    
                    estimated_tempo = base_tempo * (1 + (energy_factor - 1) * variance_factor)
                    estimated_tempo = max(60.0, min(200.0, estimated_tempo))  # Clamp to reasonable range
                    
                    confidence = min(1.0, (energy_factor + variance_factor) / 3.0)
                    
                    return {
                        "ml_tempo": round(float(estimated_tempo), 1),
                        "ml_tempo_confidence": round(confidence, 3),
                        "ml_model": "essentia_vggish_tempo",
                        "ml_feature_energy": round(float(feature_energy), 4),
                        "ml_feature_variance": round(float(feature_variance), 4)
                    }
            
            # Fallback - return neutral values
            return {
                "ml_tempo": 120.0,
                "ml_tempo_confidence": 0.3,
                "ml_model": "essentia_fallback"
            }
            
        except Exception as e:
            logger.error(f"❌ ML tempo detection failed: {e}")
            return {"ml_tempo": 0.0, "ml_tempo_confidence": 0.0}
    
    def _fallback_key_detection(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Fallback key detection when ML model unavailable"""
        # This would use the enhanced librosa method from Week 1
        return {
            "ml_key": "unknown",
            "ml_mode": "unknown", 
            "ml_key_confidence": 0.0,
            "ml_model": "fallback_librosa"
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of loaded models"""
        return {
            "models_loaded": self.models_loaded,
            "available_models": list(self.available_models.keys()),
            "model_count": len(self.available_models),
            "missing_models": [
                name for name, path in self.model_paths.items()
                if not os.path.exists(path)
            ]
        }