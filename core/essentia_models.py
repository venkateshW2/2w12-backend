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
            # These models need to be downloaded
            "key_detection": "models/discogs-effnet-key.pb",
            "genre_classification": "models/discogs-effnet-genres.pb",
            "mood_detection": "models/mood_acoustic.pb",
            "danceability": "models/danceability.pb"
        }
        
        self._load_available_models()
    
    def _load_available_models(self):
        """Load available Essentia models"""
        try:
            logger.info("🔄 Loading Essentia models...")
            
            # Key Detection Model
            if self._model_exists("key_detection"):
                self.available_models["key_detection"] = es.TensorflowPredictEffnetDiscogs(
                    graphFilename=self.model_paths["key_detection"],
                    output="predictions"
                )
                logger.info("✅ Key detection model loaded")
            
            # Genre Classification Model
            if self._model_exists("genre_classification"):
                self.available_models["genre_classification"] = es.TensorflowPredictGenreDiscogs(
                    graphFilename=self.model_paths["genre_classification"],
                    output="predictions"
                )
                logger.info("✅ Genre classification model loaded")
            
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
        """ML-based key detection using Essentia"""
        
        if "key_detection" not in self.available_models:
            logger.warning("⚠️  Key detection model not available, using fallback")
            return self._fallback_key_detection(y, sr)
        
        try:
            # Prepare audio for Essentia
            audio_vector = es.MonoLoader(filename="", sampleRate=sr)(y.astype(np.float32))
            
            # Extract features for key detection
            features = self.available_models["key_detection"](audio_vector)
            
            # Key mapping (24 keys: major and minor for each note)
            key_labels = [
                'A major', 'A minor', 'A# major', 'A# minor',
                'B major', 'B minor', 'C major', 'C minor',
                'C# major', 'C# minor', 'D major', 'D minor',
                'D# major', 'D# minor', 'E major', 'E minor',
                'F major', 'F minor', 'F# major', 'F# minor',
                'G major', 'G minor', 'G# major', 'G# minor'
            ]
            
            # Get prediction
            predicted_index = np.argmax(features)
            confidence = float(features[predicted_index])
            predicted_key = key_labels[predicted_index]
            
            # Extract key and mode
            key_parts = predicted_key.split(' ')
            key = key_parts[0]
            mode = key_parts[1]
            
            return {
                "ml_key": key,
                "ml_mode": mode,
                "ml_key_confidence": round(confidence, 3),
                "ml_full_key": predicted_key,
                "ml_model": "essentia_discogs_effnet"
            }
            
        except Exception as e:
            logger.error(f"❌ ML key detection failed: {e}")
            return self._fallback_key_detection(y, sr)
    
    def analyze_genre_ml(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """ML-based genre classification using Essentia"""
        
        if "genre_classification" not in self.available_models:
            logger.warning("⚠️  Genre classification model not available")
            return {"ml_genre": "unknown", "ml_genre_confidence": 0.0}
        
        try:
            # Prepare audio for Essentia
            audio_vector = es.MonoLoader(filename="", sampleRate=sr)(y.astype(np.float32))
            
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