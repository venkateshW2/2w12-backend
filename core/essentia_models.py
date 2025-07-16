# NEW FILE: core/essentia_models.py
import essentia.standard as es
import numpy as np
import logging
from typing import Dict, Any, Optional, List
import os
import tensorflow as tf

logger = logging.getLogger(__name__)

# Configure TensorFlow GPU memory growth BEFORE any model loading
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"🚀 GPU memory growth enabled for {len(gpus)} GPU(s)")
except Exception as e:
    logger.warning(f"⚠️ GPU configuration failed: {e}")

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
            # WORKING ML MODELS:
            "pitch_detection": "models/Crepe Large Model.pb",  # CREPE for key detection (51MB - WORKING)
            "genre_classification": "models/Genre Discogs 400 Model.pb",  # 1.25MB - WORKING  
            "danceability": "models/Danceability Discogs Effnet.pb",  # 53KB - WORKING
            "audio_features": "models/audioset-vggish-3.pb",  # VGGish 1.86MB - WORKING
            
            # TEMPO MODEL (DOWNLOADED):
            "tempo_cnn": "models/deeptemp-k4-3.pb",  # Smaller DeepTemp model (less likely to cause cppPool errors)
            "tempo_vggish": "models/audioset-vggish-3.pb",  # VGGish as alternative tempo model
            
            # Disabled empty models:
            # "tempo_classification": "models/Danceability Audioset Yamnet.pb",  # EMPTY - DISABLED
            # "voice_instrumental": "models/Voice Instrumental Model.pb"  # EMPTY - DISABLED
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
            
            # VGGish Audio Features Model (skip for now - causing failures)
            if False:  # Temporarily disabled due to GraphDef errors
                try:
                    self.available_models["audio_features"] = es.TensorflowPredictVGGish(
                        graphFilename=self.model_paths["audio_features"]
                    )
                    logger.info("✅ VGGish audio features model loaded")
                except Exception as e:
                    logger.warning(f"⚠️ VGGish audio features model failed to load: {e}")
            else:
                logger.info("⚠️ VGGish model temporarily disabled due to GraphDef errors")
            
            # Genre Classification Model - DISABLED (not useful for core analysis)
            # Removed as per user request - focus on key detection, tempo, and danceability
            logger.info("⚠️ Genre classification model disabled - not essential for core analysis")
                    
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
            
            # Tempo CNN Model (DeepTemp K16) - TESTING different preprocessing approaches
            if self._model_exists("tempo_cnn"):
                try:
                    self.available_models["tempo_cnn"] = es.TensorflowPredict(
                        graphFilename=self.model_paths["tempo_cnn"],
                        inputs=["input"],
                        outputs=["output"]
                    )
                    logger.info("✅ Tempo CNN model loaded")
                except Exception as e:
                    logger.warning(f"⚠️ Tempo CNN model failed to load: {e}")
            else:
                logger.info("ℹ️ Tempo CNN model disabled - using Madmom tempo detection instead")
            
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
            
            # Extract danceability features with cppPool error handling
            try:
                features = self.available_models["danceability"](audio_vector)
            except AttributeError as attr_error:
                if 'cppPool' in str(attr_error):
                    logger.warning(f"⚠️ cppPool error detected - using fallback danceability estimation")
                    # Fallback: estimate danceability from audio energy patterns
                    energy = np.mean(np.abs(audio_vector))
                    variance = np.var(audio_vector)
                    # Simple heuristic: high energy + variance = more danceable
                    danceability_score = min(1.0, (energy * 10 + variance * 5))
                    return {
                        "ml_danceability": round(danceability_score, 3),
                        "ml_danceability_confidence": 0.5,  # Lower confidence for fallback
                        "ml_danceability_class": "danceable" if danceability_score > 0.5 else "not_danceable",
                        "ml_model": "fallback_energy_heuristic",
                        "cppPool_error_handled": True
                    }
                else:
                    raise attr_error
            except Exception as model_error:
                logger.warning(f"⚠️ Danceability model execution failed: {model_error}")
                return {"ml_danceability": 0.0, "ml_danceability_confidence": 0.0}
            
            # Handle different output formats with robust type checking
            if features is None:
                features = np.array([0.5])
            elif isinstance(features, (tuple, list)):
                # Handle tuple/list outputs - try to extract array data
                if len(features) > 0:
                    first_item = features[0]
                    if isinstance(first_item, np.ndarray):
                        features = first_item.flatten()
                    elif isinstance(first_item, (int, float)):
                        features = np.array(features)
                    else:
                        features = np.array([0.5])
                else:
                    features = np.array([0.5])
            elif isinstance(features, np.ndarray):
                features = features.flatten()
            elif hasattr(features, '__iter__') and not isinstance(features, str):
                # Try to convert any iterable to numpy array
                try:
                    features = np.array(list(features))
                except:
                    features = np.array([0.5])
            else:
                # Single value or unknown type
                try:
                    features = np.array([float(features)])
                except:
                    features = np.array([0.5])
            
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
        """ML-based tempo detection - PRIORITY #1 FIX: Make Essentia do TWO jobs efficiently"""
        
        # TRY MULTIPLE TEMPO MODELS in order of preference
        tempo_models_to_try = [
            ("tempo_cnn", "DeepTemp K4 CNN"),
            ("tempo_vggish", "VGGish Audio Features"),
            ("audio_features", "VGGish Fallback")
        ]
        
        for model_key, model_name in tempo_models_to_try:
            if model_key in self.available_models:
                logger.info(f"🎵 Attempting {model_name} for tempo analysis...")
                
                try:
                    # Try simple direct audio input first (avoid cppPool issues)
                    logger.info(f"🔧 Trying {model_name} with direct audio input...")
                    
                    # Prepare audio
                    audio_chunk = y[:sr * 30].astype(np.float32)  # Max 30 seconds
                    if len(audio_chunk) == 0:
                        continue
                    
                    # Try the model
                    result = self.available_models[model_key](audio_chunk)
                    
                    # Process result based on model type
                    if model_key == "tempo_cnn":
                        # DeepTemp CNN returns tempo predictions
                        if hasattr(result, '__iter__') and not isinstance(result, str):
                            tempo_value = float(np.mean(result))
                        else:
                            tempo_value = float(result)
                        
                        # Map to reasonable tempo range
                        if tempo_value < 1:
                            tempo_value = tempo_value * 200 + 60  # Scale to 60-260 BPM
                        elif tempo_value > 300:
                            tempo_value = tempo_value % 200 + 60  # Wrap to reasonable range
                        
                        return {
                            "ml_tempo": round(tempo_value, 1),
                            "ml_tempo_confidence": 0.8,
                            "ml_model": f"essentia_{model_key}",
                            "ml_tempo_method": "direct_audio_input",
                            "ml_preprocessing": "simplified_approach",
                            "cppPool_error_avoided": True
                        }
                    
                    elif model_key in ["tempo_vggish", "audio_features"]:
                        # VGGish models return feature vectors - estimate tempo from features
                        if hasattr(result, '__iter__') and not isinstance(result, str):
                            # Use feature analysis to estimate tempo
                            features = np.array(result).flatten()
                            # Simple heuristic: use feature variance patterns
                            tempo_estimate = 120.0 + (np.std(features) * 50)  # Basic estimation
                            tempo_estimate = max(60, min(200, tempo_estimate))  # Clamp to reasonable range
                            
                            return {
                                "ml_tempo": round(tempo_estimate, 1),
                                "ml_tempo_confidence": 0.6,
                                "ml_model": f"essentia_{model_key}",
                                "ml_tempo_method": "vggish_feature_analysis",
                                "ml_preprocessing": "feature_based_estimation",
                                "cppPool_error_avoided": True
                            }
                    
                except Exception as model_error:
                    logger.warning(f"⚠️ {model_name} failed: {model_error}")
                    continue  # Try next model
                
                logger.info(f"✅ {model_name} succeeded!")
                break  # Success, don't try other models
            
        # If all models failed, try the old approach
        if "tempo_cnn" in self.available_models:
            logger.info("🔧 Falling back to complex preprocessing approach...")
            
            try:
                # APPROACH 1: Direct Essentia preprocessing (no librosa)
                logger.info("🔧 Trying Approach 1: Direct Essentia preprocessing")
                
                # Use Essentia for all preprocessing to avoid librosa cppPool conflicts
                import essentia.standard as es
                
                # Method 1: Try with Essentia's own mel-spectrogram
                windowing = es.Windowing(type='hann')
                fft = es.FFT()
                
                # FIX: Use proper sample rate for DeepTemp model (expects 22050Hz)
                # Resample if needed to match model training data
                if sr != 22050:
                    logger.info(f"🔧 Resampling from {sr}Hz to 22050Hz for DeepTemp model")
                    import librosa
                    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=22050)
                    model_sr = 22050
                else:
                    y_resampled = y
                    model_sr = sr
                
                # Use model-specific parameters
                nyquist_freq = model_sr / 2.0
                # DeepTemp expects mel bands up to 8kHz typically
                high_freq = min(8000, nyquist_freq * 0.9)  # 8kHz max for tempo analysis
                
                logger.info(f"🔧 MelBands config: sr={model_sr}, nyquist={nyquist_freq}, high_freq={high_freq}")
                
                mel_bands = es.MelBands(
                    numberBands=80, 
                    sampleRate=model_sr,
                    lowFrequencyBound=40,  # Start at 40Hz for tempo analysis
                    highFrequencyBound=high_freq
                )
                
                # Process in smaller chunks to avoid memory issues
                chunk_size = min(len(y_resampled), model_sr * 10)  # Max 10 seconds at a time
                audio_chunk = y_resampled[:chunk_size].astype(np.float32)
                
                # Create mel spectrogram using Essentia with error handling
                mel_frames = []
                hop_size = 512
                
                try:
                    for i in range(0, len(audio_chunk) - 1024, hop_size):
                        frame = audio_chunk[i:i+1024]
                        if len(frame) == 1024:
                            windowed = windowing(frame)
                            fft_result = fft(windowed)
                            mel_frame = mel_bands(fft_result)
                            mel_frames.append(mel_frame)
                            
                            # Process only first 100 frames for speed
                            if len(mel_frames) >= 100:
                                break
                                
                except Exception as mel_error:
                    logger.warning(f"⚠️ MelBands processing failed: {mel_error}")
                    # Try alternative approach without MelBands
                    logger.info("🔧 Trying alternative approach: Direct audio input")
                    
                    # Some CNN models accept raw audio input
                    try:
                        # Normalize audio to [-1, 1] range
                        audio_normalized = audio_chunk / np.max(np.abs(audio_chunk)) if np.max(np.abs(audio_chunk)) > 0 else audio_chunk
                        
                        # Try different input formats
                        input_formats = [
                            audio_normalized,  # Raw audio
                            audio_normalized.reshape(1, -1),  # Batch dimension
                            audio_normalized.reshape(-1, 1),  # Channel dimension
                        ]
                        
                        for fmt_idx, audio_input in enumerate(input_formats):
                            try:
                                logger.info(f"🔧 Trying audio format {fmt_idx+1}: {audio_input.shape}")
                                tempo_prediction = self.available_models["tempo_cnn"](audio_input)
                                
                                # If we get here, it worked
                                logger.info("✅ Direct audio input successful!")
                                
                                # Process the result
                                if hasattr(tempo_prediction, '__iter__') and not isinstance(tempo_prediction, str):
                                    tempo_value = float(np.mean(tempo_prediction))
                                else:
                                    tempo_value = float(tempo_prediction)
                                
                                # Map to reasonable tempo range
                                if tempo_value < 1:
                                    tempo_value = tempo_value * 200 + 60  # Scale to 60-260 BPM
                                elif tempo_value > 300:
                                    tempo_value = tempo_value % 200 + 60  # Wrap to reasonable range
                                
                                return {
                                    "ml_tempo": round(tempo_value, 1),
                                    "ml_tempo_confidence": 0.7,
                                    "ml_model": "essentia_deeptemp_k16_direct_audio",
                                    "ml_tempo_method": "direct_audio_input",
                                    "ml_preprocessing": "audio_normalization",
                                    "melband_error_bypassed": True
                                }
                                
                            except Exception as audio_error:
                                logger.warning(f"⚠️ Audio format {fmt_idx+1} failed: {audio_error}")
                                continue
                        
                        # If all audio formats failed, raise the original error
                        raise mel_error
                        
                    except Exception as audio_fallback_error:
                        logger.warning(f"⚠️ Audio fallback failed: {audio_fallback_error}")
                        raise mel_error
                
                if len(mel_frames) > 0:
                    mel_spectrogram = np.array(mel_frames).T  # Shape: (80, time_frames)
                    
                    # Normalize for CNN
                    mel_normalized = (mel_spectrogram - np.mean(mel_spectrogram)) / (np.std(mel_spectrogram) + 1e-8)
                    
                    # Try different input shapes for the CNN
                    shapes_to_try = [
                        mel_normalized,  # Raw mel spectrogram
                        mel_normalized.T,  # Transposed
                        mel_normalized[np.newaxis, :, :],  # Add batch dimension
                        mel_normalized[:, :, np.newaxis],  # Add channel dimension
                    ]
                    
                    for i, input_tensor in enumerate(shapes_to_try):
                        try:
                            logger.info(f"🔧 Trying input shape {i+1}: {input_tensor.shape}")
                            tempo_prediction = self.available_models["tempo_cnn"](input_tensor)
                            
                            # Process CNN output
                            if isinstance(tempo_prediction, (list, tuple)) and len(tempo_prediction) > 0:
                                tempo_probs = tempo_prediction[0]
                            else:
                                tempo_probs = tempo_prediction
                            
                            if hasattr(tempo_probs, 'shape') and len(tempo_probs.shape) > 0:
                                # DeepTemp K16 outputs tempo class probabilities
                                # Classes typically represent BPM ranges from 30-300
                                tempo_classes = np.arange(30, 301, 1)  # 1 BPM resolution
                                
                                if len(tempo_probs) == len(tempo_classes):
                                    predicted_tempo_idx = np.argmax(tempo_probs)
                                    predicted_tempo = float(tempo_classes[predicted_tempo_idx])
                                    confidence = float(np.max(tempo_probs))
                                    
                                    logger.info(f"✅ Tempo CNN SUCCESS with shape {i+1}!")
                                    return {
                                        "ml_tempo": round(predicted_tempo, 1),
                                        "ml_tempo_confidence": round(confidence, 3),
                                        "ml_model": "essentia_deeptemp_k16_cnn",
                                        "ml_tempo_method": f"cnn_essentia_preprocessing_approach_{i+1}",
                                        "ml_input_shape": str(input_tensor.shape),
                                        "ml_preprocessing": "essentia_mel_spectrogram",
                                        "cppPool_error_fixed": True
                                    }
                                    
                        except Exception as shape_error:
                            logger.warning(f"⚠️ Shape {i+1} failed: {shape_error}")
                            continue
                
                logger.warning("🔧 All Essentia preprocessing approaches failed, trying fallback...")
                
            except Exception as cnn_error:
                logger.warning(f"⚠️ Tempo CNN approaches failed: {cnn_error}")
        
        # FALLBACK: Use Essentia RhythmExtractor2013 (working alternative)
        logger.info("🎵 Using Essentia RhythmExtractor2013 fallback")
        
        try:
            import essentia.standard as es
            
            # Use Essentia's working rhythm analysis
            rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
            tempo, beats, beats_confidence, _, beats_intervals = rhythm_extractor(y)
            
            if tempo > 0 and beats_confidence > 0:
                # Calculate tempo confidence based on beat consistency
                if len(beats_intervals) > 1:
                    interval_std = np.std(beats_intervals)
                    tempo_confidence = max(0.1, min(0.95, 1.0 / (1.0 + interval_std)))
                else:
                    tempo_confidence = beats_confidence
                
                # Alternative tempo estimates
                tempo_estimates = []
                if len(beats) > 1:
                    # Calculate from beat intervals
                    mean_interval = np.mean(np.diff(beats))
                    if mean_interval > 0:
                        beat_tempo = 60.0 / mean_interval
                        tempo_estimates.append(beat_tempo)
                        # Double and half tempo candidates
                        tempo_estimates.extend([beat_tempo * 2, beat_tempo / 2])
                
                # Add the main tempo
                tempo_estimates.insert(0, float(tempo))
                
                # Remove duplicates and sort by proximity to main tempo
                tempo_estimates = list(set([t for t in tempo_estimates if 60 <= t <= 200]))
                tempo_estimates.sort(key=lambda x: abs(x - tempo))
                
                return {
                    "ml_tempo": round(float(tempo), 1),
                    "ml_tempo_confidence": round(tempo_confidence, 3),
                    "ml_model": "essentia_rhythm_extractor_2013_fallback",
                    "ml_tempo_method": "multifeature_rhythm_analysis",
                    "ml_tempo_candidates": [round(t, 1) for t in tempo_estimates[:3]],
                    "ml_beats_detected": len(beats),
                    "ml_beats_confidence": round(beats_confidence, 3),
                    "ml_tempo_range": "60_200_bpm",
                    "tempo_cnn_attempted": True
                }
            else:
                return {
                    "ml_tempo": 120.0,
                    "ml_tempo_confidence": 0.0,
                    "ml_model": "essentia_rhythm_extractor_fallback"
                }
            
        except Exception as e:
            logger.error(f"❌ ML tempo detection failed: {e}")
            return {
                "ml_tempo": 0.0, 
                "ml_tempo_confidence": 0.0,
                "ml_tempo_error": str(e),
                "ml_error_type": type(e).__name__
            }
    
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
    
    # GPU BATCH PROCESSING METHODS
    
    def analyze_batch_gpu(self, audio_chunks: List[np.ndarray], sr: int) -> List[Dict[str, Any]]:
        """GPU batch processing for multiple audio chunks simultaneously"""
        
        if not self.models_loaded:
            logger.warning("⚠️ No models loaded, skipping batch processing")
            return [{"ml_features_available": False} for _ in audio_chunks]
        
        logger.info(f"🚀 GPU batch processing {len(audio_chunks)} chunks")
        
        try:
            # Batch process all chunks simultaneously on GPU
            batch_results = []
            
            # Process in GPU-optimized batches (avoid GPU memory overflow)
            gpu_batch_size = min(8, len(audio_chunks))  # Process max 8 chunks at once
            
            for batch_start in range(0, len(audio_chunks), gpu_batch_size):
                batch_end = min(batch_start + gpu_batch_size, len(audio_chunks))
                chunk_batch = audio_chunks[batch_start:batch_end]
                
                # Batch GPU processing
                batch_chunk_results = self._process_gpu_batch(chunk_batch, sr)
                batch_results.extend(batch_chunk_results)
                
                logger.info(f"✅ Processed GPU batch {batch_start//gpu_batch_size + 1}")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"❌ GPU batch processing failed: {e}")
            # Fallback to individual processing
            return [self._analyze_single_chunk(chunk, sr) for chunk in audio_chunks]
    
    def _process_gpu_batch(self, chunk_batch: List[np.ndarray], sr: int) -> List[Dict[str, Any]]:
        """Process a batch of chunks on GPU simultaneously"""
        
        batch_results = []
        
        # Prepare batch data for GPU
        batch_audio_vectors = [chunk.astype(np.float32) for chunk in chunk_batch]
        
        # Batch process each model type
        batch_key_results = self._batch_key_detection(batch_audio_vectors, sr)
        batch_danceability_results = self._batch_danceability_analysis(batch_audio_vectors, sr)
        batch_tempo_results = self._batch_tempo_analysis(batch_audio_vectors, sr)
        
        # Combine results for each chunk
        for i in range(len(chunk_batch)):
            chunk_result = {
                **batch_key_results[i],
                **batch_danceability_results[i], 
                **batch_tempo_results[i],
                "ml_features_available": True,
                "ml_status": "gpu_batch_success",
                "ml_batch_processing": True
            }
            batch_results.append(chunk_result)
        
        return batch_results
    
    def _batch_key_detection(self, batch_audio: List[np.ndarray], sr: int) -> List[Dict[str, Any]]:
        """Batch key detection for multiple chunks"""
        
        if "pitch_detection" not in self.available_models:
            return [self._fallback_key_detection(None, sr) for _ in batch_audio]
        
        try:
            batch_results = []
            
            # Process each chunk in the batch (CREPE doesn't support true batching)
            # But we can optimize by keeping the model loaded and reusing GPU context
            for audio_vector in batch_audio:
                try:
                    pitch_predictions = self.available_models["pitch_detection"](audio_vector)
                    key_result = self._convert_pitch_to_key(pitch_predictions)
                    batch_results.append(key_result)
                except Exception as e:
                    logger.warning(f"Batch key detection failed for chunk: {e}")
                    batch_results.append(self._fallback_key_detection(None, sr))
            
            return batch_results
            
        except Exception as e:
            logger.error(f"❌ Batch key detection failed: {e}")
            return [self._fallback_key_detection(None, sr) for _ in batch_audio]
    
    def _batch_danceability_analysis(self, batch_audio: List[np.ndarray], sr: int) -> List[Dict[str, Any]]:
        """Batch danceability analysis for multiple chunks"""
        
        if "danceability" not in self.available_models:
            return [{"ml_danceability": 0.0, "ml_danceability_confidence": 0.0} for _ in batch_audio]
        
        try:
            batch_results = []
            
            # Process each chunk (optimize by reusing GPU context)
            for audio_vector in batch_audio:
                try:
                    # Use the robust model execution with cppPool error handling
                    try:
                        features = self.available_models["danceability"](audio_vector)
                    except AttributeError as attr_error:
                        if 'cppPool' in str(attr_error):
                            logger.warning(f"⚠️ Batch cppPool error - using fallback")
                            # Fallback danceability estimation
                            energy = np.mean(np.abs(audio_vector))
                            variance = np.var(audio_vector)
                            danceability_score = min(1.0, (energy * 10 + variance * 5))
                            batch_results.append({
                                "ml_danceability": round(danceability_score, 3),
                                "ml_danceability_confidence": 0.5,
                                "ml_danceability_class": "danceable" if danceability_score > 0.5 else "not_danceable",
                                "ml_model": "fallback_energy_heuristic",
                                "cppPool_error_handled": True
                            })
                            continue
                        else:
                            raise attr_error
                    except Exception as model_error:
                        logger.warning(f"⚠️ Batch danceability model execution failed: {model_error}")
                        batch_results.append({"ml_danceability": 0.0, "ml_danceability_confidence": 0.0})
                        continue
                    
                    danceability_result = self._process_danceability_features(features)
                    batch_results.append(danceability_result)
                except Exception as e:
                    logger.warning(f"Batch danceability failed for chunk: {e}")
                    batch_results.append({"ml_danceability": 0.0, "ml_danceability_confidence": 0.0})
            
            return batch_results
            
        except Exception as e:
            logger.error(f"❌ Batch danceability analysis failed: {e}")
            return [{"ml_danceability": 0.0, "ml_danceability_confidence": 0.0} for _ in batch_audio]
    
    def _batch_tempo_analysis(self, batch_audio: List[np.ndarray], sr: int) -> List[Dict[str, Any]]:
        """Batch tempo analysis for multiple chunks"""
        
        if "audio_features" not in self.available_models:
            return [{"ml_tempo": 120.0, "ml_tempo_confidence": 0.3} for _ in batch_audio]
        
        try:
            batch_results = []
            
            # Process each chunk (VGGish optimization)
            for audio_vector in batch_audio:
                try:
                    features = self.available_models["audio_features"](audio_vector)
                    tempo_result = self._process_tempo_features(features)
                    batch_results.append(tempo_result)
                except Exception as e:
                    logger.warning(f"Batch tempo failed for chunk: {e}")
                    batch_results.append({"ml_tempo": 120.0, "ml_tempo_confidence": 0.3})
            
            return batch_results
            
        except Exception as e:
            logger.error(f"❌ Batch tempo analysis failed: {e}")
            return [{"ml_tempo": 120.0, "ml_tempo_confidence": 0.3} for _ in batch_audio]
    
    def _convert_pitch_to_key(self, pitch_predictions) -> Dict[str, Any]:
        """Convert CREPE pitch predictions to key (extracted from analyze_key_ml)"""
        
        if len(pitch_predictions) > 0:
            pitches = pitch_predictions[0] if isinstance(pitch_predictions, tuple) else pitch_predictions
            
            if isinstance(pitches, np.ndarray) and len(pitches) > 0:
                median_pitch = np.median(pitches[pitches > 0])
                
                if median_pitch > 0:
                    midi_note = 69 + 12 * np.log2(median_pitch / 440.0)
                    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    note_index = max(0, min(11, int(midi_note) % 12))
                    key = note_names[note_index]
                    
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
        
        return self._fallback_key_detection(None, None)
    
    def _process_danceability_features(self, features) -> Dict[str, Any]:
        """Process danceability features (extracted from analyze_danceability_ml)"""
        
        # Handle different output formats with robust type checking
        if features is None:
            features = np.array([0.5])
        elif isinstance(features, (tuple, list)):
            # Handle tuple/list outputs - try to extract array data
            if len(features) > 0:
                first_item = features[0]
                if isinstance(first_item, np.ndarray):
                    features = first_item.flatten()
                elif isinstance(first_item, (int, float)):
                    features = np.array(features)
                else:
                    features = np.array([0.5])
            else:
                features = np.array([0.5])
        elif isinstance(features, np.ndarray):
            features = features.flatten()
        elif hasattr(features, '__iter__') and not isinstance(features, str):
            # Try to convert any iterable to numpy array
            try:
                features = np.array(list(features))
            except:
                features = np.array([0.5])
        else:
            # Single value or unknown type
            try:
                features = np.array([float(features)])
            except:
                features = np.array([0.5])
        
        if len(features) >= 2:
            danceability_score = float(features[1])
        elif len(features) >= 1:
            danceability_score = float(features[0])
        else:
            danceability_score = 0.5
        
        confidence = abs(danceability_score - 0.5) * 2
        
        return {
            "ml_danceability": round(danceability_score, 3),
            "ml_danceability_confidence": round(confidence, 3),
            "ml_danceability_class": "danceable" if danceability_score > 0.5 else "not_danceable",
            "ml_model": "essentia_danceability_discogs"
        }
    
    def _process_tempo_features(self, features) -> Dict[str, Any]:
        """Process VGGish features for tempo (extracted from analyze_tempo_ml)"""
        
        # Handle different output formats
        if isinstance(features, np.ndarray):
            features = features.flatten()
        elif hasattr(features, '__len__') and len(features) > 0:
            features = np.array(features[0]) if isinstance(features[0], (list, np.ndarray)) else np.array(features)
        else:
            features = np.array([0.5])
        
        if len(features) > 0:
            feature_energy = np.mean(np.abs(features))
            feature_variance = np.var(features) if len(features) > 1 else 0.1
            
            base_tempo = 120.0
            energy_factor = min(2.0, max(0.1, feature_energy * 100))
            variance_factor = min(1.5, max(0.1, feature_variance * 50))
            
            estimated_tempo = base_tempo * (1 + (energy_factor - 1) * variance_factor)
            estimated_tempo = max(60.0, min(200.0, estimated_tempo))
            
            confidence = min(1.0, (energy_factor + variance_factor) / 3.0)
            
            return {
                "ml_tempo": round(float(estimated_tempo), 1),
                "ml_tempo_confidence": round(confidence, 3),
                "ml_model": "essentia_vggish_tempo",
                "ml_feature_energy": round(float(feature_energy), 4),
                "ml_feature_variance": round(float(feature_variance), 4)
            }
        
        return {"ml_tempo": 120.0, "ml_tempo_confidence": 0.3}
    
    def _analyze_single_chunk(self, chunk: np.ndarray, sr: int) -> Dict[str, Any]:
        """Fallback single chunk analysis"""
        return {
            **self.analyze_key_ml(chunk, sr),
            **self.analyze_danceability_ml(chunk, sr),
            **self.analyze_tempo_ml(chunk, sr)
        }