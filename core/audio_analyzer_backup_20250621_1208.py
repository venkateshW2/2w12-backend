# core/audio_analyzer.py
import librosa
import numpy as np
from typing import Dict, Any, Tuple
import tempfile
import os

class AudioAnalyzer:
    """Core audio analysis functionality"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file with librosa"""
        return librosa.load(file_path, sr=self.sample_rate)
    
    def basic_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Basic audio analysis - key, tempo, features"""
        # Key detection
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_profile = np.mean(chroma, axis=1)
        key_index = np.argmax(key_profile)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        detected_key = keys[key_index]
        
        # Tempo detection
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Basic features
        rms_energy = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        return {
            "duration": len(y) / sr,
            "tempo": float(tempo),
            "key": detected_key,
            "rms_energy": float(rms_energy),
            "spectral_centroid": float(spectral_centroid),
            "zero_crossing_rate": float(zero_crossing_rate),
            "sample_rate": sr
        }
    
    def advanced_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Advanced multi-algorithm analysis"""
        # Multiple key detection methods
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
        from collections import Counter
        key_votes = [key_index_chroma, key_index_harmonic, key_index_cqt]
        most_common_key = Counter(key_votes).most_common(1)[0][0]
        consensus_strength = Counter(key_votes).most_common(1)[0][1] / 3
        
        # Enhanced tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_stability = np.std(np.diff(librosa.frames_to_time(beats, sr=sr))) if len(beats) > 1 else 0.0
        
        # Spectral features
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        return {
            "key_detection": {
                "consensus_key": keys[most_common_key],
                "consensus_strength": round(float(consensus_strength), 3),
                "methods": {
                    "chroma": {"key": keys[key_index_chroma], "confidence": round(float(np.max(key_profile_chroma) / np.sum(key_profile_chroma)), 3)},
                    "harmonic": {"key": keys[key_index_harmonic], "confidence": round(float(np.max(key_profile_harmonic) / np.sum(key_profile_harmonic)), 3)},
                    "cqt": {"key": keys[key_index_cqt], "confidence": round(float(np.max(key_profile_cqt) / np.sum(key_profile_cqt)), 3)}
                }
            },
            "tempo_analysis": {
                "tempo": round(float(tempo), 1),
                "stability": round(float(tempo_stability), 3),
                "beat_count": len(beats)
            },
            "spectral_features": {
                "bandwidth": round(float(spectral_bandwidth), 2),
                "contrast": round(float(spectral_contrast), 2),
                "rolloff": round(float(spectral_rolloff), 2)
            },
            "duration": len(y) / sr,
            "sample_rate": sr
        }
    
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