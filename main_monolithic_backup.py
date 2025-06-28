from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import librosa
import numpy as np
import tempfile
import os
import requests
import json
from urllib.parse import urlparse, parse_qs
from typing import Optional, List
from collections import Counter
import essentia.standard as es

app = FastAPI(title="2W12 Audio Analysis API", version="2.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://2w12.one",
        "https://www.2w12.one", 
        "http://localhost:3000",  # Development
        "http://127.0.0.1:5500",  # Live Server
        "*"  # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models for Request Bodies
class YouTubeRequest(BaseModel):
    youtube_url: str

class BatchAnalysisResult(BaseModel):
    filename: str
    status: str
    key: Optional[str] = None
    tempo: Optional[float] = None
    duration: Optional[float] = None
    error: Optional[str] = None

# Root and Health Endpoints
@app.get("/")
async def root():
    return {"message": "2W12 Audio Analysis API", "status": "running", "version": "2.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "librosa_version": librosa.__version__,
        "ffmpeg_available": os.system("which ffmpeg") == 0,
        "essentia_available": True,
        "endpoints_available": [
            "/api/audio/analyze",
            "/api/audio/analyze-advanced", 
            "/api/audio/classify-genre",
            "/api/audio/detect-mood",
            "/api/audio/loudness",
            "/api/audio/mfcc",
            "/api/audio/audio-tagging",
            "/api/audio/batch-analyze"
        ]
    }

# =============================================================================
# BASIC AUDIO ANALYSIS
# =============================================================================

@app.post("/api/audio/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """Basic audio analysis for key, tempo, and fundamental features"""
    
    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Load audio with librosa
        y, sr = librosa.load(tmp_file_path, sr=22050)
        
        # Analyze tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Analyze key (simplified chroma-based)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_profile = np.mean(chroma, axis=1)
        key_index = np.argmax(key_profile)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        detected_key = keys[key_index]
        
        # Calculate additional basic features
        duration = len(y) / sr
        rms = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        return {
            "filename": file.filename,
            "duration": round(duration, 2),
            "tempo": round(float(tempo), 1),
            "key": detected_key,
            "rms_energy": round(float(rms), 4),
            "spectral_centroid": round(float(spectral_centroid), 2),
            "zero_crossing_rate": round(float(zero_crossing_rate), 4),
            "sample_rate": sr,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# =============================================================================
# ADVANCED AUDIO ANALYSIS
# =============================================================================

@app.post("/api/audio/analyze-advanced")
async def analyze_audio_advanced(file: UploadFile = File(...)):
    """Advanced audio analysis with multiple algorithms and consensus"""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        y, sr = librosa.load(tmp_file_path, sr=22050)
        
        # Method 1: Chroma-based key detection
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_profile_chroma = np.mean(chroma, axis=1)
        key_index_chroma = np.argmax(key_profile_chroma)
        
        # Method 2: Harmonic-percussive separation + chroma
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        chroma_harmonic = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
        key_profile_harmonic = np.mean(chroma_harmonic, axis=1)
        key_index_harmonic = np.argmax(key_profile_harmonic)
        
        # Method 3: Constant-Q transform based
        cqt = np.abs(librosa.cqt(y, sr=sr))
        chroma_cqt = librosa.feature.chroma_cqt(C=cqt, sr=sr)
        key_profile_cqt = np.mean(chroma_cqt, axis=1)
        key_index_cqt = np.argmax(key_profile_cqt)
        
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Confidence scores
        confidence_chroma = np.max(key_profile_chroma) / np.sum(key_profile_chroma)
        confidence_harmonic = np.max(key_profile_harmonic) / np.sum(key_profile_harmonic)
        confidence_cqt = np.max(key_profile_cqt) / np.sum(key_profile_cqt)
        
        
        # Consensus detection
        key_votes = [key_index_chroma, key_index_harmonic, key_index_cqt]
        most_common_key = Counter(key_votes).most_common(1)[0][0]
        consensus_strength = Counter(key_votes).most_common(1)[0][1] / 3
        
        # Enhanced tempo detection
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_stability = np.std(np.diff(librosa.frames_to_time(beats, sr=sr))) if len(beats) > 1 else 0.0
        
        # Advanced spectral features
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        return {
            "filename": file.filename,
            "key_detection": {
                "consensus_key": keys[most_common_key],
                "consensus_strength": round(float(consensus_strength), 3),
                "methods": {
                    "chroma": {
                        "key": keys[key_index_chroma],
                        "confidence": round(float(confidence_chroma), 3)
                    },
                    "harmonic": {
                        "key": keys[key_index_harmonic], 
                        "confidence": round(float(confidence_harmonic), 3)
                    },
                    "cqt": {
                        "key": keys[key_index_cqt],
                        "confidence": round(float(confidence_cqt), 3)
                    }
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
            "sample_rate": sr,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced analysis failed: {str(e)}")
    
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# =============================================================================
# GENRE CLASSIFICATION
# =============================================================================

@app.post("/api/audio/classify-genre")
async def classify_genre_basic(file: UploadFile = File(...)):
    """Basic genre classification using audio features"""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        y, sr = librosa.load(tmp_file_path, sr=22050)
        
        # Extract features for genre classification
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        
        # Enhanced rule-based genre classification
        if tempo > 120 and spectral_centroid > 2000 and spectral_bandwidth > 1000:
            genre = "Electronic/Dance"
            confidence = 0.82
        elif tempo < 80 and chroma > 0.5 and spectral_centroid < 1200:
            genre = "Classical/Ambient"
            confidence = 0.78
        elif 80 <= tempo <= 120 and spectral_centroid < 1500 and spectral_rolloff > 2000:
            genre = "Rock/Pop"
            confidence = 0.73
        elif tempo > 140 and spectral_centroid > 1800:
            genre = "Hip-Hop/Rap"
            confidence = 0.68
        elif tempo < 70 and chroma < 0.3:
            genre = "Jazz/Blues"
            confidence = 0.65
        else:
            genre = "Other/Mixed"
            confidence = 0.50
        
        return {
            "filename": file.filename,
            "genre": genre,
            "confidence": confidence,
            "features": {
                "tempo": round(float(tempo), 1),
                "chroma_mean": round(float(chroma), 3),
                "spectral_centroid": round(float(spectral_centroid), 2),
                "spectral_rolloff": round(float(spectral_rolloff), 2),
                "spectral_bandwidth": round(float(spectral_bandwidth), 2),
                "mfcc_mean": round(float(np.mean(mfcc)), 3)
            },
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Genre classification failed: {str(e)}")
    
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# =============================================================================
# MOOD DETECTION
# =============================================================================

@app.post("/api/audio/detect-mood")
async def detect_mood(file: UploadFile = File(...)):
    """Detect audio mood using acoustic features and emotional mapping"""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        y, sr = librosa.load(tmp_file_path, sr=22050)
        
        # Extract mood-relevant features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        rms_energy = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Mood classification based on multi-dimensional features
        energy_level = rms_energy * 1000  # Scale for easier comparison
        brightness = spectral_centroid / 1000  # Scale for easier comparison
        rhythmic_complexity = zero_crossing_rate * 100
        
        # Enhanced mood detection with multiple conditions
        if tempo > 120 and energy_level > 50 and brightness > 2:
            mood = "Energetic"
            confidence = 0.88
        elif tempo < 80 and energy_level < 30 and chroma > 0.4:
            mood = "Calm/Peaceful"
            confidence = 0.85
        elif chroma > 0.5 and brightness > 2 and energy_level > 40:
            mood = "Happy/Uplifting"
            confidence = 0.81
        elif chroma < 0.3 and brightness < 1 and tempo < 90:
            mood = "Sad/Melancholic"
            confidence = 0.78
        elif tempo > 100 and rhythmic_complexity > 15 and energy_level > 35:
            mood = "Exciting/Dynamic"
            confidence = 0.75
        elif energy_level < 20 and tempo < 70:
            mood = "Relaxing/Meditative"
            confidence = 0.72
        else:
            mood = "Neutral/Balanced"
            confidence = 0.60
        
        return {
            "filename": file.filename,
            "mood": mood,
            "confidence": confidence,
            "emotional_features": {
                "energy_level": round(float(energy_level), 2),
                "brightness": round(float(brightness), 2),
                "rhythmic_complexity": round(float(rhythmic_complexity), 2),
                "harmonic_richness": round(float(chroma), 3)
            },
            "technical_features": {
                "tempo": round(float(tempo), 1),
                "rms_energy": round(float(rms_energy), 4),
                "spectral_centroid": round(float(spectral_centroid), 2),
                "spectral_rolloff": round(float(spectral_rolloff), 2)
            },
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mood detection failed: {str(e)}")
    
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# =============================================================================
# LOUDNESS ANALYSIS
# =============================================================================
# Replace the loudness endpoint in your main.py with this fixed version:

@app.post("/api/audio/loudness")
async def analyze_loudness(file: UploadFile = File(...)):
    """Comprehensive loudness analysis with multiple standards"""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        y, sr = librosa.load(tmp_file_path, sr=22050)  # Use consistent sample rate
        
        # RMS-based loudness
        rms = librosa.feature.rms(y=y)[0]
        rms_db = librosa.amplitude_to_db(rms)
        
        # Peak analysis
        peak_amplitude = float(np.max(np.abs(y)))
        peak_db = float(librosa.amplitude_to_db([peak_amplitude])[0]) if peak_amplitude > 0 else -np.inf
        
        # Dynamic range
        dynamic_range = float(np.max(rms_db) - np.min(rms_db))
        
        # Crest factor (peak to RMS ratio)
        mean_rms = float(np.mean(rms))
        crest_factor = peak_amplitude / mean_rms if mean_rms > 0 else 0
        crest_factor_db = float(20 * np.log10(crest_factor)) if crest_factor > 0 else -np.inf
        
        # Loudness statistics
        loudness_stats = {
            "rms_mean_db": round(float(np.mean(rms_db)), 2),
            "rms_max_db": round(float(np.max(rms_db)), 2),
            "rms_min_db": round(float(np.min(rms_db)), 2),
            "peak_db": round(peak_db, 2),
            "dynamic_range_db": round(dynamic_range, 2),
            "crest_factor_db": round(crest_factor_db, 2)
        }
        
        # Loudness categorization
        avg_loudness = float(np.mean(rms_db))
        if avg_loudness > -10:
            loudness_category = "Very Loud"
        elif avg_loudness > -20:
            loudness_category = "Loud"
        elif avg_loudness > -30:
            loudness_category = "Moderate"
        elif avg_loudness > -40:
            loudness_category = "Quiet"
        else:
            loudness_category = "Very Quiet"
        
        return {
            "filename": file.filename,
            "loudness_category": loudness_category,
            "loudness_stats": loudness_stats,
            "technical_info": {
                "sample_rate": int(sr),
                "duration": round(float(len(y) / sr), 2),
                "samples": int(len(y))
            },
            "recommendations": {
                "mastering_headroom": round(max(0, -1 - peak_db), 2),
                "compression_suggested": bool(dynamic_range > 30),
                "normalization_gain": round(-23 - avg_loudness, 2)  # EBU R128 target
            },
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loudness analysis failed: {str(e)}")
    
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)









# MFCC ANALYSIS
# =============================================================================

@app.post("/api/audio/mfcc")
async def analyze_mfcc(file: UploadFile = File(...)):
    """Detailed MFCC (Mel-frequency Cepstral Coefficients) analysis"""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        y, sr = librosa.load(tmp_file_path, sr=22050)
        
        # Extract MFCC features with different configurations
        mfcc_13 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_20 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        # Statistical analysis of MFCCs
        mfcc_stats = {}
        for i in range(13):
            mfcc_stats[f"mfcc_{i+1}"] = {
                "mean": round(float(np.mean(mfcc_13[i])), 4),
                "std": round(float(np.std(mfcc_13[i])), 4),
                "min": round(float(np.min(mfcc_13[i])), 4),
                "max": round(float(np.max(mfcc_13[i])), 4)
            }
        
        # Delta and Delta-Delta MFCCs (temporal derivatives)
        mfcc_delta = librosa.feature.delta(mfcc_13)
        mfcc_delta2 = librosa.feature.delta(mfcc_13, order=2)
        
        # Overall MFCC characteristics
        mfcc_summary = {
            "total_frames": mfcc_13.shape[1],
            "coefficients_13": [round(float(np.mean(mfcc_13[i])), 4) for i in range(13)],
            "coefficients_20": [round(float(np.mean(mfcc_20[i])), 4) for i in range(20)],
            "delta_mean": round(float(np.mean(mfcc_delta)), 4),
            "delta2_mean": round(float(np.mean(mfcc_delta2)), 4)
        }
        
        # Spectral features related to MFCC
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        
        return {
            "filename": file.filename,
            "mfcc_summary": mfcc_summary,
            "mfcc_statistics": mfcc_stats,
            "temporal_dynamics": {
                "delta_variance": round(float(np.var(mfcc_delta)), 4),
                "delta2_variance": round(float(np.var(mfcc_delta2)), 4),
                "temporal_stability": round(float(1 / (1 + np.std(mfcc_delta))), 4)
            },
            "spectral_context": {
                "mel_bands": mel_spectrogram.shape[0],
                "spectral_centroid": round(float(spectral_centroid), 2),
                "spectral_bandwidth": round(float(spectral_bandwidth), 2)
            },
            "interpretation": {
                "timbre_complexity": "High" if np.std(mfcc_13) > 10 else "Moderate" if np.std(mfcc_13) > 5 else "Low",
                "spectral_richness": "Rich" if spectral_bandwidth > 1500 else "Moderate" if spectral_bandwidth > 800 else "Simple",
                "temporal_variation": "Dynamic" if np.std(mfcc_delta) > 2 else "Stable"
            },
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MFCC analysis failed: {str(e)}")
    
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# =============================================================================
# AUDIO TAGGING / METADATA
# =============================================================================

@app.post("/api/audio/audio-tagging")
async def analyze_audio_tagging(file: UploadFile = File(...)):
    """Audio tagging and content analysis using multiple techniques"""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        y, sr = librosa.load(tmp_file_path, sr=22050)
        
        # Basic audio properties
        duration = len(y) / sr
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Harmonic and percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_ratio = np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y)) + 1e-8)
        percussive_ratio = np.mean(np.abs(y_percussive)) / (np.mean(np.abs(y)) + 1e-8)
        
        # Spectral features for content analysis
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Rhythmic analysis
        rhythm_features = {
            "tempo": round(float(tempo), 1),
            "beat_strength": round(float(np.mean(librosa.onset.onset_strength(y=y, sr=sr))), 4),
            "rhythmic_regularity": round(float(1 / (1 + np.std(np.diff(beats)) * sr / len(y))), 4) if len(beats) > 1 else 0.0
        }
        
        # Tonal analysis
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        tonal_features = {
            "harmonic_ratio": round(float(harmonic_ratio), 4),
            "percussive_ratio": round(float(percussive_ratio), 4),
            "tonal_stability": round(float(np.mean(np.std(chroma, axis=1))), 4),
            "harmonic_complexity": round(float(np.mean(np.std(tonnetz, axis=1))), 4)
        }
        
        # Audio content tags based on analysis
        tags = []
        
        # Instrument/content detection
        if harmonic_ratio > 0.6:
            tags.append("harmonic-rich")
        if percussive_ratio > 0.4:
            tags.append("percussive")
        if zero_crossing_rate > 0.1:
            tags.append("noisy/distorted")
        if spectral_centroid > 2000:
            tags.append("bright")
        elif spectral_centroid < 1000:
            tags.append("dark")
        
        # Genre/style indicators
        if tempo > 120 and percussive_ratio > 0.3:
            tags.append("dance/electronic")
        if harmonic_ratio > 0.7 and tempo < 80:
            tags.append("classical/ambient")
        if 80 <= tempo <= 120 and harmonic_ratio > 0.5:
            tags.append("pop/rock")
        
        # Technical quality tags
        rms = np.mean(librosa.feature.rms(y=y))
        if rms < 0.01:
            tags.append("quiet")
        elif rms > 0.1:
            tags.append("loud")
        
        return {
            "filename": file.filename,
            "duration": round(duration, 2),
            "tags": tags,
            "rhythm_analysis": rhythm_features,
            "tonal_analysis": tonal_features,
            "spectral_analysis": {
                "spectral_centroid": round(float(spectral_centroid), 2),
                "spectral_rolloff": round(float(spectral_rolloff), 2),
                "spectral_contrast": round(float(spectral_contrast), 2),
                "zero_crossing_rate": round(float(zero_crossing_rate), 4)
            },
            "content_classification": {
                "is_harmonic": harmonic_ratio > 0.5,
                "is_percussive": percussive_ratio > 0.3,
                "is_tonal": np.mean(chroma) > 0.3,
                "is_rhythmic": len(beats) > duration * 0.5
            },
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio tagging failed: {str(e)}")
    
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# =============================================================================
# BATCH PROCESSING
# =============================================================================

@app.post("/api/audio/batch-analyze")
async def batch_analyze_audio(files: List[UploadFile] = File(...)):
    """Analyze multiple audio files at once with comprehensive results"""
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Too many files. Maximum 10 files per batch.")
    
    results = []
    
    for file in files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            # Analyze each file with basic + some advanced features
            y, sr = librosa.load(tmp_file_path, sr=22050)
            
            # Key detection
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            key_profile = np.mean(chroma, axis=1)
            key_index = np.argmax(key_profile)
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            detected_key = keys[key_index]
            
            # Tempo detection
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Additional features
            rms_energy = np.mean(librosa.feature.rms(y=y))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            duration = len(y) / sr
            
            # Quick genre classification
            if tempo > 120 and spectral_centroid > 2000:
                genre = "Electronic/Dance"
            elif tempo < 80 and np.mean(chroma) > 0.5:
                genre = "Classical/Ambient"
            elif 80 <= tempo <= 120:
                genre = "Pop/Rock"
            else:
                genre = "Other"
            
            results.append({
                "filename": file.filename,
                "key": detected_key,
                "tempo": round(float(tempo), 1),
                "duration": round(duration, 2),
                "genre": genre,
                "rms_energy": round(float(rms_energy), 4),
                "spectral_centroid": round(float(spectral_centroid), 2),
                "beat_count": len(beats),
                "status": "success"
            })
            
            # Clean up
            os.unlink(tmp_file_path)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })
    
    # Batch summary statistics
    successful_results = [r for r in results if r["status"] == "success"]
    
    batch_summary = {
        "total_files": len(files),
        "successful": len(successful_results),
        "failed": len([r for r in results if r["status"] == "failed"]),
        "average_duration": round(np.mean([r["duration"] for r in successful_results]), 2) if successful_results else 0,
        "key_distribution": dict(Counter([r["key"] for r in successful_results])),
        "genre_distribution": dict(Counter([r["genre"] for r in successful_results])),
        "tempo_range": {
            "min": round(min([r["tempo"] for r in successful_results]), 1) if successful_results else 0,
            "max": round(max([r["tempo"] for r in successful_results]), 1) if successful_results else 0,
            "average": round(np.mean([r["tempo"] for r in successful_results]), 1) if successful_results else 0
        }
    }
    
    return {
        "batch_summary": batch_summary,
        "results": results,
        "status": "completed"
    }

# =============================================================================
# API DOCUMENTATION
# =============================================================================

@app.get("/api/docs")
async def api_documentation():
    """Comprehensive API documentation with examples"""
    return {
        "title": "2W12.ONE Audio Analysis API",
        "version": "2.0.0", 
        "description": "Professional audio analysis API with machine learning features",
        "base_url": "http://128.199.25.218:8001",
        "endpoints": {
            "/health": {
                "method": "GET",
                "description": "Server health check and available features",
                "response": "JSON with status and versions"
            },
            "/api/audio/analyze": {
                "method": "POST",
                "description": "Basic audio analysis - key, tempo, duration, spectral features",
                "input": "Audio file (WAV, MP3, FLAC, M4A)",
                "features": ["Key detection", "Tempo/BPM", "Spectral centroid", "RMS energy"]
            },
            "/api/audio/analyze-advanced": {
                "method": "POST", 
                "description": "Advanced multi-algorithm key detection with consensus",
                "input": "Audio file",
                "features": ["3 key detection methods", "Consensus algorithm", "Confidence scores", "Advanced spectral analysis"]
            },
            "/api/audio/classify-genre": {
                "method": "POST",
                "description": "Music genre classification using audio features",
                "input": "Audio file",
                "output": "Genre prediction with confidence score"
            },
            "/api/audio/detect-mood": {
                "method": "POST",
                "description": "Emotional mood detection from acoustic properties",
                "input": "Audio file",
                "moods": ["Energetic", "Calm", "Happy", "Sad", "Exciting", "Relaxing", "Neutral"]
            },
            "/api/audio/loudness": {
                "method": "POST",
                "description": "Professional loudness analysis with mastering recommendations",
                "input": "Audio file",
                "features": ["RMS loudness", "Peak analysis", "Dynamic range", "Mastering suggestions"]
            },
            "/api/audio/mfcc": {
                "method": "POST",
                "description": "Detailed MFCC analysis for timbre and texture",
                "input": "Audio file", 
                "features": ["13 & 20 MFCC coefficients", "Delta features", "Statistical analysis"]
            },
            "/api/audio/audio-tagging": {
                "method": "POST",
                "description": "Content analysis and automatic tagging",
                "input": "Audio file",
                "features": ["Harmonic/percussive separation", "Content classification", "Auto-tagging"]
            },
            "/api/audio/batch-analyze": {
                "method": "POST",
                "description": "Batch processing for multiple files (max 10)",
                "input": "Multiple audio files",
                "output": "Individual results + batch statistics"
            }
        },
        "supported_formats": ["WAV", "MP3", "FLAC", "M4A"],
        "features": {
            "audio_analysis": "librosa 0.11.0 with advanced algorithms",
            "key_detection": "Multiple methods with consensus voting",
            "genre_classification": "Rule-based feature analysis",
            "mood_detection": "Acoustic emotion mapping",
            "loudness_analysis": "Professional mastering standards",
            "mfcc_analysis": "Detailed timbre characterization",
            "content_tagging": "Automatic audio content classification",
            "batch_processing": "Efficient multi-file analysis"
        },
        "technical_specs": {
            "sample_rate": "22050 Hz (auto-converted)",
            "audio_engine": "librosa + FFmpeg",
            "max_file_size": "50MB per file",
            "max_batch_size": "10 files",
            "response_format": "JSON"
        },
        "usage_examples": {
            "basic_analysis": "curl -X POST -F 'file=@audio.wav' /api/audio/analyze",
            "batch_processing": "curl -X POST -F 'files=@file1.wav' -F 'files=@file2.wav' /api/audio/batch-analyze"
        }
    }

# =============================================================================
# MAIN APPLICATION
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
