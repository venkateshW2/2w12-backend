
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import tempfile
import os
import yt_dlp
import subprocess
import requests
import json
from urllib.parse import urlparse, parse_qs
from typing import Optional

app = FastAPI(title="2W12 Audio Analysis API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "2W12 Audio Analysis API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "librosa_version": librosa.__version__,
        "ffmpeg_available": os.system("which ffmpeg") == 0
    }

@app.post("/api/audio/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """Analyze uploaded audio file for key and tempo"""
    
    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        # Save uploaded file
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Load audio with librosa
        y, sr = librosa.load(tmp_file_path, sr=22050)
        
        # Analyze tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Analyze key (simplified - you can enhance this)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_profile = np.mean(chroma, axis=1)
        key_index = np.argmax(key_profile)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        detected_key = keys[key_index]
        
        # Calculate additional features
        duration = len(y) / sr
        rms = np.mean(librosa.feature.rms(y=y))
        
        return {
            "filename": file.filename,
            "duration": round(duration, 2),
            "tempo": round(float(tempo), 1),
            "key": detected_key,
            "rms_energy": round(float(rms), 4),
            "sample_rate": sr,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

#youtube helper functions

def extract_youtube_id(url: str) -> Optional[str]:
    """Extract video ID from various YouTube URL formats"""
    parsed_url = urlparse(url)
    
    # YouTube watch URLs
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif parsed_url.path.startswith('/shorts/'):
            return parsed_url.path.split('/shorts/')[1]
    
    # Short URLs
    elif parsed_url.hostname in ['youtu.be']:
        return parsed_url.path.lstrip('/')
    
    return None

async def download_audio_from_rapidapi(video_id: str) -> dict:
    """Download audio using RapidAPI YouTube service"""
    
    rapidapi_url = f"https://ytstream-download-youtube-videos.p.rapidapi.com/dl?id={video_id}"
    
    headers = {
        'X-RapidAPI-Key': '0950947a59msh129f99a31d7db49p103ec3jsn41d98dab874b',
        'X-RapidAPI-Host': 'ytstream-download-youtube-videos.p.rapidapi.com'
    }
    
    try:
        # Get video info and download URLs
        response = requests.get(rapidapi_url, headers=headers)
        response.raise_for_status()
        
        video_data = response.json()
        
        # Find best audio format
        audio_url = None
        for format_item in video_data.get('formats', []):
            if format_item.get('mimeType', '').startswith('audio/'):
                audio_url = format_item.get('url')
                break
        
        if not audio_url:
            raise Exception("No audio format found")
        
        # Download the actual audio file
        audio_response = requests.get(audio_url)
        audio_response.raise_for_status()
        
        return {
            'audio_data': audio_response.content,
            'title': video_data.get('title', 'Unknown'),
            'duration': video_data.get('lengthSeconds', 0),
            'success': True
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

#ADD YOUTUBE ANALYSIS ENDPOINT
@app.post("/api/youtube/analyze")
async def analyze_youtube_video(youtube_url: str):
    """Download YouTube video and analyze audio"""
    
    # Extract video ID
    video_id = extract_youtube_id(youtube_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    # Download audio via RapidAPI
    download_result = await download_audio_from_rapidapi(video_id)
    
    if not download_result['success']:
        raise HTTPException(status_code=500, detail=f"Download failed: {download_result['error']}")
    
    # Save audio to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as tmp_file:
        tmp_file.write(download_result['audio_data'])
        tmp_file_path = tmp_file.name
    
    try:
        # Analyze with librosa
        y, sr = librosa.load(tmp_file_path, sr=22050)
        
        # Enhanced key detection
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_profile = np.mean(chroma, axis=1)
        key_index = np.argmax(key_profile)
        key_confidence = np.max(key_profile) / np.sum(key_profile)
        
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        detected_key = keys[key_index]
        
        # Tempo detection with confidence
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_confidence = len(beats) / (len(y) / sr)  # beats per second ratio
        
        # Additional audio features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rms_energy = np.mean(librosa.feature.rms(y=y))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        return {
            "video_id": video_id,
            "title": download_result['title'],
            "duration": download_result['duration'],
            "analysis": {
                "key": detected_key,
                "key_confidence": round(float(key_confidence), 3),
                "tempo": round(float(tempo), 1),
                "tempo_confidence": round(float(tempo_confidence), 3),
                "spectral_centroid": round(float(spectral_centroid), 2),
                "rms_energy": round(float(rms_energy), 4),
                "zero_crossing_rate": round(float(zero_crossing_rate), 4)
            },
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

#ADD ENHANCED AUDIO ANALYSIS Features
#2.1 Add Genre Classification (Basic ML)
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
        
        # Simple rule-based genre classification
        if tempo > 120 and spectral_centroid > 2000:
            genre = "Electronic/Dance"
            confidence = 0.75
        elif tempo < 80 and chroma > 0.5:
            genre = "Classical/Ambient"
            confidence = 0.70
        elif 80 <= tempo <= 120 and spectral_centroid < 1500:
            genre = "Rock/Pop"
            confidence = 0.65
        else:
            genre = "Other"
            confidence = 0.50
        
        return {
            "filename": file.filename,
            "genre": genre,
            "confidence": confidence,
            "features": {
                "tempo": round(float(tempo), 1),
                "chroma_mean": round(float(chroma), 3),
                "spectral_centroid": round(float(spectral_centroid), 2),
                "spectral_rolloff": round(float(spectral_rolloff), 2)
            },
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Genre classification failed: {str(e)}")
    
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


#2.2 Add Mood Detection

@app.post("/api/audio/detect-mood")
async def detect_mood(file: UploadFile = File(...)):
    """Detect audio mood using acoustic features"""
    
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
        
        # Mood classification based on features
        energy_level = rms_energy * 1000  # Scale for easier comparison
        brightness = spectral_centroid / 1000  # Scale for easier comparison
        
        if tempo > 120 and energy_level > 50:
            mood = "Energetic"
            confidence = 0.85
        elif tempo < 80 and energy_level < 30:
            mood = "Calm"
            confidence = 0.80
        elif chroma > 0.5 and brightness > 2:
            mood = "Happy"
            confidence = 0.75
        elif chroma < 0.3 and brightness < 1:
            mood = "Melancholic"
            confidence = 0.70
        else:
            mood = "Neutral"
            confidence = 0.60
        
        return {
            "filename": file.filename,
            "mood": mood,
            "confidence": confidence,
            "features": {
                "tempo": round(float(tempo), 1),
                "energy_level": round(float(energy_level), 2),
                "brightness": round(float(brightness), 2),
                "chroma_mean": round(float(chroma), 3)
            },
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mood detection failed: {str(e)}")
    
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# STEP 3: ADD BATCH PROCESSING
# 3.1 Batch Analysis Endpoint
from typing import List

@app.post("/api/audio/batch-analyze")
async def batch_analyze_audio(files: List[UploadFile] = File(...)):
    """Analyze multiple audio files at once"""
    
    results = []
    
    for file in files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            # Analyze each file
            y, sr = librosa.load(tmp_file_path, sr=22050)
            
            # Key detection
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            key_profile = np.mean(chroma, axis=1)
            key_index = np.argmax(key_profile)
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            detected_key = keys[key_index]
            
            # Tempo detection
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Additional features
            rms_energy = np.mean(librosa.feature.rms(y=y))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            
            results.append({
                "filename": file.filename,
                "key": detected_key,
                "tempo": round(float(tempo), 1),
                "duration": len(y) / sr,
                "rms_energy": round(float(rms_energy), 4),
                "spectral_centroid": round(float(spectral_centroid), 2),
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
    
    return {
        "total_files": len(files),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "failed"]),
        "results": results
    }

# STEP 4: ADD ADVANCED KEY DETECTION
#4.1 Multi-Algorithm Key Detection
@app.post("/api/audio/analyze-advanced")
async def analyze_audio_advanced(file: UploadFile = File(...)):
    """Advanced audio analysis with multiple algorithms"""
    
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
        from collections import Counter
        most_common_key = Counter(key_votes).most_common(1)[0][0]
        consensus_strength = Counter(key_votes).most_common(1)[0][1] / 3
        
        # Enhanced tempo detection
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_stability = np.std(np.diff(librosa.frames_to_time(beats, sr=sr)))
        
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
            "duration": len(y) / sr,
            "sample_rate": sr,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced analysis failed: {str(e)}")
    
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# STEP 5: UPDATE CORS AND ADD API DOCUMENTATION
#5.1 Update CORS for Frontend Integration

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://2w12.one",
        "https://www.2w12.one",
        "http://localhost:3000",  # Development
        "http://127.0.0.1:5500",  # Live Server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#5.2 Add API Documentation Endpoint

@app.get("/api/docs")
async def api_documentation():
    """Comprehensive API documentation"""
    return {
        "title": "2W12.ONE Audio Analysis API",
        "version": "2.0.0",
        "base_url": "http://128.199.25.218:8001",
        "endpoints": {
            "/health": "Server health check",
            "/api/audio/analyze": "Basic audio analysis (key, tempo, features)",
            "/api/audio/analyze-advanced": "Advanced multi-algorithm key detection",
            "/api/youtube/analyze": "YouTube video download and analysis",
            "/api/audio/classify-genre": "Basic genre classification",
            "/api/audio/detect-mood": "Mood detection from audio features",
            "/api/audio/batch-analyze": "Batch processing multiple files"
        },
        "features": {
            "key_detection": "Multiple algorithms with consensus",
            "tempo_analysis": "BPM detection with stability metrics",
            "genre_classification": "Rule-based genre detection",
            "mood_detection": "Acoustic feature-based mood analysis",
            "youtube_integration": "RapidAPI-powered video downloading",
            "batch_processing": "Multiple file analysis"
        },
        "libraries": {
            "librosa": "0.11.0",
            "tensorflow": "2.15.0",
            "fastapi": "Latest",
            "essentia": "Available"
        }
    }




