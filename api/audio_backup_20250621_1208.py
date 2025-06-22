# api/audio.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from models.responses import BasicAudioResponse, AdvancedAudioResponse, GenreResponse
from core.audio_analyzer import AudioAnalyzer
import tempfile
import os

router = APIRouter()
analyzer = AudioAnalyzer(44100)

@router.post("/analyze", response_model=BasicAudioResponse)
async def analyze_audio_basic(file: UploadFile = File(...)):
    """Basic audio analysis endpoint"""
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Load and analyze audio
        y, sr = analyzer.load_audio(tmp_file_path)
        analysis = analyzer.basic_analysis(y, sr)
        
        return BasicAudioResponse(
            filename=file.filename,
            status="success",
            **analysis
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@router.post("/analyze-advanced", response_model=AdvancedAudioResponse) 
async def analyze_audio_advanced(file: UploadFile = File(...)):
    """Advanced multi-algorithm audio analysis"""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        y, sr = analyzer.load_audio(tmp_file_path)
        analysis = analyzer.advanced_analysis(y, sr)
        
        return AdvancedAudioResponse(
            filename=file.filename,
            status="success",
            **analysis
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced analysis failed: {str(e)}")
    
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@router.post("/classify-genre", response_model=GenreResponse)
async def classify_genre_basic(file: UploadFile = File(...)):
    """Genre classification endpoint"""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        y, sr = analyzer.load_audio(tmp_file_path)
        analysis = analyzer.classify_genre(y, sr)
        
        return GenreResponse(
            filename=file.filename,
            status="success",
            **analysis
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Genre classification failed: {str(e)}")
    
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)