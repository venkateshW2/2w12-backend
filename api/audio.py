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
         **analysis  # Remove the status="success" line
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
@router.get("/test-methods")
async def test_methods():
    """Test if enhanced methods exist"""
    methods = []
    if hasattr(analyzer, 'extract_transient_timeline'):
        methods.append("extract_transient_timeline")
    if hasattr(analyzer, 'get_memory_usage'):
        methods.append("get_memory_usage")
    if hasattr(analyzer, 'extract_mfcc_features'):
        methods.append("extract_mfcc_features")
    
    return {
        "status": "success",
        "available_methods": methods,
        "analyzer_class": str(type(analyzer)),
        "all_methods": [m for m in dir(analyzer) if not m.startswith('_')]
    }
@router.get("/memory-status")
async def get_memory_status():
    """Monitor server memory usage"""
    try:
        return analyzer.get_memory_usage()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory status failed: {str(e)}")

@router.post("/transients")
async def extract_transients(file: UploadFile = File(...)):
    """Extract transient timeline for UI visualization"""
    
    temp_file_path = None
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name
        
        # Load and analyze
        y, sr = analyzer.load_audio(temp_file_path)
        transient_markers = analyzer.extract_transient_timeline(y, sr)
        
        # Clean up memory if method exists
        if hasattr(analyzer, 'memory_cleanup'):
            analyzer.memory_cleanup(y)
        
        return {
            "filename": file.filename,
            "status": "success",
            "transient_markers": transient_markers,
            "total_transients": len(transient_markers)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transient extraction failed: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@router.post("/mfcc")
async def extract_mfcc_features_endpoint(file: UploadFile = File(...)):
    """Extract MFCC features"""
    
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name
        
        y, sr = analyzer.load_audio(temp_file_path)
        mfcc_features = analyzer.extract_mfcc_features(y, sr)
        
        if hasattr(analyzer, 'memory_cleanup'):
            analyzer.memory_cleanup(y)
        
        return {
            "filename": file.filename,
            "status": "success",
            "mfcc_analysis": mfcc_features
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MFCC analysis failed: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
# Add this endpoint to your api/audio.py file

@router.post("/loudness")
async def analyze_loudness(file: UploadFile = File(...)):
    """Professional loudness analysis with EBU R128 compliance"""
    
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name
        
        y, sr = analyzer.load_audio(temp_file_path)
        loudness_analysis = analyzer.professional_loudness_analysis(y, sr)
        
        if hasattr(analyzer, 'memory_cleanup'):
            analyzer.memory_cleanup(y)
        
        return {
            "filename": file.filename,
            "status": "success",
            "loudness_analysis": loudness_analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loudness analysis failed: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path) 


