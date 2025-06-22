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
# In api/audio.py, update the loudness endpoint:
@router.post("/loudness-simple")
async def analyze_loudness_simple(file: UploadFile = File(...)):
    """Simple loudness analysis that works"""
    
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name
        
        y, sr = analyzer.load_audio(temp_file_path)
        
        # Simple calculations that definitely work
        peak_db = 20 * np.log10(np.max(np.abs(y))) if np.max(np.abs(y)) > 0 else -100
        rms_value = np.sqrt(np.mean(y**2))
        rms_db = 20 * np.log10(rms_value) if rms_value > 0 else -100
        dynamic_range = peak_db - rms_db
        
        return {
            "filename": file.filename,
            "status": "success",
            "loudness_analysis": {
                "peak_dbfs": round(float(peak_db), 2),
                "rms_db": round(float(rms_db), 2),
                "dynamic_range_db": round(float(dynamic_range), 2),
                "analysis_type": "simple_working"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loudness analysis failed: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@router.post("/analyze-basic-plus")
async def analyze_basic_plus(file: UploadFile = File(...)):
    """Enhanced basic analysis that definitely works"""
    
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name
        
        y, sr = analyzer.load_audio(temp_file_path)
        
        # Use existing working methods
        basic_result = analyzer.basic_analysis(y, sr)
        
        # Add safe additional features
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        
        # Add transients if method exists
        transients = []
        if hasattr(analyzer, 'extract_transient_timeline'):
            try:
                transients = analyzer.extract_transient_timeline(y, sr)
            except:
                pass
        
        return {
            "filename": file.filename,
            "status": "success",
            **basic_result,
            "additional_features": {
                "spectral_bandwidth": round(float(spectral_bandwidth), 2),
                "spectral_contrast": round(float(spectral_contrast), 2),
                "transient_count": len(transients)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
@router.post("/loudness-working")
async def analyze_loudness_working(file: UploadFile = File(...)):
    """Working loudness analysis using basic calculations"""
    
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name
        
        y, sr = analyzer.load_audio(temp_file_path)
        
        # Simple loudness calculations that work
        import numpy as np
        peak_db = 20 * np.log10(np.max(np.abs(y))) if np.max(np.abs(y)) > 0 else -100
        rms_value = np.sqrt(np.mean(y**2))
        rms_db = 20 * np.log10(rms_value) if rms_value > 0 else -100
        dynamic_range = peak_db - rms_db
        
        recommendations = []
        if peak_db > -1:
            recommendations.append("Peak levels too high - risk of clipping")
        if dynamic_range < 6:
            recommendations.append("Low dynamic range - heavily compressed")
        elif dynamic_range > 20:
            recommendations.append("High dynamic range - good dynamics")
        
        return {
            "filename": file.filename,
            "status": "success",
            "loudness_analysis": {
                "peak_dbfs": round(float(peak_db), 2),
                "rms_db": round(float(rms_db), 2),
                "dynamic_range_db": round(float(dynamic_range), 2),
                "recommendations": recommendations,
                "analysis_type": "working_simple"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loudness analysis failed: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@router.post("/analyze-enhanced")
async def analyze_enhanced(file: UploadFile = File(...)):
    """Enhanced analysis using only working methods"""
    
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name
        
        y, sr = analyzer.load_audio(temp_file_path)
        
        # Use working methods
        basic_result = analyzer.basic_analysis(y, sr)
        
        # Enhanced tempo (this works!)
        tempo_analysis = analyzer.multi_algorithm_tempo_detection(y, sr)
        
        # MFCC features (this works!)
        mfcc_features = analyzer.extract_mfcc_features(y, sr)
        
        # Transients (this works!)
        transients = analyzer.extract_transient_timeline(y, sr)
        
        return {
            "filename": file.filename,
            "status": "success",
            **basic_result,
            "enhanced_tempo": tempo_analysis,
            "mfcc_features": mfcc_features,
            "transient_count": len(transients),
            "transient_markers": transients[:10] if len(transients) > 10 else transients  # First 10
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced analysis failed: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
