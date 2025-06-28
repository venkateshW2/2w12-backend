# NEW FILE: tests/test_enhanced_loader.py
import pytest
import numpy as np
from core.enhanced_audio_loader import EnhancedAudioLoader

def test_enhanced_loader_initialization():
    """Test enhanced loader initialization"""
    loader = EnhancedAudioLoader()
    assert loader.sample_rate == 22050
    assert loader.analysis_version == "v2.0_enhanced_librosa"
    assert loader.ml_models_loaded == False  # Week 1

def test_file_fingerprinting():
    """Test file fingerprinting functionality"""
    loader = EnhancedAudioLoader()
    
    # Create test audio data
    test_content = b"fake audio data for testing"
    filename = "test_audio.wav"
    
    fingerprint = loader.db.create_file_fingerprint(test_content, filename)
    assert len(fingerprint) == 32  # MD5 hash
    assert isinstance(fingerprint, str)

def test_enhanced_analysis_workflow():
    """Test complete enhanced analysis workflow"""
    # This would require actual audio file
    # For now, test the workflow structure
    loader = EnhancedAudioLoader()
    
    # Test that all required methods exist
    assert hasattr(loader, 'analyze_with_caching')
    assert hasattr(loader, '_perform_comprehensive_analysis')
    assert hasattr(loader, '_librosa_enhanced_analysis')
    assert hasattr(loader, '_enhanced_key_detection')
    assert hasattr(loader, '_enhanced_tempo_detection')

if __name__ == "__main__":
    test_enhanced_loader_initialization()
    test_file_fingerprinting()
    test_enhanced_analysis_workflow()
    print("✅ Enhanced loader tests passed!")