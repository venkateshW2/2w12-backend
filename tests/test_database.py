# NEW FILE: tests/test_database.py
import pytest
import json
from core.database_manager import SoundToolsDatabase

def test_database_connection():
    """Test Redis connection"""
    db = SoundToolsDatabase()
    stats = db.get_cache_stats()
    assert stats["status"] in ["healthy", "needs_optimization"]

def test_fingerprint_creation():
    """Test file fingerprinting"""
    db = SoundToolsDatabase()
    
    # Test data
    content1 = b"test audio data 1"
    content2 = b"test audio data 2"
    
    fingerprint1 = db.create_file_fingerprint(content1, "test1.wav")
    fingerprint2 = db.create_file_fingerprint(content2, "test2.wav")
    fingerprint3 = db.create_file_fingerprint(content1, "test1.wav")  # Same as 1
    
    assert fingerprint1 != fingerprint2  # Different content = different fingerprint
    assert fingerprint1 == fingerprint3  # Same content = same fingerprint
    assert len(fingerprint1) == 32  # MD5 hash length

def test_cache_operations():
    """Test cache store and retrieve"""
    db = SoundToolsDatabase()
    
    # Test data
    fingerprint = "test_fingerprint_123"
    analysis_data = {
        "key": "C major",
        "tempo": 120.0,
        "duration": 180.5
    }
    
    # Store
    success = db.cache_analysis_result(fingerprint, analysis_data, ttl=60)
    assert success == True
    
    # Retrieve
    retrieved = db.get_cached_analysis(fingerprint)
    assert retrieved is not None
    assert retrieved["key"] == "C major"
    assert retrieved["tempo"] == 120.0
    assert "cached_at" in retrieved
    assert "fingerprint" in retrieved

if __name__ == "__main__":
    # Run basic tests
    test_database_connection()
    test_fingerprint_creation()
    test_cache_operations()
    print("✅ All database tests passed!")