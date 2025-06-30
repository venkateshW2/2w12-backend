# NEW FILE: scripts/benchmark_performance.py
import requests
import time
import json
from pathlib import Path

def benchmark_enhanced_analysis():
    """Benchmark enhanced analysis performance"""
    
    API_BASE = "http://192.168.1.16:8001"
    
    # Test file (you'll need to create this)
    test_file = Path("test_assets/sample_audio.wav")
    
    if not test_file.exists():
        print("❌ Test file not found. Create test_assets/sample_audio.wav")
        return
    
    print("🚀 Performance Benchmark - Enhanced Analysis")
    print("=" * 50)
    
    # Test 1: First analysis (cache MISS)
    print("Test 1: First analysis (cache MISS)")
    start_time = time.time()
    
    with open(test_file, 'rb') as f:
        response = requests.post(
            f"{API_BASE}/api/audio/analyze-enhanced",
            files={'file': f}
        )
    
    first_analysis_time = time.time() - start_time
    result1 = response.json()
    
    print(f"⏱️  Time: {first_analysis_time:.2f}s")
    print(f"📊 Cache: {result1['analysis']['cache_status']}")
    print(f"🎵 Key: {result1['analysis']['key']} {result1['analysis']['mode']}")
    print(f"🥁 Tempo: {result1['analysis']['tempo']} BPM")
    print()
    
    # Test 2: Second analysis (cache HIT)
    print("Test 2: Second analysis (cache HIT)")
    start_time = time.time()
    
    with open(test_file, 'rb') as f:
        response = requests.post(
            f"{API_BASE}/api/audio/analyze-enhanced",
            files={'file': f}
        )
    
    second_analysis_time = time.time() - start_time
    result2 = response.json()
    
    print(f"⏱️  Time: {second_analysis_time:.2f}s")
    print(f"📊 Cache: {result2['analysis']['cache_status']}")
    
    # Performance improvement
    speedup = first_analysis_time / second_analysis_time
    print(f"🚀 Speedup: {speedup:.1f}x faster!")
    print()
    
    # Test 3: Cache statistics
    print("Test 3: Cache Statistics")
    stats_response = requests.get(f"{API_BASE}/api/stats/cache")
    stats = stats_response.json()
    
    performance = stats['cache_statistics']['performance']
    print(f"📈 Hit Rate: {performance['hit_rate_percent']}%")
    print(f"🎯 Hits: {performance['cache_hits']}")
    print(f"❌ Misses: {performance['cache_misses']}")
    print(f"💾 Stores: {performance['cache_stores']}")
    
    print("\n✅ Benchmark completed successfully!")

if __name__ == "__main__":
    benchmark_enhanced_analysis()