#!/usr/bin/env python3
"""
Test script to evaluate chunking strategies without modifying main code
"""

import time
import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_chunk_fast(chunk_data):
    """Fast analysis for testing different chunk sizes"""
    chunk_audio, chunk_id, sr = chunk_data
    start_time = time.time()
    
    try:
        # Simulate fast analysis (minimal librosa operations)
        duration = len(chunk_audio) / sr
        energy = float(np.mean(np.abs(chunk_audio)))
        
        # Simulate more realistic processing (similar to our current analysis)
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=chunk_audio, sr=sr)))
        spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=chunk_audio, sr=sr)))
        zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(chunk_audio)))
        
        # Simulate some key detection work (lightweight)
        chroma = librosa.feature.chroma_stft(y=chunk_audio, sr=sr, hop_length=2048, n_fft=1024)
        chroma_mean = np.mean(chroma, axis=1)
        
        processing_time = time.time() - start_time
        
        return {
            "chunk_id": chunk_id,
            "duration": duration,
            "energy": energy,
            "spectral_centroid": spectral_centroid,
            "processing_time": processing_time
        }
    except Exception as e:
        return {
            "chunk_id": chunk_id,
            "error": str(e),
            "processing_time": time.time() - start_time
        }

def test_chunking_strategy(audio_file, chunk_duration=10, max_workers=4):
    """Test different chunking strategies"""
    
    print(f"\n🧪 Testing Chunking Strategy:")
    print(f"   - Chunk Duration: {chunk_duration}s")
    print(f"   - Max Workers: {max_workers}")
    
    # Load audio
    print("📁 Loading audio...")
    y, sr = librosa.load(audio_file, sr=22050)
    total_duration = len(y) / sr
    
    print(f"   - File Duration: {total_duration:.1f}s")
    print(f"   - Sample Rate: {sr}Hz")
    
    # Create chunks
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(1 * sr)  # 1 second overlap
    
    chunks = []
    for i in range(0, len(y), chunk_samples - overlap_samples):
        chunk_start = i
        chunk_end = min(i + chunk_samples, len(y))
        chunk_audio = y[chunk_start:chunk_end]
        
        if len(chunk_audio) >= sr:  # At least 1 second
            chunks.append((chunk_audio, len(chunks), sr))
    
    print(f"   - Total Chunks: {len(chunks)}")
    
    # Test with ThreadPoolExecutor
    print(f"\n⚡ Testing ThreadPoolExecutor ({max_workers} workers):")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        thread_results = list(executor.map(analyze_chunk_fast, chunks))
    
    thread_time = time.time() - start_time
    thread_processing_times = [r.get("processing_time", 0) for r in thread_results if "processing_time" in r]
    
    print(f"   - Total Time: {thread_time:.2f}s")
    print(f"   - Avg Chunk Time: {np.mean(thread_processing_times):.2f}s")
    print(f"   - Realtime Factor: {total_duration / thread_time:.2f}x")
    
    # Test with ProcessPoolExecutor (if we want to try it)
    print(f"\n🚀 Testing ProcessPoolExecutor ({max_workers} workers):")
    start_time = time.time()
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            process_results = list(executor.map(analyze_chunk_fast, chunks))
        
        process_time = time.time() - start_time
        process_processing_times = [r.get("processing_time", 0) for r in process_results if "processing_time" in r]
        
        print(f"   - Total Time: {process_time:.2f}s")
        print(f"   - Avg Chunk Time: {np.mean(process_processing_times):.2f}s")
        print(f"   - Realtime Factor: {total_duration / process_time:.2f}x")
        
        # Compare
        speedup = thread_time / process_time if process_time > 0 else 1
        print(f"   - Speedup vs Threading: {speedup:.2f}x")
        
    except Exception as e:
        print(f"   - ProcessPool Error: {e}")
    
    return {
        "chunk_duration": chunk_duration,
        "max_workers": max_workers,
        "total_chunks": len(chunks),
        "file_duration": total_duration,
        "thread_time": thread_time,
        "thread_realtime_factor": total_duration / thread_time
    }

def compare_strategies():
    """Compare different chunking strategies"""
    
    # Test file (use larger file for realistic testing)
    test_file = "/app/large_test_5min.wav"
    
    print("🔬 Chunking Strategy Comparison")
    print("=" * 50)
    
    strategies = [
        (120, 3),  # Current strategy
        (60, 4),   # Medium chunks
        (30, 6),   # Smaller chunks  
        (10, 8),   # Aggressive chunking
        (5, 8),    # Very aggressive
    ]
    
    results = []
    for chunk_duration, max_workers in strategies:
        try:
            result = test_chunking_strategy(test_file, chunk_duration, max_workers)
            results.append(result)
        except Exception as e:
            print(f"❌ Strategy ({chunk_duration}s, {max_workers}w) failed: {e}")
    
    # Summary comparison
    print("\n📊 STRATEGY COMPARISON SUMMARY:")
    print("=" * 60)
    print(f"{'Chunk Size':<12} {'Workers':<8} {'Chunks':<8} {'Time':<8} {'Realtime':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['chunk_duration']:<12} {result['max_workers']:<8} {result['total_chunks']:<8} "
              f"{result['thread_time']:.1f}s{' ':<4} {result['thread_realtime_factor']:.2f}x")
    
    # Find best strategy
    if results:
        best = max(results, key=lambda x: x['thread_realtime_factor'])
        print(f"\n🏆 BEST STRATEGY: {best['chunk_duration']}s chunks, {best['max_workers']} workers")
        print(f"   → {best['thread_realtime_factor']:.2f}x realtime factor")

if __name__ == "__main__":
    compare_strategies()