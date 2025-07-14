#!/usr/bin/env python3

import audioflux as af
import numpy as np

print("🔍 Testing AudioFlux audio loading...")

try:
    # Test loading the audio file
    audio_data, sr = af.read("temp_audio/swiggy38sec.wav")
    
    print(f"✅ AudioFlux loaded successfully:")
    print(f"   📊 Data shape: {audio_data.shape if hasattr(audio_data, 'shape') else len(audio_data)}")
    print(f"   📊 Data type: {audio_data.dtype if hasattr(audio_data, 'dtype') else type(audio_data)}")
    print(f"   📊 Sample rate: {sr}Hz")
    print(f"   📊 Duration: {len(audio_data) / sr:.1f}s")
    print(f"   📊 Data range: min={np.min(audio_data):.6f}, max={np.max(audio_data):.6f}")
    print(f"   📊 First 10 samples: {audio_data[:10]}")
    print(f"   📊 Middle 10 samples: {audio_data[len(audio_data)//2:len(audio_data)//2+10]}")
    print(f"   📊 Last 10 samples: {audio_data[-10:]}")
    
    # Test normalization
    max_val = np.max(np.abs(audio_data))
    print(f"   📊 Max absolute value: {max_val:.6f}")
    
    # Test chunking like in the processor
    target_width = 1920
    chunk_size = len(audio_data) // target_width
    print(f"   📊 Chunk size for {target_width}px: {chunk_size}")
    
    # Test first several chunks to find where audio starts
    for i in range(0, min(20, target_width)):
        start_idx = i * chunk_size
        chunk = audio_data[start_idx:start_idx+chunk_size]
        if len(chunk) > 0:
            peak = np.max(chunk)
            valley = np.min(chunk)
            if abs(peak) > 0.001 or abs(valley) > 0.001:  # Only show non-silent chunks
                print(f"   📊 Chunk {i}: peak={peak:.6f}, valley={valley:.6f} ← AUDIO CONTENT!")
        else:
            print(f"   📊 Chunk {i}: empty")
    
    if max_val > 0:
        normalized = audio_data / max_val
        print(f"   📊 After normalization: min={np.min(normalized):.6f}, max={np.max(normalized):.6f}")
    
except Exception as e:
    print(f"❌ AudioFlux loading failed: {e}")
    import traceback
    traceback.print_exc()