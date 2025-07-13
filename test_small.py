#!/usr/bin/env python3

import requests
import json
import time
import os

def test_small_file():
    """Test with the 38-second file"""
    
    test_file = "swiggy38sec.wav"
    test_path = f"temp_audio/{test_file}"
    
    if not os.path.exists(test_path):
        print(f"❌ File not found: {test_path}")
        return
    
    print(f"🎵 Testing small file: {test_file}")
    
    url = "http://localhost:8001/api/audio/analyze-enhanced"
    
    try:
        start_time = time.time()
        
        with open(test_path, 'rb') as f:
            files = {'file': (test_file, f, 'audio/wav')}
            response = requests.post(url, files=files, timeout=120)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract key metrics
            file_info = result.get('analysis', {}).get('file_info', {})
            file_duration = file_info.get('analyzed_duration', 0)
            madmom_count = result.get('analysis', {}).get('madmom_downbeat_count', 0)
            madmom_time = result.get('analysis', {}).get('madmom_processing_time', 0)
            key_detected = result.get('analysis', {}).get('ml_key', 'unknown')
            
            realtime_factor = file_duration / processing_time if processing_time > 0 and file_duration > 0 else 0
            
            print(f"\n🚀 SMALL FILE RESULTS:")
            print(f"   📁 File: {test_file}")
            print(f"   ⏱️  Duration: {file_duration:.1f}s")
            print(f"   🕐 Processing: {processing_time:.2f}s") 
            print(f"   ⚡ Realtime Factor: {realtime_factor:.2f}x")
            print(f"   🥁 Madmom Downbeats: {madmom_count}")
            print(f"   🔧 Madmom Time: {madmom_time:.2f}s")
            print(f"   🎵 Key Detected: {key_detected}")
            
            if realtime_factor >= 1.0:
                print(f"   ✅ FASTER THAN REALTIME! ({realtime_factor:.2f}x)")
            elif realtime_factor > 0:
                print(f"   ⚠️  Slower than realtime ({realtime_factor:.2f}x)")
            else:
                print(f"   ❌ Duration issue detected")
                
            # Show file info for debugging
            print(f"\n🔍 File Info Debug:")
            print(f"   File Info: {file_info}")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_small_file()