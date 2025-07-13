#!/usr/bin/env python3

import requests
import json
import time
import os

def test_performance():
    """Test the current performance with Madmom restored"""
    
    # Check what audio files we have
    audio_dir = "temp_audio"
    if os.path.exists(audio_dir):
        files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3'))]
        print(f"Available files: {files}")
        if files:
            test_file = files[0]
            test_path = os.path.join(audio_dir, test_file)
        else:
            print("❌ No audio files found")
            return
    else:
        print("❌ temp_audio directory not found")
        return
    
    print(f"🎵 Testing performance with: {test_file}")
    
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
            file_duration = result.get('file_info', {}).get('analyzed_duration', 0)
            madmom_count = result.get('analysis', {}).get('madmom_downbeat_count', 0)
            madmom_time = result.get('analysis', {}).get('madmom_processing_time', 0)
            
            realtime_factor = file_duration / processing_time if processing_time > 0 else 0
            
            print(f"\n🚀 PERFORMANCE RESULTS:")
            print(f"   📁 File: {test_file}")
            print(f"   ⏱️  Duration: {file_duration:.1f}s")
            print(f"   🕐 Processing: {processing_time:.2f}s")
            print(f"   ⚡ Realtime Factor: {realtime_factor:.2f}x")
            print(f"   🥁 Madmom Downbeats: {madmom_count}")
            print(f"   🔧 Madmom Time: {madmom_time:.2f}s")
            
            if realtime_factor >= 1.0:
                print(f"   ✅ FASTER THAN REALTIME! ({realtime_factor:.2f}x)")
            else:
                print(f"   ⚠️  Slower than realtime ({realtime_factor:.2f}x)")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_performance()