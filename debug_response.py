#!/usr/bin/env python3
# Debug the API response
import requests
import json
import tempfile
import numpy as np
import scipy.io.wavfile

def debug_api_response():
    """Debug what's actually in the API response"""
    
    # Create simple test audio
    sample_rate = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 note
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_file_path = tmp_file.name
        scipy.io.wavfile.write(tmp_file_path, sample_rate, (audio * 32767).astype(np.int16))
    
    try:
        with open(tmp_file_path, 'rb') as audio_file:
            files = {'file': ('test.wav', audio_file, 'audio/wav')}
            response = requests.post(
                'http://localhost:8001/api/audio/analyze-enhanced',
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Response received")
            print("📋 Top-level keys in response:")
            for key in sorted(result.keys()):
                value = result[key]
                if isinstance(value, dict):
                    print(f"   {key}: dict with {len(value)} keys")
                elif isinstance(value, list):
                    print(f"   {key}: list with {len(value)} items")
                else:
                    print(f"   {key}: {type(value).__name__} = {value}")
            
            # Check analysis keys
            if 'analysis' in result:
                analysis = result['analysis']
                print(f"\n📊 Analysis keys:")
                for key in sorted(analysis.keys()):
                    value = analysis[key]
                    if isinstance(value, dict):
                        print(f"   analysis.{key}: dict with {len(value)} keys")
                    elif isinstance(value, list):
                        print(f"   analysis.{key}: list with {len(value)} items")
                    else:
                        print(f"   analysis.{key}: {type(value).__name__} = {value}")
                
                # Check specifically for content_analysis
                if 'content_analysis' in analysis:
                    print(f"\n🎯 FOUND content_analysis!")
                    content = analysis['content_analysis']
                    print(f"   content_analysis keys: {list(content.keys())}")
                else:
                    print(f"\n❌ content_analysis NOT found in analysis")
            
            # Check if there are any errors
            if 'error' in result:
                print(f"❌ Error in response: {result['error']}")
                
            # Check architecture
            if 'architecture' in result.get('analysis', {}):
                print(f"🏗️ Architecture: {result['analysis']['architecture']}")
                
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    import os
    if os.path.exists(tmp_file_path):
        os.unlink(tmp_file_path)

if __name__ == "__main__":
    debug_api_response()